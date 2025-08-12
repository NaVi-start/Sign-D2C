import os
import time
import queue
import shutil
import torch
import pickle

import numpy as np
import pandas as pd

from batch import Batch
from torch import Tensor
from loss import RegLoss
from vocabulary import Vocabulary
from torchtext.data import Dataset
from data import make_data_iter, load_data
from torch.utils.tensorboard import SummaryWriter
from stage2_models.model import Model, build_model
from plot_videos import plot_video, alter_DTW_timing
from stage2_models.stage2_prediction import validate_on_data
from constants import TARGET_PAD, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from builders import build_optimizer, build_scheduler, build_gradient_clipper
from helpers import load_config, log_cfg, load_checkpoint, make_model_dir, make_logger, \
    set_seed, symlink_update, ConfigurationError, get_latest_checkpoint

class TrainManager_Diffusion:
    def __init__(self, model: Model, config: dict, src_vocab: Vocabulary, test=False):
        train_config = config["training"]
        model_dir = train_config["model_stage2_dir"]
        model_continue = train_config.get("continue", True)

        if not os.path.isdir(model_dir):
            model_continue = False
        if test:
            model_continue = True

        self.model_dir = make_model_dir(train_config["model_stage2_dir"],
                                        overwrite=train_config.get("overwrite", False),
                                        model_continue=model_continue)

        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir+"/tensorboard/")

        self.model = model
        self.src_vocab = src_vocab

        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self._log_parameters_list()
        self.target_pad = TARGET_PAD

        self.loss = RegLoss(cfg = config, target_pad=self.target_pad)

        self.normalization = "batch"

        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=self.model.diffusion.parameters())
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.ckpt_best_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 1))
        self.ckpt_queue = queue.Queue(maxsize=1)
        self.val_on_train = config["data"].get("val_on_train", False)
        self.eval_metric = train_config.get("eval_metric", "dtw").lower()
        if self.eval_metric not in ['bleu', 'chrf', "dtw"]:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'bleu', 'chrf', 'DTW'")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                       "eval_metric")

        if self.early_stopping_metric in ["loss","dtw"]:
            self.minimize_metric = True
        else:
            raise ConfigurationError("Invalid setting for 'early_stopping_metric', "
                                    "valid options: 'loss', 'dtw',.")

        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["Diffusion"]["Denoiser"]["hidden_size"])

        self.level = "word"
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = "sentence"
        self.eval_batch_size = train_config.get("eval_batch_size",self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type",self.batch_type)
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        self.max_output_length = train_config.get("max_output_length", None)

        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            self.loss.cuda()

        self.steps = 0

        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score

        if model_continue:
            ckpt = get_latest_checkpoint(model_dir)
            if ckpt is None:
                self.logger.info("Can't find checkpoint in directory %s", ckpt)
            else:
                self.logger.info("Continuing model from %s", ckpt)
                self.init_from_checkpoint(ckpt)
        self.skip_frames = config["data"].get("skip_frames", 1)

    def _log_parameters_list(self) -> None:
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.diffusion.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.diffusion.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def init_from_checkpoint(self, path: str) -> None:
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        self.model.diffusion.load_state_dict(model_checkpoint["model_state"])
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        if model_checkpoint["scheduler_state"] is not None and \
                self.scheduler is not None:
 
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        if self.use_cuda:
            self.model.cuda()

    def _save_checkpoint(self, type="every") -> None:
        model_path = "{}/{}_{}.ckpt".format(self.model_dir, self.steps, type)
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.diffusion.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if \
            self.scheduler is not None else None,
        }
        torch.save(state, model_path)
        if type == "best":
            if self.ckpt_best_queue.full():
                to_delete = self.ckpt_best_queue.get()  
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)
            self.ckpt_best_queue.put(model_path)
            best_path = "{}/best.ckpt".format(self.model_dir)
            try:
                symlink_update("{}_best.ckpt".format(self.steps), best_path)
            except OSError:
                torch.save(state, best_path)

        elif type == "every":
            if self.ckpt_queue.full():
                to_delete = self.ckpt_queue.get()  
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)
            self.ckpt_queue.put(model_path)
            every_path = "{}/every.ckpt".format(self.model_dir)
            try:
                symlink_update("{}_best.ckpt".format(self.steps), every_path)
            except OSError:
                torch.save(state, every_path)

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) -> None:
        train_iter = make_data_iter(train_data,
                                    batch_size=self.batch_size,
                                    batch_type=self.batch_type,
                                    train=True, shuffle=self.shuffle)
        val_step = 0
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.diffusion.train()
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            count = self.batch_multiplier - 1
            epoch_loss = 0

            for batch in iter(train_iter):
                self.model.diffusion.train()
                batch = Batch(torch_batch=batch,
                              pad_index=self.pad_index,
                              model=self.model)

                update = count == 0

                batch_loss, noise = self._train_batch(batch, update=update)
                self.tb_writer.add_scalar("train/train_batch_loss", batch_loss,self.steps)
                count = self.batch_multiplier if update else count
                count -= 1
                epoch_loss += batch_loss.detach().cpu().numpy()

                if self.scheduler is not None and self.scheduler_step_at == "step" and update:
                    self.scheduler.step()

                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - start_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f "
                        "Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1, self.steps, batch_loss,
                        elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens

                if self.steps % self.validation_freq == 0 and update:

                    valid_start_time = time.time()

                    valid_score, valid_loss, valid_references, valid_hypotheses, \
                        valid_inputs, all_dtw_scores, valid_file_paths = \
                        validate_on_data(
                            batch_size=self.eval_batch_size,
                            data=valid_data,
                            eval_metric=self.eval_metric,
                            model=self.model,
                            src_vocab=self.src_vocab,
                            max_output_length=self.max_output_length,
                            loss_function=self.loss,
                            batch_type=self.eval_batch_type,
                            type="val",
                        )

                    val_step += 1
                    self.tb_writer.add_scalar("valid/valid_loss", valid_loss, self.steps)
                    self.tb_writer.add_scalar("valid/valid_score", valid_score, self.steps)
                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.early_stopping_metric == "dtw":
                        ckpt_score = valid_score
                    else:
                        ckpt_score = valid_score
                    new_best = False
                    self.best = False
                    if self.is_best(ckpt_score):
                        self.best = True
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint(type="best")

                        display = list(range(0, len(valid_hypotheses), int(np.ceil(len(valid_hypotheses) / 13.15))))
                        self.produce_validation_video(
                            output_joints=valid_hypotheses,
                            inputs=valid_inputs,
                            references=valid_references,
                            model_dir=self.model_dir,
                            steps=self.steps,
                            display=display,
                            type="val_inf",
                            file_paths=valid_file_paths,
                        )

                    self._save_checkpoint(type="every")

                    if self.scheduler is not None and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score)

                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        eval_metric=self.eval_metric,
                        new_best=new_best, report_type="val",)

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch %3d, step %8d: Val DTW Score: %6.2f, '
                        'loss: %8.4f,  duration: %.4fs',
                            epoch_no+1, self.steps, valid_score,
                            valid_loss, valid_duration)

                if self.stop:
                    break
            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                     self.learning_rate_min)
                break

            self.logger.info('Epoch %3d: total training loss %.5f', epoch_no+1,
                             epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no+1)
        self.logger.info('Best validation result at step %8d: %6.2f %s.',
                         self.best_ckpt_iteration, self.best_ckpt_score,
                         self.early_stopping_metric)
        self.tb_writer.close()  

    def _train_batch(self, batch: Batch, update: bool = True) -> Tensor:

        output, encoder_output, preds_x, mask_list = self.model(is_train=True, 
                                                                trg_input=batch.trg_input[:, :, :150], 
                                                                trg_mask=batch.trg_mask, 
                                                                lengths=batch.trg_lengths)
        
        loss_mask = (batch.trg_input != self.target_pad)[:, :, :1].any(dim=2, keepdim=True).repeat(1,1,512)
        batch_loss = self.loss(encoder_output, preds_x, loss_mask)  

        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

        norm_batch_loss = batch_loss / normalizer
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier
        norm_batch_multiply.backward()

        if self.clip_grad_fun is not None:
            self.clip_grad_fun(params=self.model.diffusion.parameters())
        if update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps += 1
        self.total_tokens += batch.ntokens

        return norm_batch_loss, None
    
    def produce_validation_video(self, output_joints, inputs, references, display, model_dir, type, steps="", file_paths=None, dtw_file=None):
        if type != "test":
            dir_name = model_dir + "/videos/Step_{}/".format(steps)
            if not os.path.exists(model_dir + "/videos/"):
                os.mkdir(model_dir + "/videos/")
        elif type == "test":
            dir_name = model_dir + "/test_videos/"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        for i in display:
            seq = output_joints[i]
            ref_seq = references[i]
            input = inputs[i]
            gloss_label = input[0]
            if input[1] is not "</s>":
                gloss_label += "_" + input[1]
            if input[2] is not "</s>":
                gloss_label += "_" + input[2]
            timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)
            video_ext = "{}_{}.mp4".format(gloss_label, "{0:.2f}".format(float(dtw_score)).replace(".", "_"))
            if file_paths is not None:
                sequence_ID = file_paths[i]
            else:
                sequence_ID = None
            print(sequence_ID + '    dtw: ' + '{0:.2f}'.format(float(dtw_score)))
            if dtw_file != None:
                dtw_file.writelines(sequence_ID + ' ' + '{0:.2f}'.format(float(dtw_score)) + '\n')
            plot_video(joints=timing_hyp_seq,
                       file_path=dir_name,
                       video_name=video_ext,
                       references=ref_seq_count,
                       skip_frames=self.skip_frames,
                       sequence_ID=sequence_ID)

    def save_skels(self, output_joints, display, model_dir, type, file_paths=None):
        picklefile = open(model_dir + "/phoenix14t.skels.%s" % type, "wb")
        csvIn = pd.read_csv(model_dir + "/csv/%s_phoenix2014t.csv" % type, sep='|',encoding='utf-8')
        pickle_list = []
        for i in display:
            name = file_paths[i]
            video = name[len(os.path.dirname(name))+1:]
            signer = csvIn[csvIn['id']==video]['signer'].item()
            gloss = csvIn[csvIn['id']==video]['annotation'].item()
            text = csvIn[csvIn['id']==video]['translation'].item()
            seq = output_joints[i].cpu()[:,:-1]
            sign = torch.tensor(seq, dtype = torch.float32)
            dict_num = {'name': name, 'signer': signer, 'gloss': gloss, 'text': text, 'sign': sign}
            pickle_list.append(dict_num)
        pickle.dump(pickle_list, picklefile)
        print("The skeletons of %s date have been save." % type)

    def _add_report(self, valid_score: float, valid_loss: float, eval_metric: str, new_best: bool = False, report_type: str = "val") -> None:
        current_lr = -1
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        if current_lr < self.learning_rate_min:
            self.stop = True
        if report_type == "val":
            with open(self.valid_report_file, 'a') as opened_file:
                opened_file.write(
                    "Steps: {} Loss: {:.5f}| DTW: {:.3f}|"
                    " LR: {:.6f} {}\n".format(
                        self.steps, valid_loss, valid_score,
                        current_lr, "*" if new_best else ""))

def train_stage2(cfg_file: str, ckpt=None) -> None:

    cfg = load_config(cfg_file)
    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)

    model = build_model(cfg)

    pre_model_dir = cfg["training"]["model_stage1_dir"]
    ckpt_pre_model = get_latest_checkpoint(pre_model_dir, post_fix="_best")
    if ckpt_pre_model is None:
        raise FileNotFoundError("No checkpoint found in directory {}."
                                .format(pre_model_dir))
    pre_model_checkpoint = load_checkpoint(ckpt_pre_model, use_cuda=True)
    model.vae.load_state_dict(pre_model_checkpoint["model_state"])

    trainer = TrainManager_Diffusion(model=model, config=cfg, src_vocab=src_vocab)

    shutil.copy2(cfg_file, trainer.model_dir+"/config.yaml")
    log_cfg(cfg, trainer.logger)

    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

def test_stage2(cfg_file, ckpt: str) -> None:

    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_stage2_dir"]

    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get("eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", True)
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)
    data_to_predict = {"dev": dev_data, "test": test_data}

    model = build_model(cfg)

    pre_model_dir = cfg["training"]["model_stage1_dir"]
    ckpt_pre_model = get_latest_checkpoint(pre_model_dir, post_fix="_best")
    if ckpt_pre_model is None:
        raise FileNotFoundError("No checkpoint found in directory {}."
                                .format(pre_model_dir))
    pre_model_checkpoint = load_checkpoint(ckpt_pre_model, use_cuda=True)
    model.vae.load_state_dict(pre_model_checkpoint["model_state"])

    model_dir = cfg["training"]["model_stage2_dir"]
    ckpt_model = get_latest_checkpoint(model_dir, post_fix="_best")
    if ckpt_model is None:
        raise FileNotFoundError("No checkpoint found in directory {}."
                                .format(model_dir))
    model_checkpoint = load_checkpoint(ckpt_model, use_cuda=True)
    model.diffusion.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    trainer = TrainManager_Diffusion(model=model, config=cfg, src_vocab=src_vocab, test=True)

    for data_set_name, data_set in data_to_predict.items():
        score, loss, references, hypotheses, inputs, all_dtw_scores, file_paths = \
            validate_on_data(
                model=model,
                data=data_set,
                batch_size=batch_size,
                max_output_length=max_output_length,
                eval_metric=eval_metric,
                src_vocab=src_vocab,
                loss_function=None,
                batch_type=batch_type,
                type="val" if not data_set_name is "train" else "train_inf"
            )

        if not os.path.exists(os.path.join(model_dir, 'test_videos')):
            os.mkdir(os.path.join(model_dir, 'test_videos'))
            
        dtw_file = open(os.path.join(model_dir, 'test_videos', data_set_name+'_dtw.txt'),'w')
        dtw_file.writelines('DTW Score of %s set: %.3f\n' %(data_set_name, score))

        print('DTW Score of %s set: %.3f' %(data_set_name, score))

        display = list(range(len(hypotheses)))

        trainer.save_skels(output_joints=hypotheses, display=display, model_dir=model_dir, type=data_set_name, file_paths=file_paths)

        '''
        trainer.produce_validation_video(
            output_joints=hypotheses,
            inputs=inputs,
            references=references,
            model_dir=model_dir,
            display=display,
            type="test",
            file_paths=file_paths,
            dtw_file=dtw_file,
        )
        '''

import torch

from torch import nn
from helpers import subsequent_mask
from stage1_models.PoseVAE import PoseVAE
from stage2_models.diffusion import GaussianDiffusion

class Model(nn.Module):
    def __init__(self, cfg: dict, VAE: PoseVAE, Diffusion: GaussianDiffusion):
        super().__init__()

        self.vae = VAE
        self.diffusion = Diffusion
        self.mask_ratio = cfg["training"].get("mask_ratio", 0.3)
        self.obs = cfg["training"].get("observation", 20)
        self.preds = cfg["training"].get("prediction", 10)

    def forward(self, is_train, trg_input, trg_mask, lengths):

        encoder_output, c, mask_list = self.encode(is_train, trg_input, trg_mask, lengths)
        preds_x = self.diffusion(x=encoder_output, c=c, mask=trg_mask, is_train=is_train)

        output = self.decode(preds_x, trg_mask)

        return output, encoder_output, preds_x, mask_list

    def encode(self, is_train, trg_input, trg_mask, lengths):
        padding_mask = trg_mask
        sub_mask = subsequent_mask(trg_input.size(1)).type_as(trg_mask)
        self.vae.eval()
        with torch.no_grad():
            encoder_output = self.vae.encode(trg_input, sub_mask, padding_mask)
            mask, mask_list = self.random_mask(input=trg_input, lengths=lengths, mask_ratio=self.mask_ratio)
            c = encoder_output * mask
            if not is_train :
                video_mask = self.board_mask(input=trg_input, lengths=lengths, obs=self.obs, preds=self.preds)
                trg_input = trg_input * video_mask
                c = self.vae.encode(trg_input, sub_mask, padding_mask)
        return encoder_output, c, mask_list

    def decode(self, x, trg_mask):
        self.vae.eval()
        padding_mask = trg_mask
        sub_mask = subsequent_mask(x.size(1)).type_as(trg_mask)
        with torch.no_grad():
            output = self.vae.decode(x, sub_mask, padding_mask)
        return output

    def random_mask(self, input, lengths, mask_ratio=0.3):
        mask_list = []
        true_ratio = 1 - mask_ratio 
        batch_size, seq_len, _ = input.size()
        feature_dim = 512
        mask = torch.zeros(batch_size, seq_len, feature_dim).bool().to(input.device)

        for i, length in enumerate(lengths):
            num_masked_frames = int(length * true_ratio)
            masked_indices = torch.randperm(length)[:num_masked_frames]
            mask[i, masked_indices, :] = True
            mask_list.append(masked_indices)
        return mask, mask_list

    def board_mask(self, input, lengths, obs, preds):
        batch_size, seq_len, feature_dim = input.size()
        mask = torch.zeros(batch_size, seq_len, feature_dim).bool().to(input.device)
        for i, length in enumerate(lengths):
            pos = 0
            while pos < length:
                end_pos = min(pos + obs, length)
                mask[i, pos:end_pos, :] = True
                pos += obs + preds
        return mask

def build_model(cfg: dict):

    full_cfg = cfg
    cfg = cfg["model"]

    pre_model = PoseVAE(args=cfg, freeze=True)
    diffusion = GaussianDiffusion(args=cfg)

    model = Model(cfg=full_cfg, VAE=pre_model, Diffusion=diffusion)

    return model

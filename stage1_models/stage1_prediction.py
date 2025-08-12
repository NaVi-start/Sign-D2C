import torch
import numpy as np

from batch import Batch
from constants import PAD_TOKEN
from data import make_data_iter
from stage1_models import PoseVAE
from helpers import calculate_dtw
from torchtext.data import Dataset
from vocabulary import Vocabulary

def validate_on_data(model: PoseVAE,
                     data: Dataset,
                     batch_size: int,
                     max_output_length: int,
                     eval_metric: str,
                     src_vocab: Vocabulary,
                     loss_function: torch.nn.Module = None,
                     batch_type: str = "sentence",
                     type = "val",
                     BT_model = None):

    valid_iter = make_data_iter(dataset=data, 
                                batch_size=batch_size, 
                                batch_type=batch_type,
                                shuffle=True, train=False)

    pad_index = src_vocab.stoi[PAD_TOKEN]
    model.eval()

    with torch.no_grad():
        valid_hypotheses = []
        valid_references = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []

        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        batches = 0
        for valid_batch in iter(valid_iter):
            batch = Batch(torch_batch=valid_batch,
                          pad_index=pad_index,
                          model=model)

            targets = batch.trg_input
            output = model(batch.trg_input[:, :, :150], batch.trg_mask)
            
            if loss_function != None:
                batch_loss = loss_function(output, targets[:, :, :150])
                valid_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs
            
            output = torch.cat((output, batch.trg_input[:, :, 150:]), dim=-1)

            valid_references.extend(targets)
            valid_hypotheses.extend(output)
            file_paths.extend(batch.file_paths)

            valid_inputs.extend([[src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))] for i in
                                 range(len(batch.src))])

            dtw_score = calculate_dtw(targets, output)
            all_dtw_scores.extend(dtw_score)
            batches += 1

        current_valid_score = np.mean(all_dtw_scores)

    return current_valid_score, valid_loss, valid_references, valid_hypotheses, \
           valid_inputs, all_dtw_scores, file_paths

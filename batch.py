# coding: utf-8

"""
Implementation of a mini-batch.
"""

import torch
import torch.nn.functional as F

from constants import TARGET_PAD

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, model):

        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths = torch_batch.src
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        self.file_paths = torch_batch.file_paths
        self.use_cuda = True
        self.target_pad = TARGET_PAD

        if hasattr(torch_batch, "trg"):
            trg = torch_batch.trg
            trg_lengths = torch_batch.lengths
            self.trg_input = trg.clone()
            self.trg_lengths = trg_lengths
            self.trg = trg.clone()
            trg_mask = (self.trg_input != self.target_pad).unsqueeze(1)
            pad_amount = self.trg_input.shape[1] - self.trg_input.shape[2]
            self.trg_mask = (F.pad(input=trg_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if self.use_cuda:
            self._make_cuda()

    # If using Cuda
    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()


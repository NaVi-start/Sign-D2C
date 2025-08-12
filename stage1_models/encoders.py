import torch

from torch import nn
from transformer_layers import TransformerEncoderLayer, PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, hidden_size, ff_size, num_heads, num_layers, dropout):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self._output_size = hidden_size

    def forward(self, x, mask = None, padding_mask = None):

        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask, padding_mask)

        return self.layer_norm(x)

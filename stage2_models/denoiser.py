import torch

from torch import nn
from helpers import subsequent_mask
from transformer_layers import TransformerDecoderLayer, PositionalEncoding, SinusoidalPositionEmbeddings


class Denoiser(nn.Module):
    def __init__(self, num_layers, num_heads, embedding_dim, hidden_size, ff_size, dropout):
        super(Denoiser, self).__init__()

        self.layers = nn.ModuleList([TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout) for _ in range(num_layers)])

        self.pe = PositionalEncoding(hidden_size,mask_count=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size*2),
            nn.GELU(),
            nn.Linear(hidden_size*2, hidden_size),
        )

        self.output_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, c, t, mask):

        assert mask is not None, "trg_mask required for Transformer"
        time_embed = self.time_mlp(t)[:, None, :].repeat(1, x.shape[1], 1)
        condition = c + time_embed
        condition = self.dropout(condition)

        x = self.pe(x)
        x = self.dropout(x)

        padding_mask = mask
        sub_mask = subsequent_mask(x.size(1)).type_as(mask)

        for layer in self.layers:
            x = layer(x=x, c=condition, mask=sub_mask, padding_mask=padding_mask)

        x = self.layer_norm(x)
        output = self.output_layer(x)

        return output

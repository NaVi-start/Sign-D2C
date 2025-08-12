import math
import torch

from torch import nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 size: int = 0,
                 ff_size: int = 0,
                 num_heads: int = 0,
                 dropout: float = 0.1):

        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, mask, padding_mask):
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask=mask, padding_mask=padding_mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 size: int = 0,
                 ff_size: int = 0,
                 num_heads: int = 0,
                 dropout: float = 0.1):

        super(TransformerDecoderLayer, self).__init__()

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.trg_trg_att = MultiHeadedAttention(num_heads, size,
                                                dropout=dropout)
        
        self.src_trg_att = MultiHeadedAttention(num_heads, size,
                                                dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, c, mask, padding_mask):

        h1 = self.x_layer_norm(x)
        h1 = self.trg_trg_att(h1, h1, h1, mask=mask, padding_mask=padding_mask)
        h1 = self.dropout(h1) + x

        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(c, c, h1_norm, mask=mask, padding_mask=padding_mask)
        o = self.feed_forward(self.dropout(h2) + h1)
        
        return o

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, k, v, q, mask = None, padding_mask = None):

        batch_size = k.size(0)
        num_heads = self.num_heads

        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        q = q / math.sqrt(self.head_size)

        scores = torch.matmul(q, k.transpose(2, 3))

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attention = self.softmax(scores)
        attention = self.dropout(attention)

        if padding_mask is not None:
            attention = attention.masked_fill(~padding_mask, 0.0)

        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size)

        output = self.output_layer(context)

        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, ff_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x

class PositionalEncoding(nn.Module):
    def __init__(self,
                 size: int = 0,
                 max_len: int = 200000, 
                 mask_count=False):

        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 2, dtype=torch.float) *
                              -(math.log(10000.0) / size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = size
        self.mask_count = mask_count

    def forward(self, emb):
        return emb + self.pe[:, :emb.size(1)]

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

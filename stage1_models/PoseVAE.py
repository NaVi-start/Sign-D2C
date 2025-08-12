from torch import nn
from helpers import freeze_params, subsequent_mask
from stage1_models.encoders import Encoder
from transformer_layers import PositionalEncoding

class PoseVAE(nn.Module):
    def __init__(self, args, freeze=False):
        super().__init__()

        trg_size = args.get("trg_size")
        embedding_dim = args["PoseVAE"].get("embedding_dim")
        num_heads = args["PoseVAE"].get("num_heads")
        num_layers = args["PoseVAE"].get("num_layers")
        dropout = args["PoseVAE"].get("dropout")
        hidden_size = args["PoseVAE"].get("hidden_size")
        ff_size = args["PoseVAE"].get("ff_size")

        self.pose_emb = nn.Linear(trg_size, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(hidden_size, ff_size=ff_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        self.decoder = Encoder(hidden_size, ff_size=ff_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout)

        self.output_layer = nn.Linear(hidden_size, trg_size)

        if freeze:
            freeze_params(self)
    
    def encode(self, trg, trg_mask = None, padding_mask = None):
        x = self.pose_emb(trg)
        x = self.pe(x)
        x = self.dropout(x)
        x = self.encoder(x, trg_mask, padding_mask)
        return x
    
    def decode(self, x, trg_mask = None, padding_mask = None):
        x = self.pe(x)
        x = self.dropout(x)
        x = self.decoder(x, trg_mask, padding_mask)
        x = self.output_layer(x)
        return x
        
    def forward(self, trg, trg_mask = None):
        
        padding_mask = trg_mask
        trg_mask = subsequent_mask(trg.size(1)).type_as(trg_mask)

        x = self.encode(trg, trg_mask, padding_mask)
        x = self.decode(x, trg_mask, padding_mask)

        return x

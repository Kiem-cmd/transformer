import torch 
import torch.nn as nn 
from encoder_layer import Encoder_Layer 
from position_embedding import PositionEmbedding
class Encoder(nn.Module):
    def __init__(self,n_layer,src_vocab_size,max_len,d_model,n_heads,ff_dim,dropout = 0.5):
        super().__init__()

        self.embed = nn.Embedding(src_vocab_size,d_model)
        self.pe = PositionEmbedding(max_len,d_model)
        self.layers = nn.ModuleList([
            Encoder_Layer(d_model,n_heads,ff_dim,dropout) for _ in range(n_layer)
        ])
    def forward(self,x,mask = None):
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x,score = layer(x)
        return x,score
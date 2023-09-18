import torch 
import torch.nn as nn 
from encoder_layer import Encoder_Layer 

class Encoder(nn.Module):
    def __init__(self,n_layer,src_vocab_size,max_len,embed_dim,d_model,n_heads,ff_dim,dropout = 0.5):
        super().__init__()

        self.embed = Embedding(src_vocab_size,embed_dim)
        self.pe = PositionEmbedding(max_len,embed_dim)
        self.layers = nn.ModuleList([
            Encoder_Layer(embed_dim,d_model,n_heads,ff_dim,dropout) for _ in range(n_layer)
        ])
    def forward(self,x,mask):
        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.layers:
            x,score = layer(x)
        return x,score
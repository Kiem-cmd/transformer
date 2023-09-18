import torch 
import torch.nn as nn
from attention import MultiHeadAttention
from Feed_Forward import FeedForward

class Encoder_Layer(nn.Module):
    def __init__(self,embed_dim,d_model,n_heads,ff_dim,dropout = 0.5):
        """ 
        """ 

        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model) 
        self.attention =  MultiHeadAttention(embed_dim,d_model,n_heads,dropout)
        self.ffn = FeedForward(d_model,ff_dim,dropout) 

    def forward(self,x):
        hidden_state = self.layernorm1(x)
        x = x + self.attention(hidden_state) 
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return x 
import torch 
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward

class Encoder_Layer(nn.Module):
    def __init__(self,d_model,n_heads,ff_dim,dropout = 0.5):
        """ 
        """ 

        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model) 
        self.attention =  MultiHeadAttention(d_model,n_heads,dropout)
        self.ffn = FeedForward(d_model,ff_dim,dropout) 
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x_ = self.layer_norm1(x)
        x_,score = self.attention(x_,x_,x_,mask = None) 
        x = x + x_ 
        x = self.layer_norm2(x)
        x = self.ffn(x)
        return x,score
import torch 
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward

class Encoder_Layer(nn.Module):
    def __init__(self,d_model,n_heads,ff_dim,dropout = 0.5):
        """ 
        Param:
        -----------------------------------------------------
        d_model: int 
        n_head:  int 
        ff_dim:  int 
        """ 

        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model) 
        self.attention =  MultiHeadAttention(d_model,n_heads,dropout)
        self.ffn = FeedForward(d_model,ff_dim,dropout) 
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,src_mask):
        x_ ,score = self.attention(x,x,x,src_mask) 
        x = self.layer_norm1(x + self.dropout(x_))
        x_ = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(x_))
        return x,score
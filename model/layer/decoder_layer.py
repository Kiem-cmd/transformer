import torch
import torch.nn as nn 
from attention import MultiHeadAttention 
from feed_forward import FeedForward 


class Decoder_Layer(nn.Module):
    def __init__(self,n_heads,d_model,ff_dim,dropout = 0.5):
        """
        Param: 
        --------------------------------------

        """
        super().__init__() 
        self.mask_attention = MultiHeadAttention(n_heads,d_model,dropout) 
        self.layer_norm1 = nn.LayerNorm(d_model) 

        self.attention = MultiHeadAttention(n_heads,d_model,dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = FeedForward(d_model,ff_dim) 
        self.layer_norm3 = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, src, tgt, src_mask, tgt_mask):
        """ 
        Param: 
        -------------------------------------------
        
        """
        tgt_, _ = self.mask_attention(tgt,tgt,tgt,tgt_mask) 
        tgt = self.layer_norm1(tgt + self.dropout(tgt_))

        tgt_,score = self.attention(tgt,src,src,src_mask) 
        tgt = self.layer_norm2(tgt + self.dropout(tgt_))

        tgt_ = self.ffn(tgt) 
        tgt = self.layer_norm3(tgt + self.dropout(tgt_))

        return tgt,score 

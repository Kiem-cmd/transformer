import torch
import torch.nn as nn 
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout):
        """
        Param:
        ----------------------------------------
        d_model: int
                            --> head_dim = d_model // n_heads
        n_heads: int 

        dropout: dropout rate 
        -----------------------------------------

        """
        super().__init__()
        self.d_model = d_model 
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) 

        assert d_model % n_heads == 0,"d_model is not divisible by num_head"

        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)

    def scaled_dot_attention(self,q,k,v,mask = None,dropout = None):
        """
        Param:
        ---------------------------------------------
        Q: Query - shape(B,num_heads,max_len,head_dim)
        K: Key   - shape(B,num_heads,max_len,head_dim)
        V: Value - shape(B,num_heads,max_len,head_dim) 

        masking: được sử dụng để xử lí mask
        dropout: dropout rate 
        ---------------------------------------------
        Return: 
        ----------------------------------------------
        Score = softmax (Q.K^T/sqrt(dk)).V 
        Attention Distribution 

        """
        d_k = q.shape[-1]
        score = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        p_attn  = score.softmax(dim = -1)
        if dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn,v),p_attn

    def forward(self, q, k, v,mask = None):
        q = self.w_q(q)    ## (B,max_len,d_model)
        k = self.w_k(k)    ## (B,max_len,d_model)
        v = self.w_v(v)    ## (B,max_len,d_model)
        
        q = q.view(q.shape[0], q.shape[1], self.n_heads,self.d_k).transpose(1,2)    ## (B,n_heads,max_len,d_k(hidden_dim))
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.d_k).transpose(1,2)   ## -----------------------------------
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.d_k).transpose(1,2)   ## ----------------------------------


        x,attn_score = self.scaled_dot_attention(q,k,v,mask,self.dropout)
        x = self.dropout(x)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        return self.w_o(x),attn_score


import torch 
import torch.nn as nn 

class PositionEmbedding(nn.Module):
    def __init__(self,max_len,embed_dim):
        super().__init__()
        self.encoding = torch.zeros(max_len,embed_dim)
        pos = torch.arange(max_len).view(-1,1)
        index = torch.arange(0,embed_dim,step = 2)

        self.encoding[:,::2] = torch.sin(pos/10000**(index/embed_dim))
        self.encoding[:,1::2] = torch.cos(pos/10000**(index/embed_dim))

    def forward(self,x):
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len,:]
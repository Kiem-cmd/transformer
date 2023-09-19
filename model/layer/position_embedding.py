import torch 
import torch.nn as nn 

class PositionEmbedding(nn.Module):
    def __init__(self,max_len,d_model):
        super().__init__()
        self.encoding = torch.zeros(max_len,d_model)
        pos = torch.arange(max_len).view(-1,1)
        index = torch.arange(0,d_model,step = 2)

        self.encoding[:,::2] = torch.sin(pos/10000**(index/d_model))
        self.encoding[:,1::2] = torch.cos(pos/10000**(index/d_model))

    def forward(self,x):
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len,:]
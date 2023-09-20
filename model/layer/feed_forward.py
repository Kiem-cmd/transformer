import torch 
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self,d_model,ff_dim,dropout = 0.5):

        """ 
        d_model : int 
        ff_dim  : int 
        """ 

        super().__init__() 
        
        self.linear = nn.Linear(d_model,ff_dim)
        self.linear_2 = nn.Linear(ff_dim,d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.linear_2(x) 
        x = self.dropout(x)
        return x
 
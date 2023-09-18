import torch 
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self,hidden_dim,ff_dim,dropout = 0.5):

        """ 
        """ 

        super().__init__() 
        
        self.linear = nn.Linear(hidden_dim,ff_dim)
        self.linear_2 = nn.Linear(ff_dim,hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.linear_2(x) 
        x = self.dropout(x)
        
        return x

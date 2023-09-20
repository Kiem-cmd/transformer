import torch 
import torch.nn as nn 
from decoder_layer import Decoder_Layer



class Decoder(nn.Module):
    def __init__(self,n_layers,tgt_vocab_size,max_len, n_heads,d_model,ff_dim,dropout = 0.5):
        """ 
        Param: 
        ---------------------------------------

        """
        
        super().__init__()
        self.embedding = nn.Embedding(tgt_vocab_size,hidden_dim) 
        self.encoding = PositionEmbedding(max_len,d_model)
        self.layers = nn.ModuleList([ 
            Decoder_Layer() for _ in range(n_layers) 
        ])
        self.linear = nn.Linear(d_model,tgt_vocab_size)
    def forward(self,tgt,src,tgt_mask,src_mask):
        """ 
        Param: 
        ----------------------------------------

        
        """
        tgt = self.embedding(tgt) 
        tgt = self.encoding(tgt) 

        for layer in self.layers:
            tgt,score = layer(src,tgt,src_mask,tgt_mask) 
        
        output = self.linear(tgt) 

        return output,score

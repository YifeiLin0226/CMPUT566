import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.lineark = nn.Linear(d_model, heads * self.d_k)
        self.linearq = nn.Linear(d_model, heads * self.d_k)
        self.linearv = nn.Linear(d_model, heads * self.d_k)
        self.final_linear = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim = 1)
        self.output = nn.Linear(d_model, d_model)

    def multihead_transform(self, x:torch.Tensor):
        shape = x.shape[:-1]
        x = x.view(*shape, self.heads, self.d_k)
        return x
    
    def forward(self, x):
        key = self.multihead_transform(self.lineark(x))
        query = self.multihead_transform(self.linearq(x))
        value = self.multihead_transform(self.linearv(x))
        
        score = torch.einsum('bihd,bjhd -> bijh', query, key) 
        score = self.softmax(score / math.sqrt(self.d_k))

        x = torch.einsum('bijh, bjhd -> bihd', score, value)
        x = x.reshape(*x.shape[:-2], -1)
        return self.final_linear(x)


        



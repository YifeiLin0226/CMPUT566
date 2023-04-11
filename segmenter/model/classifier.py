import torch
from torch import nn

class ClassificationHead(nn.Module):

    def __init__(self, d_model, n_cls):
        super().__init__()
        self.linear1 = nn.Linear(d_model, n_cls)
    
    def forward(self, x):
        return self.linear1(x)

import torch
import torch.nn.functional as F
from torch import nn


class FFN(nn.Module):

    def __init__(self, d_model, d_ff, p_drop = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_drop)

    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return self.dropout(x)



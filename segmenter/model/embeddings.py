import torch
from torch import nn

class PatchEmbeddings(nn.Module):

    def __init__(self, d_model, patch_size, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride = patch_size)
    
    def forward(self, x):
        x = self.conv(x)
        bs, c, h, w = x.shape

        x = x.view(bs, c, h * w)
        x = x.permute(0, 2, 1)
        # shape (batch_size, seq_len, d_model)
        return x

class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model, p_drop = 0.1, max_len = 5000):
        super().__init__()
        self.encodings = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(p_drop)
    
    def forward(self, x):
        positional_encodings = self.encodings[:, :x.shape[1]]
        x = x + positional_encodings
        x = self.dropout(x)
        return x


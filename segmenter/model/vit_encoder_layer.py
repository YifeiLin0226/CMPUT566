import torch
from torch import nn

from .mha import MultiHeadAttention
from .ffn import FFN

class ViTEncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
        self.d_model = d_model
        self.layernorm1 = nn.LayerNorm([d_model])
        self.layernorm2 = nn.LayerNorm([d_model])
    
    def forward(self, x):
        z = self.layernorm1(x)
        x = self.attn(z) + x
        z = self.layernorm2(x)
        x = self.ffn(z) + x
        return x


import math 

import torch
from torch import nn

from .vit_encoder_layer import ViTEncoderLayer


class MaskTransformer(nn.Module):

    def __init__(self, num_classes, d_model, d_ff, num_layers, num_heads):
        super().__init__()
        self.num_classes = num_classes
        self.cls_tokens = nn.Parameter(torch.randn(1, num_classes, d_model))
        self.layers = nn.ModuleList([ViTEncoderLayer(num_heads, d_model, d_ff) for _ in range(num_layers)])
    
    def forward(self, x):
        cls_tokens = self.cls_tokens.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_tokens, x], dim = 1)
        for layer in self.layers:
            x = layer(x)
        output_cls, output_patch = x[:, :self.num_classes], x[:, self.num_classes:]
        x = torch.einsum('bnd, bkd -> bnk', output_patch, output_cls)
        size = int(math.sqrt(x.shape[1]))
        x = x.view(x.shape[0], size, size, -1)
        return x

class LinearDecoder(nn.Module):
    
    def __init__(self, num_classes, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)
    

    
    def forward(self, x):
        x = self.linear(x)
        size = int(math.sqrt(x.shape[1]))
        x = torch.reshape(x, (x.shape[0], size, size, -1))
        #print(x.shape)
        return x
import torch
from torch import nn

from .vit_encoder_layer import ViTEncoderLayer
from .embeddings import PatchEmbeddings, PositionalEmbeddings
from .classifier import ClassificationHead

class ViT(nn.Module):

    def __init__(self, image_size, patch_size, d_model, d_ff, num_layers, n_cls, num_heads, p_drop = 0.1, in_channels = 3):
        super().__init__()
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError('The image size has to fully divide the patch size')

        self.positionEmb = PositionalEmbeddings(d_model, p_drop)
        self.patchEmb = PatchEmbeddings(d_model, patch_size, in_channels)

        self.transformer_layers = nn.ModuleList([ViTEncoderLayer(num_heads, d_model, d_ff) for _ in range(num_layers)])
        
        self.classificationHead = ClassificationHead(d_model, n_cls)

    

    def forward_features(self, x):
        x = self.patchEmb(x)
        x = self.positionEmb(x)

        for layer in self.transformer_layers:
            x = layer(x)
            

        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, 0]
        x = self.classificationHead(x)
        return x

    
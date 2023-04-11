import torch
from torch import nn

from .decoder import LinearDecoder, MaskTransformer

class Segmenter(nn.Module):
    
    def __init__(self, encoder, num_classes, d_model, num_layers, num_heads, image_size, d_ff = 2048, mask = False):
        super().__init__()
        if mask:
            self.decoder = MaskTransformer(num_classes, d_model, d_ff, num_layers, num_heads)
        else:
            self.decoder = LinearDecoder(num_classes, d_model)
        
        self.encoder = encoder
        self.upsampler = nn.UpsamplingBilinear2d(size = image_size)
    
    def forward(self, x):
        x = self.encoder(x)
        # remove the single cls token from vit
        x = x[:, 1:, :]
        #print(x.shape)
        x = self.decoder(x)
        x = x.permute(0,3, 1, 2)
        x = self.upsampler(x)
        #print(x.shape)
        return x
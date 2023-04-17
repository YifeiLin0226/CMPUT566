import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ham import NMF

class Hamburger(nn.Module):
    def __init__(self):
        super(Hamburger, self).__init__()
        self.identity = nn.Identity()
        self.nmf = NMF()
        self.lower_bread = nn.Sequential(nn.Conv2d(512, 512, 1),
                                         nn.ReLU(inplace = True))
        self.upper_bread = nn.Sequential(nn.Conv2d(512, 512, 1, bias = False),
                                         nn.BatchNorm2d(512))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
        
    def forward(self, x):
        shortcout = self.identity(x)

        x = self.lower_bread(x)
        x = self.nmf(x)
        x = self.upper_bread(x)

        x = x + shortcout

        x = F.relu(x, inplace = True)

        return x
    
    def update(self, D_matrix):
        self.nmf.update(D_matrix)



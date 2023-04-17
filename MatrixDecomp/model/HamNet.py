import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet
from .hamburger import Hamburger

class HamNet(nn.Module):
    def __init__(self, num_classes, n_layers):
        super(HamNet, self).__init__()
        backbone = resnet(n_layers)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        self.conv = nn.Conv2d(2048, 512, 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace = True)
        self.hamburger = Hamburger()
        self.align = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1, bias = False),
                                      nn.BatchNorm2d(256),
                                        nn.ReLU(inplace = True))
        self.fc = nn.Sequential(nn.Dropout2d(p = 0.1),
                                nn.Conv2d(256, num_classes, 1))

    def forward(self, x):
        size = x.shape[2:]
        x = self.backbone(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.hamburger(x)
        x = self.align(x)
        x = self.fc(x)
        x = F.interpolate(x, size = size, mode = 'bilinear', align_corners = True)

        return x



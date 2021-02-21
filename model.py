from layer import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_schedular


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=128, norm='bnorm'):
        super(Generator, self).__init__()

        self.dec1 = DECBR2d(1*in_channels, 8*nker, kernel_size=4, stride=1,
                            padding=0, norm=norm, relu=0.0, bias=False)
        self.dec2 = DECBR2d(8*in_channels, 4*nker, kernel_size=4, stride=2,
                            padding=0, norm=norm, relu=0.0, bias=False)
        self.dec3 = DECBR2d(4*in_channels, 2*nker, kernel_size=4, stride=2,
                            padding=0, norm=norm, relu=0.0, bias=False)
        self.dec4 = DECBR2d(2*in_channels, 1*nker, kernel_size=4, stride=2,
                            padding=0, norm=norm, relu=0.0, bias=False)
        self.dec5 = DECBR2d(1*in_channels, out_channels, kernel_size=4, stride=2,
                            padding=0, norm=norm, relu=0.0, bias=False)

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = torch.tanh(x)

        return x

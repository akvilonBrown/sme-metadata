import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_parts import *


""" Full assembly of the parts to form the complete network 
naked Unet3 model without channel attention modules
given for reference, not used in experiments
"""




class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        c64 = 64
        super(UNet3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TripleConv(n_channels, c64)
        self.down1 = Down(c64, c64)
        self.down2 = Down(c64, c64)
        self.down3 = Down(c64, c64)
        #factor = 2 if bilinear else 1
        self.down4 = Down(c64, c64)
        self.down5 = Down(c64, c64)        
        self.up1 = Up(c64, c64, bilinear)
        self.up2 = Up(c64, c64, bilinear)
        self.up3 = Up(c64, c64, bilinear)
        self.up4 = Up(c64, c64, bilinear)
        self.up5 = Up(c64, c64, bilinear)
        self.outc = OutConv(c64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)        
        x = self.up1(x6, x5)        
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits

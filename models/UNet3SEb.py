import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_parts import *


    

class SE_block(nn.Module):
    """squeeze and excitation block"""
    def __init__(self, num_features, reduction_factor=4):
        super(SE_block, self).__init__()
        # squeeze block
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # excitation block
        self.excite = nn.Sequential(
            nn.Linear(num_features, num_features // reduction_factor),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction_factor, num_features*2),
            #nn.Sigmoid()
        )
    def forward(self, x):
        batch, channel, _, _ = x.size()
        squeeze_res = self.squeeze(x).view(batch, channel)        
        excite_res = self.excite(squeeze_res)       
        f_scale, bias = excite_res.split(excite_res.data.size(1) // 2, dim=1) #
        f_scale = torch.sigmoid(f_scale)
        f_scale = f_scale.view(batch, channel, 1, 1)
        bias = bias.view(batch, channel, 1, 1)        
        return x * f_scale + bias    


""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F


class UNet3SEb(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        c64 = 64
        super(UNet3SEb, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TripleConv(n_channels, c64)
        self.se_inc = SE_block(c64, 2)
        self.down1 = Down(c64, c64)
        self.se_down1 = SE_block(c64, 2)
        self.down2 = Down(c64, c64)
        self.se_down2 = SE_block(c64, 2)
        self.down3 = Down(c64, c64)
        #factor = 2 if bilinear else 1
        self.se_down3 = SE_block(c64, 2)
        self.down4 = Down(c64, c64)
        self.se_down4 = SE_block(c64, 2)
        self.down5 = Down(c64, c64)
        self.se_down5 = SE_block(c64, 4)
        self.up1 = Up(c64, c64, bilinear)
        self.se_up1 = SE_block(c64, 2)
        self.up2 = Up(c64, c64, bilinear)
        self.se_up2 = SE_block(c64, 2)
        self.up3 = Up(c64, c64, bilinear)
        self.se_up3 = SE_block(c64, 2)
        self.up4 = Up(c64, c64, bilinear)
        self.se_up4 = SE_block(c64, 2)
        self.up5 = Up(c64, c64, bilinear)
        self.se_up5 = SE_block(c64, 2)
        self.outc = OutConv(c64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.se_inc(x1)
        x2 = self.down1(x1)
        x2 = self.se_down1(x2)
        x3 = self.down2(x2)
        x3 = self.se_down2(x3)
        x4 = self.down3(x3)
        x4 = self.se_down3(x4)
        x5 = self.down4(x4)
        x5 = self.se_down4(x5)
        x6 = self.down5(x5)        
        x6 = self.se_down5(x6)        
        x = self.up1(x6, x5)
        x = self.se_up1(x)        
        x = self.up2(x, x4)
        x = self.se_up2(x)
        x = self.up3(x, x3)
        x = self.se_up3(x)
        x = self.up4(x, x2)
        x = self.se_up4(x)
        x = self.up5(x, x1)
        x = self.se_up5(x)
        
        logits = self.outc(x)
        return logits

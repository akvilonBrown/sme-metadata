import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_parts import *

class SE_block(nn.Module):
    """squeeze and excitation block"""
    def __init__(self, num_features_input, num_features_output, reduction_factor):
        super(SE_block, self).__init__()
        # excitation block
        self.excite = nn.Sequential(
            nn.Linear(num_features_input, num_features_output // reduction_factor),
            nn.ReLU(inplace=True),
            nn.Linear(num_features_output // reduction_factor, num_features_output*2),
            #nn.Sigmoid()
        )
    def forward(self, x, m):
        batch, channel, _, _ = x.size()
        excite_res = self.excite(m)
        f_scale, bias = excite_res.split(excite_res.data.size(1) // 2, dim=1) #
        f_scale = torch.sigmoid(f_scale)
        f_scale = f_scale.view(batch, channel, 1, 1)
        bias = bias.view(batch, channel, 1, 1)
        return x * f_scale + bias    

"""
Full assembly of the parts to form the complete network 
only the label input will be presen tin SE block (so no squeeze)
so the label should govern the channel selection process
"""
class UNet3SEbMetaOnly(nn.Module):
    def __init__(self, n_channels, n_classes, meta_length, bilinear=True, se_reduction = 2):
        c64 = 64
        #the input size for SE block after concatenation with num of channels
        se_input_size = meta_length 
        super(UNet3SEbMetaOnly, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TripleConv(n_channels, c64)
        self.se_inc = SE_block(se_input_size, c64, se_reduction)
        self.down1 = Down(c64, c64)
        self.se_down1 = SE_block(se_input_size, c64, se_reduction)
        self.down2 = Down(c64, c64)
        self.se_down2 = SE_block(se_input_size, c64, se_reduction)
        self.down3 = Down(c64, c64)
        #factor = 2 if bilinear else 1
        self.se_down3 = SE_block(se_input_size, c64, se_reduction)
        self.down4 = Down(c64, c64)
        self.se_down4 = SE_block(se_input_size, c64, se_reduction)
        self.down5 = Down(c64, c64)
        self.se_down5 = SE_block(se_input_size, c64, se_reduction)
        self.up1 = Up(c64, c64, bilinear)
        self.se_up1 = SE_block(se_input_size, c64, se_reduction)
        self.up2 = Up(c64, c64, bilinear)
        self.se_up2 = SE_block(se_input_size, c64, se_reduction)
        self.up3 = Up(c64, c64, bilinear)
        self.se_up3 = SE_block(se_input_size, c64, se_reduction)
        self.up4 = Up(c64, c64, bilinear)
        self.se_up4 = SE_block(se_input_size, c64, se_reduction)
        self.up5 = Up(c64, c64, bilinear)
        self.se_up5 = SE_block(se_input_size, c64, se_reduction)
        self.outc = OutConv(c64, n_classes)

    def forward(self, x, m):
        x1 = self.inc(x)
        x1 = self.se_inc(x1, m)
        x2 = self.down1(x1)
        x2 = self.se_down1(x2, m)
        x3 = self.down2(x2)
        x3 = self.se_down2(x3, m)
        x4 = self.down3(x3)
        x4 = self.se_down3(x4, m)
        x5 = self.down4(x4)
        x5 = self.se_down4(x5, m)
        x6 = self.down5(x5)        
        x6 = self.se_down5(x6, m)        
        x = self.up1(x6, x5)
        x = self.se_up1(x, m)
        x = self.up2(x, x4)
        x = self.se_up2(x, m)
        x = self.up3(x, x3)
        x = self.se_up3(x, m)
        x = self.up4(x, x2)
        x = self.se_up4(x, m)
        x = self.up5(x, x1)
        x = self.se_up5(x, m)
        
        logits = self.outc(x)
        return logits

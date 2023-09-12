import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_parts import *



class FiLMgenerator(nn.Module):
    """The FiLM generator processes the conditioning information
    and produces parameters that describe how the target network should alter its computation.
    Here, the FiLM generator is a multi-layer perceptron.
    Args:
        n_features (int): Number of input channels.
        n_channels (int): Number of output channels.
        n_hid (int): Number of hidden units in layer.
    Attributes:
        linear1 (Linear): Input linear layer.
        sig (Sigmoid): Sigmoid function.
        linear2 (Linear): Hidden linear layer.
        linear3 (Linear): Output linear layer.
    """

    def __init__(self, n_features, n_channels, n_hid=64):
        super(FiLMgenerator, self).__init__()
        self.linear1 = nn.Linear(n_features, n_hid)
        self.sig = nn.Sigmoid()
        self.linear2 = nn.Linear(n_hid, n_hid // 4)
        self.linear3 = nn.Linear(n_hid // 4, n_channels * 2)

    def forward(self, x, shared_weights=None):
        if shared_weights is not None:  # weight sharing
            self.linear1.weight = shared_weights[0]
            self.linear2.weight = shared_weights[1]

        x = self.linear1(x)
        x = self.sig(x)
        x = self.linear2(x)
        x = self.sig(x)
        x = self.linear3(x)

        out = self.sig(x)
        return out, [self.linear1.weight, self.linear2.weight]


class FiLMlayer(nn.Module):
    """Applies Feature-wise Linear Modulation to the incoming data
    .. seealso::
        Perez, Ethan, et al. "Film: Visual reasoning with a general conditioning layer."
        Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
    Args:
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        n_channels (int): Number of output channels.
    Attributes:
        batch_size (int): Batch size.
        height (int): Image height.
        width (int): Image width.
        feature_size (int): Number of features in data.
        generator (FiLMgenerator): FiLM network.
        gammas (float): Multiplicative term of the FiLM linear modulation.
        betas (float): Additive term of the FiLM linear modulation.
    """

    def __init__(self, n_metadata, n_channels):
        super(FiLMlayer, self).__init__()

        self.batch_size = None
        self.height = None
        self.width = None
        self.depth = None
        self.feature_size = None
        self.generator = FiLMgenerator(n_metadata, n_channels)
        # Add the parameters gammas and betas to access them out of the class.
        self.gammas = None
        self.betas = None

    def forward(self, feature_maps, context, w_shared):
        data_shape = feature_maps.data.shape
        if len(data_shape) == 4:
            _, self.feature_size, self.height, self.width = data_shape
        elif len(data_shape) == 5:
            _, self.feature_size, self.height, self.width, self.depth = data_shape
        else:
            raise ValueError("Data should be either 2D (tensor length: 4) or 3D (tensor length: 5), found shape: {}".format(data_shape))
        '''
        if torch.cuda.is_available():
            context = torch.Tensor(context).cuda()
        else:
            context = torch.Tensor(context)
        '''
        # Estimate the FiLM parameters using a FiLM generator from the contioning metadata
        film_params, new_w_shared = self.generator(context, w_shared)

        # FiLM applies a different affine transformation to each channel,
        # consistent accross spatial locations
        if len(data_shape) == 4:
            film_params = film_params.unsqueeze(-1).unsqueeze(-1)
            film_params = film_params.repeat(1, 1, self.height, self.width)
        else:
            film_params = film_params.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            film_params = film_params.repeat(1, 1, self.height, self.width, self.depth)

        self.gammas = film_params[:, :self.feature_size, ]
        self.betas = film_params[:, self.feature_size:, ]

        # Apply the linear modulation
        output = self.gammas * feature_maps + self.betas

        return output, new_w_shared



""" Full assembly of the parts to form the complete network """

class UNet3FILM(nn.Module):
    def __init__(self, n_channels, n_classes, meta_length, bilinear=True, se_reduction = 2):
        c64 = 64
        #the input size for SE block after concatenation with num of channels
        se_input_size = meta_length 
        super(UNet3FILM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        #self.meta_length = meta_length
        self.bilinear = bilinear

        self.inc = TripleConv(n_channels, c64)
        #self.se_inc = SE_block(se_input_size, c64, se_reduction)
        self.se_inc = FiLMlayer(se_input_size, c64)
        self.down1 = Down(c64, c64)
        self.se_down1 = FiLMlayer(se_input_size, c64)
        self.down2 = Down(c64, c64)
        self.se_down2 = FiLMlayer(se_input_size, c64)
        self.down3 = Down(c64, c64)
        #factor = 2 if bilinear else 1
        self.se_down3 = FiLMlayer(se_input_size, c64)
        self.down4 = Down(c64, c64)
        self.se_down4 = FiLMlayer(se_input_size, c64)
        self.down5 = Down(c64, c64)
        self.se_down5 = FiLMlayer(se_input_size, c64)
        self.up1 = Up(c64, c64, bilinear)
        self.se_up1 = FiLMlayer(se_input_size, c64)
        self.up2 = Up(c64, c64, bilinear)
        self.se_up2 = FiLMlayer(se_input_size, c64)
        self.up3 = Up(c64, c64, bilinear)
        self.se_up3 = FiLMlayer(se_input_size, c64)
        self.up4 = Up(c64, c64, bilinear)
        self.se_up4 = FiLMlayer(se_input_size, c64)
        self.up5 = Up(c64, c64, bilinear)
        self.se_up5 = FiLMlayer(se_input_size, c64)
        self.outc = OutConv(c64, n_classes)

    def forward(self, x, m):
        x1 = self.inc(x)
        x1, w_film = self.se_inc(x1, m, None)
        x2 = self.down1(x1)
        x2, w_film = self.se_down1(x2, m, w_film)
        x3 = self.down2(x2)
        x3, w_film = self.se_down2(x3, m, w_film)
        x4 = self.down3(x3)
        x4, w_film = self.se_down3(x4, m, w_film)
        x5 = self.down4(x4)
        x5, w_film = self.se_down4(x5, m, w_film)
        x6 = self.down5(x5)        
        x6, w_film = self.se_down5(x6, m, w_film)       
        x = self.up1(x6, x5)
        x, w_film = self.se_up1(x, m, w_film) 
        #print(f'x after up1: {x.shape}')
        x = self.up2(x, x4)
        x, w_film = self.se_up2(x, m, w_film)
        x = self.up3(x, x3)
        x, w_film = self.se_up3(x, m, w_film)
        x = self.up4(x, x2)
        x, w_film = self.se_up4(x, m, w_film)
        x = self.up5(x, x1)
        x, w_film = self.se_up5(x, m, w_film)
        
        logits = self.outc(x)
        return logits
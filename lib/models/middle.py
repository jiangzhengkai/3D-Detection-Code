import torch
from torch import nn
from torch.nn import functional as F


class SpMiddleFHD(nn.Module):
    def __init__(self, 
                 output_shape,
                 use_norm,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHD'):
        super(SpMiddleFHD, self).__init__()
        self.name = name
        if use_norm:
        
        else:


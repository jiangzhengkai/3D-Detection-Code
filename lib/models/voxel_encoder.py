import numpy as np
import torch
from torch import nn
from .common import Empty


class VFELayer(nn.Module)
    def __init__(self, in_channels, out_channels, use_norm=True, name='VFE'):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            BatchNorm1d = nn.BatchNorm1d(momentum=0.01, eps=1e-3)
            Linear = nn.Linear(bias=True)
        else:
            BatchNorm = Empty
            Linear = nn.Linear(bias=False)

        self.linear = Linear
        self.batchnorm = BatchNorm1d
    def forward(self, inputs):
        voxel_count = input.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        pointwise = F.relu(x)

        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]

        repeated = aggregated.repeat(1, voxel_count, 1)
        concatenated = torch.cat([pointwise, repeated], dim=2)
        return concatenated
    
        
class VoxelFeatureExtractor(nn.Module):
    def __init__(self, 
                 num_input_features, 
                 use_norm, num_filters,
                 num_filters=[32,128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40.0, -3, 70.4, 40.0, 1.0),
                 name='VOxelFeatureExtractor'):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = nn.BatchNorm1d(momentum=0.01, eps=1e-3)
            Linear = nn.Linear(bias=True)
        else:
            BatchNorm = Empty
            Linear = nn.Linear(bias=False)
        self.linear = linear
        self.batchnorm = BatchNorm1d
        
    
    def forward(self, features, num_voxels, coordinates):
    
    return voxelwise



class VoxelFeatureExtractorV3(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40.0, -3, 70.4, 40.0, 1.0),
                 name='VoxelFeatureExtractorV3'):
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features
    def forward(self, features, num_voxels, coordinates)
        points_mean = features[:, :, :self.num_input_features].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()

import numpy as np
import torch
from torch import nn
from .common import Empty
from torch.nn import functional as F

def get_padding_indicator(actual_num, max_num, axis=0):
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num. dtype=torch.int. device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


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
       
        num_input_features += 3
        self._with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = Linear(num_filters[1], num_filters[1])
        self.norm = BatchNorm1d(num_filters[1]) 
    
    def forward(self, features, num_voxels, coordinates):
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_distance = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat([features, features_relative, points_distance], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axia=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)
        x *= mask
        voxelwise = torch.max(x, dim=1)[0]
        return voxelwise

class VoxelFeatureExtractorV2(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40.0, -3, 70.4, 40.0, 1.0),
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = nn.BatchNorm1d(eps=1e-3, momentum=0.01)
            Linear = nn.Linear(bias=False)
        else:
            BatchNorm1d = Empty
            Linear = nn.Linear(bias=True)
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
    
        num_filters = [num_input_features] + num_filters
        filters_pairs = [[num_filters[i], num_filters[i + 1]]
                         for i in range(len(num_filters) - 1)]    
        self.vfe_layers = nn.ModuleList(
            [VFELayer(i, o, use_norm) for i, o in filters_pairs])
        self.linear = Linear(num_filters[-1], num_filters[-1])
    def forward(self, features, num_voxels, coordinates):
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        feature_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_distance = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat([features, features_relative, points_distance], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_padding_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        features = self.linear(features)
        features = self.norm(features.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        features = F.relu(features)
        features *= mask
        voxelwise = torch.max(features, dim=1)[0]
        return voxelwise
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

class SimpleVoxel(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(32, 128),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40.0, -3, 70.4, 40.0, 1.0),
                 name='SimpleVoxel'):
        super(SimpleVoxel, self).__init__()
        self.num_input_features = num_input_features
        self.name = name
    def forward(self, features, num_voxels, coordinates):
        points_mean = features[:, :, :self.num_input_features].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        features = torch.norm(points_mean[:, :2], p=2, dim=1, keep_dim=True)
        res = torch.cat([features. points_mean[: 2:self.num_input_features]], dim=1)
        return res

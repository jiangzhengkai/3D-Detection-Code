import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class PIXORFeatureLayer(nn.Module):
    def __init__(self, voxel_feature_shape):
        super(PIXORFeatureLayer, self).__init__()
        self._voxel_feature_shape = voxel_feature_shape
        self.ny = self._voxel_feature_shape[2]
        self.nx = self._voxel_feature_shape[3]
        self.nchannels = self._voxel_feature_shape[1]  + 1

    def forward(self, vox_feats, num_points, coords, batch_size):
        output_shape = [batch_size] + self._voxel_feature_shape[1:]
        voxel_features = torch.ones_like(num_points.view(-1, 1)).type(torch.cuda.FloatTensor)

        vox_feats = vox_feats.mean(dim=1).view(-1, 5)

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype, device=voxel_features.device)

            # Only include non-empty voxels
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            intensity = vox_feats[batch_mask, -1:]
            intensity = intensity.t()

            # Now scatter the blob back to the canvas
            canvas[:-1, indices] = voxels
            canvas[-1:, indices] = intensity

            # Append to a list for later stacking
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch_size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column staking to fine 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)

        return batch_canvas


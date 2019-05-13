import numpy as np
import torch

def convert_batch_to_device(data_batch, dtype=torch.float32, device=None):
    device = device if device is not None else torch.device("cuda:0")

    data_batch_torch = {}
    for k, v in data_batch.items():
        if k in ["anchors", "reg_targets", "reg_weights"]:
            res = []
            for kk, vv in v.items():
                vv = np.array(vv)
                res.append(torch.tensor(vv, dtype=torch.float32,
                                        device=device))
            data_batch_torch[k] = res
        elif k in ["voxels", "bev_map"]:
            # slow when directly provide fp32 data with dtype=torch.half
            data_batch_torch[k] = torch.tensor(v,
                                            dtype=torch.float32,
                                            device=device)
        elif k in ["coordinates", "num_points"]:
            data_batch_torch[k] = torch.tensor(v,
                                            dtype=torch.int32,
                                            device=device)
        elif k == 'labels':
            res = []
            for kk, vv in v.items():
                vv = np.array(vv)
                res.append(torch.tensor(vv, dtype=torch.int32, device=device))
            data_batch_torch[k] = res
        elif k == 'points':
            data_batch_torch[k] = torch.tensor(v,
                                            dtype=torch.float,
                                            device=device)
        elif k in ["anchors_mask"]:
            res = []
            for kk, vv in v.items():
                vv = np.array(vv)
                res.append(torch.tensor(vv, dtype=torch.uint8, device=device))
            data_batch_torch[k] = res
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(v1, dtype=dtype, device=device)
            data_batch_torch[k] = calib
        elif k == "num_voxels":
            data_batch_torch[k] = torch.tensor(v,
                                            dtype=torch.int64,
                                            device=device)
        else:
            data_batch_torch[k] = v

    return data_batch_torch


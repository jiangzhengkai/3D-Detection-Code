import numpy as np
import torch
import threading, queue

class DataPrefetch(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item
    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


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
                                               dtype=torch.float32,
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


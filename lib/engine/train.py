#### python package
import torch
import argparse
import os
import numpy as np

import pathlib
import torch

from lib.datasets.loader.build_loader import build_dataloader
#from lib.models.build_model import build_model
#from lib.solver.build_optimizer import build_optimizer
#from lib.metrics.build_loss import build_detection_loss
#from utils.build_scheduler import build_scheduler

from lib.utils.logger import setup_logger
from lib.utils.dist_common import get_rank

def data_batch_convert_to_torch(data_batch, dtype=torch.float32, device=None):
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





def train(config, logger=None):
    logger = setup_logger("Training", config.output_dir, get_rank())    

    ####### dataloader #######
    train_dataloader = build_dataloader(config, training=True, logger=logger)
    #val_dataloader = build_dataloader(config, training=False, logger=logger)

    ####### build network ######
    #net = build_model(config)

    ####### optimizer #######
    #optimizer = build_optimizer(config)
    #lr_scheduler = build_scheduler(config)


    ####### criterions #######
    #self.criterion = build_detection_loss(config)


    num_epochs = config.input.train.num_epochs
    #eval_epoch = config.eval.num_epoch
    num_gpus = len(config.gpus.split(','))

    total_steps = int(num_epochs)
    logger.info("total training steps: %s" %(total_steps))
    
    device = torch.device('cuda')
    #arguments = {}
    #arguments['iter'] = 0
    #arguments['epoch'] = 0
    for epoch in range(num_epochs):
        for i, data_batch in enumerate(train_dataloader):
            data_torch = data_batch_convert_to_torch(data_batch, device=device)
            import pdb;pdb.set_trace() 
            print(1)

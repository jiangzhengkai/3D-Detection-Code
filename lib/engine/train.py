#### python package
import torch
import argparse
import os
import numpy as np

import pathlib
import torch

from lib.datasets.loader.build_loader import build_dataloader
from lib.models.build_model import build_network
#from lib.solver.build_optimizer import build_optimizer
#from lib.metrics.build_loss import build_detection_loss
#from utils.build_scheduler import build_scheduler

from lib.utils.logger import setup_logger
from lib.utils.dist_common import get_rank
from lib.engine.convert_batch_to_device import convert_batch_to_device

def train(config, logger=None):
    logger = setup_logger("Training", config.output_dir, get_rank())    

    ####### dataloader #######
    train_dataloader = build_dataloader(config, training=True, logger=logger)
    #val_dataloader = build_dataloader(config, training=False, logger=logger)

    ####### build network ######
    net = build_network(config, logger=logger)

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
    for epoch in range(num_epochs):
        for i, data_batch in enumerate(train_dataloader):
            ######## data_device ########
            #### voxels: num_voxels x max_num_points x 4 
            #### num_points: num_voxels
            #### points: max_points x (4 + 1)
            #### coordinates: num_voxels x 4
            #### num_voxels: batch_size x 1
            #### calib.rect: batch_size x 4 x 4
            #### calib.Trv2c: batch_size x 4 x 4
            #### calib.P2: batch_size x 4 x 4
            #### anchors: [batch_size x num_anchors x 7]
            #### labels: [batch_size x num_anchors]
            #### reg_targets: [batch_size x num_anchors x 7]
            #### reg_weights: [batch_size x num_anchors]
            #### meta_data: [dict_0, dict_1, ... dict_batch_size]
            data_device = convert_batch_to_device(data_batch, device=device)
            import pdb;pdb.set_trace() 
            print(1)

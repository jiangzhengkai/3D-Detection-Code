#### python package
import torch
import argparse
import os

import pathlib
import torch

from lib.datasets.loader.build_loader import build_dataloader
#from lib.models.build_model import build_model
#from lib.solver.build_optimizer import build_optimizer
#from lib.metrics.build_loss import build_detection_loss
#from utils.build_scheduler import build_scheduler

from lib.utils.logger import setup_logger
from lib.utils.dist_common import get_rank


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
    logger.info("total training steps: {total_steps}")


    #arguments = {}
    #arguments['iter'] = 0
    #arguments['epoch'] = 0
    for epoch in range(num_epochs):
        for i, data_batch in enumerate(train_dataloader):
            import pdb;pdb.set_trace() 
            print(1)

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


def train(config):
    logger = setup_logger("Training", config.output_dir, get_rank())    

    ####### dataloader #######
    train_dataset = build_dataloader(config, training=True)
    val_dataset = build_dataloader(config, training=False)

    ####### build network ######
    #net = build_model(config)

    ####### optimizer #######
    #optimizer = build_optimizer(config)
    #lr_scheduler = build_scheduler(config)


    ####### criterions #######
    #self.criterion = build_detection_loss(config)


    #num_epochs = config.train.num_epochs
    #eval_epoch = config.eval.num_epoch
    #num_gpus = len(config.train.gpus.split(','))

    #total_steps = int(num_epochs * len(train_dataset) / (config.train.batch_size * num_gpus))
    #logger.info("total training steps: {total_steps}")


    #arguments = {}
    #arguments['iter'] = 0
    #arguments['epoch'] = 0
    #for epoch in range(num_epochs):
    #    for iter, batch in enumerate(train_loader):
    #         print(1)

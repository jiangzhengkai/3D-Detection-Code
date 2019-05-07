#### python package
import torch
import argparse
import os

import pathlib
import torch

from dataset.loader import build_dataset
from models.build_model import build_model
from utils.build_optimizer import build_optimizer
from utils.build_scheduler import build_scheduler

from config import cfg
from utils.logging import create_logger


def train(config):
    logger, _ = create_logger(config.output_path, '', config.dataset.image_set)    

    ####### dataset #######
    train_dataset = build_dataset(config, training=True)
    val_dataset = build_dataset(config, training=False)
    ###### dataloader #######
    self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size,
						    shuffle=True, num_workers=config.train.num_workers,
						    pin_memory=False, collate_fn=
						    sampler=train_sampler)
    self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.eval.batch_size,
						  shuffle=True, num_workers=config.train.num_workers,
						  pin_memory=False, collate_fn=
						  sampler=train_sampler)


    ####### build network ######
    net = build_network(config)

    ####### optimizer #######
    optimizer = build_optimizer(config)
    lr_scheduler = build_scheduler(config)


    ####### criterions #######
    self.criterion = DetectionLoss(config)


    num_epochs = config.train.num_epochs
    eval_epoch = config.eval.num_epoch
    num_gpus = len(config.train.gpus.split(','))

    total_steps = int(num_epochs * len(train_dataset) / (config.train.batch_size * num_gpus))
    logger.info("total training steps: {total_steps}")


    arguments = {}
    arguments['iter'] = 0
    arguments['epoch'] = 0
    for epoch in range(num_epochs):
        for iter, batch in enumerate(train_loader):
             




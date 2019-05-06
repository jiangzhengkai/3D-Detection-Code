#### python package
import torch
import argparse
import os

import pathlib
import torch

from config import cfg
from dataset.loader import build_dataloader
from models.build_model import build_model
from utils.build_optimizer import build_optimizer
from utils.build_scheduler import build_scheduler



class Trainer():
    def __init__(self, config):
        self.config = config
        ####### dataloader #######
        self.train_loader = build_dataloader(config, training=True, dist=True)
        self.val_loader = build_dataloader(config, training=False, dist=True)

        ####### build network ######
        net = build_network(config)

        ####### optimizer #######
        optimizer = build_optimizer(config)
        lr_scheduler = build_scheduler(config)


        ####### criterions #######
        self.criterion = DetectionLoss(config)



    def training(self, epoch):



    def validation(self, epoch):


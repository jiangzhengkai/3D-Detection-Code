#### python package
import torch
import argparse
import os
import numpy as np

import pathlib
import torch

from lib.datasets.loader.build_loader import build_dataloader
from lib.models.build_model import build_network
from lib.solver.build_optimizer import build_optimizer
#from lib.metrics.build_loss import build_detection_loss
#from utils.build_scheduler import build_scheduler

from lib.utils.logger import setup_logger
from lib.utils.dist_common import get_rank
from lib.engine.convert_batch_to_device import convert_batch_to_device

from lib.engine.test import test
from lib.utils.dist_common import synchronize

def train(config, logger=None):
    logger = setup_logger("Training", config.output_dir, get_rank())    

    ####### dataloader #######
    train_dataloader = build_dataloader(config, training=True, logger=logger)
    #val_dataloader = build_dataloader(config, training=False, logger=logger)

    ####### build network ######
    model = build_network(config, logger=logger)

    ####### optimizer #######
    optimizer = build_optimizer(config, model)
    #lr_scheduler = build_scheduler(config)


    ####### criterions #######
    #self.criterion = build_detection_loss(config)


    num_epochs = config.input.train.num_epochs
    #eval_epoch = config.eval.num_epoch
    num_gpus = len(config.gpus.split(','))

    total_steps = int(num_epochs)
    logger.info("total training steps: %s" %(total_steps))
    
    device = torch.device('cuda')
    model = model.to(device)
    logger.info("Model Articutures: %s"%(model))
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
            rpn_predict_dicts = model(data_device)
            losses_dict = model.loss(data_device, rpn_predict_dicts)

            batch_size = data_device["anchors"][0].shape[0]
            losses = []
            cls_loss_reduceds, loc_loss_reduceds, cls_preds, careds = [], [], [], []
            loc_losses, cls_pos_losses, cls_neg_losses = [], [], []

            if config.model.decoder.auxiliary.use_direction_classifier:
                dir_loss_reduceds = []
            else:
                dir_loss_reduceds = None

            for task_id, loss_dict in enumerate(losses_dict):
                cls_pred = loss_dict["cls_preds"]
                loss = loss_dict["loss"].mean()
                cls_loss_reduced = loss_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = loss_dict["loc_loss_reduced"].mean()
                cls_pos_loss = loss_dict["cls_pos_loss"]
                cls_neg_loss = loss_dict["cls_neg_loss"]
                loc_loss = loss_dict["loc_loss"]
                cls_loss = loss_dict["cls_loss"]
                dir_loss_reduced = loss_dict["dir_loss_reduced"]
                labels = data_device["labels"]
                cared = loss_dict["cared"]

                losses.append(loss)
                loc_losses.append(loc_loss)
                cls_pos_losses.append(cls_pos_loss)
                cls_neg_losses.append(cls_neg_loss)
                cls_loss_reduceds.append(cls_loss_reduced)
                loc_loss_reduceds.append(loc_loss_reduced)
                cls_preds.append(cls_pred)
                careds.append(cared)
            task_loss = torch.stack(losses)
            loss_all = torch.Tensor(config.model.decoder.head.weights).to(device) * task_loss
            loss_mean = torch.mean(loss_all)
            

            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()


        torch.save(model.state_dict(), )
        if epoch % eval_epoch == 0:
            logger.info("Finish epoch %d, start eval ..." %(epoch))
            test(eval_dataloader, 
                 model, 
                 save_dir=save_dir, 
                 device=device, 
                 distributed=distributed, 
                 logger=logger)
        synchronize()


        import pdb;pdb.set_trace() 
        print(1)

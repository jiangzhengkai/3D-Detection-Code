#### python package
import torch
import argparse
import os
import numpy as np

import pathlib
import torch
import time

from lib.datasets.loader.build_loader import build_dataloader
from lib.models.build_model import build_network
from lib.solver.build_optimizer import build_optimizer
from lib.solver.build_scheduler import build_lr_scheduler
from lib.core.target.target_assigner import target_assigners_all_classes

from lib.utils.logger import setup_logger
from lib.utils.dist_common import get_rank
from lib.engine.convert_batch_to_device import convert_batch_to_device

from lib.engine.metrics import get_metrics
from lib.engine.test import test
from lib.utils.dist_common import synchronize

def train(config, logger=None):
    logger = setup_logger("Training", config.output_dir, get_rank())    

    ####### dataloader #######
    train_dataloader = build_dataloader(config, training=True, logger=logger)
    val_dataloader = build_dataloader(config, training=False, logger=logger)

    ####### build network ######
    model = build_network(config, logger=logger)

    ####### optimizer #######
    optimizer = build_optimizer(config, model)


    num_epochs = config.input.train.num_epochs
    num_gpus = len(config.gpus.split(','))

    
    total_steps = int(num_epochs * len(train_dataloader.dataset) / config.input.train.batch_size * num_gpus)
    logger.info("total training steps: %s" %(total_steps))

    lr_scheduler = build_lr_scheduler(config, optimizer, total_steps)
    
    device = torch.device('cuda')
    model = model.to(device)
    logger.info("Model Articutures: %s"%(model))

    # num_classes 
    target_assigners = target_assigners_all_classes(config)
    num_classes = [len(target_assigner.classes) for target_assigner in target_assigners]
    class_names = [target_assigner.classes for target_assigner in target_assigners]
    t = time.time()

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
            num_step = int(epoch * len(train_dataloader.dataset) / config.input.train.batch_size) + i
            lr_scheduler.step(num_step)
             
            data_device = convert_batch_to_device(data_batch, device=device)
            rpn_predict_dicts = model(data_device)
            losses_dict = model.loss(data_device, rpn_predict_dicts)
  
            batch_size = data_device["anchors"][0].shape[0]
            losses = []
            cls_loss_reduceds, loc_loss_reduceds, dir_loss_reduceds, cls_preds, careds = [], [], [], [], []
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
                dir_loss_reduceds.append(dir_loss_reduced)
                cls_preds.append(cls_pred)
                careds.append(cared)
                
            task_loss = torch.stack(losses)
            loss_all = torch.Tensor(config.model.decoder.head.weights).to(device) * task_loss
            loss_mean = torch.mean(loss_all)

            optimizer.zero_grad()                       
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            for idx, [cls_loss_reduced, loc_loss_reduced, dir_loss_reduced, cls_pred, labels, cared, loc_loss, cls_pos_loss, cls_neg_loss, loss] in enumerate(
                zip(cls_loss_reduceds, loc_loss_reduceds, dir_loss_reduceds, cls_preds, data_device["labels"], careds, loc_losses, cls_pos_losses, cls_neg_losses, losses)):

                net_metrics = get_metrics(config, cls_loss_reduced, loc_loss_reduced, cls_pred, labels, cared, num_classes[idx])
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                if "anchor_mask" not in data_device:
                    num_anchors = data_device["anchors"][idx].shape[1]
                else:
                    num_anchors = int(data_device["anchors_mask"][idx].shape[1])

                step_time = time.time() - t
                t = time.time()
                if num_step % 50 == 0:
                    logger.info("Metrics for task: {}".format(class_names[idx]))

                    loc_loss_elem = [float(loc_loss[:,:,i].sum().detach().cpu().numpy() / batch_size) for i in range(loc_loss.shape[-1])]
                    metrics["loss"] = loss
                    metrics["loc_elem"] = loc_loss_elem
                    metrics["cls_pos_rt"] = float(cls_pos_loss.sum().detach().cpu().numpy())
                    metrics["cls_neg_rt"] = float(cls_neg_loss.sum().detach().cpu().numpy())
                    if config.model.decoder.auxiliary.use_direction_classifier:
                        metrics["dir_rt"] = float(dir_loss_reduced.sum().detach().cpu().numpy())

                    logger.info("step: %d time %4f Loss_all: %6f, Loss_cls: %6f Loss_loc: %6f Loss_dir: %6f"%(
                                 num_step, step_time, loss, cls_loss_reduced, loc_loss_reduced, dir_loss_reduced))
                    logger.info("step: %d Loc_elements x: %6f y: %6f z: %6f w: %6f h: %6f l: %6f angle: %6f"%(
                                num_step, *(metrics["loc_elem"])))
                    logger.info("step: %d Cls_elements cls_neg_rt: %2f cls_pos_rt: %2f"%(
                                num_step, metrics["cls_neg_rt"], metrics["cls_pos_rt"]))
                    
                    num_voxel = int(data_device["voxels"].shape[0])
                    num_pos = int(num_pos)
                    num_neg = int(num_neg)
                    num_anchors = int(num_anchors)
                    lr = float(optimizer.lr)
                    logger.info("step: %d Auxiliraries num_voxels: %d num_pos: %d num_neg: %d num_anchors: %d lr: %6f"%(
                                 num_step, num_voxel, num_pos, num_neg, num_anchors, lr))
                    pr_metrics = net_metrics["pr"]
                    logger.info("step: %d RpnAcc: %6f PrecRec prec@30: %6f rec@30: %6f prec@50: %6f rec@50: %6f"%(
                                 num_step, net_metrics["rpn_acc"], pr_metrics["prec@30"], pr_metrics["rec@30"], pr_metrics["prec@50"], pr_metrics["rec@50"]))
                    logger.info("-------------------------------------------------------------------------------------------------------------------")

            torch.cuda.empty_cache()

        torch.save(model.state_dict(), config.output_dir+"/model_%d.pth"%epoch)
        if epoch % 5 == 0 or num_step == total_steps:
            logger.info("Finish epoch %d, start eval ..." %(epoch))
            distributed = len(config.gpus.split(',')) > 1
            test(val_dataloader, 
                 model, 
                 save_dir=config.output_dir, 
                 device=device, 
                 distributed=distributed, 
                 logger=logger)
        synchronize()

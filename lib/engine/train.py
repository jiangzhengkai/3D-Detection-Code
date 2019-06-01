#### python package
import torch
import argparse
import os
import numpy as np

import pathlib
import torch
import gc
import time

from lib.datasets.loader.build_loader import build_dataloader
from lib.models.build_model import build_network
from lib.solver.build_optimizer import build_optimizer
from lib.solver.build_scheduler import build_lr_scheduler
from lib.core.target.target_assigner import target_assigners_all_classes

from lib.utils.logger import setup_logger
from lib.utils.dist_common import get_rank
from lib.engine.convert_batch_to_device import convert_batch_to_device, DataPrefetch

#from lib.engine.metrics import get_metrics
from lib.engine.test import test
from lib.utils.dist_common import synchronize
from lib.utils.checkpoint import Det3DCheckpointer

#from apex import parallel
def train(config, logger=None, model_dir=None, distributed=False):
    logger = setup_logger("Training", model_dir, get_rank())

    ####### dataloader #######
    train_dataloader = build_dataloader(config, training=True, logger=logger)
    val_dataloader = build_dataloader(config, training=False, logger=logger)

    ####### build network ######
    device = torch.device('cuda')
    model = build_network(config, logger=logger, device=device)
    logger.info("Model Articutures: %s"%(model))
    if distributed:
        #model = parallel.convert_syncbn_model(model)
        logger.info("Using SyncBn")
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            broadcast_buffers=False,
        )
        net_module = model.module
    else:
        net_module = model.to(device)
        logger.info("Training use Single-GPU")

    ####### optimizer #######
    optimizer = build_optimizer(config, net_module)

    ####### checkpoint #######
    save_to_disk = get_rank() == 0
    arguments = {}
    arguments["iteration"] = 0
    arguments["epoch"] = 0
    checkpoint = Det3DCheckpointer(net_module,
                                   optimizer=optimizer,
                                   save_dir=model_dir,
                                   save_to_disk=save_to_disk,
                                   logger=logger)

    arguments.update(checkpoint.load())
    logger.info(f"extra arguments: {arguments}")
    net_module.clear_metrics()

    num_epochs = config.input.train.num_epochs
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    total_steps = int(num_epochs * len(train_dataloader.dataset) / (config.input.train.batch_size * num_gpus))
    logger.info("total training steps: %s" %(total_steps))

    lr_scheduler = build_lr_scheduler(config, optimizer, total_steps)

    # num_classes
    target_assigners = target_assigners_all_classes(config)
    num_classes = [len(target_assigner.classes) for target_assigner in target_assigners]
    class_names = [target_assigner.classes for target_assigner in target_assigners]
    t = time.time()
    for epoch in range(arguments["epoch"]+1, num_epochs+1):
        arguments["epoch"] = epoch
        for i, data_batch in enumerate(DataPrefetch(iter(train_dataloader), max_prefetch=4)):
            lr_scheduler.step(net_module.get_global_step())
            arguments["iteration"] += 1

            data_device = convert_batch_to_device(data_batch, device=device)

            optimizer.zero_grad()

            losses_dict = net_module(data_device)

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
            loss_mean = torch.sum(loss_all)

            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(net_module.parameters(), 10.0)
            optimizer.step()
            net_module.update_global_step()

            for idx, [cls_loss_reduced, loc_loss_reduced, dir_loss_reduced, cls_pred, labels, cared, loc_loss, cls_pos_loss, cls_neg_loss, loss] in enumerate(
                zip(cls_loss_reduceds, loc_loss_reduceds, dir_loss_reduceds, cls_preds, data_device["labels"], careds, loc_losses, cls_pos_losses, cls_neg_losses, losses)):

                net_metrics = net_module.update_metrics(cls_loss_reduced, loc_loss_reduced, cls_pred, labels, cared, idx)
                #net_metrics = get_metrics(config, cls_loss_reduced, loc_loss_reduced, cls_pred, labels, cared, num_classes[idx])
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                if "anchor_mask" not in data_device:
                    num_anchors = data_device["anchors"][idx].shape[1]
                else:
                    num_anchors = int(data_device["anchors_mask"][idx][0].sum())

                step_time = time.time() - t
                t = time.time()
                step = net_module.get_global_step()
                if step % 50 == 0:
                    logger.info(f"Metrics for task: {class_names[idx]}, saved to: {model_dir}")
                    loc_loss_elem = [float(loc_loss[:,:,i].sum().detach().cpu().numpy() / batch_size) for i in range(loc_loss.shape[-1])]
                    metrics["loss"] = loss
                    metrics["loc_elem"] = loc_loss_elem
                    metrics["cls_pos_rt"] = float(cls_pos_loss.sum().detach().cpu().numpy())
                    metrics["cls_neg_rt"] = float(cls_neg_loss.sum().detach().cpu().numpy())
                    if config.model.decoder.auxiliary.use_direction_classifier:
                        metrics["dir_rt"] = float(dir_loss_reduced.sum().detach().cpu().numpy())
                    logger.info("epoch: %d step: %d time %4f Loss_all: %6f"%(
                                 epoch, step, step_time, loss))
                    logger.info("epoch: %d step: %d Loss_cls: %6f Loss_loc: %6f Loss_dir: %6f"%(
                                 epoch, step, cls_loss_reduced, loc_loss_reduced, dir_loss_reduced))
                    if config.input.train.dataset.type == "KittiDataset":
                        logger.info("epoch: %d step: %d Loc_elements x: %6f y: %6f z: %6f w: %6f h: %6f l: %6f angle: %6f"%(
                                    epoch, step, *(metrics["loc_elem"])))
                    else:
                        logger.info("epoch: %d step: %d Loc_elements x: %6f y: %6f z: %6f w: %6f h: %6f l: %6f vx: %6f vy: %6f angle: %6f"%(
                                    epoch, step, *(metrics["loc_elem"])))
                    logger.info("epoch: %d step: %d Cls_elements cls_neg_rt: %2f cls_pos_rt: %2f"%(
                                epoch, step, metrics["cls_neg_rt"], metrics["cls_pos_rt"]))
                    num_voxel = int(data_device["voxels"].shape[0])
                    num_pos = int(num_pos)
                    num_neg = int(num_neg)
                    num_anchors = int(num_anchors)
                    lr = float(optimizer.lr)
                    metrics["misc"] = {
                        "num_vox": int(num_voxel),
                        "num_pos": int(num_pos),
                        "num_neg": int(num_neg),
                        "num_anchors": int(num_anchors),
                        "lr": float(optimizer.lr),
                    }
                    checkpoint.writer.open()
                    checkpoint.writer.log_metrics(metrics, step)

                    logger.info("epoch: %d step: %d Auxiliraries num_voxels: %d num_pos: %d num_neg: %d num_anchors: %d lr: %6f"%(
                                 epoch, step, num_voxel, num_pos, num_neg, num_anchors, lr))
                    pr_metrics = net_metrics["pr"]
                    logger.info("epoch: %d step: %d RpnAcc: %6f"%(epoch, step, net_metrics["rpn_acc"]))
                    logger.info("epoch: %d step: %d Prec prec@10: %6f prec@30: %6f prec@50: %6f prec@70: %6f prec@90: %6f"%(
                                 epoch, step, pr_metrics["prec@10"], pr_metrics["prec@30"], pr_metrics["prec@50"],
                                 pr_metrics["prec@70"], pr_metrics["prec@90"]))
                    logger.info("epoch: %d step: %d Reca reca@10: %6f reca@30: %6f reca@50: %6f reca@70: %6f reca@90: %6f"%(
                                 epoch, step, pr_metrics["rec@10"], pr_metrics["rec@30"], pr_metrics["rec@50"],
                                 pr_metrics["rec@70"], pr_metrics["rec@90"]))
                    logger.info("-------------------------------------------------------------------------------------------------------------------")

        net_module.clear_metrics()
        gc.collect()
        checkpoint.save("model_epoch_{:03d}_step_{:06d}".format(
            epoch, step, **arguments))
        if epoch % 1 == 0:
            logger.info("Finish epoch %d, start eval ..." %(epoch))
            test(val_dataloader,
                 model,
                 save_dir=model_dir,
                 device=device,
                 distributed=distributed,
                 logger=logger)
            torch.cuda.empty_cache()
        synchronize()

    checkpoint.writer.close()

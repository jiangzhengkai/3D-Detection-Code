#### python package
import torch
import argparse
import os

import pathlib
import torch

from config import cfg
from dataset.loader import build_loader
from models.build_network import build_network



  



def train(config, result_path=None, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    ####### dataloader #######
    dataloader = build_dataloader(config, training=True, voxel_generator=voxel_generator, target_assigner=target_assigner, dist=True)
    


    ####### build network ###### 
    net = build_network(config).to(device)



    ####### optimizer #######
    optimizer = build_optimizer(config)
    lr_scheduler = build_scheduler(config)



    ####### training #######
    data_iter = iter(dataloader)
    


def evaluate(config, model_dir, result_path=None, model_path=None, measure_time=False, batch_size=None):

    ####### load config #######



    ####### build network #######



    ####### dataloader #######



    ####### evaluation #######

def main():
    parser = argparse.ArgumentParser(description='3d object detection training')
    parser.add_argument('--config', default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    num_gpus = int(os.environ["WORLD_SIZE"] if "WORLD_SIZE" in os.environ else 1)
    args.distributed = num_gpus > 1    

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    output_dir = cfg.output_dir

    logger = setup_logger("3d-object-detection", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    
    with open(args.config, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, output_dir)


    

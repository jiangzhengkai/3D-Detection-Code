import os
import torch
import torch.distributed as dist
import argparse
from lib.engine.train import train
from lib.utils.dist_common import get_rank
from lib.config.config import cfg, cfg_from_file
from lib.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='3d object detection train')
    parser.add_argument('--cfg', default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpus', type=int, help="number of gpus to use")
    args = parser.parse_args()

    cfg_from_file(args.cfg)
    output_dir = cfg.output_dir
    num_gpus = args.gpus if args.gpus else len(cfg.gpus.split(','))

    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
    logger = setup_logger("3d-object-detection", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    with open(args.cfg, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    train(cfg)

if __name__ == "__main__":
    main()


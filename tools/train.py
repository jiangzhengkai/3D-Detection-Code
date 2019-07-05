import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.distributed as dist
import argparse
from lib.engine.train import train
from lib.utils.dist_common import get_rank, synchronize
from lib.config.config import cfg, cfg_from_file
from lib.utils.collect_env import collect_env_info
from lib.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='3d object detection train')
    parser.add_argument('--cfg', default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--model_dir', default="default", type=str, help="model to save dir")
    args = parser.parse_args()

    cfg_from_file(args.cfg)
    output_dir = args.model_dir
    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    logger = setup_logger("3d-object-detection", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    with open(args.cfg, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    train(cfg, logger=logger, model_dir=output_dir, local_rank=args.local_rank, distributed=args.distributed)

if __name__ == "__main__":
    main()


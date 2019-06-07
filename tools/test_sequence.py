#### python package
import torch
import argparse
import os

import torch.distributed as dist
from lib.config.config import cfg, cfg_from_file
from lib.utils.collect_env import collect_env_info
from lib.utils.logger import setup_logger

from lib.datasets.loader.build_sequence_loader import build_sequence_dataloader
from lib.models.build_sequence_model import build_sequence_network

from lib.utils.dist_common import get_rank, synchronize
from lib.engine.test import test_sequence
from lib.utils.checkpoint import Det3DCheckpointer

from apex import parallel
def test_model(config, logger=None, model_dir=None, local_rank=None, distributed=False):
    logger = setup_logger("Training", model_dir, get_rank())    

    ####### dataloader #######
    val_dataloader = build_sequence_dataloader(config, training=False, logger=logger)

    ####### build network ######
    device = torch.device('cuda')
    model = build_sequence_network(config, logger=logger, device=device)
    logger.info("Model Articutures: %s"%(model))
    if distributed:
        model = parallel.convert_syncbn_model(model)
        logger.info("Using SyncBn")
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        net_module = model.module
    else:
        net_module = model.to(device)
        logger.info("Training use Single-GPU")

    ####### checkpoint #######
    checkpoint = Det3DCheckpointer(net_module, 
                                   optimizer=None,
                                   save_dir=model_dir,
                                   save_to_disk=False,
                                   logger=logger)

    checkpoint.load()

    logger.info("Finish start eval ...")
    test_sequence(val_dataloader, 
         model, 
         save_dir=model_dir, 
         device=device, 
         distributed=distributed, 
         logger=logger)
    torch.cuda.empty_cache()
    synchronize()
    

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
    test_model(cfg, logger=logger, model_dir=output_dir, local_rank=args.local_rank, distributed=args.distributed)

if __name__ == "__main__":
    main()


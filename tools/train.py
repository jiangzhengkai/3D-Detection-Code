import torch

from lib.config import cfg



def main():
    parser = argparse.ArgumentParser(description='3d object detection train')
    parser.add_argument('--config', default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1, help="number of gpus to use")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    output_dir = cfg.output_dir
    num_gpus = args.gpus

    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger = setup_logger("3d-object-detection", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    with open(args.config, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    trainer = Trainer(cfg)
    for epoch in range(cfg.train.start_epochs, cfg.train.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)


if __name__ == "__main__":
    main()


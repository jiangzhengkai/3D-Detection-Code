import argparse
from nuscenes.nuscenes import NuScenes
import json
from lib.config.config import cfg, cfg_from_file
from lib.datasets.loader.build_loader import build_dataloader
from lib.datasets.nuscenes_dataset import eval_main
from lib.utils.logger import setup_logger
from lib.utils.dist_common import get_rank



def main():
    parser = argparse.ArgumentParser(description='3d object detection train')
    parser.add_argument('--cfg', default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument('--model_dir', default="default", type=str, help="model to save dir")
    args = parser.parse_args()
    cfg_from_file(args.cfg)

    logger = setup_logger("3d-object-detection", args.model_dir, get_rank())

    val_dataloader = build_dataloader(cfg, training=False, logger=logger)


    dataset = val_dataloader.dataset

    nusc = NuScenes(version="v1.0-trainval",
                    dataroot="/data/datasets/NuScenes",
                    verbose=True)
    #detections = json.load(open("results_new.json",'r'))
    eval_main(nusc, "cvpr_2019", "results_63.42_hand.json",
              "val", args.model_dir)
    #val_dataloader.dataset.evaluation_nusc(detections["results"], args.model_dir)
if __name__ == "__main__":
    main()

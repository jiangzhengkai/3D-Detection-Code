# create dataset information
import copy
from pathlib import Path
import pickle
import fire

import kitti_dataset as kitti_dataset
import nuscenes_dataset as nuscenes_dataset
from all_dataset import create_groundtruth_database

def kitti_data_preparation(root_path):
    kitti_dataset.create_kitti_information_file(root_path)
    kitti_dataset.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset", root_path, Path(root_path) / "kitti_infos_train.pkl")

def nuscenes_data_preparation(root_path, version, max_sweeps=9):
    nuscenes_dataset.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps) 
    create_groundtruth_database("NuscenesDataset", root_path, Path(root_path) / "nuscenes_infos_train.pkl")

if __main__ == '__main__':
    fire.Fire()


import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import fire

import lib.datasets.kitti.kitti_dataset as kitti_ds
from lib.datasets.all_dataset import create_groundtruth_database
import lib.datasets.nuscenes.nuscenes_dataset as nusc_ds
def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database('KittiDataset', root_path, Path(root_path) / "kitti_infos_train.pkl")
def nuscenes_data_prep(root_path, version, nsweeps=10):
    nusc_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps)
    create_groundtruth_database('NuScenesDataset', root_path, Path(root_path) / "infos_train_{:02d}sweeps_withvelo.pkl".format(nsweeps), nsweeps=nsweeps)

if __name__ == '__main__':
    fire.Fire()


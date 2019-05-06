# 3D-Detection-Code
General 3d object detection code for kitti dataset. Code to implement 3d object detection code for better understanding and cleaning


## Environment Setup

### 1.Install dependence python packages for environment setup

```bash
conda install scikit-image scipy numba pillow matplotlib
```
```bash
pip install fire tensorboardX protobuf opencv-python
```

### 2. NuScenes Datasets
Install [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit).



## Prepare Dataset


* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) Dataset Preparation

Download KITTI dataset and create some directories first:

```plain
|── KITTI_DATASET_ROOT
    ├── Train    <-- 7481 train data
    |   ├── image_2 <-- for visualization
    |   ├── calib
    |   ├── label_2
    |   ├── velodyne
    |   └── velodyne_reduced <-- empty directory
    └── Test     <-- 7580 test data
        ├── image_2 <-- for visualization
        ├── calib
        ├── velodyne
        └── velodyne_reduced <-- empty directory
```

Then run
```bash
python create_data.py kitti_data_preparation --data_path=KITTI_DATASET_ROOT
```

* [NuScenes](https://www.nuscenes.org) Dataset Preparation

Download NuScenes dataset:
```plain
|── NUSCENES_DATASET_ROOT
    ├── Train
    |   ├── samples       <-- key frames
    |   ├── sweeps        <-- frames without annotation
    |   ├── maps          <-- unused
    |   └── v1.0-trainval <-- metadata and annotations
    ├── Test
    |   ├── samples       <-- key frames
    |   ├── sweeps        <-- frames without annotation
    |   ├── maps          <-- unused
    |   └── v1.0-test     <-- metadata
```

Then run
```bash
python create_data.py nuscenes_data_preparation --data_path=NUSCENES_DATASET_ROOT --mode Train --version="v1.0-trainval" --max_sweeps=9
python create_data.py nuscenes_data_preparation --data_path=NUSCENES_DATASET_ROOT --mode Test --version="v1.0-test" --max_sweeps=9
```

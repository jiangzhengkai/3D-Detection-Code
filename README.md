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

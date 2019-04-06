from pathlib import Path
import pickle
import time
import numpy as np

from 3d_detection.ops import box_np_ops
from 3d_detection.dataset import kitti_common as kitti
from 3d_detection.dataset.base_dataset import Dataet

class KittiDataset(Dataset):
    num_points_feature = 4
    def __init__(self, root_path, info_path, class_names=None, prep_func=None, num_point_features=None):
    def __len__(self):
    @property
    def num_point_features(self):

    @property
    def ground_truth_annotations(self):
    
    def convert_detection_to_kitti_annotations(self, detection):
    

    def evaluation(self, detections, output_dir):


    def __getitem__(self, idx):



    def get_sensor_data(self, query):
 

    


def _calculate_num_points_in_gt(data_path,
                                infos, 
                                relative_path, 
                                remove_outsize=True, 
                                num_features=4):
    for info in infos:
        point_cloud_info = info["point_cloud"]
        image_info = info['image']
        calib = info['calib']
        if relative_path:
            velodyne_path = str(Path(data_path) / point_cloud_info['velodyne_path'])
        else:
            velodyne_path = point_cloud_info['velodyne_path']
        points_velodyne = np.fromfile(velodyne_path, dtype=np.float32).reshape([-1, num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_output:
            points_velodyne = box_np_ops.remove_outside_points(points_velodyne, rect, Trv2c, P2, image_info['image_shape'])
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
    
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([num_ignored])])
        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)


def _read_imageset(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def create_kitti_info_file(data_path, save_path, relative_path=True):
    imageset_folder = Path(__file__).resolve().parent / "imagesets"
    train_img_ids = _read_imageset(str(imageset_folder / "train.txt")) 
    val_img_ids = _read_imageset(str(imageset_folder / "val.txt"))
    test_img_ids = _read_imageset(str(imageset_folder / "test.txt"))

    print("Generate kitti information--------")
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    kitti_infos_train = kitti.get_kitti_image_info(data_path, 
                                                   training=True, 
                                                   velodyne=True, 
                                                   calib=True, 
                                                   image_ids=train_image_ids)
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = os.path.join(save_path, 'kitti_infos_train.pkl')
    print('kitti info train file is saved to %s' % filename)
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    kitti_infos_val = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / 'kitti_infos_val.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    filename = save_path / 'kitti_infos_trainval.pkl'
    print(f"Kitti info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)

    kitti_infos_test = kitti.get_kitti_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path)
    filename = save_path / 'kitti_infos_test.pkl'
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)

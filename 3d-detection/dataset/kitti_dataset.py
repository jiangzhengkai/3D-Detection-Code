import os
import pickle
import time

from 3d_detection.dataset import kitti_common as kitti
from 3d_detection.dataset.base_dataset import Dataet

class KittiDataset(Dataset):
    def __init__():
    def __len__(self):
    @property
    def num_point_features(self):

    @property
    def ground_truth_annotations(self):
    
    def convert_detection_to_kitti_annotations(self, detection):
    

    def evaluation(self, detections, output_dir):


    def __getitem__(self, idx):



    def get_sensor_data(self, query):
 

    



def read_imageset(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def create_kitti_information_file(data_path, save_path):
    imageset_foler = os.path.dirname(os.path.abspath(__file__))
    train_image_ids = read_imageset(os.path.join(imageset_folder, "imagesets", "train.txt"))
    val_image_ids = read_imageset(os.path.join(imageset_folder, "imagesets", "val.txt"))
    test_image_ids = read_imageset(os.path.join(imageset_folder, "imagesets", "test.txt"))
    print("Generate kitti information--------")
    if save_path is None:
        save_path = data_path
    else:
        save_path = save_path
    kitti_information_train = kitti.get_kitti_image_information(data_path, 
                                                                training=True, 
                                                                velodyne=True, 
                                                                calib=True, 
                                                                image_ids=train_image_ids)

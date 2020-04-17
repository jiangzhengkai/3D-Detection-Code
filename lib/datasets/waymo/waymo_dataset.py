import lmdb
import lz4framed
import numpy as np
import pickle
import os

from copy import deepcopy

from torch.utils.data import Dataset

from .waymo_common import DetectionMetricsEstimatorTest


def deserialize(lmdb_value, compressed=True):
    if lmdb_value is None:
        return None
    if compressed:
        return pickle.loads(lz4framed.decompress(lmdb_value))
    else:
        return pickle.loads(lmdb_value)



class WaymoDataset(Dataset):

    NumPointFeatures = 8

    def __init__(
        self,
        root_path,
        info_path,
	dataset_list_path=None,
        prep_func=None,
        test_mode=False,
        class_names=None,
        **kwargs
    ):
        super(WaymoDataset, self).__init__(
            root_path, info_path, dataset_list_path, pipeline, test_mode
        )
        self.root_path = root_path
        self._num_point_features = __class__.NumPointFeatures
        self._class_names = class_names
        self._info_path = info_path
        self.dataset_list_path = dataset_list_path

        xx = lmdb.open(info_path,
               readonly=True,
               max_readers=1,
               lock=False,
               readahead=False,
               meminit=False)
        self.xx = xx
        with xx.begin(write=False) as txn:
           all_list = deserialize(txn.get(b'all_list'))
           all_count = deserialize(txn.get(b'all_count'))
        self.all_list = all_list

    def __len__(self):
        dataset_list = open(self.dataset_list_path, "r").readlines()[:1]
        self.dataset_list = []
        for line in dataset_list:
            line = line.strip()
            self.dataset_list.append(line)
        return len(self.dataset_list)


    @property
    def num_point_features(self):
        return self._num_point_features

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def get_sensor_data(self, idx):
        key = self.dataset_list[idx]

        with self.xx.begin(write=False) as txn:
           data_dict = deserialize(txn.get('data-{}'.format(key).encode("ascii")))
           label_dict = deserialize(txn.get('label-{}'.format(key).encode("ascii")))
        points = data_dict["points"]["first_return"]
        labels = []
        for _, value in label_dict["label_dict"].items():
            bbox = value["obj_bbox"][0]
            cx = bbox["cx"]
            cy = bbox["cy"]
            cz = bbox["cz"]
            length = bbox["length"]
            height = bbox["height"]
            width = bbox["width"]
            heading = bbox["heading"]
            obj_type = value["obj_type"]
            
            label = np.array([cx, cy, cz, length, width, height, heading, obj_type], dtype=np.float32)
            labels.append(label[np.newaxis, :])
        points = np.concatenate(points, axis=0)
        labels = np.concatenate(labels, axis=0)
        num_points_per_gt_box = label_dict['num_points_per_box_first_return']

        # filter sign class
        gt_classes = labels[:, -1]
        labels = labels[gt_classes == 1, :]
        num_points_per_gt_box = num_points_per_gt_box[gt_classes == 1]

       
        gt_classes_name = [self._class_names[int(gt_class) - 1] for gt_class in labels[:, -1]]
        #print("num_points_per_gt", num_points_per_gt_box)
        # to do ignore points 0
        labels[num_points_per_gt_box == 0, -1] = -1
        
         
        res = {
            "lidar": {
                "type": "lidar",
                "points": points,
            },
            "annotations": {
                'boxes': labels[:,:7],
                'names': gt_classes_name,
            },
            "metadata": {
                "token": key,
            },
            "mode": "val" if self.test_mode else "train",
        }
        #np.save("points_origin", points)
        #np.save("gt_boxes_origin", labels[:, :7])
        #np.save("points_after", data["points"])
        #np.save("gt_boxes_after", np.concatenate(data["annos"]["gt_boxes"]))
        return res
    
    def ground_truth_annotations(self):
        gt_boxes = []
        gt_types = []
        gt_frameids = []
        for i, key in enumerate(self.dataset_list):
            with self.xx.begin(write=False) as txn:
                label_dict = deserialize(txn.get('label-{}'.format(key).encode("ascii")))
                data_dict = deserialize(txn.get('data-{}'.format(key).encode("ascii")))
            points = data_dict["points"]["first_return"]
            points = np.concatenate(points, axis=0)
            labels = []
            for key, value in label_dict["label_dict"].items():
                bbox = value["obj_bbox"][0]
                cx = bbox["cx"]
                cy = bbox["cy"]
                cz = bbox["cz"]
                length = bbox["length"]
                height = bbox["height"]
                width = bbox["width"]
                heading = bbox["heading"]
                obj_type = value["obj_type"]
                if obj_type in [1, 2, 4]:
                    box = np.array([cx, cy, cz, length, width, height, heading])
                    if obj_type == 4:
                        box_type = np.array([3])
                    else:
                        box_type = np.array([obj_type])
                    frame_id = np.array([i])

                    gt_boxes.append(box[np.newaxis, :])
                    gt_types.append(box_type)
                    gt_frameids.append(frame_id)
        boxes = np.concatenate(gt_boxes, axis=0) 
        types = np.concatenate(gt_types, axis=0)
        frameids = np.concatenate(gt_frameids, axis=0)
        return boxes, types, frameids
    
    def reformate_results_for_evaluation(self, detections):
        pred_boxes = []
        pred_types = []
        pred_frameids = []
        pred_scores = []
        for i, key in enumerate(self.dataset_list):
            detection = detections[key]
            box = detection["box3d_lidar"]
            score = detection["scores"]
            box_type = detection["label_preds"]
            frame_id = len(score) * [i]
        
            pred_boxes.append(box)
            pred_scores.append(score)
            pred_types.append(box_type + 1)
            pred_frameids.append(frame_id)
        pred_boxes = np.concatenate(pred_boxes, axis=0)
        pred_scores = np.concatenate(pred_scores, axis=0)
        pred_types = np.concatenate(pred_types, axis=0)
        pred_frameids = np.concatenate(pred_frameids, axis=0)
    
        return pred_boxes, pred_types, pred_frameids, pred_scores

    def evaluation(self, detections, output_dir=None, testset=False):
        # pd_bbox, pd_type, pd_frameid, pd_score
        gt_boxes, gt_types, gt_frameids = self.ground_truth_annotations()
        pred_boxes, pred_types, pred_frameids, pred_scores = \
            self.reformate_results_for_evaluation(detections)

        np.save("gt_boxes", gt_boxes)
        np.save("gt_types", gt_types)
        np.save("pred_boxes", pred_boxes)
        np.save("pred_types", pred_types)
        # eval metrics
        mAP = DetectionMetricsEstimatorTest()
        GT_BOXES = (gt_boxes, gt_types, gt_frameids)
        PRED_BOXES = (pred_boxes, pred_types, pred_frameids, pred_scores)
        
        mAP.testAPBasic(PRED_BOXES, GT_BOXES)
    



    
    

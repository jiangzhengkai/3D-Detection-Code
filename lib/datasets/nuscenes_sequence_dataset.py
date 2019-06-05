import os
import random
import os.path as osp
from functools import reduce
from pathlib import Path
import pickle
import time
from functools import partial
from copy import deepcopy
import numpy as np
import json
import random
from lib.core.bbox import box_np_ops
from lib.datasets import preprocess as prep
from lib.datasets import kitti_common as kitti
from torch.utils.data import Dataset
from lib.utils.eval import get_coco_eval_result, get_official_eval_result
from lib.utils.progress_bar import progress_bar_iter as prog_bar
import pyquaternion
from pyquaternion import Quaternion
from lib.deps.nuscenes.nuscenes import NuScenes
from lib.deps.nuscenes.utils import splits
from lib.deps.nuscenes.utils.data_classes import LidarPointCloud
from lib.deps.nuscenes.eval.detection.config import eval_detection_configs
from lib.deps.nuscenes.utils.geometry_utils import view_points, transform_matrix
from lib.deps.nuscenes.utils.data_classes import Box
from lib.deps.nuscenes.eval.detection.config import config_factory
from lib.deps.nuscenes.eval.detection.evaluate import NuScenesEval
from typing import Tuple, List
import time
import operator
from tqdm import tqdm


def eval_main(nusc, eval_version, res_path, eval_set, output_dir):
    cfg = config_factory(eval_version)

    nusc_eval = NuScenesEval(nusc,
                             config=cfg,
                             result_path=res_path,
                             eval_set=eval_set,
                             output_dir=output_dir,
                             verbose=True)
    metrics_summary = nusc_eval.main(plot_examples=10, )


general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}

cls_attr_dist = {
    'barrier': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0
    },
    'bicycle': {
        'cycle.with_rider': 2791,
        'cycle.without_rider': 8946,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0
    },
    'bus': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 9092,
        'vehicle.parked': 3294,
        'vehicle.stopped': 3881
    },
    'car': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 114304,
        'vehicle.parked': 330133,
        'vehicle.stopped': 46898
    },
    'construction_vehicle': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 882,
        'vehicle.parked': 11549,
        'vehicle.stopped': 2102
    },
    'ignore': {
        'cycle.with_rider': 307,
        'cycle.without_rider': 73,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 165,
        'vehicle.parked': 400,
        'vehicle.stopped': 102
    },
    'motorcycle': {
        'cycle.with_rider': 4233,
        'cycle.without_rider': 8326,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0
    },
    'pedestrian': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 157444,
        'pedestrian.sitting_lying_down': 13939,
        'pedestrian.standing': 46530,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0
    },
    'traffic_cone': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0
    },
    'trailer': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 3421,
        'vehicle.parked': 19224,
        'vehicle.stopped': 1895
    },
    'truck': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 21339,
        'vehicle.parked': 55626,
        'vehicle.stopped': 11097
    }
}


class NuScenesSequenceDataset(Dataset):
    NumPointFeatures = 5

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 minor_classes=None,
                 prep_func=None,
                 num_point_features=None,
                 subset=False,
                 **kwargs):
        if 'nsweeps' in kwargs:
            self.nsweeps = kwargs['nsweeps']
        else:
            self.nsweeps = 1
        assert self.nsweeps > 0, "At least input one sweep please!"
        self.logger = None
        if 'logger' in kwargs:
            self.logger = kwargs['logger']
            if self.logger is not None:
                self.logger.info(
                    f"{NuScenesSequenceDataset}'s traning with {self.nsweeps} nsweeps")

        self._root_path = Path(root_path)
        with open(info_path, 'rb') as f:
            self._nusc_infos_all = pickle.load(f)
        if subset:  # if training
            self.frac = int(len(self._nusc_infos_all) * 0.25)

            self._cls_infos = {name: [] for name in class_names}
            for info in self._nusc_infos_all:
                for name in set(info["gt_names"]):
                    if name in class_names:
                        self._cls_infos[name].append(info)

            self.duplicated_samples = sum(
                [len(v) for _, v in self._cls_infos.items()])
            self._cls_dist = {
                k: len(v) / self.duplicated_samples
                for k, v in self._cls_infos.items()
            }

            self._nusc_infos = []

            frac = 1. / len(class_names)
            ratios = [frac / v for v in self._cls_dist.values()]

            for cls_infos, ratio in zip(list(self._cls_infos.values()),
                                        ratios):
                self._nusc_infos += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)).tolist()

            self._cls_infos = {name: [] for name in class_names}
            for info in self._nusc_infos:
                for name in set(info["gt_names"]):
                    if name in class_names:
                        self._cls_infos[name].append(info)

            self._cls_dist = {
                k: len(v) / len(self._nusc_infos)
                for k, v in self._cls_infos.items()
            }
        else:
            self._nusc_infos = self._nusc_infos_all

        if self.logger is not None:
            self.logger.info(f"Using {len(self._nusc_infos)} for training")

        self._num_point_features = 5
        self._class_names = class_names
        self._prep_func = prep_func
        self._name_mapping = general_to_detection
        self._kitti_name_mapping = {}
        for k, v in self._name_mapping.items():
            if v.lower() in ["car", "pedestrian"]:  # we only eval these classes in kitti
                self._kitti_name_mapping[k] = v

        self.version = "v1.0-trainval"
        self.eval_version = "cvpr_2019"

    def reset(self):
        self.logger.info(f"re-sample {self.frac} frames from full set")
        random.shuffle(self._nusc_infos_all)
        self._nusc_infos = self._nusc_infos_all[:self.frac]

    def __len__(self):
        return len(self._nusc_infos)

    def box_velocity(self,
                     nusc,
                     sample_annotation_token: str,
                     max_time_diff: float = 1.5) -> np.ndarray:
        """
        Estimate the velocity for an annotation.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
        :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
        """

        current = nusc.get('sample_annotation', sample_annotation_token)
        has_prev = current['prev'] != ''
        has_next = current['next'] != ''

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = nusc.get('sample_annotation', current['prev'])
        else:
            first = current

        if has_next:
            last = nusc.get('sample_annotation', current['next'])
        else:
            last = current

        pos_last = np.array(last['translation'])
        pos_first = np.array(first['translation'])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * nusc.get('sample',
                                    last['sample_token'])['timestamp']
        time_first = 1e-6 * nusc.get('sample',
                                     first['sample_token'])['timestamp']
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff

    def remove_close(self, points, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """
        x_filt = np.abs(points[0, :]) < radius
        y_filt = np.abs(points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        points = points[:, not_close]
        return points

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        cls_range_map = eval_detection_configs[
            self.eval_version]["class_range"]
        gt_annos = []
        for info in self._nusc_infos:
            gt_names = np.array(info["gt_names"])
            gt_boxes = info["gt_boxes"]
            mask = np.array([n != "ignore" for n in gt_names], dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            det_range = np.array([cls_range_map[n] for n in gt_names])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            N = int(np.sum(mask))
            gt_annos.append({
                "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                "alpha": np.full(N, -10),
                "occluded": np.zeros(N),
                "truncated": np.zeros(N),
                "name": gt_names[mask],
                "location": gt_boxes[mask][:, :3],
                "dimensions": gt_boxes[mask][:, 3:6],
                "rotation_y": gt_boxes[mask][:, 6],
                "token": info['token'],
            })
        return gt_annos

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx, self.nsweeps)
        example = self._prep_func(input_dict=input_dict, logger=self.logger)
        example["current_frame"]["metadata"] = input_dict["current_frame"]["metadata"]
        if "anchors_mask" in example["current_frame"]:
            example["current_frame"]["anchors_mask"] = [
                am.astype(np.uint8) for am in example["current_frame"]["anchors_mask"]
            ]
        return example

    def get_sensor_data(self, query, nsweeps=10):
        idx = query
        read_test_image = False
        if isinstance(query, dict):
            assert "lidar" in query
            idx = query["lidar"]["idx"]
            read_test_image = "cam" in query

        info = self._nusc_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "token": info["token"]
            },
        }
        
        res_keyframe = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "token": info["token"]
            },
        }


        # read for current frame
        lidar_path = Path(info['lidar_path'])
        points = read_file(str(lidar_path))
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        min_distance = 1.0
        assert (nsweeps - 1) <= len(
            info["sweeps"]
        ), "nsweeps {} should not greater than sweep list length {}.".format(
            nsweeps, len(info["sweeps"]))

        def read_sweep(sweep):
            points_sweep = read_file(str(sweep["lidar_path"])).T

            nbr_points = points_sweep.shape[1]
            if sweep["transform_matrix"] is not None:
                points_sweep[:3, :] = sweep["transform_matrix"].dot(
                    np.vstack(
                        (points_sweep[:3, :], np.ones(nbr_points))))[:3, :]
            points_sweep = self.remove_close(points_sweep, min_distance)
            curr_times = sweep["time_lag"] * np.ones(
                (1, points_sweep.shape[1]))

            return points_sweep.T, curr_times.T

        for i in range(nsweeps - 1):
            sweep = info["sweeps"][i]
            points_sweep, times_sweep = read_sweep(sweep)
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        if read_test_image:
            if Path(info["cam_front_path"]).exists():
                with open(str(info["cam_front_path"]), 'rb') as f:
                    image_str = f.read()
            else:
                image_str = None
            res["cam"] = {
                "type": "camera",
                "data": image_str,
                # "datatype": "jpg",
                "datatype": Path(info["cam_front_path"]).suffix[1:],
            }
        res["lidar"]["points"] = points
        res["lidar"]["times"] = times
        res["lidar"]["combined"] = np.hstack([points, times])

        if 'gt_boxes' in info:
            res["lidar"]["annotations"] = {
                'boxes': info["gt_boxes"].astype(np.float32),
                'names': info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
       

        res_squences = {} 
        res_squences["current_frame"] = res

        # for last_keyframe read
       
        random_index = random.randint(0,nsweeps-2)
        sweep_keyframe = info["sweeps"][random_index]
        key_points_sweep, key_times_sweep = read_sweep(sweep_keyframe)
        key_times_sweep = key_times_sweep.astype(key_points_sweep.dtype)

        res_keyframe["lidar"]["points"] = key_points_sweep
        res_keyframe["lidar"]["times"] = key_times_sweep
        res_keyframe["lidar"]["combined"] = np.hstack([key_points_sweep, key_times_sweep])
        res_squences['keyframe'] = res_keyframe

        return res_squences

    def evaluation_kitti(self, detections, output_dir):
        """eval by kitti evaluation tool
        """
        class_names = self._class_names
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None

        dets = detections
        detections = []

        miss = 0
        for gt in gt_annos:
            try:
                detections.append(dets[gt['token']])
            except:
                miss += 1

        assert miss == 0

        dt_annos = []
        for det in detections:
            final_box_preds = det["box3d_lidar"].numpy()
            label_preds = det["label_preds"].numpy()
            scores = det["scores"].detach().numpy()
            anno = kitti.get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                anno["bbox"].append(np.array([0, 0, 50, 50]))
                anno["alpha"].append(-10)
                anno["dimensions"].append(box3d_lidar[j, 3:6])
                anno["location"].append(box3d_lidar[j, :3])
                anno["rotation_y"].append(box3d_lidar[j, 6])
                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])
                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                dt_annos.append(anno)
            else:
                dt_annos.append(kitti.empty_result_anno())
            num_example = dt_annos[-1]["name"].shape[0]
            dt_annos[-1]["metadata"] = det["metadata"]

        for anno in gt_annos:
            names = anno["name"].tolist()
            mapped_names = []
            for n in names:
                if n in self._name_mapping:
                    mapped_names.append(self._name_mapping[n])
                else:
                    mapped_names.append(n)
            anno["name"] = np.array(mapped_names)
        for anno in dt_annos:
            names = anno["name"].tolist()
            mapped_names = []
            for n in names:
                if n in self._name_mapping:
                    mapped_names.append(self._name_mapping[n])
                else:
                    mapped_names.append(n)
            anno["name"] = np.array(mapped_names)
        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        z_axis = 2
        z_center = 0.5
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        result_official_dict = get_official_eval_result(gt_annos,
                                                        dt_annos,
                                                        mapped_class_names,
                                                        z_axis=z_axis,
                                                        z_center=z_center)
        result_coco = get_coco_eval_result(gt_annos,
                                           dt_annos,
                                           mapped_class_names,
                                           z_axis=z_axis,
                                           z_center=z_center)
        return {
            "results": {
                "official": result_official_dict["result"],
                "coco": result_coco["result"],
            },
            "detail": {
                "official": result_official_dict["detail"],
                "coco": result_coco["detail"],
            },
        }

    def evaluation_nusc(self, detections, output_dir, testset=False):
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        if not testset:
            dets = []

            gt_annos = self.ground_truth_annotations
            assert gt_annos is not None

            miss = 0
            for gt in gt_annos:
                try:
                    dets.append(detections[gt['token']])
                except:
                    miss += 1

            assert miss == 0
        else:
            dets = [v for _, v in detections.items()]
            assert len(detections) == 6008

        nusc_annos = {
            'results': {},
            'meta': None,
        }

        nusc = NuScenes(version=version,
                        dataroot=str(self._root_path),
                        verbose=True)

        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        for det in dets:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(nusc, boxes,
                                              det["metadata"]["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            "car", "construction_vehicle", "bus", "truck",
                            "trailer"
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ["bicycle", "motorcycle"]:
                        attr = 'cycle.with_rider'
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = 'pedestrian.standing'
                    else:
                        attr = None

                nusc_anno = {
                    "sample_token":
                    det["metadata"]["token"],
                    "translation":
                    box.center.tolist(),
                    "size":
                    box.wlh.tolist(),
                    "rotation":
                    box.orientation.elements.tolist(),
                    "velocity":
                    box.velocity[:2].tolist(),
                    "detection_name":
                    name,
                    "detection_score":
                    box.score,
                    "attribute_name":
                    attr if attr is not None else max(
                        cls_attr_dist[name].items(),
                        key=operator.itemgetter(1))[0],
                }
                annos.append(nusc_anno)
            nusc_annos['results'].update({det["metadata"]["token"]: annos})

        nusc_annos['meta'] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        res_path = str(Path(output_dir) / "results.json")
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(nusc, self.eval_version, res_path,
                      eval_set_map[self.version], output_dir)

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ', '.join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ', '.join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            return {
                "results": {
                    "nusc": result
                },
                "detail": {
                    "nusc": detail
                },
            }
        else:
            return None

    def evaluation(self, detections, output_dir, testset=False):
        res_nusc = self.evaluation_nusc(detections,
                                        output_dir,
                                        testset=testset)

        if res_nusc is not None:
            res = {
                "results": {
                    "nusc": res_nusc["results"]["nusc"],
                    # "kitti.official": res_kitti["results"]["official"],
                    # "kitti.coco": res_kitti["results"]["coco"],
                },
                "detail": {
                    "eval.nusc": res_nusc["detail"]["nusc"],
                    # "eval.kitti": {
                    # "official": res_kitti["detail"]["official"],
                    # "coco": res_kitti["detail"]["coco"],
                    # },
                },
            }
        else:
            res = None

        return res


def read_file(path):
    points = None
    try_cnt = 0
    while points is None:
        try_cnt += 1
        try:
            points = \
                np.fromfile(path, dtype=np.float32).reshape([-1, 5])[:, :4]
        except:
            points = None
        if try_cnt > 3:
            break

    return points


def _second_det_to_nusc_box(detection):
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    box3d[:, -1] = -box3d[:, -1] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(box3d[i, :3],
                  box3d[i, 3:6],
                  quat,
                  label=labels[i],
                  score=scores[i],
                  velocity=velocity)
        box_list.append(box)
    return box_list


def _lidar_nusc_box_to_global(nusc, boxes, sample_token):
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record["data"]["LIDAR_TOP"]
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec['next'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def get_sample_data(nusc,
                    sample_data_token: str,
                    selected_anntokens: List[str] = None) -> \
        Tuple[str, List[Box], np.array]:
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param selected_anntokens: If provided only return the selected annotation.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:

        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         nsweeps=10):
    from nuscenes.utils.geometry_utils import transform_matrix

    train_nusc_infos = []
    val_nusc_infos = []

    ref_chan = "LIDAR_TOP"  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for sample in prog_bar(nusc.sample):
        """ Manual save info["sweeps"] """
        # Get reference pose and timestamp
        # ref_chan == "LIDAR_TOP"
        ref_sd_token = sample["data"][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_cs_rec = nusc.get('calibrated_sensor',
                              ref_sd_rec['calibrated_sensor_token'])
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample["data"]["CAM_FRONT"]
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(
            ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'],
                                        Quaternion(ref_cs_rec['rotation']),
                                        inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'],
                                           Quaternion(
                                               ref_pose_rec['rotation']),
                                           inverse=True)

        info = {
            "lidar_path": ref_lidar_path,
            "cam_front_path": ref_cam_path,
            "cam_intrinsic": ref_cam_intrinsic,
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
        }

        sample_data_token = sample['data'][chan]
        curr_sd_rec = nusc.get('sample_data', sample_data_token)
        sweeps = []
        while len(sweeps) < nsweeps - 1:
            if curr_sd_rec['prev'] == "":
                if len(sweeps) == 0:
                    sweep = {
                        "lidar_path": ref_lidar_path,
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": None,
                        "time_lag": curr_sd_rec['timestamp'] * 0,
                        # time_lag: 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])

                # Get past pose
                current_pose_rec = nusc.get('ego_pose',
                                            curr_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(
                    current_pose_rec['translation'],
                    Quaternion(current_pose_rec['rotation']),
                    inverse=False)

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    'calibrated_sensor',
                    curr_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(
                    current_cs_rec['translation'],
                    Quaternion(current_cs_rec['rotation']),
                    inverse=False)

                tm = reduce(np.dot, [
                    ref_from_car, car_from_global, global_from_car,
                    car_from_current
                ])

                lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])

                time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']

                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": curr_sd_rec['token'],
                    "transform_matrix": tm,
                    "global_from_car": global_from_car,
                    "car_from_current": car_from_current,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        assert len(
            info["sweeps"]
        ) == nsweeps - 1, f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, you should duplicate to sweep num {nsweeps-1}"
        """ read from api """
        # sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        #
        # # Get boxes in lidar frame.
        # lidar_path, boxes, cam_intrinsic = nusc.get_sample_data(
        #     sample['data']['LIDAR_TOP'])
        #
        # # Get aggregated point cloud in lidar frame.
        # sample_rec = nusc.get('sample', sd_record['sample_token'])
        # chan = sd_record['channel']
        # ref_chan = 'LIDAR_TOP'
        # pc, times = LidarPointCloud.from_file_multisweep(nusc,
        #                                                  sample_rec,
        #                                                  chan,
        #                                                  ref_chan,
        #                                                  nsweeps=nsweeps)
        # lidar_path = osp.join(nusc.dataroot, "sample_10sweeps/LIDAR_TOP",
        #                       sample['data']['LIDAR_TOP'] + ".bin")
        # pc.points.astype('float32').tofile(open(lidar_path, "wb"))
        #
        # info = {
        #     "lidar_path": lidar_path,
        #     "token": sample["token"],
        #     # "timestamp": times,
        # }

        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)
            # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation)
                             for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate(
                [locs, dims, velocity[:, :2], -rots - np.pi / 2], axis=1)
            # gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            assert len(annotations) == len(gt_boxes) == len(velocity)

            info["gt_boxes"] = gt_boxes
            info["gt_boxes_velocity"] = velocity
            info["gt_names"] = np.array(
                [general_to_detection[name] for name in names])
            info["gt_boxes_token"] = tokens

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def create_nuscenes_infos_test(root_path, version="v1.0-trainval", nsweeps=10):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        # random.shuffle(train_scenes)
        # train_scenes = train_scenes[:int(len(train_scenes)*0.2)]
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(
            f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(nusc,
                                                            train_scenes,
                                                            val_scenes,
                                                            test,
                                                            nsweeps=nsweeps)

    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        with open(
                root_path /
                "infos_test_{:02d}sweeps_withvelo.pkl".format(nsweeps),
                'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        with open(
                root_path /
                "infos_train_{:02d}sweeps_withvelo.pkl".format(nsweeps),
                'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(
                root_path /
                "infos_val_{:02d}sweeps_withvelo.pkl".format(nsweeps),
                'wb') as f:
            pickle.dump(val_nusc_infos, f)

def create_nuscenes_infos(root_path, version="v1.0-trainval", nsweeps=10):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        # random.shuffle(train_scenes)
        # train_scenes = train_scenes[:int(len(train_scenes)*0.2)]
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(
            f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(nusc,
                                                            train_scenes,
                                                            val_scenes,
                                                            test,
                                                            nsweeps=nsweeps)

    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        with open(
                root_path /
                "infos_test_{:02d}sweeps_withvelo.pkl".format(nsweeps),
                'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        with open(
                root_path /
                "infos_train_{:02d}sweeps_withvelo.pkl".format(nsweeps),
                'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(
                root_path /
                "infos_val_{:02d}sweeps_withvelo.pkl".format(nsweeps),
                'wb') as f:
            pickle.dump(val_nusc_infos, f)


def get_box_mean(info_path, class_name="vehicle.car"):
    with open(info_path, 'rb') as f:
        nusc_infos = pickle.load(f)

    gt_boxes_list = []
    for info in nusc_infos:
        mask = np.array([s == class_name for s in info["gt_names"]],
                        dtype=np.bool_)
        gt_boxes_list.append(info["gt_boxes"][mask].reshape(-1, 7))
    gt_boxes_list = np.concatenate(gt_boxes_list, axis=0)
    print(gt_boxes_list.mean(0))

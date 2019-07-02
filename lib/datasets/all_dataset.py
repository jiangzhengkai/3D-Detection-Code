import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np

from lib.core.bbox import box_np_ops
from lib.datasets.kitti.kitti_dataset import KittiDataset
from lib.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from lib.datasets.nuscenes.nuscenes_sequence_dataset import NuScenesSequenceDataset
from lib.utils.progress_bar import progress_bar_iter as prog_bar

def get_dataset_class(name):
    return {
        "KittiDataset": KittiDataset,
        "NuScenesDataset": NuScenesDataset,
        "NuScenesSequenceDataset": NuScenesSequenceDataset,
    }[name]

def create_groundtruth_database(
        dataset_class_name,
        data_path,
        info_path=None,
        used_classes=None,
        database_save_path=None,
        db_info_save_path=None,
        relative_path=True,
        add_rgb=False,
        lidar_only=False,
        bev_only=False,
        coors_range=None,
        **kwargs,
):
    if 'nsweeps' in kwargs:
        dataset = get_dataset_class(dataset_class_name)(
            info_path=info_path,
            root_path=data_path,
            nsweeps=kwargs['nsweeps'])
        nsweeps = dataset.nsweeps
    else:
        dataset = get_dataset_class(dataset_class_name)(info_path=info_path,
                                                        root_path=data_path)
        nsweeps = 1

    root_path = Path(data_path)

    if dataset_class_name == 'NuScenesDataset':
        database_save_path = root_path / 'gt_database_10sweeps_withvelo'
        db_info_save_path = root_path / "dbinfos_train_10sweeps_withvelo.pkl"
    else:
        database_save_path = root_path / 'gt_database'
        db_info_save_path = root_path / "dbinfos_train.pkl"

    database_save_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0

    # def prepare_single_data(index):
    for index in tqdm(range(len(dataset))):
        image_idx = index
        # modified to nuscenes
        # sensor_datas = dataset.get_sensor_datas(j, nsweeps) # return max_sweeps length list
        sensor_data = dataset.get_sensor_data(index)
        # for nsweep, sensor_data in enumerate(sensor_datas):
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]
        points = sensor_data["lidar"]["points"]
        #points = sensor_data["lidar"]["combined"]
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]
        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            filepath = database_save_path / filename
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_path = str(database_save_path.stem + "/" + filename)
                else:
                    db_path = str(filepath)

                db_info = {
                    "name": names[i],
                    "path": db_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
        # print(f"Finish {index}th sample")

    print("dataset length: ", len(dataset))

    print(f"finish {nsweeps}'s dbinfos preprocess")

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


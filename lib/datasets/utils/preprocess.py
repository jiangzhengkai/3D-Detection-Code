import itertools
import numpy as np
from lib.datasets.kitti import kitti_common as kitti
from lib.core.bbox import box_np_ops
from lib.datasets.utils import preprocess as prep
from collections import defaultdict
from lib.utils import simplevis
import numba
from lib.core.bbox.geometry import points_in_convex_polygon_3d_jit, points_in_convex_polygon_jit


def collate_batch_fn(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    batch_size = len(batch_list)
    ret = {}
    for key, elems in example_merged.items():
        if key in ['voxels', 'num_points', 'num_gt', 'voxel_labels']:
            ret[key] = np.concatenate(elems, axis=0)
        elif key in ["gt_boxes",]:
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 9))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, :len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res
        elif key == 'metadata':
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key in ['coordinates', 'points']:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)),
                                  mode='constant',
                                  constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key in [
                "anchors", "anchors_mask", "reg_targets", "reg_weights",
                "labels"
        ]:
            ret[key] = defaultdict(list)
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(ele)
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


def collate_sequence_batch_fn(batch_list):
    ################ current frame ##################
    example_current_frame_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example["current_frame"].items():
            example_current_frame_merged[k].append(v)
    batch_size = len(batch_list)
    ret_current_frame = {}
    for key, elems in example_current_frame_merged.items():
        if key in ['voxels', 'num_points', 'num_gt', 'voxel_labels']:
            ret_current_frame[key] = np.concatenate(elems, axis=0)
        elif key in ["gt_boxes",]:
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 9))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, :len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret_current_frame[key] = res
        elif key == 'metadata':
            ret_current_frame[key] = elems
        elif key == "calib":
            ret_current_frame[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret_current_frame[key][k1] = [v1]
                    else:
                        ret_current_frame[key][k1].append(v1)
            for k1, v1 in ret_current_frame[key].items():
                ret_current_frame[key][k1] = np.stack(v1, axis=0)
        elif key in ['coordinates', 'points']:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)),
                                  mode='constant',
                                  constant_values=i)
                coors.append(coor_pad)
            ret_current_frame[key] = np.concatenate(coors, axis=0)
        elif key in [
                "anchors", "anchors_mask", "reg_targets", "reg_weights",
                "labels"
        ]:
            ret_current_frame[key] = defaultdict(list)
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret_current_frame[key][str(idx)].append(ele)
        else:
            ret_current_frame[key] = np.stack(elems, axis=0)

    ################ key frame ##################
    example_keyframe_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example["keyframe"].items():
            example_keyframe_merged[k].append(v)
    batch_size = len(batch_list)
    ret_keyframe = {}
    for key, elems in example_keyframe_merged.items():
        if key in ['voxels', 'num_points', 'num_gt', 'voxel_labels']:
            ret_keyframe[key] = np.concatenate(elems, axis=0)
        elif key == "calib":
            ret_keyframe[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret_keyframe[key][k1] = [v1]
                    else:
                        ret_keyframe[key][k1].append(v1)
            for k1, v1 in ret_keyframe[key].items():
                ret_keyframe[key][k1] = np.stack(v1, axis=0)
        elif key in ['coordinates', 'points']:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)),
                                  mode='constant',
                                  constant_values=i)
                coors.append(coor_pad)
            ret_keyframe[key] = np.concatenate(coors, axis=0)
        else:
            ret_keyframe[key] = np.stack(elems, axis=0)


    rets = {}
    rets["current_frame"] = ret_current_frame
    rets["keyframe"] = ret_keyframe
    return rets

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def prep_pointcloud(config,
		    input_dict,
		    root_path,
		    voxel_generator,
		    target_assigners,
		    db_sampler=None,
		    remove_outside_points=False,
		    training=True,
		    create_targets=True,
		    num_point_features=4,
		    anchor_cache=None,
		    random_crop=False,
		    reference_detections=None,
		    out_size_factor=2,
		    out_dtype=np.float32,
		    min_points_in_gt=-1,
		    logger=None):
    """
    convert point cloud to voxels, create targets if ground truths exists
    input_dict format: dataset.get_sensor_data format
    """
    prep_config = config.input.train.preprocess if training  else config.input.eval.preprocess
    remove_environment = prep_config.remove_environment

    max_num_voxels = config.input.train.preprocess.max_num_voxels if training else config.input.eval.preprocess.max_num_voxels
    shuffle_points = prep_config.shuffle
    anchor_area_threshold = prep_config.anchor_area_threshold

    if training:
        remove_unknown = prep_config.remove_unknow_examples
        gt_rotation_noise = prep_config.gt_rotation_noise
        gt_location_noise_std = prep_config.gt_location_noise
        global_rotation_noise = prep_config.global_rotation_noise
        global_scale_noise = prep_config.global_scale_noise
        global_random_rot_range = prep_config.global_rotation_per_object_range
        global_translate_noise_std = prep_config.global_translation_noise
        gt_points_drop = prep_config.gt_drop_percentage
        gt_drop_max_keep = prep_config.gt_drop_max_keep_points
        remove_points_after_sample = prep_config.remove_points_after_sample

    task_class_names = [target_assigner.classes for target_assigner in target_assigners]
    class_names = list(itertools.chain(*task_class_names))

    if config.input.train.dataset.type == "KittiDataset":
        points = input_dict["lidar"]["points"]
    else:
        points = input_dict["lidar"]["combined"]

    if training:
        anno_dict = input_dict["lidar"]["annotations"]
        gt_dict = {
            "gt_boxes": anno_dict["boxes"],
            "gt_names": np.array(anno_dict["names"]).reshape(-1),
        }
        if "difficulty" not in anno_dict:
            difficulty = np.zeros([anno_dict["boxes"].shape[0]], dtype=np.int32)
            gt_dict["difficulty"] = difficulty
        else:
            gt_dict["difficulty"] = anno_dict["difficulty"]

    calib = None
    if "calib" in input_dict:
        calib = input_dict["calib"]

    if reference_detections is not None:
        assert calib is not None and "image" in input_dict
        C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
        frustums = box_np_ops.get_frustum_v2(reference_detections, C)
        frustums -= T
        frustums = np.einsum('ij, akj->aki', np.linalg.inv(R), frustums)
        frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
        masks = points_in_convex_polygon_3d_jit(points, surfaces)
        points = points[masks.any(-1)]

    if remove_outside_points:
        assert calib is not None
        image_shape = input_dict["image"]["image_shape"]
        points = box_np_ops.remove_outside_points(points, calib["rect"],
                                                  calib["Trv2c"], calib["P2"],
                                                  image_shape)
    if remove_environment is True and training:
        selected = kitti.keep_arrays_by_name(gt_names, target_assigner.classes)
        _dict_select(gt_dict, selected)
        masks = box_np_ops.points_in_rbbox(points, gt_dict["gt_boxes"])
        points = points[masks.any(-1)]

    if training:
        # boxes_lidar = gt_dict["gt_boxes"]
        # bev_map = simplevis.nuscene_vis(points, boxes_lidar)
        # cv2.imshow('pre-noise', bev_map)
        selected = kitti.drop_arrays_by_name(gt_dict["gt_names"], ["DontCare", "ignore"])
        _dict_select(gt_dict, selected)
        if remove_unknown:
            remove_mask = gt_dict["difficulty"] == -1
            keep_mask = np.logical_not(remove_mask)
            _dict_select(gt_dict, keep_mask)
        gt_dict.pop("difficulty")

        if min_points_in_gt > 0:
            # points_count_rbbox takes 10ms with 10 sweeps
            # nuscenes data
            point_counts = box_np_ops.points_count_rbbox(points, gt_dict["gt_boxes"])
            mask = point_counts >= min_points_in_gt
            _dict_select(gt_dict, mask)
        gt_boxes_mask = np.array([n in class_names for n in gt_dict["gt_names"]], dtype=np.bool_)
        if db_sampler is not None:
            group_ids = None
            sampled_dict = db_sampler.sample_all(
                root_path,
                gt_dict["gt_boxes"],
                gt_dict["gt_names"],
                num_point_features,
                random_crop,
                gt_group_ids=group_ids,
                calib=calib,
            )
            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                gt_dict["gt_names"] = np.concatenate(
                    [gt_dict["gt_names"], sampled_gt_names], axis=0)
                gt_dict["gt_boxes"] = np.concatenate(
                    [gt_dict["gt_boxes"], sampled_gt_boxes])
                gt_boxes_mask = np.concatenate(
                    [gt_boxes_mask, sampled_gt_masks], axis=0)

                if remove_points_after_sample:
                    masks = box_np_ops.points_in_rbbox(points, sampled_gt_boxes)
                    points = points[np.logical_not(masks.any(-1))]
                points = np.concatenate([sampled_points, points], axis=0)
        pc_range = voxel_generator.point_cloud_range
        prep.noise_per_object_v3_(
            gt_dict["gt_boxes"],
            points,
            gt_boxes_mask,
            rotation_perturb=gt_rotation_noise,
            center_noise_std=gt_location_noise_std,
            global_random_rot_range=global_random_rot_range,
            group_ids=group_ids,
            num_try=100)

        _dict_select(gt_dict, gt_boxes_mask)
        gt_classes = np.array([class_names.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32)
        gt_dict["gt_classes"] = gt_classes
        gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"], points)
        gt_dict["gt_boxes"], points = prep.global_rotation(gt_dict["gt_boxes"], points, rotation=global_rotation_noise)
        gt_dict["gt_boxes"], points = prep.global_scaling_v2(gt_dict["gt_boxes"], points, *global_scale_noise)
        #prep.global_translate_(gt_dict["gt_boxes"], points, global_translate_noise_std)

        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
        _dict_select(gt_dict, mask)

        task_masks = []
        flag = 0
        for class_name in task_class_names:
            task_masks.append([np.where(gt_dict['gt_classes'] == class_name.index(i) + 1 + flag) for i in class_name])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        task_names = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            task_name = []
            for m in mask:
                task_box.append(gt_dict['gt_boxes'][m])
                task_class.append(gt_dict['gt_classes'][m] - flag2)
                task_name.append(gt_dict['gt_names'][m])
            task_boxes.append(np.concatenate(task_box, axis=0))
            task_classes.append(np.concatenate(task_class))
            task_names.append(np.concatenate(task_name))
            flag2 += len(mask)

        for task_box in task_boxes:
            # limit rad to [-pi, pi]
            task_box[:, -1] = box_np_ops.limit_period(task_box[:, -1],
                                                          offset=0.5,
                                                           period=2 * np.pi)
        # boxes_lidar = gt_dict["gt_boxes"]
        # bev_map = simplevis.nuscene_vis(points, boxes_lidar)
        # cv2.imshow('post-noise', bev_map)
        # cv2.waitKey(0)
        gt_dict["gt_classes"] = task_classes
        gt_dict["gt_names"] = task_names
        gt_dict["gt_boxes"] = task_boxes

    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    voxels, coordinates, num_points = voxel_generator.generate(points, max_num_voxels)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

    example = {
        'voxels': voxels,
        'num_points': num_points,
        'points': points,
        'coordinates': coordinates,
        "num_voxels": num_voxels,
        #"gt_boxes": gt_dict["gt_boxes"],
        #"gt_dict": gt_dict,
    }
    if calib is not None:
        example["calib"] = calib

    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    if anchor_cache is not None:
        anchorss = anchor_cache["anchors"]
        anchors_bvs = anchor_cache["anchors_bv"]
        anchors_dicts = anchor_cache["anchors_dict"]
    else:
        rets = [
            target_assigner.generate_anchors(feature_map_size)
            for target_assigner in target_assigners
        ]
        anchorss = [ret["anchors"].reshape([-1, 7]) for ret in rets]
        anchors_dicts = [
            target_assigner.generate_anchors_dict(feature_map_size)
            for target_assigner in target_assigners
        ]
        anchors_bvs = [
            box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
            for anchors in anchorss
        ]

    example["anchors"] = anchorss

    if anchor_area_threshold >= 0:
        example["anchors_mask"] = []
        for idx, anchors_bv in enumerate(anchors_bvs):
            anchors_mask = None
            # slow with high resolution. recommend disable this forever.
            coors = coordinates
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                coors, tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = box_np_ops.fused_get_anchors_area(
                dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
            anchors_mask = anchors_area > anchor_area_threshold
            # example['anchors_mask'] = anchors_mask.astype(np.uint8)
            example['anchors_mask'].append(anchors_mask)

    if not training:
        return example
    if create_targets:
        targets_dicts = []
        for idx, target_assigner in enumerate(target_assigners):
            if "anchors_mask" in example:
                anchors_mask = example["anchors_mask"][idx]
            else:
                anchors_mask = None
            targets_dict = target_assigner.assign_v2(
                anchors_dicts[idx],
                gt_dict["gt_boxes"][idx],
                anchors_mask,
                gt_classes=gt_dict["gt_classes"][idx],
                gt_names=gt_dict["gt_names"][idx])
            targets_dicts.append(targets_dict)

        example.update({
            'labels':
            [targets_dict['labels'] for targets_dict in targets_dicts],
            'reg_targets':
            [targets_dict['bbox_targets'] for targets_dict in targets_dicts],
            'reg_weights': [
                targets_dict['bbox_outside_weights']
                for targets_dict in targets_dicts
            ],
        })
    return example


def prep_sequence_pointcloud(config,
		             input_dict,
		             root_path,
		             voxel_generator,
		             target_assigners,
		             db_sampler=None,
		             remove_outside_points=False,
		             training=True,
		             create_targets=True,
		             num_point_features=4,
		             anchor_cache=None,
		             random_crop=False,
		             reference_detections=None,
		             out_size_factor=2,
		             out_dtype=np.float32,
		             min_points_in_gt=-1,
		             logger=None):
    """
    convert point cloud to voxels, create targets if ground truths exists
    input_dict format: dataset.get_sensor_data format
    """
    prep_config = config.input.train.preprocess if training  else config.input.eval.preprocess
    remove_environment = prep_config.remove_environment

    max_num_voxels = config.input.train.preprocess.max_num_voxels if training else config.input.eval.preprocess.max_num_voxels
    shuffle_points = prep_config.shuffle
    anchor_area_threshold = prep_config.anchor_area_threshold
    calib = None
    if "calib" in input_dict:
        calib = input_dict["current_frame"]["calib"]


    if training:
        remove_unknown = prep_config.remove_unknow_examples
        gt_rotation_noise = prep_config.gt_rotation_noise
        gt_location_noise_std = prep_config.gt_location_noise
        global_rotation_noise = prep_config.global_rotation_noise
        global_scale_noise = prep_config.global_scale_noise
        global_random_rot_range = prep_config.global_rotation_per_object_range
        global_translate_noise_std = prep_config.global_translation_noise
        gt_points_drop = prep_config.gt_drop_percentage
        gt_drop_max_keep = prep_config.gt_drop_max_keep_points
        remove_points_after_sample = prep_config.remove_points_after_sample

    task_class_names = [target_assigner.classes for target_assigner in target_assigners]
    class_names = list(itertools.chain(*task_class_names))
    #################### for current frame ##############################################
    points = input_dict["current_frame"]["lidar"]["combined"]
    keyframe_points = input_dict["keyframe"]["lidar"]["combined"]

    if training:
        anno_dict = input_dict["current_frame"]["lidar"]["annotations"]
        gt_dict = {
            "gt_boxes": anno_dict["boxes"],
            "gt_names": np.array(anno_dict["names"]).reshape(-1),
        }
        if "difficulty" not in anno_dict:
            difficulty = np.zeros([anno_dict["boxes"].shape[0]], dtype=np.int32)
            gt_dict["difficulty"] = difficulty
        else:
            gt_dict["difficulty"] = anno_dict["difficulty"]


    if training:
        selected = kitti.drop_arrays_by_name(gt_dict["gt_names"], ["DontCare", "ignore"])
        _dict_select(gt_dict, selected)
        if remove_unknown:
            remove_mask = gt_dict["difficulty"] == -1
            keep_mask = np.logical_not(remove_mask)
            _dict_select(gt_dict, keep_mask)
        gt_dict.pop("difficulty")

        if min_points_in_gt > 0:
            # points_count_rbbox takes 10ms with 10 sweeps nuscenes data
            point_counts = box_np_ops.points_count_rbbox(points, gt_dict["gt_boxes"])
            mask = point_counts >= min_points_in_gt
            _dict_select(gt_dict, mask)

        gt_boxes_mask = np.array([n in class_names for n in gt_dict["gt_names"]], dtype=np.bool_)
        if db_sampler is not None:
            group_ids = None
            sampled_dict = db_sampler.sample_all(
                root_path,
                gt_dict["gt_boxes"],
                gt_dict["gt_names"],
                num_point_features,
                random_crop,
                gt_group_ids=group_ids,
                calib=calib,
            )
            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                gt_dict["gt_names"] = np.concatenate(
                    [gt_dict["gt_names"], sampled_gt_names], axis=0)
                gt_dict["gt_boxes"] = np.concatenate(
                    [gt_dict["gt_boxes"], sampled_gt_boxes])
                gt_boxes_mask = np.concatenate(
                    [gt_boxes_mask, sampled_gt_masks], axis=0)

                if remove_points_after_sample:
                    masks = box_np_ops.points_in_rbbox(points, sampled_gt_boxes)
                    points = points[np.logical_not(masks.any(-1))]
                points = np.concatenate([sampled_points, points], axis=0)
        pc_range = voxel_generator.point_cloud_range

        prep.noise_per_object_v3_(
            gt_dict["gt_boxes"],
            points,
            gt_boxes_mask,
            rotation_perturb=gt_rotation_noise,
            center_noise_std=gt_location_noise_std,
            global_random_rot_range=global_random_rot_range,
            group_ids=group_ids,
            num_try=100)

        _dict_select(gt_dict, gt_boxes_mask)
        gt_classes = np.array([class_names.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32)
        gt_dict["gt_classes"] = gt_classes

        ######## current_frame and keyframe data concatenate
        num_points_current = points.shape[0]
        points = np.concatenate((points, keyframe_points), axis=0)


        gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"], points)
        gt_dict["gt_boxes"], points = prep.global_rotation(gt_dict["gt_boxes"], points, rotation=global_rotation_noise)
        gt_dict["gt_boxes"], points = prep.global_scaling_v2(gt_dict["gt_boxes"], points, *global_scale_noise)
        #prep.global_translate_(gt_dict["gt_boxes"], points, global_translate_noise_std)
        ######## slice augumented data ########
        points_keyframe = points[num_points_current:, :]
        points = points[:num_points_current, :]

        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
        _dict_select(gt_dict, mask)

        task_masks = []
        flag = 0
        for class_name in task_class_names:
            task_masks.append([np.where(gt_dict['gt_classes'] == class_name.index(i) + 1 + flag) for i in class_name])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        task_names = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            task_name = []
            for m in mask:
                task_box.append(gt_dict['gt_boxes'][m])
                task_class.append(gt_dict['gt_classes'][m] - flag2)
                task_name.append(gt_dict['gt_names'][m])
            task_boxes.append(np.concatenate(task_box, axis=0))
            task_classes.append(np.concatenate(task_class))
            task_names.append(np.concatenate(task_name))
            flag2 += len(mask)

        for task_box in task_boxes:
            # limit rad to [-pi, pi]
            task_box[:, -1] = box_np_ops.limit_period(task_box[:, -1],
                                                      offset=0.5,
                                                      period=2 * np.pi)
        gt_dict["gt_classes"] = task_classes
        gt_dict["gt_names"] = task_names
        gt_dict["gt_boxes"] = task_boxes

    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    voxels, coordinates, num_points = voxel_generator.generate(points, max_num_voxels)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

    ################################keyframe voxels #################################
    keyframe_voxels, keyframe_coordinates, keyframe_num_points = voxel_generator.generate(keyframe_points, max_num_voxels)
    keyframe_num_voxels = np.array([keyframe_voxels.shape[0]], dtype=np.int64)
    example = {
        'voxels': voxels,
        'num_points': num_points,
        'points': points,
        'coordinates': coordinates,
        "num_voxels": num_voxels,
        #"gt_boxes": gt_dict["gt_boxes"],
        #"gt_dict": gt_dict,
    }
    example_keyframe = {
        'voxels': keyframe_voxels,
        'num_points': keyframe_num_points,
        'points': keyframe_points,
        'coordinates': keyframe_coordinates,
        'num_voxels': keyframe_num_voxels
    }

    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    if anchor_cache is not None:
        anchorss = anchor_cache["anchors"]
        anchors_bvs = anchor_cache["anchors_bv"]
        anchors_dicts = anchor_cache["anchors_dict"]
    else:
        rets = [
            target_assigner.generate_anchors(feature_map_size)
            for target_assigner in target_assigners
        ]
        anchorss = [ret["anchors"].reshape([-1, 7]) for ret in rets]
        anchors_dicts = [
            target_assigner.generate_anchors_dict(feature_map_size)
            for target_assigner in target_assigners
        ]
        anchors_bvs = [
            box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
            for anchors in anchorss
        ]

    example["anchors"] = anchorss

    if anchor_area_threshold >= 0:
        example["anchors_mask"] = []
        for idx, anchors_bv in enumerate(anchors_bvs):
            anchors_mask = None
            # slow with high resolution. recommend disable this forever.
            coors = coordinates
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                coors, tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = box_np_ops.fused_get_anchors_area(
                dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
            anchors_mask = anchors_area > anchor_area_threshold
            example['anchors_mask'].append(anchors_mask)
    example_sequences = {}
    example_sequences["current_frame"] = example
    example_sequences["keyframe"] = example_keyframe

    if not training:
        return example_sequences
    if create_targets:
        targets_dicts = []
        for idx, target_assigner in enumerate(target_assigners):
            if "anchors_mask" in example:
                anchors_mask = example["anchors_mask"][idx]
            else:
                anchors_mask = None
            targets_dict = target_assigner.assign_v2(
                anchors_dicts[idx],
                gt_dict["gt_boxes"][idx],
                anchors_mask,
                gt_classes=gt_dict["gt_classes"][idx],
                gt_names=gt_dict["gt_names"][idx])
            targets_dicts.append(targets_dict)

        example_sequences["current_frame"].update({
            'labels':
            [targets_dict['labels'] for targets_dict in targets_dicts],
            'reg_targets':
            [targets_dict['bbox_targets'] for targets_dict in targets_dicts],
            'reg_weights': [
                targets_dict['bbox_outside_weights']
                for targets_dict in targets_dicts
            ],
        })
    ###################################### for keyframe #########################################

    return example_sequences


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (
                                C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (
                                C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret

def global_translate_(gt_boxes, points, noise_translate_std):
    """
    Apply global translation to gt_boxes and points.
    """

    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array([noise_translate_std, noise_translate_std, noise_translate_std])
    if all([e == 0 for e in noise_translate_std]):
        return gt_boxes, points
    noise_translate = np.array([np.random.normal(0, noise_translate_std[0], 1),
                                np.random.normal(0, noise_translate_std[1], 1),
                                np.random.normal(0, noise_translate_std[0], 1)]).T

    points[:, :3] += noise_translate
    gt_boxes[:, :3] += noise_translate

def global_rotation(gt_boxes, points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2)
    if gt_boxes.shape[1] > 7:
         gt_boxes[:, 6:8] = box_np_ops.rotation_points_single_angle(
             np.hstack([gt_boxes[:, 6:8], np.zeros((gt_boxes.shape[0], 1))]), noise_rotation, axis=2)[:, :2]
    gt_boxes[:, -1] += noise_rotation
    return gt_boxes, points


def random_flip(gt_boxes, points, probability=0.5):
    enable = np.random.choice([False, True],
                              replace=False,
                              p=[1 - probability, probability])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, -1] = -gt_boxes[:, -1] + np.pi
        points[:, 1] = -points[:, 1]
        if gt_boxes.shape[1] > 7: # y axis: x, y, z, w, h, l, vx, vy, r
            gt_boxes[:, 7] = -gt_boxes[:, 7]
    return gt_boxes, points


def global_scaling_v2(gt_boxes, points, min_scale=0.95, max_scale=1.05):
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :-1] *= noise_scale
    return gt_boxes, points


def global_rotation_v2(gt_boxes, points, min_rad=-np.pi / 4,
                       max_rad=np.pi / 4):
    noise_rotation = np.random.uniform(min_rad, max_rad)
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2)
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points
def filter_gt_box_outside_range(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_boxes_bv = box_np_ops.center_to_corner_box2d(
        gt_boxes[:, [0, 1]], gt_boxes[:, [3, 3 + 1]], gt_boxes[:, -1])
    bounding_box = box_np_ops.minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...])
    ret = points_in_convex_polygon_jit(
        gt_boxes_bv.reshape(-1, 2), bounding_box)
    return np.any(ret.reshape(-1, 4), axis=1)

def set_group_noise_same_(loc_noise, rot_noise, group_ids):
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict[group_ids[i]]]
        rot_noise[i] = rot_noise[gid_to_index_dict[group_ids[i]]]


def set_group_noise_same_v2_(loc_noise, rot_noise, grot_noise, group_ids):
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict[group_ids[i]]]
        rot_noise[i] = rot_noise[gid_to_index_dict[group_ids[i]]]
        grot_noise[i] = grot_noise[gid_to_index_dict[group_ids[i]]]
	
def get_group_center(locs, group_ids):
    num_groups = 0
    group_centers = np.zeros_like(locs)
    group_centers_ret = np.zeros_like(locs)
    group_id_dict = {}
    group_id_num_dict = OrderedDict()
    for i, gid in enumerate(group_ids):
        if gid >= 0:
            if gid in group_id_dict:
                group_centers[group_id_dict[gid]] += locs[i]
                group_id_num_dict[gid] += 1
            else:
                group_id_dict[gid] = num_groups
                num_groups += 1
                group_id_num_dict[gid] = 1
                group_centers[group_id_dict[gid]] = locs[i]
    for i, gid in enumerate(group_ids):
        group_centers_ret[i] = group_centers[
            group_id_dict[gid]] / group_id_num_dict[gid]
    return group_centers_ret, group_id_num_dict

@numba.njit
def group_transform_(loc_noise, rot_noise, locs, rots, group_center,
                     valid_mask):
    # loc_noise: [N, M, 3], locs: [N, 3]
    # rot_noise: [N, M]
    # group_center: [N, 3]
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x**2 + y**2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j]) - np.sin(rot_center))
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j]) - np.cos(rot_center))


@numba.njit
def group_transform_v2_(loc_noise, rot_noise, locs, rots, group_center,
                        grot_noise, valid_mask):
    # loc_noise: [N, M, 3], locs: [N, 3]
    # rot_noise: [N, M]
    # group_center: [N, 3]
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x**2 + y**2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j] + grot_noise[i, j]) -
                    np.sin(rot_center + grot_noise[i, j]))
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j] + grot_noise[i, j]) -
                    np.cos(rot_center + grot_noise[i, j]))

@numba.njit
def noise_per_box_group(boxes, valid_mask, loc_noises, rot_noises, group_nums):
    # WARNING: this function need boxes to be sorted by group id.
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_groups = group_nums.shape[0]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    # print(valid_mask)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_corners[i] = box_corners[i + idx]
                    current_corners[i] -= boxes[i + idx, :2]
                    _rotation_box2d_jit_(current_corners[i],
                                         rot_noises[idx + i, j], rot_mat_T)
                    current_corners[
                        i] += boxes[i + idx, :2] + loc_noises[i + idx, j, :2]
                coll_mat = box_collision_test(
                    current_corners[:num].reshape(num, 4, 2), box_corners)
                for i in range(num):  # remove self-coll
                    coll_mat[i, idx:idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                    break
        idx += num
    return success_mask
		
@numba.njit
def noise_per_box_group_v2_(boxes, valid_mask, loc_noises, rot_noises,
                            group_nums, global_rot_noises):
    # WARNING: this function need boxes to be sorted by group id.
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((max_group_num, 2), dtype=boxes.dtype)

    current_grot = np.zeros((max_group_num, ), dtype=boxes.dtype)
    dst_grot = np.zeros((max_group_num, ), dtype=boxes.dtype)

    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)

    # print(valid_mask)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_box[0, :] = boxes[i + idx]
                    current_radius = np.sqrt(current_box[0, 0]**2 +
                                             current_box[0, 1]**2)
                    current_grot[i] = np.arctan2(current_box[0, 0],
                                                 current_box[0, 1])
                    dst_grot[i] = current_grot[i] + global_rot_noises[idx +
                                                                      i, j]
                    dst_pos[i, 0] = current_radius * np.sin(dst_grot[i])
                    dst_pos[i, 1] = current_radius * np.cos(dst_grot[i])
                    current_box[0, :2] = dst_pos[i]
                    current_box[0, -1] += (dst_grot[i] - current_grot[i])

                    rot_sin = np.sin(current_box[0, -1])
                    rot_cos = np.cos(current_box[0, -1])
                    rot_mat_T[0, 0] = rot_cos
                    rot_mat_T[0, 1] = -rot_sin
                    rot_mat_T[1, 0] = rot_sin
                    rot_mat_T[1, 1] = rot_cos
                    current_corners[i] = current_box[
                        0, 2:4] * corners_norm @ rot_mat_T + current_box[0, :2]
                    current_corners[i] -= current_box[0, :2]

                    _rotation_box2d_jit_(current_corners[i],
                                         rot_noises[idx + i, j], rot_mat_T)
                    current_corners[
                        i] += current_box[0, :2] + loc_noises[i + idx, j, :2]
                coll_mat = box_collision_test(
                    current_corners[:num].reshape(num, 4, 2), box_corners)
                for i in range(num):  # remove self-coll
                    coll_mat[i, idx:idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                        loc_noises[i + idx, j, :2] += (
                            dst_pos[i] - boxes[i + idx, :2])
                        rot_noises[i + idx, j] += (
                            dst_grot[i] - current_grot[i])
                    break
        idx += num
    return success_mask

@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def _select_transform(transform, indices):
    result = np.zeros((transform.shape[0], *transform.shape[2:]),
                      dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result

@numba.njit
def points_transform_(points, centers, point_masks, loc_transform,
                      rot_transform, valid_mask):
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


def noise_per_object_v3_(gt_boxes,
                         points=None,
                         valid_mask=None,
                         rotation_perturb=np.pi / 4,
                         center_noise_std=1.0,
                         global_random_rot_range=np.pi / 4,
                         num_try=5,
                         group_ids=None):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_

    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_
        points: [M, 4], point cloud in lidar.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [
            -global_random_rot_range, global_random_rot_range
        ]
    enable_grot = np.abs(global_random_rot_range[0] -
                         global_random_rot_range[1]) >= 1e-3
    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [
            center_noise_std, center_noise_std, center_noise_std
        ]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes, ), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(
        scale=center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try])
    if group_ids is not None:
        if enable_grot:
            set_group_noise_same_v2_(loc_noises, rot_noises, global_rot_noises,
                                     group_ids)
        else:
            set_group_noise_same_(loc_noises, rot_noises, group_ids)
        group_centers, group_id_num_dict = get_group_center(
            gt_boxes[:, :3], group_ids)
        if enable_grot:
            group_transform_v2_(loc_noises, rot_noises, gt_boxes[:, :3],
                                gt_boxes[:, 6], group_centers,
                                global_rot_noises, valid_mask)
        else:
            group_transform_(loc_noises, rot_noises, gt_boxes[:, :3],
                             gt_boxes[:, 6], group_centers, valid_mask)
        group_nums = np.array(list(group_id_num_dict.values()), dtype=np.int64)

    origin = [0.5, 0.5, 0.5]
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=origin,
        axis=2)
    if group_ids is not None:
        if not enable_grot:
            selected_noise = noise_per_box_group(gt_boxes[:, [0, 1, 3, 4, 6]],
                                                 valid_mask, loc_noises,
                                                 rot_noises, group_nums)
        else:
            selected_noise = noise_per_box_group_v2_(
                gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises,
                rot_noises, group_nums, global_rot_noises)
    else:
        if not enable_grot:
            selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]],
                                           valid_mask, loc_noises, rot_noises)
        else:
            selected_noise = noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
                                               valid_mask, loc_noises,
                                               rot_noises, global_rot_noises)
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms,
                          rot_transforms, valid_mask)

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)

@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises,
                      global_rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2, ), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0]**2 + boxes[i, 1]**2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += (dst_grot - current_grot)

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = -rot_sin
                rot_mat_T[1, 0] = rot_sin
                rot_mat_T[1, 1] = rot_cos
                current_corners[:] = current_box[
                    0, 2:4] * corners_norm @ rot_mat_T + current_box[0, :2]
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j],
                                     rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += (dst_pos - boxes[i, :2])
                    rot_noises[i, j] += (dst_grot - current_grot)
                    break
    return success_mask
@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j],
                                     rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                # print(coll_mat)
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask




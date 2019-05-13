import itertools
import numpy as np
from lib.datasets import kitti_common as kitti
from lib.core.bbox import box_np_ops
from lib.datasets import preprocess as prep
from collections import defaultdict
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
        elif key in [
                "gt_boxes",
        ]:
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
                    batch_task_gt_boxes3d[i, :len(elems[i][idx]
                                                  ), :] = elems[i][idx]
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
    max_num_voxels = prep_config.max_num_voxels
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
        gt_points_max_keep = prep_config.gt_drop_percentage
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
        selected = kitti.drop_arrays_by_name(gt_dict["gt_names"], ["DontCare", "ignore"])
        _dict_select(gt_dict, selected)
        if remove_unknown:
            remove_mask = gt_dict["difficulty"] == -1
            keep_mask = np.logical_not(remove_mask)
            _dict_select(gt_dict, keep_mask)
        gt_dict.pop("difficulty")
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

            if sampled_dict is not None:
                points = np.concatenate([sampled_points, points], axis=0)

        pc_range = voxel_generator.point_cloud_range

        _dict_select(gt_dict, gt_boxes_mask)
        gt_classes = np.array([class_names.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32)
        gt_dict["gt_classes"] = gt_classes

        gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"], points)
        gt_dict["gt_boxes"], points = prep.global_rotation(gt_dict["gt_boxes"], points, rotation=global_rotation_noise)
        gt_dict["gt_boxes"], points = prep.global_scaling_v2(gt_dict["gt_boxes"], points, *global_scale_noise)
        prep.global_translate_(gt_dict["gt_boxes"], points, global_translate_noise_std)

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

    example = {
        'voxels': voxels,
        'num_points': num_points,
        'points': points,
        'coordinates': coordinates,
        "num_voxels": num_voxels,
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


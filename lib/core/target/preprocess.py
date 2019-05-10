import itertools


def prep_pointcloud(config,
		    input_dict,
		    root_path,
		    voxel_generator,
		    target_assiginers,
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
    prep_config = config.train.preprocess if training  else config.eval.preprocess
    
    

    remove_environment = prep_config.remove_environment
    max_num_voxels = prep_config.max_num_voxels
    shuffle_points = prep_config.shuffle
    anchor_area_threshold = prep_config.anchor_area_threshold
    
    if training:
        remove_unkonw = prep_config.remove_unkonw_examples 
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
        assert calib is not None and "image" in input_dict:
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
    

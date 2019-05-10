


def prep_pointcloud(config,
		    input_dict,
		    root_path,
		    voxel_generator,
		    target_assiginers,
		    db_sampler=None,
		    remove_outside_points,
		    training,
		    create_targets,
		    num_point_features,
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


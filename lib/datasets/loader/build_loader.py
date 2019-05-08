from functools import partial
from lib.core.voxel.voxel_generator import VoxelGenerator
from lib.core.target.target_assigner import TargetAssiginer
from .sampler import GroupSampler, DistributedGroupSampler


def build_dataset(config, training):

    voxel_generator = VoxelGenerator(voxel_size=config.voxel.voxel_size,
				     point_cloud_range=config.voxel.point_cloud_range,
				     max_num_points=config.voxel.max_num_points,
				     max_voxels=config.voxel.max_voxels)

    target_assiginer = TargetAssiginer(box_coder=,
				       anchor_generators=,
				       region_similarity_calculator=,
				       positive_fraction=,
				       sample_size=512)

    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // 2
    
    prep_func = partial(pre_pointcloud,
			root_paths=,
			class_names=,
			voxel_generator=,
			target_assiginer=,
			training=,
			max_voxels=,
			remove_outsize_points=,
			remove_unknow=,
			create_target=,
			shuffle_points,
			gt_rotation_noise=,
			gt_loc_noise_std=,
			global_scaling_noise=,
			global_random_rot_range=,
			db_sampler=,
			unlabeled_db_sampler=,
			generate_bev=,
			without_reflectivity=,
			num_point_feature=,
			anchor_area_threshold=,
			gt_points_drop=,
			gt_drop_max_keep,
			remove_points_after_sampler,
			remove_revironment=,
			use_group_id=,
			out_size_factor=)

    dataset = KittiDataset(info_path=,
			   root_path=,
                           num_point_features,
			   target_assiginer=,
			   feature_map_size=,
			   prep_func=prep_func)

    return dataset

def build_dataloader(config, training):
    dataset = build_dataset(config, training)
    
    batch_size = config.train.batch_size if training else config.eval.batch_size
    num_workers = config.train.num_workers if training else config.eval.num_workers
    sampler = train_sampler if training else eval_sampler
    distributed = len(config.gpus.split(',')) > 1
    if distributed:
	sampler = DistributedGroupSampler(dataset)
    else:
        sampler = GroupSampler(dataset)


    dataloader = torch.utils.data.Dataloader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=collate_batch_fn,
					     sampler=sampler)
    return dataloader

from functools import partial
from lib.core.voxel.voxel_generator import VoxelGenerator
from lib.core.target.target_assigner import TargetAssiginer


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
    
    prep_func = 


    dataset = KittiDataset()


def build_dataloader(config, training):
    dataset = build_dataset(config, training)
    
    batch_size = config.train.batch_size if training else config.eval.batch_size
    num_workers = config.train.num_workers if training else config.eval.num_workers
    sampler = train_sampler if training else eval_sampler

    dataloader = torch.utils.data.Dataloader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=collate_batch_fn,
					     sampler=sampler)
    return dataloader

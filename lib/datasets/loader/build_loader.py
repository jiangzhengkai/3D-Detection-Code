from functools import partial
from lib.core.voxel.voxel_generator import VoxelGenerator

from lib.core.bbox.box_coder import box_coder
from lib.core.anchor.anchor_generators import anchor_generators
from lib.core.bbox.region_similarity import region_similarity_calculator
from lib.core.target.target_assigner import TargetAssiginer
from lib.core.voxel.db_sampler import DataBaseSampler
from .sampler import GroupSampler, DistributedGroupSampler


def build_dataset(config, training):

    voxel_generator = VoxelGenerator(
	voxel_size=config.voxel.voxel_size,
	point_cloud_range=config.voxel.point_cloud_range,
	max_num_points=config.voxel.max_num_points,
	max_voxels=config.voxel.max_voxels)

    box_coder = box_coder(config)
    anchor_generators = anchor_generators(config)
    region_similarity_calculator =

    target_assiginer = TargetAssiginer(
	box_coder=box_coder,
	anchor_generators=anchor_generators,
	region_similarity_calculator=region_similarity_calculator,
	positive_fraction=,
	sample_size=512)

    if training:
        db_sampler = DataBaseSampler(config)
    else:
        db_sampler = None

    
    prep_func = partial(
	pre_pointcloud,
	root_paths=,
	voxel_generator=,
	target_assiginer=,
	training=,
	remove_outsize_points=,
	db_sampler=,
	num_point_feature=,
	out_size_factor=)

    dataset = KittiDataset(
	info_path=,
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

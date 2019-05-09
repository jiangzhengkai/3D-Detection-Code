from functools import partial
from lib.core.voxel.voxel_generator import VoxelGenerator

from lib.core.bbox.box_coder import box_coder
from lib.core.anchor.anchor_generators import anchor_generators
from lib.core.bbox.region_similarity import region_similarity_calculator
from lib.core.target.target_assigner import TargetAssigner
from lib.core.voxel.db_sampler import DataBaseSampler
from .sampler import GroupSampler, DistributedGroupSampler


def build_dataset(config, training):

    voxel_generator = VoxelGenerator(
	voxel_size=config.input.voxel.voxel_size,
	point_cloud_range=config.input.voxel.point_cloud_range,
	max_num_points=config.input.voxel.max_num_points,
	max_voxels=config.input.voxel.max_voxels)

    bbox_coder = box_coder(config)
    region_similarity = region_similarity_calculator(config.target_assiginer.anchor_generators.region_similarity_calculator)


    anchor_generators_all_class = anchor_generators(config.target_assiginer.anchor_generators)

    target_assiginers = []
    for i, task in enumerate(config.tasks):
        target_assiginer = TargetAssiginer(
	    box_coder=bbox_coder,
	    anchor_generators=anchor_generators_all_class[i],
	    region_similarity_calculator=region_similarity,
	    positive_fraction=None,
	    sample_size=512)
        target_assiginers.append(target_assiginer)
    if training:
        db_sampler = DataBaseSampler(config)
    else:
        db_sampler = None

    out_size_factor = 8
    grid_size = voxel_generator.grid_size
    feature_map = grid_size[:2] // out_size_factor
   
    dataset_class = get_dataset_class(config.dataset.type)    

    prep_func = partial(
	pre_pointcloud,
        prep_cfg=config,
	root_paths=config.dataset.root_paths,
	voxel_generator=vixel_generators,
	target_assiginers=target_assiginers,
	training=training,
	remove_outsize_points=False,
	db_sampler=db_sampler,
	num_point_feature=dataset_class.NumPointsFeatures,
	out_size_factor=out_size_factor)

    dataset = dataset_class(
	info_path=config.dataset.info_path,
	root_path=config.dataset.root_path,
	target_assiginer=target_assiginers,
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

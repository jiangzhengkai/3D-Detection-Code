from functools import partial
from lib.core.voxel.voxel_generator import VoxelGenerator
from lib.core.bbox.box_coder import box_coder
from lib.core.anchor.anchor_generators import anchor_generators
from lib.core.bbox.region_similarity import region_similarity_calculator
from lib.core.target.target_assigner import TargetAssigner
from lib.core.sampler.db_sampler import DBSampler
from lib.datasets.preprocess import prep_pointcloud
from .sampler import Sampler, DistributedSampler

from lib.datasets.all_dataset import get_dataset_class
from lib.datasets.preprocess import collate_batch_fn
import itertools
import torch
import torch.utils.data

def build_dataset(config, training):
    ######## voxel generator ########
    voxel_generator = VoxelGenerator(
	voxel_size=config.input.voxel.voxel_size,
	point_cloud_range=config.input.voxel.point_cloud_range,
	max_num_points=config.input.voxel.max_num_points,
	max_voxels=config.input.voxel.max_voxels)

    ######## box coder ########
    bbox_coder = box_coder(config)

    ######## region similarity ########
    region_similarity = region_similarity_calculator(config.target_assigner.anchor_generators.region_similarity_calculator)

    ######## anchor generators ########
    anchor_generators_all_class = anchor_generators(config.target_assigner.anchor_generators)

    ######## target assiginers ########
    target_assigners = []
    class_names = []
    flag = 0
    for num_class, class_name in zip(config.tasks.num_classes, config.tasks.class_names):
        target_assigner = TargetAssigner(
	    box_coder=bbox_coder,
	    anchor_generators=anchor_generators_all_class[flag:flag+len(class_name)],
	    region_similarity_calculator=region_similarity,
	    positive_fraction=config.target_assigner.anchor_generators.sample_positive_fraction,
	    sample_size=config.target_assigner.anchor_generators.sample_size)
    
        flag += len(class_name)
        target_assigners.append(target_assigner)
        class_names.append(class_name)

    ######## database sampler ########
    if training:
        db_sampler = DBSampler(config)
    else:
        db_sampler = None

    out_size_factor = 8
    grid_size = voxel_generator.grid_size
    feature_map = grid_size[:2] // out_size_factor
    dataset_class = get_dataset_class(config.input.train.dataset.type)    
    config_dataset = config.input.train.dataset if training else config.input.eval.dataset

    ####### anchor_caches #######
    rets = [
	target_assigner.generate_anchors(feature_map_size) 
	for target_assigner in target_assigners
    ]
    class_namess = [
	target_assigner.classes 
	for target_assigner in target_assigners
    ]
    anchors_dicts = [
	target_assigner.generate_anchors_dict(feature_map_size) 
	for target_assigner in target_assigners
    ]
    anchors = [
	ret["anchors"].reshape([-1, ret["anchors"].shape[-1]]) 
	for ret in rets
    ]
    matched_thresholdss = [
	ret["matched_thresholds"] 
	for ret in rets
    ]
    unmatched_thresholdss = [
	ret["unmatched_thresholds"] 
	for ret in rets
    ]
    anchors_bvs = [
        box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, -1]])
        for anchors in anchorss
    ]
    anchor_cache = {
        "anchors": anchorss,
        "anchors_bv": anchors_bvs,
        "matched_thresholds": matched_thresholdss,
        "unmatched_thresholds": unmatched_thresholdss,
        "anchors_dict": anchors_dicts,
    }
    ######## prep_pointcloud ########
    prep_func = partial(
	prep_pointcloud,
        cfg=config,
	root_paths=config_dataset.root_path,
	voxel_generator=voxel_generator,
	target_assigners=target_assigners,
	training=training,
        anchor_cache=anchor_cache,
	remove_outsize_points=False,
	db_sampler=db_sampler,
	num_point_feature=config.input.num_point_features,
	out_size_factor=out_size_factor)

    ######## dataset ########
    dataset = dataset_class(
	info_path=config_dataset.info_path,
	root_path=config_dataset.root_path,
        num_point_features=config.input.num_point_features,
	class_names=list(itertools.chain(*class_names)),
	prep_func=prep_func)
    return dataset

def build_dataloader(config, training):
    dataset = build_dataset(config, training)
    
    batch_size = config.input.train.batch_size if training else config.input.eval.batch_size
    num_workers = config.input.train.preprocess.num_workers if training else config.input.eval.preprocess.num_workers
    distributed = len(config.gpus.split(',')) > 1
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = Sampler(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=collate_batch_fn,
					     sampler=sampler)
    return dataloader

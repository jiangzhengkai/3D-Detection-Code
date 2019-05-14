import torch
import torch.utils.data
import itertools
from functools import partial
from lib.core.voxel.voxel_generator import VoxelGenerator
from lib.core.target.target_assigner import target_assigners_all_classes
from lib.core.sampler.db_sampler import DBSampler
from lib.datasets.preprocess import prep_pointcloud
from lib.datasets.loader.sampler import Sampler, DistributedSampler
from lib.datasets.all_dataset import get_dataset_class
from lib.datasets.preprocess import collate_batch_fn
from lib.core.bbox import box_np_ops

def build_dataset(config, training, logger=None):
    ######## voxel generator ########
    voxel_generator = VoxelGenerator(
	voxel_size=config.input.voxel.voxel_size,
	point_cloud_range=config.input.voxel.point_cloud_range,
	max_num_points=config.input.voxel.max_num_points,
	max_voxels=config.input.voxel.max_voxels)
    ####### target assigners ########
    target_assigners = target_assigners_all_classes(config)

    ######## database sampler ########
    if training:
        if logger is not None:
            logger.info("Enable db sampler: db_sampler")
        db_sampler = DBSampler(config, logger=logger)
    else:
        db_sampler = None

    out_size_factor = 8
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    if logger is not None:
        logger.info("feature_map_size: {}".format(feature_map_size))
    dataset_class = get_dataset_class(config.input.train.dataset.type)    
    config_dataset = config.input.train.dataset if training else config.input.eval.dataset

    ####### anchor_caches #######
    rets = [
	target_assigner.generate_anchors(feature_map_size) 
	for target_assigner in target_assigners
    ]
    class_names = [
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
        box_np_ops.rbbox2d_to_near_bbox(anchor[:, [0, 1, 3, 4, -1]])
        for anchor in anchors
    ]
    anchor_cache = {
        "anchors": anchors,
        "anchors_bv": anchors_bvs,
        "matched_thresholds": matched_thresholdss,
        "unmatched_thresholds": unmatched_thresholdss,
        "anchors_dict": anchors_dicts,
    }
    
    ######## prep_pointcloud ########
    prep_func = partial(
	prep_pointcloud,
        config=config,
	root_path=config_dataset.root_path,
	voxel_generator=voxel_generator,
	target_assigners=target_assigners,
	training=training,
        anchor_cache=anchor_cache,
	remove_outside_points=False,
	db_sampler=db_sampler,
	num_point_features=config.input.num_point_features,
	out_size_factor=out_size_factor)

    ######## dataset ########
    dataset = dataset_class(
	info_path=config_dataset.info_path,
	root_path=config_dataset.root_path,
        num_point_features=config.input.num_point_features,
	class_names=list(itertools.chain(*class_names)),
	prep_func=prep_func)
    return dataset

def build_dataloader(config, training, logger=None):
    dataset = build_dataset(config, training, logger=logger)
    
    batch_size = config.input.train.batch_size if training else config.input.eval.batch_size
    num_workers = config.input.train.preprocess.num_workers if training else config.input.eval.preprocess.num_workers
    distributed = len(config.gpus.split(',')) > 1
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=collate_batch_fn,
					     sampler=sampler)
    return dataloader

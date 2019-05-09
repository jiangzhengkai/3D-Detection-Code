import os
from yacs.config import CfgNode as CN

# --------------------------------------------------------------------
# Input
# --------------------------------------------------------------------
_C.input = CN()

_C.input.voxel = CN()
_C.input.voxel.voxel_size = 
_C.input.voxel.point_cloud_range = 
_C.input.voxel.max_num_points = 
_C.input.voxel.max_voxels = 


_C.box_coder = CN()
_C.box_coder.type = 
_C.box_coder.value = CN()
_C.box_coder.value.linear_dim = 
_C.box_coder.value.encode_angle_vector = 


_C.target_assiginer = CN()
_C.target_assiginer.anchor_generators = CN()
_C.target_assiginer.anchor_generators.type = 
_C.target_assiginer.anchor_generators.value = CN()
_C.target_assiginer.anchor_generators.value.sizes = 
_C.target_assiginer.anchor_generators.value.anchor_ranges = 
_C.target_assiginer.anchor_generators.value.rotations = 
_C.target_assiginer.anchor_generators.value.velocities = 
_C.target_assiginer.anchor_generators.value.matched_threshold = 
_C.target_assiginer.anchor_generators.value.unmatched_threshold = 
_C.target_assiginer.anchor_generators.value.class_names = 


_C.tasks = CN()
_C.tasks.num_class = 
_C.tasks.class_names = 


# --------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ()
_C.DATASETS.TEST = ()


# --------------------------------------------------------------------
# Dataloader
# --------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4


# --------------------------------------------------------------------
# Anchors options
# --------------------------------------------------------------------
_C.ANCHORS = CN()


# --------------------------------------------------------------------
# Backbone options
# --------------------------------------------------------------------
_C.MODEL.BACKBONE = CN()




# --------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------

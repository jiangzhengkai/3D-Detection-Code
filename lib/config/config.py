from easydict import EasyDict as edict
import numpy as np

_C = edict()
_C.output_dir = ''
_C.gpus = '0'
_C.devices = [0, 1]
# --------------------------------------------------------------------
# Input
# --------------------------------------------------------------------
_C.input = edict()
_C.input.num_point_features = 4
# --------------------------------------------------------------------
# voxel
# --------------------------------------------------------------------
_C.input.voxel = edict()
_C.input.voxel.voxel_size = [0.05, 0.05, 0.1]
_C.input.voxel.point_cloud_range = [0, -40.0, -3.0, 70.4, 40.0, 1.0]
_C.input.voxel.max_num_points = 5
_C.input.voxel.max_voxels = 20000
# --------------------------------------------------------------------
# train
# --------------------------------------------------------------------
_C.input.train = edict()
_C.input.train.batch_size = 6
_C.input.train.dataset = edict()
_C.input.train.dataset.type = "KittiDataset"
_C.input.train.dataset.root_path = " "
_C.input.train.dataset.info_path = " "
_C.input.train.dataset.nsweeps = 1
# --------------------------------------------------------------------
# train preprocess
# --------------------------------------------------------------------
_C.input.train.preprocess = edict()
_C.input.train.preprocess.max_num_voxels = 40000
_C.input.train.preprocess.shuffle = False
_C.input.train.preprocess.num_workers = 0
_C.input.train.preprocess.gt_location_noise = [0.25, 0.25, 0.25]
_C.input.train.preprocess.gt_rotation_noise = [-0.15707963267, 0.15707963267]
_C.input.train.preprocess.global_rotation_noise = [-0.78539816, 0.78539816]    
_C.input.train.preprocess.global_scale_noise = [0.95, 1.05]
_C.input.train.preprocess.global_rotation_per_object_range = [0, 0]
_C.input.train.preprocess.global_translation_noise = [0.2, 0.2, 0.2]
_C.input.train.preprocess.anchor_area_threshold = -1
_C.input.train.preprocess.remove_points_after_sample =  False
_C.input.train.preprocess.gt_drop_percentage = 0.0
_C.input.train.preprocess.gt_drop_max_keep_points = 15
_C.input.train.preprocess.remove_unknow_examples = False
_C.input.train.preprocess.remove_environment = False
_C.input.train.preprocess.db_sampler = edict()
_C.input.train.preprocess.db_sampler.enable = True
_C.input.train.preprocess.db_sampler.db_info_path = " "
_C.input.train.preprocess.db_sampler.sample_group_names = ["car",]
_C.input.train.preprocess.db_Sampler.sample_group_values = [2,]
_C.input.train.preprocess.db_sampler.db_preprocess_steps = edict()
_C.input.train.preprocess.db_sampler.db_preprocess_steps.type = ["car",]
_C.input.train.preprocess.db_sampler.db_preprocess_steps.value = [5, ]
# --------------------------------------------------------------------
# eval
# --------------------------------------------------------------------
_C.input.eval = edict()
_C.input.eval.batch_size = 6
# --------------------------------------------------------------------
# eval preprocess
# --------------------------------------------------------------------
_C.input.eval.preprocess = edict()
_C.input.eval.preprocess.num_workers = 0
_C.input.eval.preprocess.shuffle = False
_C.input.eval.preprocess.max_num_voxels = 40000
_C.input.eval.preprocess.anchor_area_threshold = -1
_C.input.eval.preprocess.remove_environment = False
# --------------------------------------------------------------------
# eval dataset
# --------------------------------------------------------------------
_C.input.eval.dataset = edict()
_C.input.eval.dataset.type = "KittiDataset"
_C.input.eval.dataset.root_path = " "
_C.input.eval.dataset.info_path = " "
_C.input.eval.dataset.nsweeps = 1 
# --------------------------------------------------------------------
# box_coder
# --------------------------------------------------------------------
_C.box_coder = edict()
_C.box_coder.type = "ground_box3d_coder"
_C.box_coder.value = edict()
_C.box_coder.value.linear_dim = False
_C.box_coder.value.encode_angle_vector = False
# --------------------------------------------------------------------
# target assigner
# --------------------------------------------------------------------
_C.target_assigner = edict()
_C.target_assigner.anchor_generators = edict()
_C.target_assigner.anchor_generators.anchor_types = ["anchor_generator_range",]
_C.target_assigner.anchor_generators.anchor_sizes = [(1.97, 4.63, 1.74),]
_C.target_assigner.anchor_generators.anchor_ranges = [(-51.2, -51.2, -0.92, 51.2, 51.2, -0.92),]
_C.target_assigner.anchor_generators.anchor_rotations = [(0, 1.57),]
_C.target_assigner.anchor_generators.anchor_velocities = [(0, 0),]
_C.target_assigner.anchor_generators.anchor_matched_thresholds = [0.6,]
_C.target_assigner.anchor_generators.anchor_unmatched_thresholds = [0.45,]
_C.target_assigner.anchor_generators.anchor_class_names = ["car",]
_C.target_assigner.anchor_generators.sample_positive_fraction = -1
_C.target_assigner.anchor_generators.sample_size = 512
_C.target_assigner.anchor_generators.region_similarity_calculator = edict()
_C.target_assigner.anchor_generators.region_similarity_calculator.type = "nearest_iou_similarity"
_C.target_assigner.anchor_generators.region_similarity_calculator.value = 0
# --------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------
_C.tasks = edict()
_C.tasks.num_classes = [1,]
_C.tasks.class_names = ["car",]
# --------------------------------------------------------------------
# functions
# --------------------------------------------------------------------
def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, _C)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return
    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]), type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


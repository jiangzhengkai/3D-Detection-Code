from easydict import EasyDict as edict
import numpy as np

# --------------------------------------------------------------------
# Input
# --------------------------------------------------------------------
_C = edict()
cfg = _C

_C.output_dir = ''

_C.input = edict()
_C.input.voxel = edict()
_C.input.voxel.voxel_size = [0.05, 0.05, 0.1]
_C.input.voxel.point_cloud_range = [0, -40.0, -3.0, 70.4, 40.0, 1.0]
_C.input.voxel.max_num_points = 5
_C.input.voxel.max_voxels = 20000


_C.box_coder = edict()
_C.box_coder.type = "ground_box3d_coder"
_C.box_coder.value = edict()
_C.box_coder.value.linear_dim = False
_C.box_coder.value.encode_angle_vector = False


_C.target_assiginer = edict()
_C.target_assiginer.anchor_generators = edict()
_C.target_assiginer.anchor_generators.type = ["anchor_generator_range",]
_C.target_assiginer.anchor_generators.value = edict()
_C.target_assiginer.anchor_generators.value.sizes = [(1.97, 4.63, 1.74),]
_C.target_assiginer.anchor_generators.value.anchor_ranges = [(-51.2, -51.2, -0.92, 51.2, 51.2, -0.92),]
_C.target_assiginer.anchor_generators.value.rotations = [(0, 1.57),]
_C.target_assiginer.anchor_generators.value.velocities = [(0, 0),]
_C.target_assiginer.anchor_generators.value.matched_threshold = [0.6,]
_C.target_assiginer.anchor_generators.value.unmatched_threshold = [0.45,]
_C.target_assiginer.anchor_generators.value.class_names = ["car",]


_C.tasks = edict()
_C.tasks.num_class = [1,]
_C.tasks.class_names = ["car",]


# --------------------------------------------------------------------
# Dataset
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


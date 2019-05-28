import numpy as np
from lib.core.bbox import box_np_ops


def anchor_generators(config):
    anchor_generators = []
    for i in range(len(config.anchor_types)):
        if config.anchor_types[i] == "anchor_generator_stride":
            anchor = AnchorGeneratorStride(
			        sizes=config.anchor_sizes[i],
                                strides=config.anchor_strides[i],
                                offsets=config.anchor_offsets[i],
                                rotations=config.anchor_rotations[i],
                                velocities=config.anchor_velocities[i],
                                match_threshold=config.anchor_match_thresholds[i],
                                unmatch_threshold=config.anchor_unmatch_thresholds[i],
                                class_name=config.anchor_class_names[i])
        elif config.anchor_types[i] == "anchor_generator_range":
            anchor = AnchorGeneratorRange(
                                anchor_dim=config.anchor_dims[i],
                                ranges=config.anchor_ranges[i],
                                sizes=config.anchor_sizes[i],
                                velocities=config.anchor_velocities[i],
                                rotations=config.anchor_rotations[i],
                                match_threshold=config.anchor_matched_thresholds[i],
                                unmatch_threshold=config.anchor_unmatched_thresholds[i],
                                class_name=config.anchor_class_names[i])
        else:
            raise ValueError(" unknown anchor generator type")
        anchor_generators.append(anchor)
    return anchor_generators

class AnchorGeneratorStride:
    def __init__(self,
                 sizes=[1.6, 3.9, 1.56],
                 strides=[0.4, 0.4, 1.0],
                 offsets=[0.2, -39.8, -1.78],
                 rotations=[0, np.pi / 2],
                 velocities=[0, 0],
                 class_name=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 dtype=np.float32):
        self._sizes = sizes
        self._anchor_strides = strides
        self._anchor_offsets = offsets
        self._rotations = rotations
        self._velocities = velocities
        self._dtype = dtype
        self._class_name = class_name
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold

    @property
    def class_name(self):
        return self._class_name

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    @property
    def ndim(self):
        return self._anchors.shape[-1]

    def generate(self, feature_map_size):
        self._anchors = box_np_ops.create_anchors_3d_stride(
            feature_map_size, self._sizes, self._anchor_strides,
            self._anchor_offsets, self._rotations, self._velocities, self._dtype)
        return self._anchors

class AnchorGeneratorRange:
    def __init__(self,
                 anchor_dim,
                 ranges,
                 sizes=[1.6, 3.9, 1.56],
                 rotations=[0, np.pi / 2],
                 velocities=[0, 0],
                 class_name=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 dtype=np.float32):
        self._sizes = sizes
        self._anchor_ranges = ranges
        self._rotations = rotations
        self._velocities = velocities
        self._dtype = dtype
        self._class_name = class_name
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold
        self._anchor_dim = anchor_dim

    @property
    def class_name(self):
        return self._class_name

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    @property
    def ndim(self):
        return self._anchors.shape[-1]

    def generate(self, feature_map_size):
        self._anchors = box_np_ops.create_anchors_3d_range(
            feature_map_size, self._anchor_ranges, self._sizes,
            self._rotations, self._velocities, self._dtype)
        if self._anchor_dim == 7:
            self._anchors = np.concatenate((self._anchors[...,:6], self._anchors[...,-1:]), axis=-1)
        return self._anchors


import torch
from torch import nn
from lib.models import voxel_encoder
from lib.models import middle, rpn


class VoxelNet(nn.Module):
    def __init__(self, 
                 output_shape,
                 config,
                 num_classes=[2],
                 use_norm=True,
                 target_assigners=[],
                 voxel_generator=None,
                 name='voxelnet',
                 device=None,
                 logger=None):
        super().__init__()
        self.name = name
        self.logger = logger
        self._num_classes = num_classes


        self.target_assigners = target_assigners
        self.voxel_generator = voxel_generator


        ####### voxel feature encoder #######
        vfe_class_dict = {
            "VoxelFeatureExtractor": voxel_encoder.VoxelFeatureExtractor,
            "VoxelFeatureExtractorV2": voxel_encoder.VoxelFeatureExtractorV2,
            "VoxelFeatureExtractorV3": voxel_encoder.VoxelFeatureExtractorV3, 
            "SimpleVoxel": voxel_encoder.SimpleVoxel,
        }
        vfe_class_name = config.model.encoder.vfe.type
        vfe_num_filters = config.model.encoder.vfe.num_filters
        with_distance = config.model.encoder.vfe.with_distance
        vfe_class = vfe_class_dict[vfe_class_name]
        self._vfe_class_name = vfe_class_name
        logger.info("Voxel Feature Encoder: {}".format(self._vfe_class_name))
    
        if "Pixor" in self._vfe_class_name:
            self.voxel_feature_extractor = vfe_class(output_shape[:-1])
            rpn_num_input_features = self.voxel_feature_extractor.nchannels
        else:
            self.voxel_feature_extractor = vfe_class(
            	config.model.encoder.vfe.num_input_features,
            	use_norm,
            	num_filters=vfe_num_filters,
            	with_distance=with_distance,
            	voxel_size=self.voxel_generator.voxel_size,
            	pc_range=self.voxel_generator.point_cloud_range)
  
        middle_class_dict = {
            "SpMiddleFHD": middle.SpMiddleFHD,
        }
    
        middle_class_name = config.model.encoder.middle.type
        middle_num_input_features = config.model.encoder.middle.num_input_features
        middle_num_filters_d1 = config.model.encoder.middle.num_filters_down1
        middle_num_filters_d2 = config.model.encoder.middle.num_filters_down2
    
        middle_class = middle_class_dict[middle_class_name]
        self._middle_class_name = middle_class_name
        logger.info("Middle class name: {}".format(middle_class_name))
    
        if "Pixor" in self._middle_class_name:
            self.middle_feature_extractor = middle_class(
                output_shape=output_shape,
                num_input_features=vfe_num_filters[-1])
        elif middle_class:
            self.middle_feature_extractor = middle_class(
                output_shape,
                use_norm,
                num_input_features=middle_num_input_features,
                num_filters_down1=middle_num_filters_d1,
                num_filters_down2=middle_num_filters_d2)
        else:
            logger.info("Pixor do not need Middle Layer")
        ######## rpn ########
        rpn_class_dict = {
            "RPNV2": rpn.RPNV2,
        }

        rpn_class_name = config.model.decoder.rpn.type
        rpn_num_input_features = config.model.decoder.rpn.num_input_features
        rpn_layer_nums = config.model.decoder.rpn.layer_nums
        rpn_layer_strides = config.model.decoder.rpn.downsample_layer_strides
        rpn_num_filters = config.model.decoder.rpn.downsample_num_filters
        rpn_upsample_strides = config.model.decoder.rpn.upsample_layer_strides
        rpn_num_upsample_filters = config.model.decoder.rpn.upsample_num_filters

        num_groups = config.model.decoder.rpn.num_groups
        use_groupnorm = config.model.decoder.rpn.group_norm

        rpn_class = rpn_class_dict[rpn_class_name]
        self._rpn_class_name = rpn_class_name
        logger.info("RPN class name: {}".format(self._rpn_class_name))
    
        self.rpn = rpn_class(
            use_norm=True,
            num_classes=num_classes,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_filters=rpn_num_upsample_filters,
            num_input_features=rpn_num_input_features,
            num_anchor_per_locs = [
                target_assigner.num_anchors_per_location
                for target_assigner in target_assigners
            ],
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_sizes=[
                target_assigner.box_coder.code_size
                for target_assigner in target_assigners
            ],
            logger=logger,
            )
    def forward(self, example):
        voxels = example["voxels"]
        num_points = example["num_points"]
        coordinates = example["coordinates"]
        
        if "Pixor" in self._vfe_class_name:
            voxel_features = self.voxel_feature_extractor(
                voxels, num_points, coordinates, batch_size_dev)
        else:
            voxel_features = self.voxel_feature_extractor(
                voxels, num_points, coordinates)
    
        if "Pixor" not in self._vfe_class_name:
            spatial_features = self.middle_feature_extractor(
                voxel_features, coordinates, batch_size_dev)
        else:
            spatial_features = voxel_features

        predict_dicts = self.rpn(spatial_features)
 
        return predict_dicts


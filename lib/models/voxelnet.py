import torch
from torch import nn
from lib.models import voxel_encoder
from lib.models import middle, rpn
from lib.solver import build_losses

from lib.core.loss.loss import prepare_loss_weights, create_loss, get_pos_neg_loss, get_direction_target
from lib.solver.losses import (WeightedSigmoidClassificationLoss,
                               WeightedSmoothL1LocalizationLoss,
                               WeightedSoftmaxClassificationLoss)

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
        self._target_assigners = target_assigners
        self._voxel_generator = voxel_generator
        self._encode_background_as_zeros = config.model.loss.encode_background_as_zeros
        self._encode_rad_error_by_sin = config.model.loss.encode_rad_error_by_sin
        self._use_direction_classifier = config.model.decoder.auxiliary.use_direction_classifier
        ######## args ########
        self._pos_cls_weight = config.model.loss.pos_class_weight
        self._neg_cls_weight = config.model.loss.neg_class_weight
        self._loss_norm_type = config.model.loss.loss_norm_type
        self._direction_offset = config.model.decoder.auxiliary.direction_offset
        self._use_direction_classifier = config.model.decoder.auxiliary.use_direction_classifier
        logger.info("Using direction offset %f to fix aoe" %(self._direction_offset))
        self._box_coders = [
            target_assigner.box_coder for target_assigner in target_assigners
        ]
       
        self._dir_loss_function = WeightedSoftmaxClassificationLoss()
        self._diff_loss_function = WeightedSmoothL1LocalizationLoss(device=device)

        ######## classifization and localization function ########
        cls_loss_function, loc_loss_function = build_losses.build(config)
        
        self._loc_loss_function = cls_loss_function
        self._cls_loss_function = loc_loss_function
        self._cls_loss_weight = config.model.loss.classification_loss_weight
        self._loc_loss_weight = config.model.loss.localization_loss_weight
        self._direction_loss_weight = config.model.loss.direction_loss_weight
        self._post_center_range = config.model.post_process.post_center_limit_range
     
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
            self._voxel_feature_extractor = vfe_class(output_shape[:-1])
            rpn_num_input_features = self.voxel_feature_extractor.nchannels
        else:
            self._voxel_feature_extractor = vfe_class(
            	config.model.encoder.vfe.num_input_features,
            	use_norm,
            	num_filters=vfe_num_filters,
            	with_distance=with_distance,
            	voxel_size=self._voxel_generator.voxel_size,
            	pc_range=self._voxel_generator.point_cloud_range)
  
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
            self._middle_feature_extractor = middle_class(
                output_shape=output_shape,
                num_input_features=vfe_num_filters[-1])
        elif middle_class:
            self._middle_feature_extractor = middle_class(
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
    
        self._rpn = rpn_class(
            use_norm=True,
            num_classes=num_classes,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
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
        batch_anchors = example["anchors"]
        batch_size_dev = batch_anchors[0].shape[0]
        
        if "Pixor" in self._vfe_class_name:
            voxel_features = self._voxel_feature_extractor(
                voxels, num_points, coordinates, batch_size_dev)
        else:
            voxel_features = self._voxel_feature_extractor(
                voxels, num_points, coordinates)
    
        if "Pixor" not in self._vfe_class_name:
            spatial_features = self._middle_feature_extractor(
                voxel_features, coordinates, batch_size_dev)
        else:
            spatial_features = voxel_features

        predict_dicts = self._rpn(spatial_features) 
        return predict_dicts

    def loss(self, example, preds_dicts):
        voxels = example["voxels"]
        num_points = example["num_points"]
        coordinates = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_dev = batch_anchors[0].shape[0]
        rets = []
        for task_id, pred_dict in enumerate(preds_dicts): 
            box_preds = pred_dict["box_preds"] 
            cls_preds = pred_dict["cls_preds"]
     
            labels = example["labels"][task_id]
            reg_targets = example["reg_targets"][task_id]
            
            cls_weights, reg_weights, cared = prepare_loss_weights(labels,
                                                                   pos_cls_weight=self._pos_cls_weight,
                                                                   neg_cls_weight=self._neg_cls_weight,
                                                                   loss_norm_type=self._loss_norm_type,
                                                                   dtype=voxels.dtype)

            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)

            loc_loss, cls_loss = create_loss(self._loc_loss_function,
                                             self._cls_loss_function,
                                             box_preds=box_preds,
                                             reg_targets=reg_targets,
                                             reg_weights=reg_weights,
                                             cls_preds=cls_preds,
                                             cls_targets=cls_targets,
                                             cls_weights=cls_weights,
                                             num_class=self._num_classes[task_id],
                                             encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                                             encode_background_as_zeros=self._encode_background_as_zeros,
                                             box_code_size=self._box_coders[task_id].code_size)
             
            loc_loss_reduced = loc_loss.sum() / batch_size_dev
            loc_loss_reduced *= self._loc_loss_weight
    
            cls_pos_loss, cls_neg_loss = get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self._pos_cls_weight
            cls_neg_loss /= self._neg_cls_weight
           
            cls_loss_reduced = cls_loss.sum() / batch_size_dev
            cls_loss_reduced *= self._cls_loss_weight

            loss = loc_loss_reduced + cls_loss_reduced  
       
            if self._use_direction_classifier:
                dir_targets = get_direction_target(example["anchors"][task_id],
                                                   reg_targets,
                                                   dir_offset=self._direction_offset)
                dir_logits = pred_dict["dir_cls_preds"].view(batch_size_dev, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self._dir_loss_function(dir_logits,
                                                   dir_targets,
                                                   weights=weights)
                dir_loss = dir_loss.sum() / batch_size_dev
                loss += dir_loss * self._direction_loss_weight
            ret = {
                "loss": loss,
                "cls_loss": cls_loss,
                "loc_loss": loc_loss,
                "cls_pos_loss": cls_pos_loss,
                "cls_neg_loss": cls_neg_loss,
                "cls_preds": cls_preds,
                "dir_loss_reduced": dir_loss if self._use_direction_classifier else None,
                "cls_loss_reduced": cls_loss_reduced,
                "loc_loss_reduced": loc_loss_reduced,
                "cared": cared,
                  }
            rets.append(ret)
        return rets

    def predict(self, example, preds_dict, task_id):
        batch_size = example["anchors"][task_id].shape[0]
        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]

        batch_anchors = example["anchors"][task_id].view(batch_size, -1, 9)
        
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"][task_id].view(batch_size, -1)
     
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1, self._box_coders[task_id].code_size)
        num_class_with_bg = self._num_classes[task_id]
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_classes[task_id] + 1
        batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)

        batch_box_preds = self._box_coders[task_id].decode_torch(batch_box_preds, batch_anchors)

        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        if len(self._post_center_range) > 0:
            post_center_range = torch.tensor(self._post_center_range,
                                             dtype=batch_box_preds.dtype,
                                             device=batch_box_preds.device)


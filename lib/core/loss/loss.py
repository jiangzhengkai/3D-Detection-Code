import torch
import numpy as np

def get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss



def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type='',
                         dtype=torch.float32):
    cared = labels >= 0
    positives = labels > 0
    negatives = labels == 0
    cls_weights = neg_cls_weight * negatives.type(dtype) + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == "NormByNumExamples":
        num_examples = cared.type(dtype).sum(1, keep_dim=True)
        cls_weights /= torch.clamp(num_examples, min=1.0)
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == "NormByNumPositives":
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)        
    elif loss_norm_type == "NormByNumPosNeg":
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keep_dim=True)
        cls_normalizer = (pos_neg * normalizer).sum(-1)
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    elif loss_norm_type == "DontNorm":
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    else:
        raise ValueError("unkonw loss norm type")
    return cls_weights, reg_weights, cared

def create_loss(loc_loss_factor,
                cls_loss_factor,
                box_preds,
                reg_targets,
                reg_weights,
                cls_preds,
                cls_targets,
                cls_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=9):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)

    one_hot_targets = one_hot_function(cls_targets,
                                       depth=num_class + 1,
                                       dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[...,1:]
    if encode_rad_error_by_sin:
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_factor(box_preds, reg_targets, weights=reg_weights)
    cls_losses = cls_loss_factor(cls_preds, one_hot_targets, weights=cls_weights)
    return loc_losses, cls_losses

def one_hot_function(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape),
                                depth,
                                dtype=dtype,
                                device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot

def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
        boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2
def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=torch.float32):
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    for label, weight in weight_per_class:
        positives = (labels == label).type(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = torch.clamp(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period


def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0.0):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    # dir_cls_targets = (rot_gt > 0).long()
    dir_cls_targets = (limit_period(rot_gt - dir_offset, 0.5, 2 * np.pi) >
                       0).long()
    if one_hot:
        dir_cls_targets = one_hot_function(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets


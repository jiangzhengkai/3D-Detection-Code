import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F

from lib.models.common import Empty, GroupNorm, Sequential
from lib.models.common import change_default_args

class RPNHead(nn.Module):
    def __init__(self,
                 num_input,
                 num_pred,
                 num_cls,
                 use_dir=False,
                 num_dir=0,
                 use_rc=False,
                 use_focal_loss_init=False,
                 name='',
                 **kwargs):
        super(RPNHead, self).__init__(**kwargs)
        self.use_dir = use_dir
        self.use_rc = use_rc

        self.conv_box = nn.Conv2d(num_input, num_pred, 1)
        self.conv_cls = nn.Conv2d(num_input, num_cls, 1)
        if self.use_dir:
            self.conv_dir = nn.Conv2d(num_input, num_dir, 1)
        if self.use_rc:
            self.conv_dir = nn.Conv2d(num_input, num_pred, 1)
        # initialization for focal loss
        if use_focal_loss_init:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.conv_cls.bias, bias_value)


    def forward(self, x):
        ret_list = []
        box_preds = self.conv_box(x).permute(0, 2, 3, 1).contiguous()
        cls_preds = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()
        ret_dict = {"box_preds": box_preds, "cls_preds": cls_preds}
        if self.use_dir:
            dir_preds = self.conv_dir(x).permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_preds
        if self.use_rc:
            rc_preds = self.conv_rc(x).permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds

        return ret_dict

class RPNBase(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_classes=[2,],
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_locs=[2, ],
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_sizes=[7],
                 name='rpn',
                 logger=None):
        super(RPNBase, self).__init__()
        self._num_anchor_per_locs = num_anchor_per_locs
        self._use_direction_classifier = use_direction_classifier

        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)

        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self._upsample_start_idx + 1]))

        for val in must_equal_list:
            assert val == must_equal_list[0]

        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                deblock = nn.Sequential(
                    ConvTranspose2d(
                        num_out_filters,
                        num_upsample_filters[i - self._upsample_start_idx],
                        upsample_strides[i - self._upsample_start_idx],
                        stride=upsample_strides[i - self._upsample_start_idx]),
                    BatchNorm2d(
                        num_upsample_filters[i - self._upsample_start_idx]),
                    nn.ReLU6(),
                )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        num_clss = []
        num_preds = []
        num_dirs = []

        for num_c, num_a, box_cs in zip(num_classes, num_anchor_per_locs, box_code_sizes):
            if encode_background_as_zeros:
                num_cls = num_a * num_c
            else:
                num_cls = num_a * (num_c + 1)
            num_clss.append(num_cls)

            num_pred = num_a * box_cs
            num_preds.append(num_pred)

            if use_direction_classifier:
                num_dir = num_a * 2
                num_dirs.append(num_dir)
            else:
                num_dir = None

        logger.info("num_classes: {}".format(num_classes))
        logger.info("num_preds: {}".format(num_preds))
        logger.info("num_dirs: {}".format(num_dirs))

        self.tasks = nn.ModuleList()

        for task_id, (num_pred, num_cls) in enumerate(
                zip(num_preds, num_clss)):
            self.tasks.append(
                RPNHead(
                    sum(num_upsample_filters),
                    num_pred,
                    num_cls,
                    use_dir=self._use_direction_classifier,
                    num_dir=num_dirs[task_id] if self._use_direction_classifier else None,
                ))
        logger.info("Finish RPNBase Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        raise NotImplementedError

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        ret_dicts = []
        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts


class RPNV2(RPNBase):
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU6())
        return block, planes


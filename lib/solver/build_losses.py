# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A function to build localization and classification losses from config."""

from lib.solver import losses
from lib.solver.ghm_loss import GHMCLoss, GHMRLoss


def build(config):
    """Build losses based on the config.

  Builds classification, localization losses and optionally a hard example miner
  based on the config.

  Args:
    loss_config: A losses_pb2.Loss object.

  Returns:
    classification_loss: Classification loss object.
    localization_loss: Localization loss object.
    classification_weight: Classification loss weight.
    localization_weight: Localization loss weight.
    hard_example_miner: Hard example miner object.

  Raises:
    ValueError: If hard_example_miner is used with sigmoid_focal_loss.
  """
    loss_config = config.model.loss
    classification_loss = _build_classification_loss(
        loss_config.classification_loss)
    localization_loss = _build_localization_loss(loss_config.localization_loss)

    return (classification_loss, localization_loss)


def build_faster_rcnn_classification_loss(loss_config):
    """Builds a classification loss for Faster RCNN based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
    loss_type = loss_config.type
    config = loss_config.value

    # By default, Faster RCNN second stage classifier uses Softmax loss
    # with anchor-wise outputs.
    return losses.WeightedSoftmaxClassificationLoss(
        logit_scale=config.logit_scale)


def _build_localization_loss(loss_config):
    """Builds a localization loss based on the loss config.

  Args:
    loss_config: A losses_pb2.LocalizationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
    loss_type = loss_config.type
    config = loss_config.value

    if loss_type == 'weighted_l2':
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedL2LocalizationLoss(code_weight)

    if loss_type == 'weighted_smooth_l1':
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedSmoothL1LocalizationLoss(config.sigma,
                                                       code_weight)
    if loss_type == 'weighted_ghm':
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return GHMRLoss(config.mu, config.bins, config.momentum, code_weight)

    raise ValueError('Empty loss config.')


def _build_classification_loss(loss_config):
    """Builds a classification loss based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
    loss_type = loss_config.type
    config = loss_config.value

    if loss_type == 'weighted_sigmoid':
        return losses.WeightedSigmoidClassificationLoss()
    elif loss_type == 'weighted_sigmoid_focal':
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SigmoidFocalClassificationLoss(gamma=config.gamma,
                                                     alpha=alpha)
    elif loss_type == 'weighted_softmax_focal':
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SoftmaxFocalClassificationLoss(gamma=config.gamma,
                                                     alpha=alpha)
    elif loss_type == 'weighted_ghm':
        return GHMCLoss(bins=config.bins, momentum=config.momentum)
    elif loss_type == 'weighted_softmax':
        return losses.WeightedSoftmaxClassificationLoss(
            logit_scale=config.logit_scale)
    elif loss_type == 'bootstrapped_sigmoid':
        return losses.BootstrappedSigmoidClassificationLoss(
            alpha=config.alpha,
            bootstrap_type=('hard' if config.hard_bootstrap else 'soft'))

    raise ValueError('Empty loss config.')

"""Region Similarity Calculators for BoxLists.
Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""
from abc import ABCMeta
from abc import abstractmethod

from lib.core.bbox import box_np_ops

class RegionSimilarityCalculator(object):
  """Abstract base class for 2d region similarity calculator."""
  __metaclass__ = ABCMeta

  def compare(self, boxes1, boxes2):
    """Computes matrix of pairwise similarity between BoxLists.
    This op (to be overriden) computes a measure of pairwise similarity between
    the boxes in the given BoxLists. Higher values indicate more similarity.
    Note that this method simply measures similarity and does not explicitly
    perform a matching.
    Args:
      boxes1: [N, 5] [x,y,w,l,r] tensor.
      boxes2: [M, 5] [x,y,w,l,r] tensor.
    Returns:
      a (float32) tensor of shape [N, M] with pairwise similarity score.
    """
    return self._compare(boxes1, boxes2)

  @abstractmethod
  def _compare(self, boxes1, boxes2):
    pass


class RotateIouSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on Intersection over Union (IOU) metric.
  This class computes pairwise similarity between two BoxLists based on IOU.
  """

  def _compare(self, boxes1, boxes2):
    """Compute pairwise IOU similarity between the two BoxLists.
    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.
    Returns:
      A tensor with shape [N, M] representing pairwise iou scores.
    """

    return box_np_ops.riou_cc(boxes1, boxes2)


class NearestIouSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on the squared distance metric.
  This class computes pairwise similarity between two BoxLists based on the
  negative squared distance metric.
  """

  def _compare(self, boxes1, boxes2):
    """Compute matrix of (negated) sq distances.
    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.
    Returns:
      A tensor with shape [N, M] representing negated pairwise squared distance.
    """
    boxes1_bv = box_np_ops.rbbox2d_to_near_bbox(boxes1)
    boxes2_bv = box_np_ops.rbbox2d_to_near_bbox(boxes2)
    ret = box_np_ops.iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
    return ret


class DistanceSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on Intersection over Area (IOA) metric.
  This class computes pairwise similarity between two BoxLists based on their
  pairwise intersections divided by the areas of second BoxLists.
  """
  def __init__(self, distance_norm, with_rotation=False, rotation_alpha=0.5):
      self._distance_norm = distance_norm
      self._with_rotation = with_rotation
      self._rotation_alpha = rotation_alpha

  def _compare(self, boxes1, boxes2):
    """Compute matrix of (negated) sq distances.
    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.
    Returns:
      A tensor with shape [N, M] representing negated pairwise squared distance.
    """
    return box_np_ops.distance_similarity(
        boxes1[..., [0, 1, -1]], 
        boxes2[..., [0, 1, -1]], 
        dist_norm=self._distance_norm,
        with_rotation=self._with_rotation,
        rot_alpha=self._rotation_alpha)


def region_similarity_calculator(config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    similarity_type = config.type
    if similarity_type == 'rotate_iou_similarity':
        return RotateIouSimilarity()
    elif similarity_type == 'nearest_iou_similarity':
        return NearestIouSimilarity()
    elif similarity_type == 'distance_similarity':
        cfg = config.distance_similarity
        return DistanceSimilarity(distance_norm=cfg.distance_norm,
                                  with_rotation=cfg.with_rotation,
                                  rotation_alpha=cfg.rotation_alpha)
    else:
        raise ValueError("unknown similarity type")


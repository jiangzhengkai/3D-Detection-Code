"""Waymo Open Dataset tensorflow ops python interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

ERROR = 1e-6


class DetectionMetricsEstimatorTest():

  def _build_config(self):
    config = metrics_pb2.Config()
    config_text = """
    num_desired_score_cutoffs: 11
    breakdown_generator_ids: OBJECT_TYPE
    breakdown_generator_ids: ONE_SHARD
    breakdown_generator_ids: RANGE
    breakdown_generator_ids: RANGE
    difficulties {
      levels: LEVEL_1
      levels: LEVEL_2
    }
    difficulties {
      levels: LEVEL_1
      levels: LEVEL_2
    }
    difficulties {
      levels: LEVEL_1
      levels: LEVEL_2
    }
    difficulties {
      levels: LEVEL_1
      levels: LEVEL_2
    }
    matcher_type: TYPE_HUNGARIAN
    iou_thresholds: 0.5
    iou_thresholds: 0.7
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    box_type: TYPE_3D
    """
    text_format.Merge(config_text, config)
    return config

  def _BuildGraph(self, graph):
    with graph.as_default():
      self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
      self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
      self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
      self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
      self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
      self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
      self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)

      metrics = detection_metrics.get_detection_metric_ops(
          config=self._build_config(),
          prediction_frame_id=self._pd_frame_id,
          prediction_bbox=self._pd_bbox,
          prediction_type=self._pd_type,
          prediction_score=self._pd_score,
          prediction_overlap_nlz=tf.zeros_like(
              self._pd_frame_id, dtype=tf.bool),
          ground_truth_bbox=self._gt_bbox,
          ground_truth_type=self._gt_type,
          ground_truth_frame_id=self._gt_frame_id,
          ground_truth_difficulty=tf.ones_like(
              self._gt_frame_id, dtype=tf.uint8),
          recall_at_precision=0.7,
      )
      return metrics

  def _EvalUpdateOps(
      self,
      sess,
      graph,
      metrics,
      prediction_frame_id,
      prediction_bbox,
      prediction_type,
      prediction_score,
      ground_truth_frame_id,
      ground_truth_bbox,
      ground_truth_type,
  ):
    sess.run(
        [tf.group([value[1] for value in metrics.values()])],
        feed_dict={
            self._pd_bbox: prediction_bbox,
            self._pd_frame_id: prediction_frame_id,
            self._pd_type: prediction_type,
            self._pd_score: prediction_score,
            self._gt_bbox: ground_truth_bbox,
            self._gt_type: ground_truth_type,
            self._gt_frame_id: ground_truth_frame_id,
        })

  def _EvalValueOps(self, sess, graph, metrics):
    return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}

 
  def testAPBasic(self, predictions, gt_boxes):
    
    pd_bbox, pd_type, pd_frameid, pd_score = predictions
    gt_bbox, gt_type, gt_frameid = gt_boxes

    graph = tf.Graph()
    metrics = self._BuildGraph(graph)
    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self._EvalUpdateOps(sess, graph, metrics, pd_frameid, pd_bbox, pd_type,
                          pd_score, gt_frameid, gt_bbox, gt_type)
      aps = self._EvalValueOps(sess, graph, metrics)
      for key, value in aps.items():
        print(key, value)



if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()

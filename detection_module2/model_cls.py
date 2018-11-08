# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.models import (
    Conv2D, FullyConnected, layer_register, GlobalAvgPooling)
from tensorpack.utils.argtools import memoized

from basemodel import GroupNorm
from utils.box_ops import pairwise_iou
from model_box import encode_bbox_target, decode_bbox_target
from config import config as cfg


@layer_register(log_shape=True)
def img_level_cls_outputs(feature, num_classes):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1

    Returns:
        cls_logits: N x num_class classification logits
    """
    feature = GlobalAvgPooling('global_avg_pooling',feature, data_format='channels_first')

    classification = FullyConnected(
        'img_level_class', feature, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    return classification


@under_name_scope()
def img_level_cls_losses(labels, label_logits):
    """
    Args:
        labels: n,
        label_logits: nxC

    Returns:
        label_loss
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss * 0.05, name='img_level_cls_loss')

    with tf.name_scope('img_level_cls_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='img_level_cls_prediction')
        correct = tf.to_float(tf.equal(prediction, labels))  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy')

    add_moving_summary(label_loss, accuracy)
    return label_loss


"""
img_level_cls heads for FPN:
"""

class Img_Level_CLS_Head(object):
    """
    A class to process & decode inputs/outputs of a fastrcnn classification+regression head.
    """

    def __init__(self, gt_labels, label_logits):
        """
        Args:
            label_logits: Nx#class, the output of the head
        """
        for k, v in locals().items():
            if k != 'self' and v is not None:
                setattr(self, k, v)

    @memoized
    def losses(self):
        return img_level_cls_losses(
            self.gt_labels, self.label_logits,
        )

    @memoized
    def output_scores(self, name=None):
        """ Returns: N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.label_logits, name=name)

    @memoized
    def predicted_labels(self, name=None):
        """ Returns: N ints """
        return tf.argmax(self.label_logits, axis=1, name=name)

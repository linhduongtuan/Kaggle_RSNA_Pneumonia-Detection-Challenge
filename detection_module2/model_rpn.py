# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope, auto_reuse_variable_scope
from tensorpack.models import Conv2D, layer_register

from model_box import clip_boxes
from config import config as cfg

from utils.box_ops import pairwise_iou


@layer_register(log_shape=True)
@auto_reuse_variable_scope
def rpn_head(featuremap, channel, num_anchors):
    """
    Returns:
        label_logits: fHxfWxNA
        box_logits: fHxfWxNAx4
    """
    with argscope(Conv2D, data_format='channels_first',
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01)):
        hidden = Conv2D('conv0', featuremap, channel, 3, activation=tf.nn.relu)

        label_logits = Conv2D('class', hidden, num_anchors, 1)
        box_logits = Conv2D('box', hidden, 4 * num_anchors, 1)
        # 1, NA(*4), im/16, im/16 (NCHW)

        label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # 1xfHxfWxNA
        label_logits = tf.squeeze(label_logits, 0)  # fHxfWxNA

        shp = tf.shape(box_logits)  # 1x(NAx4)xfHxfW
        box_logits = tf.transpose(box_logits, [0, 2, 3, 1])  # 1xfHxfWx(NAx4)
        box_logits = tf.reshape(box_logits, tf.stack([shp[2], shp[3], num_anchors, 4]))  # fHxfWxNAx4
    return label_logits, box_logits


@layer_register(log_shape=True)
@auto_reuse_variable_scope
def rpn_head_iou(featuremap, channel, num_anchors):
    """
    Returns:
        label_logits: fHxfWxNA
        box_logits: fHxfWxNAx4
    """
    with argscope(Conv2D, data_format='channels_first',
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01)):
        hidden = Conv2D('conv0', featuremap, channel, 3, activation=tf.nn.relu)

        label_logits = Conv2D('class', hidden, num_anchors, 1)
        box_logits = Conv2D('box', hidden, 4 * num_anchors, 1)
        iou_logits = Conv2D('iou', hidden, num_anchors, 1)
        # 1, NA(*4), im/16, im/16 (NCHW)

        label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # 1xfHxfWxNA
        label_logits = tf.squeeze(label_logits, 0)  # fHxfWxNA

        shp = tf.shape(box_logits)  # 1x(NAx4)xfHxfW
        box_logits = tf.transpose(box_logits, [0, 2, 3, 1])  # 1xfHxfWx(NAx4)
        box_logits = tf.reshape(box_logits, tf.stack([shp[2], shp[3], num_anchors, 4]))  # fHxfWxNAx4

        iou_logits = tf.transpose(iou_logits, [0, 2, 3, 1])  # 1xfHxfWxNA
        iou_logits = tf.squeeze(iou_logits, 0)  # fHxfWxNA

    return label_logits, box_logits, iou_logits


@under_name_scope()
def rpn_losses_iou(anchor_labels, anchor_boxes, gt_boxes, rpn_boxes, label_logits, box_logits, iou_logits):
    """
    Args:
        anchor_labels: fHxfWxNA
        anchor_boxes: fHxfWxNAx4, encoded
        gt_boxes:
        rpn_boxes: fHxfWxNA decoded
        label_logits:  fHxfWxNA
        box_logits: fHxfWxNAx4
        iou_logits:  fHxfWxNA

    Returns:
        label_loss, box_loss, iou_loss
    """
    with tf.device('/cpu:0'):
        valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
        pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
        nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name='num_valid_anchor')
        nr_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')
        # nr_pos is guaranteed >0 in C4. But in FPN. even nr_valid could be 0.

        valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)
    valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

    with tf.name_scope('label_metrics'):
        valid_label_prob = tf.nn.sigmoid(valid_label_logits)
        summaries = []
        with tf.device('/cpu:0'):
            for th in [0.5, 0.2, 0.1]:
                valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
                nr_pos_prediction = tf.reduce_sum(valid_prediction, name='num_pos_prediction')
                pos_prediction_corr = tf.count_nonzero(
                    tf.logical_and(
                        valid_label_prob > th,
                        tf.equal(valid_prediction, valid_anchor_labels)),
                    dtype=tf.int32)
                placeholder = 0.5  # A small value will make summaries appear lower.
                recall = tf.to_float(tf.truediv(pos_prediction_corr, nr_pos))
                recall = tf.where(tf.equal(nr_pos, 0), placeholder, recall, name='recall_th{}'.format(th))
                precision = tf.to_float(tf.truediv(pos_prediction_corr, nr_pos_prediction))
                precision = tf.where(tf.equal(nr_pos_prediction, 0),
                                     placeholder, precision, name='precision_th{}'.format(th))
                summaries.extend([precision, recall])
        add_moving_summary(*summaries)

    # Per-level loss summaries in FPN may appear lower due to the use of a small placeholder.
    # But the total RPN loss will be fine.  TODO make the summary op smarter
    placeholder = 0.
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(valid_anchor_labels), logits=valid_label_logits)
    # label_loss = tf.reduce_sum(label_loss) * (1. / cfg.RPN.BATCH_PER_IM)
    # label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name='label_loss')

#    alpha = 0.75
#    gamma = 2.0
#    probs = tf.sigmoid(valid_label_logits)
#    alpha_t = tf.ones_like(valid_label_logits) * alpha
#    alpha_t = tf.where(valid_anchor_labels > 0, alpha_t, 1.0 - alpha_t)
#    probs_t = tf.where(valid_anchor_labels > 0, probs, 1.0 - probs)
#    weight_matrix = alpha_t * tf.pow((1.0 - probs_t), gamma)
#    # label_loss = tf.reduce_sum(weight_matrix * label_loss) * (1. / cfg.RPN.BATCH_PER_IM)
#
#    label_loss = weight_matrix * ce_loss
#    
#    #n_pos = tf.reduce_sum(valid_anchor_labels)
#    n_false = tf.reduce_sum(tf.cast(tf.greater(ce_loss, -tf.log(0.5)), tf.float32))
#    def has_pos():
#        return tf.reduce_sum(label_loss) / tf.cast(n_false, tf.float32)
#    def no_pos():
#        return tf.reduce_sum(label_loss)
#    label_loss = tf.cond(n_false > 0, has_pos, no_pos)
#    label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name='label_loss')
    # find the most wrongly classified examples:

    n_selected = cfg.FRCNN.BATCH_PER_IM
    n_selected = tf.cast(n_selected, tf.int32)
    n_selected = tf.minimum(n_selected, tf.size(valid_anchor_labels))

#    label_loss = alpha_t * label_loss

    vals, _ = tf.nn.top_k(ce_loss, k=n_selected)
    try:
        th = vals[-1]
    except:
        th = 1
    selected_mask = ce_loss >= th
    loss_weight = tf.cast(selected_mask, tf.float32)
    label_loss = tf.reduce_sum(ce_loss * loss_weight) * 1. / tf.reduce_sum(loss_weight)
    label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name='label_loss')

    pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
    delta = 1.0 / 9
    # box_loss = tf.losses.huber_loss(
    #    pos_anchor_boxes, pos_box_logits, delta=delta,
    #    reduction=tf.losses.Reduction.SUM) / delta
    box_loss = tf.losses.huber_loss(
        pos_anchor_boxes, pos_box_logits,
        reduction=tf.losses.Reduction.SUM)
    box_loss = box_loss * (50. / cfg.RPN.BATCH_PER_IM)
    box_loss = tf.where(tf.equal(nr_pos, 0), placeholder, box_loss, name='box_loss')

    # iou loss: smooth l1 loss
    rpn_boxes = tf.reshape(rpn_boxes, [-1, 4])
    gt_boxes = tf.reshape(gt_boxes, [-1, 4])
    iou = pairwise_iou(rpn_boxes, gt_boxes)  # nxm
    max_iou = tf.reduce_max(iou, axis=1)
    # if only bg gt_boxes, all ious are 0.
    max_iou = tf.where(tf.equal(nr_pos, 0), tf.zeros_like(max_iou), max_iou)
    max_iou = tf.stop_gradient(tf.reshape(max_iou, [-1]), name='rpn_box_gt_iou')

    iou_logits = tf.nn.sigmoid(iou_logits)
    iou_logits = tf.reshape(iou_logits, [-1])
    iou_loss = tf.losses.huber_loss(max_iou, iou_logits, reduction='none')

    n_selected = cfg.FRCNN.BATCH_PER_IM
    n_selected = tf.cast(n_selected, tf.int32)

    vals, _ = tf.nn.top_k(iou_loss, k=n_selected)
    th = vals[-1]
    selected_mask = iou_loss >= th
    loss_weight = tf.cast(selected_mask, tf.float32)
    iou_loss = tf.reduce_sum(iou_loss * loss_weight) * 1. / tf.reduce_sum(loss_weight)
    iou_loss = tf.identity(iou_loss, name='iou_loss')

    add_moving_summary(label_loss, box_loss, iou_loss, nr_valid, nr_pos)
    return label_loss, box_loss, iou_loss


@under_name_scope()
def generate_rpn_proposals(boxes, scores, img_shape,
                           pre_nms_topk, post_nms_topk=None):
    """
    Sample RPN proposals by the following steps:
    1. Pick top k1 by scores
    2. NMS them
    3. Pick top k2 by scores. Default k2 == k1, i.e. does not filter the NMS output.

    Args:
        boxes: nx4 float dtype, the proposal boxes. Decoded to floatbox already
        scores: n float, the logits
        img_shape: [h, w]
        pre_nms_topk, post_nms_topk (int): See above.

    Returns:
        boxes: kx4 float
        scores: k logits
    """
    assert boxes.shape.ndims == 2, boxes.shape
    if post_nms_topk is None:
        post_nms_topk = pre_nms_topk

    topk = tf.minimum(pre_nms_topk, tf.size(scores))
    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
    topk_boxes = tf.gather(boxes, topk_indices)
    topk_boxes = clip_boxes(topk_boxes, img_shape)

    topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))
    topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)
    # nx1x2 each
    wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)
    valid = tf.reduce_all(wbhb > cfg.RPN.MIN_SIZE, axis=1)  # n,
    topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
    topk_valid_scores = tf.boolean_mask(topk_scores, valid)

    # TODO not needed
    topk_valid_boxes_y1x1y2x2 = tf.reshape(
        tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),
        (-1, 4), name='nms_input_boxes')
    nms_indices = tf.image.non_max_suppression(
        topk_valid_boxes_y1x1y2x2,
        topk_valid_scores,
        max_output_size=post_nms_topk,
        iou_threshold=cfg.RPN.PROPOSAL_NMS_THRESH)

    topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1, 4))
    proposal_boxes = tf.gather(topk_valid_boxes, nms_indices)
    proposal_scores = tf.gather(topk_valid_scores, nms_indices)
    tf.sigmoid(proposal_scores, name='probs')  # for visualization
    return tf.stop_gradient(proposal_boxes, name='boxes'), tf.stop_gradient(proposal_scores, name='scores')


@under_name_scope()
def generate_rpn_proposals_iou(boxes, scores, ious, img_shape,
                               pre_nms_topk, post_nms_topk=None):
    """
    Sample RPN proposals by the following steps:
    1. Pick top k1 by a * scores + b * ious,
    2. NMS them
    3. Pick top k2 by a * scores + b * ious. Default k2 == k1, i.e. does not filter the NMS output.

    Args:
        boxes: nx4 float dtype, the proposal boxes. Decoded to floatbox already
        scores: n float, the logits
        ious: n float, the logits
        img_shape: [h, w]
        pre_nms_topk, post_nms_topk (int): See above.

    Returns:
        boxes: kx4 float
        scores: k logits
        ious: k logits
    """
    assert boxes.shape.ndims == 2, boxes.shape
    if post_nms_topk is None:
        post_nms_topk = pre_nms_topk

    topk = tf.minimum(pre_nms_topk, tf.size(scores))

    a, b = 1., 1.
    scores_ious = a * tf.sigmoid(scores) + b * tf.sigmoid(ious)

    topk_scores_ious, topk_indices = tf.nn.top_k(scores_ious, k=topk, sorted=False)
    topk_scores = tf.gather(scores, topk_indices)
    topk_ious = tf.gather(ious, topk_indices)
    topk_boxes = tf.gather(boxes, topk_indices)
    topk_boxes = clip_boxes(topk_boxes, img_shape)

    topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))
    topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)
    # nx1x2 each
    wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)
    valid = tf.reduce_all(wbhb > cfg.RPN.MIN_SIZE, axis=1)  # n,
    topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
    topk_valid_scores = tf.boolean_mask(topk_scores, valid)
    topk_valid_ious = tf.boolean_mask(topk_ious, valid)
    topk_scores_ious = tf.boolean_mask(topk_scores_ious, valid)

    # TODO not needed
    topk_valid_boxes_y1x1y2x2 = tf.reshape(
        tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),
        (-1, 4), name='nms_input_boxes')
    # iou score nms
    nms_indices = tf.image.non_max_suppression(
        topk_valid_boxes_y1x1y2x2,
        topk_scores_ious,
        max_output_size=post_nms_topk,
        iou_threshold=cfg.RPN.PROPOSAL_NMS_THRESH)

    topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1, 4))
    proposal_boxes = tf.gather(topk_valid_boxes, nms_indices)
    proposal_scores = tf.gather(topk_valid_scores, nms_indices)
    proposal_ious = tf.gather(topk_valid_ious, nms_indices)

    tf.sigmoid(proposal_scores, name='probs')  # for visualization
    tf.sigmoid(proposal_ious, name='overlaps')  # for visualization
    return tf.stop_gradient(proposal_boxes, name='boxes'), \
           tf.stop_gradient(proposal_scores, name='scores'), \
           tf.stop_gradient(proposal_ious, name='ious')

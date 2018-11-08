import os
import cv2
import numpy as np
import pandas as pd

classes = ['Pneumonia']


def evaluate(gtfile,predfile):

    gt = load_data(gtfile, 'gt')
    pred = load_data(predfile, 'pred')

    iou_thresh = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, ]
    performance = cal_performance(gt, pred, score_thresh=0.5, iou_thresh=iou_thresh)

    # cal_performance_acc(gt, pred, score_thresh=0.25)
    print_perf(performance)

    return performance


def load_data(filename, mode):
    list = open(filename, 'r')
    all_imgs = {}
    for line in list:
        line_split = line.strip().split(',')
        (imgID, data) = line_split
        if imgID == 'patientId':
            continue
        all_imgs[imgID] = {}
        all_imgs[imgID]['boxes'] = []
        all_imgs[imgID]['num'] = 0
        if mode is 'gt':
            mm = 4
        if mode is 'pred':
            mm = 5
            all_imgs[imgID]['scores'] = []

        data = data.split(' ')
        for i in range(len(data) // mm):
            if mode is 'pred':
                all_imgs[imgID]['scores'].append(float(data[i * mm]))
            x1 = int(float(data[mm - 4 + i * mm]))
            y1 = int(float(data[mm - 3 + i * mm]))
            x2 = x1 + int(float(data[mm - 2 + i * mm]) - 1)
            y2 = y1 + int(float(data[mm - 1 + i * mm]) - 1)
            all_imgs[imgID]['boxes'].append([x1, y1, x2, y2])
            all_imgs[imgID]['num'] = all_imgs[imgID]['num'] + 1
    return all_imgs


def bbox_overlaps(boxesA, boxesB):
    """  From tf-faster-rcnn
    Parameters
    ----------
    boxesA: (N, 4) ndarray of float [N, [x1, y1, x2, y2]]
    boxesB: (K, 4) ndarray of float [K, [x1, y1, x2, y2]]
    mode: under area: 'union' or 'min'
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxesA and boxesB
    """
    overlaps = np.zeros((boxesA.shape[0], boxesB.shape[0]), dtype=np.float32)
    for k in range(boxesB.shape[0]):
        boxB_area = (
                (boxesB[k, 2] - boxesB[k, 0] + 1) *
                (boxesB[k, 3] - boxesB[k, 1] + 1)
        )
        for n in range(boxesA.shape[0]):
            iw = (
                    min(boxesA[n, 2], boxesB[k, 2]) -
                    max(boxesA[n, 0], boxesB[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                        min(boxesA[n, 3], boxesB[k, 3]) -
                        max(boxesA[n, 1], boxesB[k, 1]) + 1
                )
                if ih > 0:
                    boxA_area = (boxesA[n, 2] - boxesA[n, 0] + 1) * (boxesA[n, 3] - boxesA[n, 1] + 1)
                    ua = float((boxA_area + boxB_area - iw * ih))
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def cal_performance(gt, pred, score_thresh, iou_thresh):
    performance = {}
    for i in iou_thresh:
        performance[str(i)] = {'TP': 0, 'FP': 0, 'FN': 0, 'AP': 0}
    performance['bg'] = {'FP': 0}
    for id in gt:
        gt_boxes = np.array(gt[id]['boxes'])
        gt_num = gt[id]['num']
        pred_boxes = np.array(pred[id]['boxes'])
        pred_scores = np.array(pred[id]['scores'])
        pred_num = pred[id]['num']

        # delete boxes whose score < score_thresh
        if pred_num > 0:
            s_inds = np.where(pred_scores >= score_thresh)[0]
            pred_boxes = pred_boxes[s_inds, :]
            pred_scores = pred_scores[s_inds]
            pred_num = len(s_inds)

        for iou_th in iou_thresh:
            if pred_num > 0 and gt_num > 0:
                # iou
                overlaps = bbox_overlaps(pred_boxes, gt_boxes)
                # # get pred_boxes max iou
                # max_overlaps = overlaps.max(axis=1)
                # get gt_boxes max iou
                gt_max_overlaps = overlaps.max(axis=0)
                #
                tp_inds = np.where(gt_max_overlaps > iou_th)[0]
                fn_inds = np.where(gt_max_overlaps <= iou_th)[0]
                # tp_inds = np.where(max_overlaps > iou_th)[0]
                # fp_inds = np.where(max_overlaps <= iou_th)[0]
                # fn_inds = np.where(gt_max_overlaps <= iou_th)[0]

                # one gt box can only be calculated once if tp, so use gt_max_overlaps
                performance[str(iou_th)]['TP'] = performance[str(iou_th)]['TP'] + np.size(tp_inds)
                performance[str(iou_th)]['FP'] = performance[str(iou_th)]['FP'] + (
                        np.size(pred_scores) - np.size(tp_inds))
                # performance[str(iou_th)]['FP'] = performance[str(iou_th)]['FP'] + np.size(fp_inds)
                performance[str(iou_th)]['FN'] = performance[str(iou_th)]['FN'] + np.size(fn_inds)

            else:
                # pred_num == 0 and gt_num > 0:
                performance[str(iou_th)]['FN'] = performance[str(iou_th)]['FN'] + gt_num
                # pred_num > 0 and gt_num == 0:
                performance[str(iou_th)]['FP'] = performance[str(iou_th)]['FP'] + pred_num

        if pred_num > 0 and gt_num == 0:
            performance['bg']['FP'] = performance['bg']['FP'] + pred_num

    for perf in performance:
        if perf == 'bg':
            continue
        eps = 1e-5
        TP = performance[perf]['TP']
        FP = performance[perf]['FP']
        FN = performance[perf]['FN']
        performance[perf]['AP'] = 1. * TP / (TP + FP + FN + eps)
        performance[perf]['Precision'] = 1. * TP / (TP + FP - performance['bg']['FP'] + eps)
        performance[perf]['Recall'] = 1. * TP / (TP + FN + eps)
        performance[perf]['F1-Score'] = 2. * performance[perf]['Precision'] * performance[perf]['Recall'] / (
                performance[perf]['Precision'] + performance[perf]['Recall'] + eps)

    return performance


def cal_performance_acc(gt, pred, score_thresh):
    performance = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'ACC': 0}
    for id in gt:
        gt_boxes = np.array(gt[id]['boxes'])
        gt_num = gt[id]['num']
        pred_boxes = np.array(pred[id]['boxes'])
        pred_scores = np.array(pred[id]['scores'])
        pred_num = pred[id]['num']

        # delete boxes whose score < score_thresh
        if pred_num > 0:
            s_inds = np.where(pred_scores >= score_thresh)[0]
            pred_boxes = pred_boxes[s_inds, :]
            pred_scores = pred_scores[s_inds]
            pred_num = len(s_inds)

        if pred_num > 0 and gt_num > 0:
            performance['TP'] = performance['TP'] + 1
        elif pred_num == 0 and gt_num == 0:
            performance['TN'] = performance['TN'] + 1
        elif pred_num > 0 and gt_num == 0:
            performance['FP'] = performance['FP'] + 1
        elif pred_num == 0 and gt_num > 0:
            performance['FN'] = performance['FN'] + 1

    TP = performance['TP']
    TN = performance['TN']
    FP = performance['FP']
    FN = performance['FN']
    performance['ACC'] = 1. * (TP + TN) / (TP + TN + FP + FN)
    performance['ACC'] = 1. * (TP + TN) / (TP + TN + FP + FN)

    print('------------------Binary--------------')
    print(performance)


def print_perf(performance):
    MAP = 0
    precision = 0
    recall = 0
    f1 = 0
    print('------------------Kaggle Performance--------------')
    for perf in performance:
        if perf == 'bg':
            continue
        print('IOU: %s,\tTP: %d,\tFP: %d,\tFN: %d,\tAP: %.4f,\tprecision: %.4f,\trecall: %.4f,\tf1-score: %.4f' % (
            perf,
            performance[perf]['TP'],
            performance[perf]['FP'],
            performance[perf]['FN'],
            performance[perf]['AP'],
            performance[perf]['Precision'],
            performance[perf]['Recall'],
            performance[perf]['F1-Score'],)
              )
        # print('IOU: %.2f,\tAP: %.6f' % (float(perf), performance[perf]['AP']))
        MAP += performance[perf]['AP']
        precision += performance[perf]['Precision']
        recall += performance[perf]['Recall']
        f1 += performance[perf]['F1-Score']

    print('BG: \tFP: %d, P: %.4f,' % (performance['bg']['FP'], performance['bg']['FP'] / 1600.))
    print('MAP: %.4f, precision: %.4f, recall: %.4f, f1-score: %.4f' % (MAP / 8., precision / 8., recall / 8., f1 / 8.))

def summary_per(performance):
    MAP = 0
    precision = 0
    recall = 0
    f1 = 0
    for perf in performance:
        if perf == 'bg':
            continue
        MAP += performance[perf]['AP']
        precision += performance[perf]['Precision']
        recall += performance[perf]['Recall']
        f1 += performance[perf]['F1-Score']

    return MAP / 8., precision / 8., recall / 8., f1 / 8.

if __name__ == '__main__':
    evaluate()
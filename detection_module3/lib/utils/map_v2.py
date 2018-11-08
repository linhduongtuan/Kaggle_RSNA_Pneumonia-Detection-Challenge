import numpy as np
import pandas as pd


def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2 - xi1) * (yi2 - yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, AP, TP, FP, FN, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    TP[t] += 1
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN
                FN[t] += 1

        fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
        FP[t] += fp
        m = 1.0 * tp / (tp + fn + fp)
        AP[t] += m
        map_total += m

    return map_total / len(thresholds), AP, TP, FP, FN


def evaluation(labels, preds, cls_threshold_list=[0.5,]):
    labels = np.asarray(pd.read_csv(labels))
    preds = np.asarray(pd.read_csv(preds))
    names = list(set(preds[:, 0]))

    for cls_threshold in cls_threshold_list:
        print('------This is for cls_threshold:', cls_threshold, '--------')

        MAP = []
        AP = dict.fromkeys([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], 0)
        AP_count = 0
        TP = dict.fromkeys([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], 0)
        FP = dict.fromkeys([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], 0)
        FN = dict.fromkeys([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], 0)

        GT = 0
        PRED = 0
        for name in names:
            label = labels[labels[:, 0] == name]
            pred = preds[preds[:, 0] == name][0][1]

            boxes_true = []
            boxes_pred = []
            scores = []

            has_gt = False
            for idx, x, y, w, h, target in label:
                if target == 1:
                    boxes_true.append([x, y, w, h])
                    GT += 1
                    has_gt = True

            has_pred = False if pd.isnull(pred) else True
            if has_pred == True:
                pred = np.asarray(str(pred).split(), dtype='float')
                length = len(pred)
                assert length % 5 == 0
                length = length // 5
                valid_pred = False
                for k in range(length):
                    if pred[k * 5] >= cls_threshold:
                        scores.append(pred[k * 5])
                        boxes_pred.append([pred[k * 5 + 1], pred[k * 5 + 2], pred[k * 5 + 3], pred[k * 5 + 4]])
                        valid_pred = True
                has_pred = has_pred & valid_pred

            if has_pred == True:
                PRED += 1

            if has_gt == False and has_pred == False:
                continue
            elif has_gt == False and has_pred == True:
                MAP.append(0)
                AP_count += 1
            elif has_gt == True and has_pred == False:
                MAP.append(0)
                AP_count += 1
            else:
                AP_count += 1
                boxes_true = np.asarray(boxes_true)
                boxes_pred = np.asarray(boxes_pred)
                scores = np.asarray(scores)

                map_score, AP, TP, FP, FN = map_iou(boxes_true, boxes_pred, scores, AP, TP, FP, FN)
                MAP.append(map_score)

        MAP = np.asarray(MAP).mean()

        for threshold in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
            print('threshold: %.2f AP: %.3f TP: %d FP: %d FN: %d GT: %d' % (
                threshold, AP[threshold] / AP_count, TP[threshold], FP[threshold], FN[threshold], GT))

        print('MAP:', MAP)
        print('Pred_img_num: ', PRED)

    return MAP
#evaluation('stage_2_train_labels_for_map.csv', 'res_kaggle_s1_test/yj_iounet_0.099.csv',
#           cls_threshold_list=[0.3,0.4,0.5,0.6,0.7,])

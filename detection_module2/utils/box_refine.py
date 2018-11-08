import numpy as np


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


def flip_boxes(boxes, im_width):
    """Flip boxes horizontally."""
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped


def box_voting(top_dets, top_scores, all_dets, all_scores, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 4] each row is [x1 y1 x2 y2]
    # all_dets is [N, 4] each row is [x1 y1 x2 y2]
    # all_probs is [N] each row is [score]
    top_dets_out = top_dets.copy()
    top_scores_out = top_scores.copy()
    top_boxes = top_dets
    all_boxes = all_dets

    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_scores_out[k] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_scores_out[k] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_scores_out[k] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws ** beta) ** (1.0 / beta)
            top_scores_out[k] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_scores_out[k] = ws.sum() / float(len(ws)) ** beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out, top_scores_out


def soft_nms_py(
        boxes, probs, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='gaussian'
):
    """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
    if boxes.shape[0] == 0:
        return boxes, []

    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    assert method in methods, 'Unknown soft_nms method: {}'.format(method)
    method = methods[method]

    boxes = boxes.copy()
    N = boxes.shape[0]
    inds = np.arange(N)

    for i in range(N):
        maxscore = probs[i]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = probs[i]
        ti = inds[i]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < probs[pos]:
                maxscore = probs[pos]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        probs[i] = probs[maxpos]
        inds[i] = inds[maxpos]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        probs[maxpos] = ts
        inds[maxpos] = ti

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below
        # threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > overlap_thresh:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > overlap_thresh:
                            weight = 0
                        else:
                            weight = 1

                    probs[pos] = weight * probs[pos]

                    # if box score falls below threshold, discard the box by
                    # swapping with last box update N
                    if probs[pos] < score_thresh:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        probs[pos] = probs[N - 1]
                        inds[pos] = inds[N - 1]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    return boxes[:N], probs[:N], inds[:N]


if __name__ == '__main__':

    boxes = [[101, 101, 201, 201], [111, 111, 211, 211], [211, 211, 311, 311]]
    probs = [0.6, 0.8, 0.99]
    boxes = np.array(boxes).reshape([-1, 4])
    probs = np.array(probs).reshape([-1])
    box, prob, keep = soft_nms_py(boxes, probs)
    print(boxes)
    print(box)
    print(keep)
    print(prob)
    top_det, top_score = box_voting(box, prob, boxes, probs, thresh=0.3)
    print(top_det)
    print(top_score)
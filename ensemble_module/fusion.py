import numpy as np
import pandas as pd
import argparse
from pandas.core.frame import DataFrame
from sys import argv

def iou(box1, box2):
    # convert str to float
    for i in range(5):
        box1[i] = float(box1[i])
        box2[i] = float(box2[i])

    # get the coordinate
    b1_x1, b1_x2 = box1[1], box1[1] + box1[3]
    b1_y1, b1_y2 = box1[2], box1[2] + box1[4]
    b2_x1, b2_x2 = box2[1], box2[1] + box2[3]
    b2_y1, b2_y2 = box2[2], box2[2] + box2[4]

    # get the intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    # intersection area
    inter_area = max(inter_x2-inter_x1+1, 0) * max(inter_y2-inter_y1+1, 0)

    # union area
    box1_area = box1[3] * box1[4]
    box2_area = box2[3] * box2[4]
    union_area = box1_area + box2_area - inter_area

    result = inter_area / union_area

    return result


parser = argparse.ArgumentParser()
parser.add_argument("-model_num", type=int, default=11)
parser.add_argument("-iou_threshold", type=float, default=0.3)
parser.add_argument("-vote_threshold", type=int, default=7)
args = parser.parse_args()
args = vars(args).items()
cfg = {}
for name, val in args:
    cfg[name] = val

path = 'solution/result_'
patient_id = []
bbox = []
origin = []             					# record the original model of each box
model_num = cfg['model_num']           		# number of results
threshold = cfg['iou_threshold']         	# threshold for iou
vote_threshold = cfg['vote_threshold']      # threshold for vote

print('---------------------------------------')
print('model_num: ', model_num)
print('vote_threshold: ', vote_threshold)
print('threshold: ', threshold)

# ---------------------- Load Data ---------------------- #
for t in range(model_num):
    data = pd.read_csv(path+str(t)+'.csv', header=0)
    data = np.array(data)
    id = data[:, 0].tolist()
    box = data[:, 1].tolist()

    for i in range(len(box)):
        if not(isinstance(box[i], float)):        # split numbers into list
            box[i] = box[i].split(' ')
            if box[i][-1] == '':                  # delete '' at the end of the list
                box[i] = box[i][:-1]
            if box[i][0] == '':                   # delete '' at the beginning of the list
                box[i] = box[i][1:]

        if id[i] not in patient_id:             	# merge data
            patient_id.append(id[i])
            if not (isinstance(box[i], float)):     # check nan, make sure box[index] is a list
                bbox.append(box[i])
                origin.append([len(box[i])//5])
            else:
                bbox.append([])
                origin.append([])
        else:
            index = patient_id.index(id[i])
            if not (isinstance(box[i], float)):
                for k in box[i]:
                    bbox[index].append(k)
                origin[index].append(len(box[i])//5)

for i in range(len(origin)):        # prefix sum, in order to distinguish each box belong to whose model
    for j in range(1, len(origin[i])):
        origin[i][j] = origin[i][j-1] + origin[i][j]

# output
s = []
for i in range(len(patient_id)):
    s.append(' '.join(bbox[i]))
output = {'patientId': patient_id, 'PredictionString': s}
output_data = DataFrame(output)
#output_data.to_csv('result_all.csv', index=False)

for i in range(len(bbox)):          # convert str to float
    bbox[i] = [float(item) for item in bbox[i]]

# ---------------------- Vote ---------------------- #
choice = []     # store the whole vote results
box_iou = []    # store the sum of iou for each box
for i in range(len(patient_id)):
    vote = [0 for j in range(len(bbox[i])//5)]
    vote_from = [[0 for jj in range(model_num)] for ii in range(len(bbox[i])//5)]     # ensure each box only be voted once by each model
    box_iou.append([0 for j in range(len(bbox[i])//5)])

    for j in range(len(bbox[i])//5):
        for k in range(j+1, len(bbox[i])//5):
            iou_tmp = iou(bbox[i][j*5:j*5+5], bbox[i][k*5:k*5+5])
            box_iou[i][j] += iou_tmp
            box_iou[i][k] += iou_tmp
            if iou_tmp > threshold:
                s, t = 0, 0
                # find j, k belong to whose model, ensure each model has one vote, boxes from the same model won't vote for each other
                for l in range(len(bbox[i])//5):
                    if j < origin[i][l]:
                        s = l
                        break
                for l in range(len(bbox[i])//5):
                    if k < origin[i][l]:
                        t = l
                        break
                if s != t:      # not from the same model
                    if vote_from[j][t] == 0:    # box j haven't voted by model t
                        vote[j] += 1
                    if vote_from[k][s] == 0:    # box k haven't voted by model s
                        vote[k] += 1
                    vote_from[j][t] = 1
                    vote_from[k][s] = 1

    print(patient_id[i], len(bbox[i])//5, ':', vote)
    choice.append(vote)


# ---------------------- Merge ---------------------- #
result_merge = []
for i in range(len(patient_id)):
    use = [0 for j in range(len(bbox[i])//5)]   # record whether box j has been merged
    result_merge.append([])

    for j in range(len(bbox[i])//5):            # if the box has few votes, then this box will be marked as merged, won't be merged any more
        if choice[i][j] < vote_threshold:
            use[j] = 1

    while (sum(use) != len(bbox[i])//5):
        max_vote = 0
        max_iou = 0
        max_index = 0
        for j in range(len(bbox[i])//5):            # choose a box with largest iou to start merge
            if use[j] == 0 and (choice[i][j] > max_vote or (choice[i][j] == max_vote and box_iou[i][j] > max_iou)):
                max_iou = box_iou[i][j]
                max_index = j
                max_vote = choice[i][j]
        use[max_index] = 1
        box_merge = bbox[i][max_index*5:max_index*5+5]

        num = 1                                     # record the number of the merged boxes
        for j in range(len(bbox[i])//5):
            if choice[i][j] >= vote_threshold and use[j] == 0 and iou(bbox[i][max_index*5:max_index*5+5], bbox[i][j*5:j*5+5]) > threshold:
                box_merge = [box_merge[k]+bbox[i][j*5+k] for k in range(5)]
                use[j] = 1
                num += 1
        if num >= vote_threshold:           
            box_merge = [box_merge[k]/num for k in range(5)]
            for k in range(5):
                result_merge[i].append(box_merge[k])

for i in range(len(result_merge)):          # convert str to float
    result_merge[i] = [str(item) for item in result_merge[i]]

# output merged result
s = []
for i in range(len(patient_id)):
    s.append(' '.join(result_merge[i]))
output = {'patientId': patient_id, 'PredictionString': s}
output_data = DataFrame(output)
output_data.to_csv('result_merge.csv', index=False, float_format='%.2f')


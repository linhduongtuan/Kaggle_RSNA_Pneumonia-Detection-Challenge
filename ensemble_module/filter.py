import numpy as np
import pandas as pd
import argparse


def classify(predsfile, clsesfile, wc_cls_thresh):
    preds = np.asarray(pd.read_csv(predsfile))
    clses = np.asarray(pd.read_csv(clsesfile))
    names = list(set(preds[:, 0]))

    tt = open('./%s' % predsfile.split('/')[-1], 'w')
    tt.write('patientId,PredictionString\n')

    final_num = 0

    for name in names:
        tt.write('%s,' % name)
        pred = preds[preds[:, 0] == name][0][1]
        cls = clses[clses[:, 0] == name][0][1]
        if pd.isnull(pred) or cls < wc_cls_thresh:
            save_pred = False
        else:
            final_num += 1
            save_pred = True

        if save_pred:
            tt.write('%s' % pred)
        tt.write('\n')
    tt.close()

    print('num_after_cls:', final_num)


"""
wc_cls_thresh:  >=
"""

parser = argparse.ArgumentParser()
parser.add_argument("-predsfile", type=str, default='result_0')
parser.add_argument("-classfile", type=str, default='result_1')
parser.add_argument("-threshold", type=float, default=0.55)
args = parser.parse_args()
args = vars(args).items()
cfg = {}
for name, val in args:
    cfg[name] = val
#classify('res_kaggle_s1_test/submission_gn_5W.csv', 'test1000_v1p1.csv', wc_cls_thresh=0.55)
classify(cfg['predsfile'], cfg['classfile'], cfg['threshold'])

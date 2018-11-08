
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils.boxes as box_utils
import utils.keypoints as keypoint_utils
import utils.segms as segm_utils
import utils.blob as blob_utils
from core.config import cfg
from datasets.json_dataset import JsonDataset
from matplotlib.patches import Rectangle

import solt.transforms as slt
import solt.data as  sld
import solt.core as slc

def get_roidb(dataset_name, proposal_file):
    ds = JsonDataset(dataset_name)
    roidb = ds.get_roidb(
        gt=True,
        proposal_file=proposal_file,
        crowd_filter_thresh=cfg.TRAIN.CROWD_FILTER_THRESH
    )
    return roidb


if __name__ == '__main__':

    dataset_name = 'RSNA_2018_pos_val'

    roidb = get_roidb(dataset_name,proposal_file=None)

    for i, entry in enumerate(roidb):
        data = cv2.imread(entry['image'])
        # bboxes = entry['boxes']

        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot(1,1,1)
        # ax.imshow(data)
        #
        # text_bbox = dict(facecolor='red',alpha=0.7, lw=0)
        # for i, bbox in enumerate(entry['boxes']):
        #     ax.add_patch(Rectangle((bbox[0],bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],fill=False,color='r',lw=2))
        #     ax.text(bbox[0]+6, bbox[1]-15,'lesion',bbox=text_bbox)

        bbox = entry['boxes'][0]

        #left_bottom
        left_bottom = (bbox[0],bbox[1])
        right_bottom = (bbox[2], bbox[1])
        left_top = (bbox[0], bbox[3])
        right_top = (bbox[2], bbox[3])

        kpts = sld.KeyPoints(np.vstack((left_top, right_top, right_bottom, left_bottom)), entry['height'], entry['width'])
        print (kpts.data)
        pass

        dc = sld.DataContainer((data, kpts, 0),'IPL')

        stream = slc.Stream([
            slt.RandomProjection(
                slc.Stream([
                    slt.RandomScale(range_x=(0.8, 1.1), p=1),
                    slt.RandomRotate(rotation_range=(-90, 90), p=1),
                    slt.RandomShear(range_x=(-0.2, 0.2), range_y=None, p=0.7),
                ]),
                v_range=(1e-6, 3e-4), p=1),
            # Various cropping and padding tricks
            slt.PadTransform(1000, 'z'),
            slt.CropTransform(1000, crop_mode='c'),
            slt.CropTransform(950, crop_mode='r'),
            slt.PadTransform(1000, 'z'),
            # Intensity augmentations
            slt.ImageGammaCorrection(p=1, gamma_range=(0.5, 3)),
            slc.SelectiveStream([
                slc.SelectiveStream([
                    slt.ImageSaltAndPepper(p=1, gain_range=0.01),
                    slt.ImageBlur(p=0.5, blur_type='m', k_size=(11,)),
                ]),
                slt.ImageAdditiveGaussianNoise(p=1, gain_range=0.5),
            ]),
        ])

        for i in range(10):
            res = stream(dc)
            img_res, kp_c, lbl_c = res.data
            fig = plt.figure(figsize=(10, 10))

            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(img_res, cmap=plt.cm.Greys_r)

            for pts, cls in zip([kp_c.data], ['Cardiomegaly']):
                text_bbox = dict(facecolor='red', alpha=0.7, lw=0)
                # Let's clip the points so that they will not
                # violate the image borders
                pts[:, 0] = np.clip(pts[:, 0], 0, img_res.shape[1] - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, img_res.shape[0] - 1)
                x, y = pts[:, 0].min(), pts[:, 1].min()
                w, h = pts[:, 0].max() - x, pts[:, 1].max() - y
                ax.add_patch(Rectangle((x, y), w, h, fill=False, color='r', lw=2))
                ax.text(x + 6, y - 15, cls, fontsize=12, bbox=text_bbox)
            plt.show()




















        # plt.imshow(data)
        # plt.show()




    pass

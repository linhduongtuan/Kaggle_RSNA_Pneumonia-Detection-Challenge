
# coding: utf-8

# *Copyright 2018 MD.ai, Inc.
# Licensed under the Apache License, Version 2.0*


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import glob


ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.abspath('./data')

test_dicom_dir = os.path.join(DATA_DIR, 'stage_1_test_images')


# Import Mask RCNN ohem version
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))
from mrcnn_OHEM.config import Config
from mrcnn_OHEM import utils
import mrcnn_OHEM.model as modellib
from mrcnn_OHEM import visualize
from mrcnn_OHEM.model import log


def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations


class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """

    # Give the configuration a recognizable name
    NAME = 'pneumonia'

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BACKBONE = 'resnet50'

    NUM_CLASSES = 2  # background + 1 pneumonia classes


    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    TRAIN_ROIS_PER_IMAGE = 16

    MAX_GT_INSTANCES = 5

    DETECTION_MAX_INSTANCES = 4
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.3

    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    STEPS_PER_EPOCH = 1000
    TOP_DOWN_PYRAMID_SIZE = 64
    STEPS_PER_EPOCH = 1000

    MEAN_PIXEL = np.array([122.94, 122.94, 122.94])

config = DetectorConfig()
config.display()



# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024
# Resnet 50 weights
model_path = '../weights/r50_ohem_finetune_ep71_score0.209.h5'


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Get filenames of test dataset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)


# Make predictions on test images, write out submission file
def predict(image_fps, filepath='r50_ohem_ep71.csv', min_conf=0.9):

    # assume square image

    with open(filepath, 'w') as file:
      for image_id in tqdm(image_fps):
        ds = pydicom.read_file(image_id)
        image = ds.pixel_array

        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

        patient_id = os.path.splitext(os.path.basename(image_id))[0]

        results = model.detect([image])
        r = results[0]

        out_str = ""
        out_str += patient_id
        assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
        if len(r['rois']) == 0:
            pass
        else:
            num_instances = len(r['rois'])
            out_str += ","
            for i in range(num_instances):
                if r['scores'][i] > min_conf:
                    out_str += ' '
                    out_str += str(round(r['scores'][i], 2))
                    out_str += ' '

                    # x1, y1, width, height
                    x1 = r['rois'][i][1]
                    y1 = r['rois'][i][0]
                    width = r['rois'][i][3] - x1
                    height = r['rois'][i][2] - y1
                    bboxes_str = "{} {} {} {}".format(x1, y1,                                                       width, height)
                    out_str += bboxes_str

        file.write(out_str+"\n")



sample_submission_fp = 'r50_ohem_ep71.csv'
predict(test_image_fps, filepath=sample_submission_fp)

# In[ ]:

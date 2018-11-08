
# coding: utf-8
#
# The [Radiological Society of North America](http://www.rsna.org/) Pneumonia Detection Challenge: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
#
# This notebook covers the basics of parsing the competition dataset, training using a detector basd on the [Mask-RCNN algorithm](https://arxiv.org/abs/1703.06870) for object detection and instance segmentation.
#
# This notebook is developed by [MD.ai](https://www.md.ai), the platform for medical AI.
# This notebook requires Google's [TensorFlow](https://www.tensorflow.org/) machine learning framework.
#
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


# directory
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.abspath('./data')
MODEL_DIR = os.path.abspath('./log')
train_dicom_dir = os.path.join(DATA_DIR, 'stage_1_train_images')


# Import Mask RCNN OHEM version
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn_OHEM.config import Config
from mrcnn_OHEM import utils
import mrcnn_OHEM.model as modellib
from mrcnn_OHEM import visualize
from mrcnn_OHEM.model import log

# Some setup functions and classes for Mask-RCNN
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


# HyperParameters selection
class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """

    # Give the configuration a recognizable name
    NAME = 'pneumonia'

    # Train on 1 GPU and 8 images per GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BACKBONE = 'resnet50' # 'resnet101'

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


class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)

        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')

        # add images
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp,
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)

# training dataset
anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv')) #all data
image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)

# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024


# Split the data into training and validation datasets

image_fps_list = list(image_fps)

# split dataset into training vs. validation dataset
# split ratio is set to 0.9 vs. 0.1 (train vs. validation, respectively)
sorted(image_fps_list)
random.seed(42)
random.shuffle(image_fps_list)

validation_split = 0.1
split_index = int((1 - validation_split) * len(image_fps_list))

image_fps_train = image_fps_list[:split_index]
image_fps_val = image_fps_list[split_index:]


# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()


# Build up the models
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

# Image augmentation
augmentation = iaa.SomeOf((0, 1), [
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    ),
    iaa.Multiply((0.9, 1.1))
])



# Train
NUM_EPOCHS = 100
import warnings
warnings.filterwarnings("ignore")
# use pretrained resnet50 weights
model_path = '../weights/r50_base.h5' # the weights trained on positive images via train_positive.py
model.load_weights(model_path, by_name=True)
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=NUM_EPOCHS,
            layers='all',
            augmentation=augmentation)

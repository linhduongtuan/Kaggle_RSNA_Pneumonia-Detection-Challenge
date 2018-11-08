from keras.preprocessing.image import random_zoom, random_shift
from keras.preprocessing.image import random_rotation, load_img
from keras.preprocessing.image import img_to_array
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas
import os
from PIL import Image
#from PIL import Image
from keras import backend as K
import tensorflow as tf
from scipy import ndimage
import matplotlib.pyplot as plt
from tools import threadsafe_generator



#gray = False 
mean = 122.94
scale = 0.017
img_rows = 512
img_cols = 512
feat_map_threshold = 70
# get all CSV file paths
# path: all label's path
# return: each CSV files path
def get_csv_path(path):
    files_1 = os.listdir(path)
    files_1.sort()
    path_list = []
    for file_1 in files_1:
        path_list.append(path+"/"+file_1)
    path_list.sort()
    path_list = np.array(path_list)
    return path_list


# get index array from each CSV file
# label_path: each CSV files path
# return: CSV's information
def get_index_array(label_path):
    dataframe = pandas.read_csv(label_path, header=0)
    dataset = dataframe.values
    label = np.array(dataset)
    return label


# split dataset to train validation and test by 7:1:2
def random_split(label):
    train, test = train_test_split(
        label, test_size=0.1, random_state=1)
    train, validation = train_test_split(
       train, test_size=0.1, random_state=1)
    return train, validation,test




# get image and label paths
# label_path: all label's path
# return: each sample's path and 14 dim labels
def get_image_and_label(label_path):
    # Get every csv file path
    label_files_path_list = get_csv_path(label_path)
    train = []
    validation = []
    test = []
    for file in label_files_path_list:
        index_array = get_index_array(file)
    # Randomly(seed = 0) split to train, validation and test 70%,10%,20%
        data = random_split(index_array)
        train.extend(data[0])
        validation.extend(data[1])
        test.extend(data[2])
    train=np.array(train)
    validation=np.array(validation)
    test = np.array(test)
    X_train = train[:, :1]
    X_validation = validation[:, :1]
    X_test = test[:,:1]
    Y_train = train[:, 1:]
    Y_validation = validation[:, 1:]
    Y_test = test[:,1:]
    return X_train, X_validation,X_test, Y_train, Y_validation, Y_test


def get_image_and_label_all(label_path, path_pre):
    # Get every csv file path
    label_files_path_list = get_csv_path(label_path)
    train = []
    validation = []
    test = []
    for file in label_files_path_list:
        index_array = get_index_array(file)
    # Randomly(seed = 0) split to train, validation and test 70%,10%,20%
#         data = random_split(index_array)
#         train.extend(data[0])
#         validation.extend(data[1])
#         test.extend(data[2])
    index_array=np.array(index_array)
    
    X_train = index_array[:, :1]
    for index, img in enumerate(X_train):
        img =  path_pre + img +".png"
        X_train[index] = img
    Y_train = index_array[:, 1:]

    return X_train, Y_train


# def get_test_image_and_label(label_path):
#     label_files_path_list = get_csv_path(label_path)
#     test = []
#     for file in label_files_path_list:
#         index_array = get_index_array(file)
#         test.extend(index_array)
#     test = np.array(test)
#     X_test = test[:, :1]
#     Y_test = test[:, 1:]
#     return X_test, Y_test


# Data Augmentation:
# Randomly translated in 4 directions by 25 pixels
# Randomly rotated from -15 to 15 degrees
# Randomly scaled between 80% and 120%
def distorted_image(image):
   image1 = random_rotation(
       image, 15, row_axis=0, col_axis=1, channel_axis=2)
   image2 = random_zoom(
       image1, (0.8, 1.2), row_axis=0, col_axis=1, channel_axis=2)
   image3 = random_shift(
       image2, 0.05, 0.05, row_axis=0, col_axis=1, channel_axis=2)
   return image3


# change array to list
def array_to_list(target_array):
    target_list = []
    for i in target_array:
        for a in i:
            target_list.append(a)
    return target_list








def slice_to_bbox(slices):
    if (slices is None) or (len(slices) < 1):
        return [0,0,img_cols,img_rows]
    x1 = img_cols
    y1 = img_rows
    x2 = 0
    y2 = 0
    for s in slices:
        dy, dx = s[:2]
        dx1 = dx.start
        dy1 = dy.start
        dx2 = dx.stop
        dy2 = dy.stop
        x1 = min(dx1, x1)
        y1 = min(dy1, y1)
        x2 = max(dx2, x2)
        y2 = max(dy2, y2)
    return [x1+1, y1+1, x2-1, y2-1]

def get_bbox_from_heatmap(heatmap, percentage):
    #plt.matshow(heatmap, cmap='jet')
    #plt.show()
    #print(" ori heatmap ", heatmap)
    val = np.percentile(heatmap, percentage)
    heatmap[heatmap > val] = 1*255
    heatmap[heatmap <= val] = 0
    # fill the hole of binary images and find bbox of blobs in it
    filled = ndimage.morphology.binary_fill_holes(heatmap)
    #plt.matshow(filled, cmap=plt.cm.hot)
    #plt.show() 
    coded_paws, num_paws = ndimage.label(filled)
    #print("coded paws ", coded_paws[500])
    #print("num paws", num_paws)
    data_slices = ndimage.find_objects(coded_paws)
    #print(data_slices)
    #print("len slices ", len(data_slices))
    bboxes = slice_to_bbox(data_slices)
    return bboxes, filled

def get_local_image(graph, image, image_array_minus_mean, feature_model, img_rows, img_cols):
    #image_ori = image_array_minus_mean
    img_array_minus_mean = np.expand_dims(image_array_minus_mean, axis=0)
    #K.clear_session()
    with graph.as_default():
        feat_map = np.squeeze(feature_model.predict(img_array_minus_mean))
    feat_map = np.amax(feat_map, axis = 2)
    feat_map = ndimage.zoom(feat_map, (32, 32), order=1)
    bbox, filled = get_bbox_from_heatmap(feat_map, feat_map_threshold)
    #print(bbox)
    image_crop = image.crop(bbox)
    image_crop = image_crop.resize((img_rows,img_cols),Image.ANTIALIAS)
#     (x1,y1,x2,y2), filled = get_bbox_from_heatmap(feat_map, 70)
#     img_ori_crop = np.array(image_ori[ y1:y2, x1:x2, :])
#     img_ori_crop = Image.fromarray(img_ori_crop.astype('uint8'))
#     img_crop = img_ori_crop.resize((img_rows,img_cols), Image.ANTIALIAS)
#     #img_crop= densenet.preprocess_input(img_crop)
    plt.imshow(image_crop)
    plt.show()
    image_crop_arr = np.array(image_crop)
    return image_crop_arr
    
                

# def generate_from_source(image_path_list, label_path_list):
#     x = []
#     y = []

#     for image_path, label_path in zip(image_path_list, label_path_list):
#         #print image_path
#         image_path = image_path[0]
#         image_array = img_to_array(
#             load_img(image_path, grayscale=gray, target_size=(512, 512)))
#         label = label_path
#         #label_16 = label[:-1]
#         #label_abnormal = label[-1]

#         x.append(image_array)
#         y.append(label)
     
#     return (np.array(x), np.array(y))
@threadsafe_generator
def generate_from_source_for_global(image_path_list, label_path_list, batch_size):
    cnt = 0
    x = []
    y = []


    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    while True:
        for image_path, label_path in zip(image_path_list, label_path_list):
#             image_array = img_to_array(
#                 load_img(image_path, grayscale=gray, target_size=(512, 512)))
            image = Image.open(image_path).convert("RGB")
            image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
            image_array = img_to_array(image)
            #print(type(image_array)) 
            image_array = image_array - mean 
            image_array *= scale
            label = label_path

            x.append(distorted_image(image_array))
            y.append(label)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x),np.array(y))
                x = []
                y = []
                
@threadsafe_generator                
def generate_from_source_for_local(graph, feature_model, image_path_list, label_path_list, batch_size):
    cnt = 0
    x = []
    y = []


    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    while True:
        for image_path, label_path in zip(image_path_list, label_path_list):
            image = Image.open(image_path).convert("RGB")
            image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
            #plt.imshow(image)
            #image = load_img(image_path, grayscale=gray, target_size=(512, 512))
            image_array = img_to_array(image)
            #print(type(image_array)) 
            image_array_minus_mean = image_array - mean 
            crop_array = get_local_image( graph, image,image_array_minus_mean, feature_model, img_rows, img_cols) 
            crop_array_minus_mean = crop_array - mean
            crop_array_minus_mean *= scale
            label = label_path

            x.append(distorted_image(crop_array_minus_mean))
            y.append(label)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x),np.array(y))
                x = []
                y = []

                
@threadsafe_generator                
def generate_from_source_for_fusion(graph, feature_model, image_path_list, label_path_list, batch_size):
    cnt = 0
    x = []
    x_local = []
    y = []


    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    while True:
        for image_path, label_path in zip(image_path_list, label_path_list):
            image = Image.open(image_path).convert("RGB")
            image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
            #plt.imshow(image)
            #image = load_img(image_path, grayscale=gray, target_size=(512, 512))
            image_array = img_to_array(image)
            #print(type(image_array)) 
            image_array_minus_mean = image_array - mean 
            image_array_minus_mean *= scale
            crop_array = get_local_image( graph, image,image_array_minus_mean, feature_model, img_rows, img_cols)  
            crop_array_minus_mean = crop_array - mean
            crop_array_minus_mean *= scale
            label = label_path

            x.append(distorted_image(image_array_minus_mean))
            x_local.append(distorted_image(crop_array_minus_mean))
            y.append(label)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield ([np.array(x), np.array(x_local)],np.array(y))
                x = []
                x_local = []
                y = []
               
                
                

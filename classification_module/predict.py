
# coding: utf-8

# In[13]:

from __future__ import print_function
import os.path
import densenet
import numpy as np
import sklearn.metrics as metrics
import data_generate 
#from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import os
import time
from keras.models import Model
import scipy.ndimage as ndimage
from keras.utils import multi_gpu_model
from PIL import Image
import matplotlib.patches as mpatches
from keras.layers import concatenate, Flatten
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
import tensorflow as tf
import pandas
import matplotlib.pyplot as pl
from sklearn.metrics import roc_auc_score,roc_curve,auc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
model_v= "p1"
data_v = "stage2"
gpu_num = 2
batch_size =1
time_str = time.ctime()

#global_multi_weights="/home/wccui/densenet_attention_pneumonia/result/classification_model/p1/global_multigpu_Sun Oct 28 03:13:23 2018.h5"
#global_multi_weights ="/home/wccui/densenet_attention_pneumonia/result/classification_model/p2/global_multigpu_Fri Oct 26 07:56:37 2018.h5"
#global_multi_weights = "/home/wccui/densenet_attention_pneumonia/result/classification_model/v1/global_multigpu_Mon Sep 10 05:44:11 2018.h5"  
global_multi_weights ="/home/wccui/classification_module/result/classification_model/p1/global_multigpu_Mon Oct 29 02:29:46 2018.h5"
#global_multi_weights = "/home/wccui/densenet_attention_pneumonia/result/classification_model/p2/global_multigpu_Mon Oct 29 02:35:40 2018.h5"

global_single_weights = global_multi_weights.replace('multi', 'single')
#label_path = "/home/wccui/densenet_attention_pneumonia/train_csv/"+model_v +"/valid"
label_path = "/home/wccui/classification_module/test_csv/"+data_v #+"/valid"
save_path = '/home/wccui/classification_module/result/test_result/'+model_v + '/rsna_'
#path_pre = "/home1/data/Kaggle_RSNA_Pneumonia/stage_2_test_images_ori_crop/"
path_pre = "/home/wccui/classification_module/imgs/test_datas"

nb_classes = 3 
nb_epoch = 1
img_rows, img_cols = 512 ,512 
img_channels = 3
img_dim = (img_rows, img_cols, img_channels)
#img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)


# In[2]:

model_global1 = densenet.DenseNet(img_dim, depth=169, nb_dense_block=4, growth_rate=32, nb_filter=64,
                   nb_layers_per_block=[6, 12, 32, 32], bottleneck=True,reduction=0.5,
                   dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=True,
                   include_top=False, weights=None,input_tensor=None,
                   classes=nb_classes, activation='softmax')

model_global2 = multi_gpu_model(model_global1, gpus=gpu_num)
model_global2.load_weights(global_multi_weights,by_name=True)
#model_global1.save(global_single_weights)
print("imagenet densenet weights for Model global loaded.")




# In[14]:

# image_and_label = data_generate.get_image_and_label(label_path)
# test_generator_global= data_generate.generate_from_source_for_global( image_and_label[2],
#                                              image_and_label[5], batch_size)
# test_num = image_and_label[2].shape[0]
# print("data generation done")


image_and_label = data_generate.get_image_and_label_all(label_path,path_pre)
test_generator_global= data_generate.generate_from_source_for_global( image_and_label[0],
                                             image_and_label[1], batch_size)
test_num = image_and_label[0].shape[0]
print("data generation done")


# In[15]:

predict_test = model_global1.predict_generator(test_generator_global,
                    steps=test_num//batch_size, verbose=1)

predict_res = np.hstack((image_and_label[0],predict_test))
gt_res = np.hstack((image_and_label[0],image_and_label[1]))
pandas.DataFrame(
    predict_res).to_csv(
    save_path+model_v+"_"+data_v+'y_pred.csv', header=None, index=None)

pandas.DataFrame(
    gt_res).to_csv(
    save_path+model_v+"_"+data_v+'y_true.csv', header=None, index=None)
print('Predict end')



# In[16]:

from sklearn.metrics import roc_auc_score,roc_curve,auc
cls_begin = 0
cls_end = 3

auc_name = save_path+model_v+"_"+data_v+"_cls"+str(cls_begin)+"_"+str(cls_end)+"_auc.png"
comp_name = save_path+model_v+"_"+data_v+"_cls"+str(cls_begin)+"_"+str(cls_end)+"compare.csv"
thres_name = save_path+model_v+"_"+data_v+"_cls"+str(cls_begin)+"_"+str(cls_end)+"threshold.csv"



# Name=['Atelectasis','Cardiomegaly','Effusion','Infiltration',\
#       'Mass','Nodule','Pneumonia','Pneumothorax',\
#       'Consolidation','Edema','Emphysema','Fibrosis',\
#       'Pleural_Thickening','Hermia','old Tuberculosis', 'new Tuberculosis','abnormal']
Name=['Normal','No Lung Opacity / Not Normal','Lung Opacity']

y_pred = predict_test
y_true = image_and_label[1]

#res = []
fig, ax = pl.subplots()
thres_all = []

for i in range(cls_begin,cls_end):
    #print("################## ",i)
    fpr, tpr, thresholds = roc_curve(y_true[:,i], y_pred[:, i])

     # find optimal cutoff
    p = np.arange(len(tpr))
    roc = pandas.DataFrame({'tf' : pandas.Series(tpr-(1-fpr), index=p),
                            'thresholds' : pandas.Series(thresholds, index=p)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    print(roc_t['thresholds'])
    thres_all.append(roc_t['thresholds'].values)

    roc_auc = auc(fpr, tpr)

    #res.append(roc_auc)
    ax.plot(fpr, tpr, lw=1, label=Name[i]+' ROC (area = %0.2f)' % (roc_auc))
    
    
fp_all = []
fn_all = []
count_all = []
pred_binary_all = np.zeros_like(y_pred)
images_num = y_true.shape[0]
for i in range(cls_begin,cls_end):
   pred_binary = (y_pred[:,i] >=thres_all[i])*1
   pred_binary_all[:,i] = pred_binary
   gt_binary = y_true[:,i]
   fp = np.sum((pred_binary > gt_binary)*1)
   fn = np.sum((gt_binary > pred_binary)*1)  
   fp_all.append(fp)
   fn_all.append(fn) 
   count_all.append(images_num)
name = ['fp', 'fn','image numbers']
name = np.array(name).reshape(3,1)
pandas.DataFrame(np.hstack((name,
   np.vstack((fp_all,fn_all,count_all))))).to_csv(
   comp_name, header=None, index=None)

pandas.DataFrame(
     thres_all).to_csv(
     thres_name, header=None, index=None)


pl.legend(fontsize='x-small')
pl.savefig(auc_name, bbox_inches='tight')


# In[9]:

fn


# In[10]:

fp


# In[11]:

images_num


# In[12]:

1-(fn+fp)/float(images_num)


# In[21]:

y_true


# In[ ]:




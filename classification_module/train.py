
# coding: utf-8

# In[1]:

from __future__ import print_function
import os.path
import densenet
import numpy as np
import sklearn.metrics as metrics
import data_generate 
from keras.utils import multi_gpu_model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import os
import time
from keras.models import Model
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
from keras.layers import concatenate, Flatten
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
model_v= "p1"
#data_v = "v-1"
gpu_num = 2
batch_size = gpu_num * 8
time_str = time.ctime()

out_dir="/home/wccui/densenet_attention_pneumonia/result/classification_model/" + model_v +"/" 
multi_weights_file = "/home/wccui/densenet_attention_pneumonia/result/classification_model/v1/global_multigpu_Mon Sep 10 23:02:00 2018.h5"
global_multi_name = out_dir + 'global_multigpu_' + time_str + ".h5"
label_path_train = "/home/wccui/classification_module/train_csv/"
label_path_test = "/home/wccui/classification_module/val_csv/"
path_pre = "/home/wccui/classification_module/imgs/train_val_datas"



nb_classes = 3
nb_epoch = 100

img_rows, img_cols = 512,512
img_channels = 3
img_dim = (img_rows, img_cols, img_channels)


# In[ ]:

image_and_label_train = data_generate.get_image_and_label_all(label_path_train, path_pre)
image_and_label_test = data_generate.get_image_and_label_all(label_path_test, path_pre)
A=np.concatenate((image_and_label_train[1].astype(np.float32),image_and_label_test[1].astype(np.float32)),axis=0)
Pos=np.sum(np.max(A,axis=1))   # num of pathological labels assigned to each data example
Neg=np.sum(1-np.max(A,axis=1))
SUMPos=np.sum(A,axis=0)        # num of data examples labeled as a specific pathology class
SUMNeg=np.sum(1-A,axis=0)
Total=A.shape[0]               # num of data examples
RatioPos=Total/SUMPos          # corresponding to wi1
RatioNeg=Total/SUMNeg          # wi2




train_generator_global = data_generate.generate_from_source_for_global(image_and_label_train[0],
                                            image_and_label_train[1], batch_size)
valid_generator_global= data_generate.generate_from_source_for_global(image_and_label_test[0],
                                            image_and_label_test[1], batch_size)

train_num = image_and_label_train[0].shape[0]
valid_num = image_and_label_test[0].shape[0]



def WeightedLoss(y_true,y_pred):
    Loss=K.zeros([1])
    #inverse class frequency
    epsilon= 1e-5*K.ones([1]) #smooth term to avoid log0 = NaN
    for i in range(batch_size):
        for j in range(nb_classes):
            if (j == nb_classes - 1):
                Loss -= 5*(RatioPos[j]*y_true[i,j]*K.log(y_pred[i,j]+epsilon)+RatioNeg[j]*(1-y_true[i,j])*K.log((1-y_pred[i,j])+epsilon))/(RatioPos[j]+RatioNeg[j])
            else:
                Loss -= (RatioPos[j]*y_true[i,j]*K.log(y_pred[i,j]+epsilon)+RatioNeg[j]*(1-y_true[i,j])*K.log((1-y_pred[i,j])+epsilon))/(RatioPos[j]+RatioNeg[j])
    Loss=Loss/(batch_size)
    
    return Loss


print("data generation done")



# In[ ]:

model_global = densenet.DenseNet(img_dim, depth=169, nb_dense_block=4, growth_rate=32, nb_filter=64,
                   nb_layers_per_block=[6, 12, 32, 32], bottleneck=True,reduction=0.5,
                   dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=True,
                   include_top=False, weights=None,input_tensor=None,
                   classes=nb_classes, activation='softmax')
#model_global.load_weights(single_weights_file, by_name=True)



#model_global.summary()
optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training

model_global2 = multi_gpu_model(model_global, gpus=gpu_num)
model_global2.load_weights(multi_weights_file, by_name=True)

model_global2.compile(loss=WeightedLoss, optimizer=optimizer, metrics=["accuracy"])
print("Model_global created")

lr_reducer      = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                    cooldown=0, patience=5, min_lr=1e-5)
model_checkpoint= ModelCheckpoint(global_multi_name, monitor="val_loss", save_best_only=True,
                                  save_weights_only=True, verbose=1)

callbacks=[lr_reducer, model_checkpoint]

model_global2.fit_generator(train_generator_global,
                    steps_per_epoch=train_num // batch_size, epochs=nb_epoch,
                    callbacks=callbacks,
                    validation_data=valid_generator_global,
                    validation_steps=valid_num // batch_size, verbose=1,
                    pickle_safe=True,
                    workers=4)


# In[ ]:




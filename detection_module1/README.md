# Requirements
numpy
scipy
Pillow
cython
matplotlib
scikit-image
tensorflow>=1.3.0
keras>=2.0.8
opencv-python
h5py
imgaug
IPython[all]

# Detection Submodels summary
1) Backbone: ResNet50 & ResNet101
2) Adopt the code https://github.com/matterport/Mask_RCNN and modify it in order
to cater for this task. Main changes include:
-- OHEM
-- NMS

# Train
### commands
cd RSNA
python train.py

### training strategy
1) Firstly train on positive images only, using pre-trained ResNet model
2) Then continue to train against all images


# Predict
### commands
cd RSNA
python predict.py
### weights
1) pre-trained model: resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
2) trained models
-- backbone resnet50: r50_ohem_finetune_ep71_score0.209.h5
-- backbone resnet101: r101_ohem_fintune_ep116_score0.210.h5
### results
the csv results generated via the code would be further filtered using classification result

Hello! Below you can find a outline of how to reproduce my solution for the RSNA Pneumonia Detection Challenge competition.
If you run into any trouble with the setup/code or have any questions please contact me at jiyuanfeng@imsightmed.com 

---
### ARCHIVE CONTENTS

* classfication_module:   classify each image(Normal, No lung, Lung Opacity) ,the folder contain the train.py and test.py. and the Usage will recommend in the readme.md in folder.
* detection_module1: 2 stage detecter, the code is borrowed from (https://github.com/matterport/Mask_RCNN),  also, the Usage will recommend in its README.md file.
* detection_module2: 2 stage detecter, the code is borrowed from(https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN),  also, the Usage will recommend in its README.md file.
* detection_module3: 2 stage detecter, the code is borrowed from(https://github.com/roytseng-tw/Detectron.pytorch),  also, the Usage will recommend in its README.md file.
* emsemble_module: in this module, we use the classifacation results to filter the detection results. the usage will recommend in the sub README.md
---
### HARDWARE:
* Intel(R) Xeon(R) CPU E5-2697 v4 36 cores 8 Nvidia Titan Xp
* Ubuntu 16.04
* necessary 3rd-party software: the repository below can be installed by pip
  * classfication_module:
	* tensorboard==1.9.0
	* tensorflow-gpu==1.9.0
  * detection_module1:
     * tensorflow>=1.3.0 
     * keras>=2.0.8
     * opencv
     * pydicom
     * tqdm
     * imgaug
  * detection_module2:
     * python3
     * opencv
     * tensorflow >=1.6
     * pycocotools
     * tqdm
  * detection_module3:
    * pytorch>=0.3.1
    * torchvision>=0.2.0
    * cython
    * matplotlib
    * opencv
    * pyyaml
    * packaging
    * pycocotools 
    * tensorboardX
   * the more detail requirement.txt is under each module folder
---
### How to train our model and make predictions

We have four modules together, one for classification, the other three for detection.

Classification model is trained as usual with 3 different classes(normal, lung opacity and not normal no opacity). Detection models are first fine-tuned from imagenet weights with positive images, than again with both positive and negative images.
* **For classfication module:**
>1.prepare csv for training and testing as the example file in train_csv and test_csv ( csv file is recognised by folder, unnecessary files need to be deleted from the fold)
>2.Change the directory in train.py to corresponding folder name for training and run train.py.
The result model will be produced in ./result/classification_model/
>3. Change the directory in predict.py to corresponding folder name for testing and run test.py.
Prediction result will locate in ./result/test_result/
>4. **And more details are describe in the README.md in this module 's folder**
* **For detection_module1:**
> 1.  **To train**:
> cd detection_module1/RSNA
> python train.py
> 2. **To Predict:**
> cd detection_module1/RSNA
python predict.py
> 3. **And more details are describe in the README.md in this module 's folder**
* **For detection_module2:**
> 1. **To train**:
> python train.py --config \
    DATA.BASEDIR=/path/to/data_prepare/ \
	DATA.IMGDIR=images \
    --load imageNet_model/ImageNet-ResNet101.npz
 > Options can be changed by either the command line or the `config.py` file.
>2. **To predict**:
>To predict on image dir:
python train.py --config TEST.RESULT_SCORE_THRESH=0.5 TEST.FRCNN_NMS_THRESH=0.1 \
	--load /path/to/model.index --predictDir path/to/IMAGE_TEST_DIR
Then, it will generate a csv file(./results/test_res.csv).
>3. **And more details are describe in the README.md in this module 's folder**
* **For detection_module3:**
> 1. **To train:**
>python ./tools/train_net_step_rsna.py \
	--dataset=rsna2018_pos_neg \
	--cfg ./configs/RSNA/e2e_frcnn.yaml \
	--use_tfboard \
    --load_ckpt /path/to/model
>2. **To predict:**
>python ./tools/create_submition.py \
    --dataset rsna2018 \
    --submit_threshold 0.4\
    --cfg ./configs/RSNA/e2e_frcnn.yaml \
    --load_ckpt /path/to/model \
    --image_dir path/to/IMAGE_TEST_DIR
    Then, it will generate a csv file(./Submission/submission.csv).
>3. **And more details are describe in the README.md in this module 's folder**
* **For emsemble_module:**
> 1. **to filter false positive samples**
> python filter.py \
	-predsfile=solution/result_1.csv \
	-classfile=solution/test3000_v1p1.csv \
	-threshold=0.55
>2. **to ensemble results**
>python fusion.py 
>-model_num=11  
>-iou_threshold=0.3
>-vote_threshold=7
> 3. **the details are describe in the README.md in this module 's folder**



  

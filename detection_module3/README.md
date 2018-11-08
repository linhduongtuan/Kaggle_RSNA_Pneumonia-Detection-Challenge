
###########  prepare data  ###############

## prepare data

On a single machine:
```
cd data_prepare
python prepare_data.py

ln -s /path/to/IMAGE images
```

It should have the following directory structure:
```
data_prepare/
  prepare_data.py
  csv2json_bg.py
  csv2json_fg.py
  stage_2_train_labels.csv
  annotations/
    train_neg.json
    train_pos.json
    val_neg.json
    val_pos.json
  images/
    1.png
	2.png
	……
	……
```


###########  pytorch.FasterRCNN  ###################

## Installation

For environment requirements, data preparation and compilation, please refer to [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).

+ Download pre-trained [ImageNet model](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl)
  to path /pytorch.FasterRCNN/data/pretrained_model/

  
## to prepare data

edit /pytorch.FasterRCNN/lib/datasets/dataset_catalog.py
from line 221 to 244:
```
    'rsna2018_train_pos': {
        IM_DIR:
            'path/to/data_prepare/images/',
        ANN_FN:
            'path/to/data_prepare/annotations/train_pos.json',
    },
    'rsna2018_train_neg': {
        IM_DIR:
            'path/to/data_prepare/images/',
        ANN_FN:
            'path/to/data_prepare/annotations/train_neg.json',
    },
    'rsna2018_val_pos': {
        IM_DIR:
            'path/to/data_prepare/images/',
        ANN_FN:
            'path/to/data_prepare/annotations/val_pos.json',
    },
    'rsna2018_val_neg': {
        IM_DIR:
            'path/to/data_prepare/images/',
        ANN_FN:
            'path/to/data_prepare/annotations/val_neg.json',
    },
```

change the path to data_prepare in your machine


## to train

On a single machine:
```
python ./tools/train_net_step_rsna.py \
	--dataset=rsna2018_pos_neg \
	--cfg ./configs/RSNA/e2e_frcnn.yaml \
	--use_tfboard \
    --load_ckpt /path/to/model
```


## to predict

On a single machine:
```
python ./tools/create_submition.py \
    --dataset rsna2018 \
    --submit_threshold 0.4\
    --cfg ./configs/RSNA/e2e_frcnn.yaml \
    --load_ckpt /path/to/model \
    --image_dir path/to/IMAGE_TEST_DIR

```
Then, it will generate a csv file(./Submission/submission.csv).

##################  END  ###################










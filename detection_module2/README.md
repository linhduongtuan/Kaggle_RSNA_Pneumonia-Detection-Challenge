
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

##################  END  ###################





###########  tensorpack.FasterRCNN  ###################

## Dependencies
+ Python 3; OpenCV.
+ TensorFlow >= 1.6 (1.4 or 1.5 can run but may crash due to a TF bug);
+ pycocotools: `pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
+ Pre-trained [ImageNet ResNet model](http://models.tensorpack.com/FasterRCNN/) from tensorpack model zoo.
  to path /tensorpack.FasterRCNN/imageNet_model/

  
## to train

On a single machine:
```
python train.py --config \
    DATA.BASEDIR=/path/to/data_prepare/ \
	DATA.IMGDIR=images \
    --load imageNet_model/ImageNet-ResNet101.npz \
```

Options can be changed by either the command line or the `config.py` file.


## to predict

To predict on image dir:
```
python train.py --config TEST.RESULT_SCORE_THRESH=0.5 TEST.FRCNN_NMS_THRESH=0.1 \
	--load /path/to/model.index --predictDir path/to/IMAGE_TEST_DIR
```
Then, it will generate a csv file(./results/test_res.csv).


## Notes

[NOTES.md](NOTES.md) has some notes about implementation details & speed.

##################  END  ###################














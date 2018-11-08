## prepare data
1. put images into the folder images
2. ./train_csv/p1/train/train_label.csv is training csv file
3. ./train_csv/p1/valid/val_label.csv is validation csv file
4. ./test_csv/stage2/test_3000.csv is test csv file
It should have the following directory structure:
```
  train_csv/
    p1/
      train/
      val/
      
  test_csv/
  	stage2/

  images/
    1.png
	2.png
	……
	……
```	


## to train

Change the directory in train.py to corresponding folder name for training and run train.py.
The result model will be produced in ./result/classification_model/

## to predict

Change the directory in predict.py to corresponding folder name for testing and run test.py.
Prediction result will locate in ./result/test_result/



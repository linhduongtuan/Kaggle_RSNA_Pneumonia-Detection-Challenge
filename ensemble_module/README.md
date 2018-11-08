
###########  ensemble  ###############

## prepare inputs
Fetch `result.csv` from the directory of each single detection model, and put them in the `solution` folder. Remember to rename these csv files as `result_x` (x is a number).


Fetch `result.csv` from the directory of  single classification model, and put it in the `solution` folder. Remember to rename csv file as `cls_results.csv`


It should have the following directory structure:

```
solution/
  result_0.csv
  result_1.csv
  result_2.csv
  ……
  ……
  cls_results.csv
```

Currently, there are 12 csv files in the `solution` folder. We ensemble these 12 csv files as our final submission.



## to filter false positive samples  

On a single machine:

```
python filter.py \
	-predsfile=solution/result_1.csv
	-classfile=solution/cls_results.csv
	-threshold=0.55
```

**Note that the results in the solution folder are already filltered, so we don't need to filter them again



## to ensemble results

On a single machine:

```
python fusion.py \ 
	-model_num=11
	-iou_threshold=0.3
	-vote_threshold=7
```

 Then, it will generate a csv file named `result_merge.csv` which is used for our final submission.
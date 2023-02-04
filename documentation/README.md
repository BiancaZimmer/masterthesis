# Documentation
(as done in the master thesis)

Steps 2.-6. can be done via _preprocessing.py_ if all parameters have been tried and tested before
1. Sort images
2. Cut images into square format
3. Train Model
4. Create Feature Embeddings
5. Create LRP Heatmaps
6. Create Prototypes
7. Generate Near Hits & Near Misses
8. Evaluation

## 1. Sort images
Sorted the images into the appropriate folders. 
Data of each patient was either put into train, test OR validation split. </br>
Parameters used in the makeimagesplits() function : `splitbypatient=True, maxiter=10, deviation=0.05)`

Command used: </br>
`python sort_ttv.py ./data_kermany/all ./data_kermany_split 0.03 -s 0.1`

This was used since in Original Dataset 250 images were used per class for testing ->
that is at most 3% of the original size </br>
For our training we thus use 0.03 testdata and 0.1 validation data

Which resulted in the following output:
```
Cycle  0  failed. Proportion of  0  Trying again ...
Cycle  1  failed. Proportion of  0.01716576532945893  Trying again ...
Cycle  2  failed. Proportion of  0.02712083299701348  Trying again ...
Winning proportion for  CNV :  0.03126429359377943
Cycle  0  failed. Proportion of  0  Trying again ...
Cycle  1  failed. Proportion of  0.023901243214848537  Trying again ...
Cycle  2  failed. Proportion of  0.05235510418490632  Trying again ...
Winning proportion for  DME :  0.03090527053055507
Cycle  0  failed. Proportion of  0  Trying again ...
Cycle  1  failed. Proportion of  0.041647331786542924  Trying again ...
Cycle  2  failed. Proportion of  0.01786542923433875  Trying again ...
Cycle  3  failed. Proportion of  0.020069605568445475  Trying again ...
Cycle  4  failed. Proportion of  0.03306264501160093  Trying again ...
Cycle  5  failed. Proportion of  0.022273781902552203  Trying again ...
Cycle  6  failed. Proportion of  0.02679814385150812  Trying again ...
Winning proportion for  DRUSEN :  0.030394431554524363
Cycle  0  failed. Proportion of  0  Trying again ...
Cycle  1  failed. Proportion of  0.03298288230159031  Trying again ...
Cycle  2  failed. Proportion of  0.031768322769195736  Trying again ...
Winning proportion for  NORMAL :  0.0285041940258853
Total number of all images: 83556
['CNV', 'DME', 'DRUSEN', 'NORMAL']
[37167, 11422, 8620, 26347]
['Split1', 'Split2']
81028,2528
Copying train files ...
Copying test files ...
Cycle  0  failed. Proportion of  0  Trying again ...
Cycle  1  failed. Proportion of  0.1072906540758228  Trying again ...
Winning proportion for  CNV :  0.10140258297458686
Cycle  0  failed. Proportion of  0  Trying again ...
Cycle  1  failed. Proportion of  0.09440780558316017  Trying again ...
Cycle  2  failed. Proportion of  0.07408076610353238  Trying again ...
Cycle  3  failed. Proportion of  0.0751648748757792  Trying again ...
Winning proportion for  DME :  0.10145451260276447
Cycle  0  failed. Proportion of  0  Trying again ...
Winning proportion for  DRUSEN :  0.10134003350083752
Cycle  0  failed. Proportion of  0  Trying again ...
Winning proportion for  NORMAL :  0.09649945303953743
Total number of all images: 81028
['CNV', 'DME', 'DRUSEN', 'NORMAL']
[36005, 11069, 8358, 25596]
['Split1', 'Split2']
72937,8091
Moving validation files ...
Pictures to move:  83556
Train pictures:  72937
Test pictures:  2528
Validation pictures:  8091
0h 29.0min 53.89623975753784sec
```

## 2. Cut images into square format

## 3. Train Model

## 4. Create Feature Embeddings

## 5. Create LRP Heatmaps

## 6. Create Prototypes

## 7. Generate Near Hits & Near Misses

## 8. Evaluation
Please see my master thesis for the evaluation


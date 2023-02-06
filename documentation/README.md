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
No logfiles available

## 3. Train Model
### MNIST
```
(venv) 130 zimmer@hydra:~/masterthesis/code_mh_main$ CUDA_VISIBLE_DEVICES=1 python3 preprocessing.py 
Which data set would you like to choose? Type 'help' if you need more information. mnist_1247
Do you want to fit (f) a model or load (l) an exciting one? [f/l] f
What kind of model do you want to train? [cnn/vgg/inception] cnn
What should the suffix of your model be? Type a string. e.g. _testcnn _cnn_seed3871
Do you want to correct the training for imbalanced data set? [y/n] y
Do you want to run the evaluation of your model? [y/n] y
Do you want to plot the loss and accuracy of your model? [y/n] y
Do you want to plot the evaluation of the miss-classified data of your model? [y/n] y
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Preprocessing images ...
Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
----- TRAINING OF MODEL -----
Clear Session & Setup Model ...
WARNING:tensorflow:From /home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/layers/normalization/batch_normalization.py:562: _colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 42, 42, 8)        32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 21, 21, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Fitting model ...
Class weights for imbalanced data:  {0: 1.0409828422230512, 1: 1.0615728413845569, 2: 0.9898474906898386, 3: 0.9198665128543178}
2022-10-26 16:29:07.266576: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-10-26 16:29:07.267014: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-10-26 16:29:07.267348: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-10-26 16:29:07.267728: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-10-26 16:29:07.268067: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-10-26 16:29:07.268354: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9636 MB memory:  -> device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:41:00.0, compute capability: 7.5
Epoch 1/50
2022-10-26 16:29:08.957109: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500
698/698 [==============================] - ETA: 0s - batch: 348.5000 - size: 31.9871 - loss: 0.1089 - accuracy: 0.9656/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2332: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates = self.state_updates

Epoch 1: val_loss improved from inf to 0.03554, saving model to /home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5
698/698 [==============================] - 14s 16ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.1089 - accuracy: 0.9656 - val_loss: 0.0355 - val_accuracy: 0.9903
Epoch 2/50
697/698 [============================>.] - ETA: 0s - batch: 348.0000 - size: 31.9871 - loss: 0.0354 - accuracy: 0.9906    
Epoch 2: val_loss improved from 0.03554 to 0.02957, saving model to /home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5
698/698 [==============================] - 12s 15ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0355 - accuracy: 0.9906 - val_loss: 0.0296 - val_accuracy: 0.9927
Epoch 3/50
697/698 [============================>.] - ETA: 0s - batch: 348.0000 - size: 31.9871 - loss: 0.0256 - accuracy: 0.9925     
Epoch 3: val_loss improved from 0.02957 to 0.01415, saving model to /home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5
698/698 [==============================] - 11s 15ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0256 - accuracy: 0.9925 - val_loss: 0.0141 - val_accuracy: 0.9948
Epoch 4/50
697/698 [============================>.] - ETA: 0s - batch: 348.0000 - size: 31.9871 - loss: 0.0196 - accuracy: 0.9947     
Epoch 4: val_loss did not improve from 0.01415
698/698 [==============================] - 11s 15ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0197 - accuracy: 0.9947 - val_loss: 0.0312 - val_accuracy: 0.9887
Epoch 5/50
698/698 [==============================] - ETA: 0s - batch: 348.5000 - size: 31.9871 - loss: 0.0180 - accuracy: 0.9955     
Epoch 5: val_loss improved from 0.01415 to 0.01034, saving model to /home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5
698/698 [==============================] - 12s 15ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0180 - accuracy: 0.9955 - val_loss: 0.0103 - val_accuracy: 0.9960
Epoch 6/50
698/698 [==============================] - ETA: 0s - batch: 348.5000 - size: 31.9871 - loss: 0.0147 - accuracy: 0.9953     
Epoch 6: val_loss did not improve from 0.01034
698/698 [==============================] - 12s 15ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0147 - accuracy: 0.9953 - val_loss: 0.0195 - val_accuracy: 0.9931
Epoch 7/50
697/698 [============================>.] - ETA: 0s - batch: 348.0000 - size: 31.9871 - loss: 0.0100 - accuracy: 0.9972     
Epoch 7: val_loss did not improve from 0.01034
698/698 [==============================] - 12s 15ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0100 - accuracy: 0.9972 - val_loss: 0.0467 - val_accuracy: 0.9847
Epoch 8/50
697/698 [============================>.] - ETA: 0s - batch: 348.0000 - size: 31.9871 - loss: 0.0115 - accuracy: 0.9965     
Epoch 8: val_loss did not improve from 0.01034
698/698 [==============================] - 12s 15ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0115 - accuracy: 0.9965 - val_loss: 0.0187 - val_accuracy: 0.9952
Epoch 9/50
697/698 [============================>.] - ETA: 0s - batch: 348.0000 - size: 31.9871 - loss: 0.0090 - accuracy: 0.9973         
Epoch 9: val_loss did not improve from 0.01034
698/698 [==============================] - 12s 16ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0090 - accuracy: 0.9973 - val_loss: 0.0263 - val_accuracy: 0.9911
Epoch 10/50
697/698 [============================>.] - ETA: 0s - batch: 348.0000 - size: 31.9871 - loss: 0.0105 - accuracy: 0.9974     
Epoch 10: val_loss did not improve from 0.01034
698/698 [==============================] - 12s 15ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0105 - accuracy: 0.9974 - val_loss: 0.0140 - val_accuracy: 0.9956
Epoch 11/50
697/698 [============================>.] - ETA: 0s - batch: 348.0000 - size: 31.9871 - loss: 0.0066 - accuracy: 0.9981     
Epoch 11: val_loss did not improve from 0.01034
698/698 [==============================] - 13s 16ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0066 - accuracy: 0.9981 - val_loss: 0.0108 - val_accuracy: 0.9964
Epoch 12/50
696/698 [============================>.] - ETA: 0s - batch: 347.5000 - size: 31.9871 - loss: 0.0068 - accuracy: 0.9978     
Epoch 12: val_loss did not improve from 0.01034
698/698 [==============================] - 13s 16ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0068 - accuracy: 0.9979 - val_loss: 0.0114 - val_accuracy: 0.9968
Epoch 13/50
694/698 [============================>.] - ETA: 0s - batch: 346.5000 - size: 31.9870 - loss: 0.0043 - accuracy: 0.9985         
Epoch 13: val_loss did not improve from 0.01034
698/698 [==============================] - 12s 15ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0043 - accuracy: 0.9985 - val_loss: 0.0160 - val_accuracy: 0.9968
Epoch 14/50
697/698 [============================>.] - ETA: 0s - batch: 348.0000 - size: 31.9871 - loss: 0.0059 - accuracy: 0.9983         
Epoch 14: val_loss did not improve from 0.01034
698/698 [==============================] - 12s 16ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0059 - accuracy: 0.9983 - val_loss: 0.0141 - val_accuracy: 0.9956
Epoch 15/50
697/698 [============================>.] - ETA: 0s - batch: 348.0000 - size: 31.9871 - loss: 0.0053 - accuracy: 0.9984         
Epoch 15: val_loss did not improve from 0.01034
698/698 [==============================] - 13s 17ms/step - batch: 348.5000 - size: 31.9871 - loss: 0.0053 - accuracy: 0.9984 - val_loss: 0.0152 - val_accuracy: 0.9952
Training needed:  0h 3min 1.92899751663208sec 
Evaluating model ...
        loss  accuracy  val_loss  val_accuracy
1   0.108860  0.965602  0.035536      0.990323
2   0.035477  0.990594  0.029571      0.992742
3   0.025627  0.992475  0.014149      0.994758
4   0.019659  0.994670  0.031170      0.988710
5   0.017968  0.995476  0.010339      0.995968
6   0.014754  0.995297  0.019526      0.993145
7   0.009975  0.997178  0.046713      0.984677
8   0.011480  0.996462  0.018657      0.995161
9   0.008995  0.997313  0.026266      0.991129
10  0.010484  0.997402  0.013981      0.995565
11  0.006635  0.998074  0.010763      0.996371
12  0.006744  0.997850  0.011425      0.996774
13  0.004291  0.998477  0.016006      0.996774
14  0.005873  0.998343  0.014112      0.995565
15  0.005256  0.998388  0.015215      0.995161
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates=self.state_updates,
              precision    recall  f1-score   support

           1      0.999     0.998     0.999      1135
           2      0.994     0.994     0.994      1032
           4      0.999     0.999     0.999       982
           7      0.994     0.995     0.995      1028

    accuracy                          0.997      4177
   macro avg      0.997     0.997     0.997      4177
weighted avg      0.997     0.997     0.997      4177

[=>] 14 misclassified images with names: ['2182.jpg', '2343.jpg', '2488.jpg', '321.jpg', '3511.jpg', '659.jpg', '924.jpg', '9839.jpg', '247.jpg', '1903.jpg', '3328.jpg', '3767.jpg', '6576.jpg', '9015.jpg']
```
### OCT
```
(venv) zimmer@hydra:~/masterthesis/code_mh_main$ CUDA_VISIBLE_DEVICES=1 python3 preprocessing.py 
Which data set would you like to choose? Type 'help' if you need more information. oct_cc
Do you want to fit (f) a model or load (l) an exciting one? [f/l] f
What kind of model do you want to train? [cnn/vgg/inception] cnn
What should the suffix of your model be? Type a string. e.g. _testcnn _cnn_seed3871
Do you want to correct the training for imbalanced data set? [y/n] y
Do you want to run the evaluation of your model? [y/n] y
Do you want to plot the loss and accuracy of your model? [y/n] y
Do you want to plot the evaluation of the miss-classified data of your model? [y/n] y
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Preprocessing images ...
Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
----- TRAINING OF MODEL -----
Clear Session & Setup Model ...
WARNING:tensorflow:From /home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/layers/normalization/batch_normalization.py:562: _colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 299, 299, 8)       80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 149, 149, 8)      0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 149, 149, 8)      32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 149, 149, 16)      1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 74, 74, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 74, 74, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 74, 74, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 37, 37, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 37, 37, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 37, 37, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 18, 18, 64)       0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 18, 18, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 18, 18, 128)       73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 9, 9, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 10368)             0         
                                                                 
 dense (Dense)               (None, 256)               2654464   
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 2,787,108
Trainable params: 2,786,612
Non-trainable params: 496
_________________________________________________________________
None
Fitting model ...
Class weights for imbalanced data:  {0: 2.427672746638264, 1: 1.8333249547556807, 2: 0.7884740119346191, 3: 0.5635856462879396}
2022-10-26 18:45:33.811799: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-10-26 18:45:33.812233: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-10-26 18:45:33.812567: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-10-26 18:45:33.812950: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-10-26 18:45:33.813296: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-10-26 18:45:33.813584: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9636 MB memory:  -> device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:41:00.0, compute capability: 7.5
Epoch 1/50
2022-10-26 18:45:35.607005: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.7031 - accuracy: 0.7538/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2332: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates = self.state_updates

Epoch 1: val_loss improved from inf to 0.47025, saving model to /home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5
2280/2280 [==============================] - 336s 146ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.7031 - accuracy: 0.7538 - val_loss: 0.4702 - val_accuracy: 0.8291
Epoch 2/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.3971 - accuracy: 0.8674    
Epoch 2: val_loss did not improve from 0.47025
2280/2280 [==============================] - 317s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.3971 - accuracy: 0.8674 - val_loss: 0.5720 - val_accuracy: 0.8192
Epoch 3/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.3164 - accuracy: 0.8979    
Epoch 3: val_loss improved from 0.47025 to 0.34487, saving model to /home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5
2280/2280 [==============================] - 315s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.3164 - accuracy: 0.8979 - val_loss: 0.3449 - val_accuracy: 0.8859
Epoch 4/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.2783 - accuracy: 0.9121    
Epoch 4: val_loss improved from 0.34487 to 0.28781, saving model to /home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5
2280/2280 [==============================] - 318s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.2783 - accuracy: 0.9121 - val_loss: 0.2878 - val_accuracy: 0.9037
Epoch 5/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.2589 - accuracy: 0.9192    
Epoch 5: val_loss did not improve from 0.28781
2280/2280 [==============================] - 319s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.2589 - accuracy: 0.9192 - val_loss: 0.3643 - val_accuracy: 0.8784
Epoch 6/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.2444 - accuracy: 0.9242    
Epoch 6: val_loss did not improve from 0.28781
2280/2280 [==============================] - 317s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.2444 - accuracy: 0.9242 - val_loss: 0.3269 - val_accuracy: 0.8973
Epoch 7/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.2253 - accuracy: 0.9291    
Epoch 7: val_loss improved from 0.28781 to 0.26370, saving model to /home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5
2280/2280 [==============================] - 317s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.2253 - accuracy: 0.9291 - val_loss: 0.2637 - val_accuracy: 0.8962
Epoch 8/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.2189 - accuracy: 0.9317    
Epoch 8: val_loss did not improve from 0.26370
2280/2280 [==============================] - 316s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.2189 - accuracy: 0.9317 - val_loss: 0.2845 - val_accuracy: 0.9006
Epoch 9/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.2110 - accuracy: 0.9340    
Epoch 9: val_loss did not improve from 0.26370
2280/2280 [==============================] - 316s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.2110 - accuracy: 0.9340 - val_loss: 0.3032 - val_accuracy: 0.9026
Epoch 10/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.2039 - accuracy: 0.9367    
Epoch 10: val_loss did not improve from 0.26370
2280/2280 [==============================] - 318s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.2039 - accuracy: 0.9367 - val_loss: 0.3305 - val_accuracy: 0.8867
Epoch 11/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1969 - accuracy: 0.9385    
Epoch 11: val_loss did not improve from 0.26370
2280/2280 [==============================] - 316s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1969 - accuracy: 0.9385 - val_loss: 0.2783 - val_accuracy: 0.9001
Epoch 12/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1945 - accuracy: 0.9398    
Epoch 12: val_loss did not improve from 0.26370
2280/2280 [==============================] - 315s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1945 - accuracy: 0.9398 - val_loss: 0.2807 - val_accuracy: 0.9072
Epoch 13/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1886 - accuracy: 0.9406    
Epoch 13: val_loss did not improve from 0.26370
2280/2280 [==============================] - 317s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1886 - accuracy: 0.9406 - val_loss: 0.3093 - val_accuracy: 0.8936
Epoch 14/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1882 - accuracy: 0.9408    
Epoch 14: val_loss did not improve from 0.26370
2280/2280 [==============================] - 318s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1882 - accuracy: 0.9408 - val_loss: 0.2708 - val_accuracy: 0.9061
Epoch 15/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1830 - accuracy: 0.9434      
Epoch 15: val_loss did not improve from 0.26370
2280/2280 [==============================] - 319s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1830 - accuracy: 0.9434 - val_loss: 0.3490 - val_accuracy: 0.8853
Epoch 16/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1778 - accuracy: 0.9443      
Epoch 16: val_loss did not improve from 0.26370
2280/2280 [==============================] - 318s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1778 - accuracy: 0.9443 - val_loss: 0.2974 - val_accuracy: 0.9026
Epoch 17/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1747 - accuracy: 0.9438      
Epoch 17: val_loss improved from 0.26370 to 0.25624, saving model to /home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5
2280/2280 [==============================] - 318s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1747 - accuracy: 0.9438 - val_loss: 0.2562 - val_accuracy: 0.9048
Epoch 18/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1721 - accuracy: 0.9463      
Epoch 18: val_loss did not improve from 0.25624
2280/2280 [==============================] - 317s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1721 - accuracy: 0.9463 - val_loss: 0.3086 - val_accuracy: 0.8893
Epoch 19/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1666 - accuracy: 0.9465      
Epoch 19: val_loss did not improve from 0.25624
2280/2280 [==============================] - 316s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1666 - accuracy: 0.9465 - val_loss: 0.2684 - val_accuracy: 0.9029
Epoch 20/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1662 - accuracy: 0.9483      
Epoch 20: val_loss did not improve from 0.25624
2280/2280 [==============================] - 317s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1662 - accuracy: 0.9483 - val_loss: 0.2897 - val_accuracy: 0.9025
Epoch 21/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1635 - accuracy: 0.9488      
Epoch 21: val_loss did not improve from 0.25624
2280/2280 [==============================] - 318s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1635 - accuracy: 0.9488 - val_loss: 0.3409 - val_accuracy: 0.8884
Epoch 22/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1616 - accuracy: 0.9503      
Epoch 22: val_loss did not improve from 0.25624
2280/2280 [==============================] - 318s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1616 - accuracy: 0.9503 - val_loss: 0.2901 - val_accuracy: 0.8972
Epoch 23/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1579 - accuracy: 0.9505      
Epoch 23: val_loss did not improve from 0.25624
2280/2280 [==============================] - 318s 139ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1579 - accuracy: 0.9505 - val_loss: 0.3485 - val_accuracy: 0.9008
Epoch 24/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1557 - accuracy: 0.9497      
Epoch 24: val_loss did not improve from 0.25624
2280/2280 [==============================] - 318s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1557 - accuracy: 0.9497 - val_loss: 0.2899 - val_accuracy: 0.9057
Epoch 25/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1551 - accuracy: 0.9508      
Epoch 25: val_loss did not improve from 0.25624
2280/2280 [==============================] - 317s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1551 - accuracy: 0.9508 - val_loss: 0.2645 - val_accuracy: 0.9089
Epoch 26/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1502 - accuracy: 0.9521      
Epoch 26: val_loss did not improve from 0.25624
2280/2280 [==============================] - 318s 138ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1502 - accuracy: 0.9521 - val_loss: 0.3130 - val_accuracy: 0.8859
Epoch 27/50
2280/2280 [==============================] - ETA: 0s - batch: 1139.5000 - size: 31.9899 - loss: 0.1553 - accuracy: 0.9519      
Epoch 27: val_loss did not improve from 0.25624
2280/2280 [==============================] - 321s 140ms/step - batch: 1139.5000 - size: 31.9899 - loss: 0.1553 - accuracy: 0.9519 - val_loss: 0.2953 - val_accuracy: 0.8991
Training needed:  2h 23min 7.302985668182373sec 
Evaluating model ...
        loss  accuracy  val_loss  val_accuracy
1   0.703022  0.753815  0.470247      0.829069
2   0.397170  0.867365  0.571975      0.819182
3   0.316374  0.897898  0.344871      0.885923
4   0.278373  0.912075  0.287807      0.903720
5   0.259009  0.919191  0.364277      0.878383
6   0.244252  0.924236  0.326886      0.897293
7   0.225279  0.929089  0.263700      0.896181
8   0.218815  0.931681  0.284541      0.900630
9   0.211055  0.933970  0.303166      0.902608
10  0.203774  0.936685  0.330475      0.886664
11  0.196944  0.938467  0.278319      0.900136
12  0.194513  0.939838  0.280745      0.907181
13  0.188670  0.940593  0.309321      0.893585
14  0.188153  0.940798  0.270820      0.906068
15  0.182906  0.943444  0.348967      0.885305
16  0.177836  0.944267  0.297449      0.902608
17  0.174657  0.943815  0.256237      0.904833
18  0.172152  0.946255  0.308606      0.889260
19  0.166630  0.946461  0.268417      0.902855
20  0.166277  0.948298  0.289717      0.902484
21  0.163507  0.948764  0.340855      0.888395
22  0.161612  0.950341  0.290140      0.897170
23  0.157963  0.950533  0.348455      0.900754
24  0.155733  0.949669  0.289940      0.905698
25  0.154917  0.950807  0.264490      0.908911
26  0.150205  0.952096  0.313043      0.885923
27  0.154880  0.951863  0.295319      0.899147
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates=self.state_updates,
              precision    recall  f1-score   support

         CNV      0.955     0.987     0.971      1162
         DME      0.968     0.870     0.916       353
      DRUSEN      0.965     0.744     0.841       262
      NORMAL      0.918     0.988     0.952       751

    accuracy                          0.946      2528
   macro avg      0.952     0.897     0.920      2528
weighted avg      0.947     0.946     0.944      2528

[=>] 137 misclassified images with names: ['CNV-4585256-4.jpeg', 'CNV-4967228-1.jpeg', 'CNV-4967228-2.jpeg', 'CNV-4967228-3.jpeg', 'CNV-4967228-4.jpeg', 'CNV-4967228-5.jpeg', 'CNV-4967228-7.jpeg', 'CNV-4967228-8.jpeg', 'CNV-5851527-16.jpeg', 'CNV-5851527-43.jpeg', 'CNV-620790-3.jpeg', 'CNV-6897828-17.jpeg', 'CNV-6897828-22.jpeg', 'CNV-6897828-3.jpeg', 'CNV-773983-2.jpeg', 'DME-4244491-16.jpeg', 'DME-4283050-1.jpeg', 'DME-5103899-1.jpeg', 'DME-5465575-32.jpeg', 'DME-5580182-7.jpeg', 'DME-5975636-2.jpeg', 'DME-6314049-2.jpeg', 'DME-7434518-12.jpeg', 'DME-7434518-13.jpeg', 'DME-7434518-4.jpeg', 'DME-7434518-7.jpeg', 'DME-7434518-8.jpeg', 'DME-7635774-1.jpeg', 'DME-7635774-2.jpeg', 'DME-7635774-3.jpeg', 'DME-7635774-4.jpeg', 'DME-7635774-5.jpeg', 'DME-8177380-49.jpeg', 'DME-8177380-9.jpeg', 'DME-8487067-1.jpeg', 'DME-8487067-2.jpeg', 'DME-8487067-3.jpeg', 'DME-8487067-4.jpeg', 'DME-8732456-10.jpeg', 'DME-8732456-11.jpeg', 'DME-8732456-12.jpeg', 'DME-8732456-13.jpeg', 'DME-8732456-14.jpeg', 'DME-8732456-15.jpeg', 'DME-8732456-16.jpeg', 'DME-8732456-21.jpeg', 'DME-8732456-23.jpeg', 'DME-8732456-24.jpeg', 'DME-8732456-25.jpeg', 'DME-8732456-26.jpeg', 'DME-8732456-27.jpeg', 'DME-8732456-28.jpeg', 'DME-8732456-29.jpeg', 'DME-8732456-30.jpeg', 'DME-8732456-32.jpeg', 'DME-8732456-39.jpeg', 'DME-8732456-5.jpeg', 'DME-8732456-6.jpeg', 'DME-8732456-7.jpeg', 'DME-8732456-8.jpeg', 'DME-8732456-9.jpeg', 'DRUSEN-1020679-2.jpeg', 'DRUSEN-1020679-3.jpeg', 'DRUSEN-1020679-5.jpeg', 'DRUSEN-1295395-1.jpeg', 'DRUSEN-1295395-2.jpeg', 'DRUSEN-1352319-1.jpeg', 'DRUSEN-1380233-7.jpeg', 'DRUSEN-2120559-1.jpeg', 'DRUSEN-2120559-2.jpeg', 'DRUSEN-2120559-3.jpeg', 'DRUSEN-2120559-4.jpeg', 'DRUSEN-2126186-1.jpeg', 'DRUSEN-228939-28.jpeg', 'DRUSEN-3300060-1.jpeg', 'DRUSEN-3342858-1.jpeg', 'DRUSEN-3342858-2.jpeg', 'DRUSEN-3342858-3.jpeg', 'DRUSEN-3342858-4.jpeg', 'DRUSEN-3342858-5.jpeg', 'DRUSEN-3342858-6.jpeg', 'DRUSEN-3342858-7.jpeg', 'DRUSEN-3424668-17.jpeg', 'DRUSEN-3424668-31.jpeg', 'DRUSEN-3424668-32.jpeg', 'DRUSEN-3424668-37.jpeg', 'DRUSEN-3424668-38.jpeg', 'DRUSEN-3424668-4.jpeg', 'DRUSEN-3424668-47.jpeg', 'DRUSEN-3424668-59.jpeg', 'DRUSEN-3424668-60.jpeg', 'DRUSEN-3424668-61.jpeg', 'DRUSEN-3424668-69.jpeg', 'DRUSEN-3424668-70.jpeg', 'DRUSEN-3424668-76.jpeg', 'DRUSEN-3833074-1.jpeg', 'DRUSEN-3833074-2.jpeg', 'DRUSEN-4951152-7.jpeg', 'DRUSEN-5392647-1.jpeg', 'DRUSEN-5983793-2.jpeg', 'DRUSEN-5983793-6.jpeg', 'DRUSEN-601865-2.jpeg', 'DRUSEN-6256161-1.jpeg', 'DRUSEN-6256161-11.jpeg', 'DRUSEN-6256161-16.jpeg', 'DRUSEN-6256161-2.jpeg', 'DRUSEN-6256161-4.jpeg', 'DRUSEN-6256161-6.jpeg', 'DRUSEN-7880345-1.jpeg', 'DRUSEN-7990001-10.jpeg', 'DRUSEN-7990001-11.jpeg', 'DRUSEN-7990001-12.jpeg', 'DRUSEN-7990001-13.jpeg', 'DRUSEN-7990001-14.jpeg', 'DRUSEN-7990001-15.jpeg', 'DRUSEN-7990001-17.jpeg', 'DRUSEN-7990001-18.jpeg', 'DRUSEN-7990001-2.jpeg', 'DRUSEN-7990001-3.jpeg', 'DRUSEN-7990001-4.jpeg', 'DRUSEN-7990001-5.jpeg', 'DRUSEN-7990001-7.jpeg', 'DRUSEN-7990001-8.jpeg', 'DRUSEN-7990001-9.jpeg', 'DRUSEN-9138933-12.jpeg', 'DRUSEN-9138933-5.jpeg', 'DRUSEN-9138933-9.jpeg', 'DRUSEN-9609006-1.jpeg', 'NORMAL-1663564-2.jpeg', 'NORMAL-1699976-11.jpeg', 'NORMAL-1699976-13.jpeg', 'NORMAL-1699976-3.jpeg', 'NORMAL-1699976-33.jpeg', 'NORMAL-1699976-41.jpeg', 'NORMAL-4412751-8.jpeg', 'NORMAL-4982430-6.jpeg', 'NORMAL-568518-23.jpeg']
```

## 4. Create Feature Embeddings
### MNIST
```
Do you want to create the feature embeddings for this model? [y/n/help] y
----- CREATION OF FEATURE EMBEDDINGS -----
FeatureModel input shape:  (None, 84, 84, 1)
Initiating  MultiCNN
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates=self.state_updates,
0  feature embeddings created
1000  feature embeddings created
…
27000  feature embeddings created
28000  feature embeddings created

Creating Feature Embeddings needed:  0h 2min 7.237229585647583sec 
Do you want to create the feature embeddings for the general VGG16? [y/n]y
FeatureModel input shape:  (None, 224, 224, 3)
Initiating  VGG16
0  feature embeddings created
…
27000  feature embeddings created
28000  feature embeddings created

Creating Feature Embeddings needed:  0h 4min 5.310938596725464sec 
Do you want to create the feature embeddings for the general VGG16? [y/n]n
```

### OCT
```
Do you want to create the feature embeddings for this model? [y/n/help] y
----- CREATION OF FEATURE EMBEDDINGS -----
FeatureModel input shape:  (None, 299, 299, 1)
Initiating  MultiCNN
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates=self.state_updates,
0  feature embeddings created
…
82000  feature embeddings created
83000  feature embeddings created

Creating Feature Embeddings needed:  0h 11min 16.264573574066162sec 
Do you want to create the feature embeddings for the general VGG16? [y/n]y
FeatureModel input shape:  (None, 224, 224, 3)
Initiating  VGG16
…
82000  feature embeddings created
83000  feature embeddings created

Creating Feature Embeddings needed:  0h 18min 34.67244052886963sec 
Do you want to create the feature embeddings for the general VGG16? [y/n]n
```

## 5. Create LRP Heatmaps
### MNIST
```
Do you want to create LRP heatmaps for your current data set and trained model now? [y/n] y
Which method would you ike to use? We propose: 
lrp.sequential_preset_a for the mnist data
lrp.sequential_preset_a_flat for the oct data
lrp.sequential_preset_a
Which epsilon value would you like to use? We propose 0.1 0.1
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...
FeatureModel input shape:  (None, 84, 84, 1)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Preprocessing images ...
Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 42, 42, 8)        32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 21, 21, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 84, 84, 1)
Preparing data ...
Preparing analyzer ...
Heatmaps will be saved to:
 /home/zimmer/masterthesis/code_mh_main/static/heatmaps/MultiCNN/mnist_1247
0 * 4  LRP heatmaps created
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/innvestigate/backend/graph.py:452: UserWarning: Ignore dtype <dtype: 'float32'> as bias type.
  warnings.warn(f"Ignore dtype {dtype} as bias type.")
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/innvestigate/backend/graph.py:465: UserWarning: Ignore dtype <dtype: 'float32'> as bias type.
  warnings.warn(f"Ignore dtype {dtype} as bias type.")
…
22000 * 4  LRP heatmaps created
Heatmaps needed:  0h 27min 46.170135736465454sec 
Do you want to create LRP heatmaps for a general (untrained) VGG16? [y/n] y
Which method would you ike to use? We propose: 
lrp.sequential_preset_a for the mnist data
lrp.sequential_preset_a_flat for the oct data
lrp.sequential_preset_a
Which epsilon value would you like to use? We propose 0.1 0.1
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Preprocessing images ...
Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 42, 42, 8)        32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 21, 21, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 84, 84, 1)
Preparing data ...
Preparing analyzer ...
Heatmaps will be saved to:
 /home/zimmer/masterthesis/code_mh_main/static/heatmaps/VGG16/mnist_1247
0 * 4  LRP heatmaps created
…
22000 * 4  LRP heatmaps created
Heatmaps needed:  0h 56min 1.9578850269317627sec 

########## LRP for Test data #######

Heatmaps will be saved to:
 /home/zimmer/masterthesis/code_mh_main/static/heatmaps/MultiCNN/mnist_1247/test                                                                          
0 * 4  LRP heatmaps created
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/innvestigate/backend/graph.py:452: UserWarning: Ignore dtype <dtype: 'float32'> as bias type.  
  warnings.warn(f"Ignore dtype {dtype} as bias type.")
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/innvestigate/backend/graph.py:465: UserWarning: Ignore dtype <dtype: 'float32'> as bias type.  
  warnings.warn(f"Ignore dtype {dtype} as bias type.")
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates=self.state_updates,
2022-11-09 17:11:36.170785: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500
1000 * 4  LRP heatmaps created
1000 * 4  LRP heatmaps created
2000 * 4  LRP heatmaps created
3000 * 4  LRP heatmaps created
4000 * 4  LRP heatmaps created
Heatmaps needed:  0h 4min 59.189966678619385sec
Do you want to create LRP heatmaps for a general (untrained) VGG16? [y/n] y
Which method would you ike to use? We propose: 
lrp.sequential_preset_a for the mnist data
lrp.sequential_preset_a_flat for the oct data
lrp.sequential_preset_a
Which epsilon value would you like to use? We propose 0.1 0.1
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Preprocessing images ...
Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 42, 42, 8)        32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 21, 21, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 84, 84, 1)
Preparing data ...
Preparing analyzer ...
Heatmaps will be saved to:
 /home/zimmer/masterthesis/code_mh_main/static/heatmaps/VGG16/mnist_1247/test
0 * 4  LRP heatmaps created
1000 * 4  LRP heatmaps created
2000 * 4  LRP heatmaps created
3000 * 4  LRP heatmaps created
4000 * 4  LRP heatmaps created
Heatmaps needed:  0h 10min 21.293186902999878sec
```
### OCT
```
Do you want to create LRP heatmaps for your current data set and trained model now? [y/n] y
Which method would you ike to use? We propose: 
lrp.sequential_preset_a for the mnist data
lrp.sequential_preset_a_flat for the oct data
lrp.sequential_preset_a_flat
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...
FeatureModel input shape:  (None, 299, 299, 1)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Preprocessing images ...
Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 299, 299, 8)       80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 149, 149, 8)      0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 149, 149, 8)      32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 149, 149, 16)      1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 74, 74, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 74, 74, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 74, 74, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 37, 37, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 37, 37, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 37, 37, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 18, 18, 64)       0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 18, 18, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 18, 18, 128)       73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 9, 9, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 10368)             0         
                                                                 
 dense (Dense)               (None, 256)               2654464   
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 2,787,108
Trainable params: 2,786,612
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 299, 299, 1)
Preparing data ...
71000 * 4  LRP heatmaps created
72000 * 4  LRP heatmaps created
Heatmaps needed:  2h 11min 21.317805767059326sec
Do you want to create LRP heatmaps for a general (untrained) VGG16? [y/n] y
Which method would you ike to use? We propose: 
lrp.sequential_preset_a for the mnist data
lrp.sequential_preset_a_flat for the oct data
lrp.sequential_preset_a_flat
FeatureModel input shape:  (None, 224, 224, 3)
==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Preprocessing images ...
Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 299, 299, 8)       80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 149, 149, 8)      0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 149, 149, 8)      32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 149, 149, 16)      1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 74, 74, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 74, 74, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 74, 74, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 37, 37, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 37, 37, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 37, 37, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 18, 18, 64)       0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 18, 18, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 18, 18, 128)       73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 9, 9, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 10368)             0         
                                                                 
 dense (Dense)               (None, 256)               2654464   
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 2,787,108
Trainable params: 2,786,612
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 299, 299, 1)
Preparing data ...
Preparing analyzer ...
Heatmaps will be saved to:
 /home/zimmer/masterthesis/code_mh_main/static/heatmaps/VGG16/oct_cc
0 * 4  LRP heatmaps created
…
72000 * 4  LRP heatmaps created
Heatmaps needed:  3h 24min 47.23978352546692sec 

###### For Test images #####

Do you want to create LRP heatmaps for your current data set and trained model now? [y/n] y
Which method would you ike to use? We propose: 
lrp.sequential_preset_a for the mnist data
lrp.sequential_preset_a_flat for the oct data
lrp.sequential_preset_a_flat
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...
FeatureModel input shape:  (None, 299, 299, 1)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Preprocessing images ...
Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 299, 299, 8)       80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 149, 149, 8)      0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 149, 149, 8)      32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 149, 149, 16)      1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 74, 74, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 74, 74, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 74, 74, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 37, 37, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 37, 37, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 37, 37, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 18, 18, 64)       0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 18, 18, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 18, 18, 128)       73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 9, 9, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 10368)             0         
                                                                 
 dense (Dense)               (None, 256)               2654464   
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 2,787,108
Trainable params: 2,786,612
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 299, 299, 1)
Preparing data ...
Preparing analyzer ...
Heatmaps will be saved to:
 /home/zimmer/masterthesis/code_mh_main/static/heatmaps/MultiCNN/oct_cc/test
0 * 4  LRP heatmaps created
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/innvestigate/backend/graph.py:452: UserWarning: Ignore dtype <dtype: 'float32'> as bias type.
  warnings.warn(f"Ignore dtype {dtype} as bias type.")
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/innvestigate/backend/graph.py:465: UserWarning: Ignore dtype <dtype: 'float32'> as bias type.
  warnings.warn(f"Ignore dtype {dtype} as bias type.")
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates=self.state_updates,
2022-11-09 16:52:30.513106: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500
1000 * 4  LRP heatmaps created
2000 * 4  LRP heatmaps created
Heatmaps needed:  0h 5min 1.8661301136016846sec
Do you want to create LRP heatmaps for a general (untrained) VGG16? [y/n] y
Which method would you ike to use? We propose:
lrp.sequential_preset_a for the mnist data
lrp.sequential_preset_a_flat for the oct data
lrp.sequential_preset_a_flat
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Preprocessing images ...
Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 299, 299, 8)       80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 149, 149, 8)      0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 149, 149, 8)      32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 149, 149, 16)      1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 74, 74, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 74, 74, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 74, 74, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 37, 37, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 37, 37, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 37, 37, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 18, 18, 64)       0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 18, 18, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 18, 18, 128)       73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 9, 9, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 10368)             0         
                                                                 
 dense (Dense)               (None, 256)               2654464   
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 2,787,108
Trainable params: 2,786,612
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 299, 299, 1)
Preparing data ...
Preparing analyzer ...
Heatmaps will be saved to:
 /home/zimmer/masterthesis/code_mh_main/static/heatmaps/VGG16/oct_cc/test
0 * 4  LRP heatmaps created
1000 * 4  LRP heatmaps created
2000 * 4  LRP heatmaps created
Heatmaps needed:  0h 7min 7.173657417297363sec 
```

## 6. Create Prototypes
```

(venv) zimmer@hydra:~/masterthesis/code_mh_main$ CUDA_VISIBLE_DEVICES=1 python3 prototype_selection.py 
FeatureModel input shape:  (None, 84, 84, 1)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

FeatureModel input shape:  (None, 299, 299, 1)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

... selecting MMD Prototypes
======= Parameters =======
num_prototypes:4
gamma:0.01
use_image_embeddings:False
################ RESULTS for 4 Prototypes and classes dict_keys(['2', '4', '7', '1']) #############                                                                                 
Accuracys for selected prototypes: 0.9104620541058176
Recalls for prototypes: [0.82364341 0.92668024 0.88618677 0.99735683]
F1-score averaged for prototypes: 0.9097674715508369
F1-scores for prototypes per class: [0.89757128 0.91457286 0.8953317  0.92977413]
#############################################################################################                                                                                       
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/rawData/mnist_1247/4_MultiCNN.json                                                                                         
... selecting MMD Prototypes
======= Parameters =======
num_prototypes:4
gamma:1e-06
use_image_embeddings:True
################ RESULTS for 4 Prototypes and classes dict_keys(['2', '4', '7', '1']) ############# 
Accuracys for selected prototypes: 0.7220493176921235
Recalls for prototypes: [0.6744186  0.59470468 0.78793774 0.81585903]
F1-score averaged for prototypes: 0.7212116124629357
F1-scores for prototypes per class: [0.71457906 0.60051414 0.80039526 0.75995076]
############################################################################################# 
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/MultiCNN/mnist_1247/4_MultiCNN.json
... selecting MMD Prototypes
======= Parameters =======
num_prototypes:4
gamma:0.0001417233560090703
use_image_embeddings:False
################ RESULTS for 4 Prototypes and classes dict_keys(['2', '4', '7', '1']) ############# 
Accuracys for selected prototypes: 0.7900406990663156
Recalls for prototypes: [0.62209302 0.75560081 0.76945525 0.99118943]
F1-score averaged for prototypes: 0.7917598727639217
F1-scores for prototypes per class: [0.74305556 0.82951369 0.83350896 0.76556652]
############################################################################################# 
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/MultiCNN/mnist_1247/4_lrp.json
... selecting MMD Prototypes
======= Parameters =======
num_prototypes:4
gamma:0.001
use_image_embeddings:False
################ RESULTS for 4 Prototypes and classes dict_keys(['2', '4', '7', '1']) ############# 
Accuracys for selected prototypes: 0.9123773042853722
Recalls for prototypes: [0.84302326 0.89816701 0.90272374 0.99647577]
F1-score averaged for prototypes: 0.9122938111584248
F1-scores for prototypes per class: [0.90342679 0.91970803 0.91473632 0.91172914]
############################################################################################# 
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/rawData/mnist_1247/4_VGG16.json
... selecting MMD Prototypes
======= Parameters =======
num_prototypes:4
gamma:3.985969387755102e-05
use_image_embeddings:True
################ RESULTS for 4 Prototypes and classes dict_keys(['2', '4', '7', '1']) ############# 
Accuracys for selected prototypes: 0.9107014603782619
Recalls for prototypes: [0.78682171 0.901222   0.94941634 0.99647577]
F1-score averaged for prototypes: 0.9096513655409506
F1-scores for prototypes per class: [0.87783784 0.9252483  0.90622098 0.9281904 ]
############################################################################################# 
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/VGG16/mnist_1247/4_VGG16.json
... selecting MMD Prototypes
======= Parameters =======
num_prototypes:4
gamma:0.001
use_image_embeddings:False
################ RESULTS for 4 Prototypes and classes dict_keys(['2', '4', '7', '1']) ############# 
Accuracys for selected prototypes: 0.8161359827627483
Recalls for prototypes: [0.97674419 0.63849287 0.71692607 0.91365639]
F1-score averaged for prototypes: 0.8129810825098566
F1-scores for prototypes per class: [0.93031841 0.67893882 0.71207729 0.91365639]
#############################################################################################
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/VGG16/mnist_1247/4_lrp.json

(venv) zimmer@hydra:~/masterthesis/code_mh_main$ CUDA_VISIBLE_DEVICES=0 python3 prototype_selection.py                
FeatureModel input shape:  (None, 84, 84, 1)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

FeatureModel input shape:  (None, 299, 299, 1)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

... selecting MMD Prototypes
======= Parameters =======
num_prototypes:5
gamma:0.001
use_image_embeddings:False
################ RESULTS for 5 Prototypes and classes dict_keys(['DRUSEN', 'DME', 'NORMAL', 'CNV']) #############                                                                   
Accuracys for selected prototypes: 0.3314873417721519
Recalls for prototypes: [0.15267176 0.29461756 0.27696405 0.41824441]
F1-score averaged for prototypes: 0.3413721999655766
F1-scores for prototypes per class: [0.12779553 0.24270712 0.29482636 0.44958372]
#############################################################################################  
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/rawData/oct_cc/5_MultiCNN.json
... selecting MMD Prototypes
======= Parameters =======
num_prototypes:5
gamma:1e-10
use_image_embeddings:True
################ RESULTS for 5 Prototypes and classes dict_keys(['DRUSEN', 'DME', 'NORMAL', 'CNV']) ############# 
Accuracys for selected prototypes: 0.34533227848101267
Recalls for prototypes: [0.12977099 0.15014164 0.09054594 0.61790017]
F1-score averaged for prototypes: 0.316386430866192
F1-scores for prototypes per class: [0.1030303  0.13965744 0.15044248 0.52542993]
############################################################################################# 
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/MultiCNN/oct_cc/5_MultiCNN.json
... selecting MMD Prototypes
======= Parameters =======
num_prototypes:5
gamma:0.001
use_image_embeddings:False
################ RESULTS for 5 Prototypes and classes dict_keys(['DRUSEN', 'DME', 'NORMAL', 'CNV']) ############# 
Accuracys for selected prototypes: 0.3299050632911392
Recalls for prototypes: [0.14503817 0.3427762  0.51531292 0.24784854]
F1-score averaged for prototypes: 0.3347137764236973
F1-scores for prototypes per class: [0.13743219 0.25155925 0.39249493 0.36711281]
############################################################################################# 
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/MultiCNN/oct_cc/5_lrp.json
... selecting MMD Prototypes
======= Parameters =======
num_prototypes:5
gamma:0.01
use_image_embeddings:False
################ RESULTS for 5 Prototypes and classes dict_keys(['DRUSEN', 'DME', 'NORMAL', 'CNV']) ############# 
Accuracys for selected prototypes: 0.3623417721518987
Recalls for prototypes: [0.27099237 0.23512748 0.3355526  0.43889845]
F1-score averaged for prototypes: 0.3800939872251481
F1-scores for prototypes per class: [0.16884661 0.22802198 0.36788321 0.48181389]
############################################################################################# 
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/rawData/oct_cc/5_VGG16.json
... selecting MMD Prototypes
======= Parameters =======
num_prototypes:5
gamma:1e-05
use_image_embeddings:True
################ RESULTS for 5 Prototypes and classes dict_keys(['DRUSEN', 'DME', 'NORMAL', 'CNV']) ############# 
Accuracys for selected prototypes: 0.4576740506329114
Recalls for prototypes: [0.2480916  0.20963173 0.54460719 0.52409639]
F1-score averaged for prototypes: 0.48127751359885784
F1-scores for prototypes per class: [0.14755959 0.24503311 0.59795322 0.55288243]
############################################################################################# 
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/VGG16/oct_cc/5_VGG16.json
... selecting MMD Prototypes
======= Parameters =======
num_prototypes:5
gamma:0.1
use_image_embeddings:False
################ RESULTS for 5 Prototypes and classes dict_keys(['DRUSEN', 'DME', 'NORMAL', 'CNV']) ############# 
Accuracys for selected prototypes: 0.6214398734177216
Recalls for prototypes: [0.24427481 0.52407932 0.54593875 0.7848537 ]
F1-score averaged for prototypes: 0.6420103812321211
F1-scores for prototypes per class: [0.22816399 0.38341969 0.60966543 0.83478261]
############################################################################################# 
Saving ...
/home/zimmer/masterthesis/code_mh_main/static/prototypes/VGG16/oct_cc/5_lrp.json
```

## 7. Generate Near Hits & Near Misses
### MNIST
```
FeatureModel input shape:  (None, 84, 84, 1)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 42, 42, 8)        32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 21, 21, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 84, 84, 1)
------------- START -------------
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates=self.state_updates,
2023-01-02 20:23:44.145277: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500
5
0.0h 0.0min 11.389426231384277sec 
10
0.0h 0.0min 14.94316816329956sec 
15
0.0h 0.0min 18.449624061584473sec 
20
0.0h 0.0min 21.933353662490845sec 
25
0.0h 0.0min 25.40114140510559sec 
30
0.0h 0.0min 28.89528489112854sec 
35
0.0h 0.0min 32.38681745529175sec 
40
0.0h 0.0min 35.87310600280762sec 
45
0.0h 0.0min 39.39665198326111sec 
50
0.0h 0.0min 42.91023635864258sec 
Maximum iteration of 10 reached
0.0h 0.0min 42.91149401664734sec
55
0.0h 0.0min 13.19010329246521sec 
60
0.0h 0.0min 16.738559246063232sec 
65
0.0h 0.0min 20.283756256103516sec 
70
0.0h 0.0min 23.819716453552246sec 
75
0.0h 0.0min 27.346009254455566sec 
80
0.0h 0.0min 30.865437030792236sec 
85
0.0h 0.0min 34.384021282196045sec 
90
0.0h 0.0min 37.93021535873413sec 
95
0.0h 0.0min 41.47471737861633sec 
100
0.0h 0.0min 45.0104284286499sec 
Maximum iteration of 20 reached
0.0h 0.0min 45.01169228553772sec 


------------- FINISHED -------------
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
------------- START -------------
5
0.0h 0.0min 13.461151123046875sec 
10
0.0h 0.0min 16.705812215805054sec 
15
0.0h 0.0min 19.9459171295166sec 
20
0.0h 0.0min 23.18537950515747sec 
25
0.0h 0.0min 26.42236018180847sec 
30
0.0h 0.0min 29.668489456176758sec 
35
0.0h 0.0min 32.909945487976074sec 
40
0.0h 0.0min 36.1539421081543sec 
45
0.0h 0.0min 39.401164293289185sec 
50
0.0h 0.0min 42.62935447692871sec 
Maximum iteration of 10 reached
0.0h 0.0min 42.630530834198sec 
55
0.0h 0.0min 13.365906715393066sec 
60
0.0h 0.0min 16.665499210357666sec 
65
0.0h 0.0min 19.945485830307007sec 
70
0.0h 0.0min 23.22565722465515sec 
75
0.0h 0.0min 26.501275062561035sec 
80
0.0h 0.0min 29.766016721725464sec 
85
0.0h 0.0min 33.0336709022522sec 
90
0.0h 0.0min 36.300108432769775sec 
95
0.0h 0.0min 39.56943106651306sec 
100
0.0h 0.0min 42.84086203575134sec 
Maximum iteration of 20 reached
0.0h 0.0min 42.84205412864685sec 

------------- FINISHED -------------
FeatureModel input shape:  (None, 84, 84, 1)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 42, 42, 8)        32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 21, 21, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 84, 84, 1)
------------- START -------------
5
0.0h 0.0min 34.025043964385986sec 
10
0.0h 1.0min 1.5204901695251465sec 
15
0.0h 1.0min 29.13583993911743sec 
20
0.0h 1.0min 56.52608370780945sec 
25
0.0h 2.0min 23.97851824760437sec
30
0.0h 2.0min 51.480507612228394sec
35
0.0h 3.0min 18.93913435935974sec
40
0.0h 3.0min 46.53764748573303sec
45
0.0h 4.0min 14.387082815170288sec
50
0.0h 4.0min 42.98845672607422sec 
Maximum iteration of 10 reached
0.0h 4.0min 42.98998427391052sec 
55
0.0h 0.0min 34.08389663696289sec 
60
0.0h 1.0min 1.3828060626983643sec 
65
0.0h 1.0min 28.729649782180786sec 
70
0.0h 1.0min 56.14008188247681sec 
75
0.0h 2.0min 24.416038274765015sec 
80
0.0h 2.0min 52.418941497802734sec 
85
0.0h 3.0min 19.930182456970215sec 
90
0.0h 3.0min 47.27004861831665sec 
95
0.0h 4.0min 14.504844188690186sec 
100
0.0h 4.0min 42.38769292831421sec 
Maximum iteration of 20 reached
0.0h 4.0min 42.38901615142822sec 

------------- FINISHED -------------
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
------------- START -------------
5
0.0h 0.0min 44.71902775764465sec 
10
0.0h 1.0min 12.124488353729248sec 
15
0.0h 1.0min 46.74555706977844sec 
20
0.0h 2.0min 22.884599924087524sec 
25
0.0h 2.0min 55.08869171142578sec 
30
0.0h 3.0min 25.048551321029663sec 
35
0.0h 3.0min 52.63500928878784sec 
40
0.0h 4.0min 29.675851345062256sec 
45
0.0h 5.0min 4.430556535720825sec 
50
0.0h 5.0min 33.44232153892517sec 
Maximum iteration of 10 reached
0.0h 5.0min 33.44354701042175sec 
55
0.0h 0.0min 37.79430031776428sec 
60
0.0h 1.0min 5.305842876434326sec 
65
0.0h 1.0min 32.72656202316284sec 
70
0.0h 2.0min 0.44859766960144043sec 
75
0.0h 2.0min 27.93812656402588sec 
80
0.0h 2.0min 55.47353458404541sec 
85
0.0h 3.0min 22.983174324035645sec 
90
0.0h 3.0min 50.72269058227539sec 
95
0.0h 4.0min 18.460933923721313sec 
100
0.0h 4.0min 46.34673452377319sec 
Maximum iteration of 20 reached
0.0h 4.0min 46.34794497489929sec 


------------- FINISHED -------------
FeatureModel input shape:  (None, 84, 84, 1)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...                                                     
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80

 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0
 )

 batch_normalization (BatchN  (None, 42, 42, 8)        32
 ormalization)

 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168

 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0
 2D)

 batch_normalization_1 (Batc  (None, 21, 21, 16)       64
 hNormalization)

 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640

 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0
 2D)

 batch_normalization_2 (Batc  (None, 10, 10, 32)       128
 hNormalization)

 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496

 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0
 2D)

 batch_normalization_3 (Batc  (None, 5, 5, 64)         256
 hNormalization)

 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856

 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0
 2D)

 flatten (Flatten)           (None, 512)               0

 dense (Dense)               (None, 256)               131328

 dropout (Dropout)           (None, 256)               0

 dense_1 (Dense)             (None, 128)               32896

 dropout_1 (Dropout)         (None, 128)               0

 batch_normalization_4 (Batc  (None, 128)              512
 hNormalization)

 activation (Activation)     (None, 128)               0

 dense_2 (Dense)             (None, 4)                 516

 activation_1 (Activation)   (None, 4)                 0

=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 84, 84, 1)
------------- START -------------
5
0.0h 13.0min 30.81589126586914sec 
10
0.0h 26.0min 55.61746573448181sec 
15
0.0h 40.0min 12.998539209365845sec
20
0.0h 53.0min 27.4389328956604sec
25
1.0h 6.0min 57.04194664955139sec
30
1.0h 20.0min 19.570916175842285sec
35
1.0h 33.0min 34.93576502799988sec
40
1.0h 47.0min 11.121551275253296sec 
45
2.0h 0.0min 31.815716981887817sec 
50
2.0h 13.0min 49.113675594329834sec 
Maximum iteration of 10 reached
2.0h 13.0min 49.11462903022766sec 
55
0.0h 11.0min 8.80941128730774sec 
60
0.0h 22.0min 8.142057657241821sec 
65
0.0h 33.0min 6.4186811447143555sec 
70
0.0h 44.0min 5.678251504898071sec 
75
0.0h 55.0min 3.3728716373443604sec 
80
1.0h 6.0min 1.8226909637451172sec 
85
1.0h 17.0min 1.0887830257415771sec 
90
1.0h 28.0min 1.070406198501587sec 
95
1.0h 39.0min 1.2919409275054932sec 
100
1.0h 50.0min 1.2201299667358398sec 
Maximum iteration of 20 reached
1.0h 50.0min 1.2211081981658936sec 

------------- FINISHED -------------
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
------------- START -------------
5
0.0h 13.0min 38.94897985458374sec 
10
0.0h 27.0min 7.615281581878662sec 
15
0.0h 40.0min 35.09681153297424sec 
20
0.0h 54.0min 2.571282386779785sec 
25
1.0h 7.0min 30.2903950214386sec 
30
1.0h 20.0min 51.13400888442993sec 
35
1.0h 34.0min 18.88610005378723sec 
40
1.0h 47.0min 49.928412199020386sec 
45
1.0h 59.0min 40.23275113105774sec 
50
2.0h 10.0min 49.0356822013855sec 
55
2.0h 21.0min 59.18298101425171sec 
60
2.0h 33.0min 9.277307033538818sec 
65
2.0h 44.0min 14.970084190368652sec 
70
2.0h 55.0min 23.725703954696655sec 
75
3.0h 6.0min 30.416300773620605sec 
80
3.0h 17.0min 38.082685470581055sec 
85
3.0h 28.0min 45.64925265312195sec 
90
3.0h 39.0min 54.892945289611816sec 
95
3.0h 51.0min 4.323294401168823sec 
100
4.0h 2.0min 15.50270938873291sec 
Maximum iteration of 20 reached
4.0h 2.0min 15.503751754760742sec 
100
0.0h 11.0min 9.992713689804077sec 
Maximum iteration of 20 reached
0.0h 11.0min 9.993827819824219sec 
------------- FINISHED -------------
FeatureModel input shape:  (None, 84, 84, 1)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

... newly loaded feature embeddings, which were not considered yet :  0
Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 42, 42, 8)        32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 21, 21, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 84, 84, 1)
------------- START -------------
5
0.0h 0.0min 20.60198402404785sec 
10
0.0h 0.0min 21.670543909072876sec 
15
0.0h 0.0min 22.740233659744263sec 
20
0.0h 0.0min 23.82032585144043sec 
25
0.0h 0.0min 24.888814449310303sec 
30
0.0h 0.0min 25.952594757080078sec 
35
0.0h 0.0min 27.01596474647522sec 
40
0.0h 0.0min 28.08603858947754sec 
45
0.0h 0.0min 29.150248527526855sec 
50
0.0h 0.0min 30.21361494064331sec 
Maximum iteration of 10 reached
0.0h 0.0min 30.214325428009033sec 
55
0.0h 0.0min 16.955950260162354sec 
60
0.0h 0.0min 17.986992359161377sec 
65
0.0h 0.0min 19.02727174758911sec 
70
0.0h 0.0min 20.075976848602295sec 
75
0.0h 0.0min 21.121464014053345sec 
80
0.0h 0.0min 22.182676315307617sec 
85
0.0h 0.0min 23.23568558692932sec 
90
0.0h 0.0min 24.291221141815186sec 
95
0.0h 0.0min 25.3491792678833sec 
100
0.0h 0.0min 26.397902250289917sec 
Maximum iteration of 20 reached
0.0h 0.0min 26.39862632751465sec 
------------- FINISHED -------------
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

... newly loaded feature embeddings, which were not considered yet :  0
Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
------------- START -------------
5
0.0h 0.0min 27.825244426727295sec 
10
0.0h 0.0min 30.780086517333984sec 
15
0.0h 0.0min 33.75183439254761sec 
20
0.0h 0.0min 36.70050096511841sec 
25
0.0h 0.0min 39.647759675979614sec 
30
0.0h 0.0min 42.577094078063965sec 
35
0.0h 0.0min 45.50978755950928sec 
40
0.0h 0.0min 48.442283630371094sec 
45
0.0h 0.0min 51.373602867126465sec 
50
0.0h 0.0min 54.31148052215576sec 
Maximum iteration of 10 reached
0.0h 0.0min 54.3122079372406sec 
55
0.0h 0.0min 25.980172395706177sec 
60
0.0h 0.0min 28.803551197052002sec 
65
0.0h 0.0min 31.646101474761963sec 
70
0.0h 0.0min 34.481523275375366sec 
75
0.0h 0.0min 37.31861352920532sec 
80
0.0h 0.0min 40.14854168891907sec 
85
0.0h 0.0min 42.96565794944763sec 
90
0.0h 0.0min 45.78728747367859sec 
95
0.0h 0.0min 48.62374520301819sec 
100
0.0h 0.0min 51.40302014350891sec 
Maximum iteration of 20 reached
0.0h 0.0min 51.40376138687134sec 
------------- FINISHED -------------
FeatureModel input shape:  (None, 84, 84, 1)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 42, 42, 8)        32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 21, 21, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 84, 84, 1)
------------- START -------------
5
0.0h 0.0min 31.283336877822876sec 
10
0.0h 0.0min 45.79001522064209sec 
15
0.0h 0.0min 58.843727111816406sec 
20
0.0h 1.0min 11.817845582962036sec 
25
0.0h 1.0min 24.93188786506653sec 
30
0.0h 1.0min 37.92557239532471sec 
35
0.0h 1.0min 50.83231854438782sec 
40
0.0h 2.0min 3.748704195022583sec 
45
0.0h 2.0min 16.73530387878418sec 
50
0.0h 2.0min 29.730880737304688sec 
Maximum iteration of 10 reached
0.0h 2.0min 29.731919050216675sec 
55
0.0h 0.0min 28.92638659477234sec 
60
0.0h 0.0min 41.855019092559814sec
65
0.0h 0.0min 54.75043725967407sec
70
0.0h 1.0min 7.497093915939331sec
75
0.0h 1.0min 20.334691286087036sec
80
0.0h 1.0min 33.21188163757324sec
85
0.0h 1.0min 46.1156280040741sec
90
0.0h 1.0min 59.01142978668213sec
95
0.0h 2.0min 11.735270023345947sec
100
0.0h 2.0min 24.618515968322754sec
Maximum iteration of 20 reached
0.0h 2.0min 24.619526624679565sec
------------- FINISHED -------------
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
------------- START -------------
5
0.0h 1.0min 50.865269899368286sec 
10
0.0h 3.0min 22.69039034843445sec 
15
0.0h 4.0min 51.373881340026855sec 
20
0.0h 6.0min 22.21960163116455sec 
25
0.0h 7.0min 52.06134796142578sec 
30
0.0h 9.0min 22.97470784187317sec 
35
0.0h 10.0min 53.13441014289856sec 
40
0.0h 12.0min 23.78626012802124sec 
45
0.0h 13.0min 53.397645711898804sec 
50
0.0h 15.0min 23.299001693725586sec 
Maximum iteration of 10 reached
0.0h 15.0min 23.30074167251587sec 
55
0.0h 1.0min 40.76160645484924sec
60
0.0h 3.0min 9.659339904785156sec
65
0.0h 4.0min 38.35260319709778sec
70
0.0h 6.0min 8.480293273925781sec
75
0.0h 7.0min 38.427629470825195sec 
80
0.0h 9.0min 8.501458168029785sec 
85
0.0h 10.0min 36.871121883392334sec 
90
0.0h 12.0min 7.890830993652344sec
95
0.0h 13.0min 39.83172821998596sec 
100
0.0h 15.0min 9.867118835449219sec 
Maximum iteration of 20 reached
0.0h 15.0min 9.868803262710571sec 
------------- FINISHED -------------
FeatureModel input shape:  (None, 84, 84, 1)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 42, 42, 8)        32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 21, 21, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 84, 84, 1)
------------- START -------------
5
0.0h 1.0min 13.226014614105225sec 
10
0.0h 2.0min 11.870126962661743sec 
15
0.0h 3.0min 10.18721318244934sec 
20
0.0h 4.0min 8.49584698677063sec 
25
0.0h 5.0min 8.105380296707153sec 
30
0.0h 6.0min 6.323283910751343sec 
35
0.0h 7.0min 5.479820489883423sec 
40
0.0h 8.0min 5.056214332580566sec 
45
0.0h 9.0min 4.704175710678101sec 
50
0.0h 10.0min 3.0833702087402344sec 
Maximum iteration of 10 reached
0.0h 10.0min 3.084399461746216sec 
55
0.0h 1.0min 11.644383907318115sec 
60
0.0h 2.0min 12.115899085998535sec 
65
0.0h 3.0min 10.978604078292847sec 
70
0.0h 4.0min 12.235047578811646sec 
75
0.0h 5.0min 9.737166166305542sec 
80
0.0h 6.0min 7.31626033782959sec 
85
0.0h 7.0min 5.03568696975708sec 
90
0.0h 8.0min 2.8109095096588135sec 
95
0.0h 9.0min 2.122014045715332sec 
100
0.0h 10.0min 2.1079823970794678sec 
Maximum iteration of 20 reached
0.0h 10.0min 2.109031915664673sec 
------------- FINISHED -------------
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
------------- START -------------
5
0.0h 4.0min 55.28175497055054sec 
10
0.0h 9.0min 39.82090473175049sec 
15
0.0h 14.0min 23.942431688308716sec 
20
0.0h 19.0min 5.755471706390381sec 
25
0.0h 23.0min 49.82682132720947sec 
30
0.0h 28.0min 33.237335443496704sec 
35
0.0h 33.0min 18.464378595352173sec 
40
0.0h 38.0min 5.135389566421509sec 
45
0.0h 42.0min 54.87409806251526sec 
50
0.0h 47.0min 45.35963988304138sec 
Maximum iteration of 10 reached
0.0h 47.0min 45.360883951187134sec 
55
0.0h 4.0min 51.504671573638916sec
60
0.0h 4.0min 54.485809564590454sec 
65
0.0h 9.0min 40.623345136642456sec 
70
0.0h 14.0min 24.291901111602783sec 
75
0.0h 19.0min 9.33557677268982sec 
80
0.0h 23.0min 54.727158069610596sec 
85
0.0h 28.0min 38.92715573310852sec 
90
0.0h 33.0min 25.48782730102539sec 
95
0.0h 38.0min 10.535347700119019sec 
100
0.0h 42.0min 56.740997314453125sec 
Maximum iteration of 20 reached
0.0h 42.0min 56.7421019077301sec

------------- FINISHED -------------
FeatureModel input shape:  (None, 84, 84, 1)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_mnist_1247_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 84, 84, 8)         80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 42, 42, 8)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 42, 42, 8)        32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 42, 42, 16)        1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 21, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 21, 21, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 10, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 128)         73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 2, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 263,972
Trainable params: 263,476
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 84, 84, 1)
------------- START -------------
5
0.0h 39.0min 51.458850622177124sec 
10
1.0h 19.0min 17.848004579544067sec 
15
1.0h 59.0min 1.1224069595336914sec 
20
2.0h 38.0min 44.47058343887329sec 
25
3.0h 18.0min 25.619669675827026sec 
30
3.0h 58.0min 14.136878967285156sec 
35
4.0h 37.0min 34.81802701950073sec 
40
5.0h 17.0min 6.543123483657837sec 
45
5.0h 56.0min 25.232823371887207sec 
50
6.0h 35.0min 50.79627013206482sec 
Maximum iteration of 10 reached
6.0h 35.0min 50.79922795295715sec 
55
0.0h 39.0min 18.660112380981445sec 
60
1.0h 18.0min 41.68474769592285sec 
65
1.0h 58.0min 0.22744393348693848sec 
70
2.0h 37.0min 16.603106260299683sec 
75
3.0h 16.0min 33.54632568359375sec 
80
3.0h 56.0min 1.8422112464904785sec 
85
4.0h 35.0min 6.641971588134766sec 
90
5.0h 14.0min 15.13011384010315sec 
95
5.0h 53.0min 27.647212266921997sec 
100
6.0h 32.0min 55.7496383190155sec 
Maximum iteration of 20 reached
6.0h 32.0min 55.75067758560181sec

------------- FINISHED -------------

FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: mnist_1247
Available Classes: ['2', '4', '7', '1']
Length of Train Data: 22327
Length of Validation Data: 2480
Length of Test Data: 4177
==============

Initializing Image Generator ...
Found 22327 images belonging to 4 classes.
Found 2480 images belonging to 4 classes.
Found 4177 images belonging to 4 classes.
------------- START -------------
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.                                                 
  updates=self.state_updates,
2023-01-04 09:43:48.562653: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500                                                
5
4.0h 12.0min 29.090702772140503sec
10
8.0h 24.0min 12.451000690460205sec
15
12.0h 35.0min 24.02212929725647sec
20
16.0h 53.0min 39.8876166343689sec
25
21.0h 8.0min 1.6420307159423828sec
30
25.0h 22.0min 18.639802932739258sec 
35
29.0h 51.0min 2.218240976333618sec
40
34.0h 7.0min 53.54738235473633sec
45
38.0h 19.0min 36.72968602180481sec
50
42.0h 31.0min 24.027393102645874sec
55
47.0h 11.0min 28.266498804092407sec
60
51.0h 49.0min 1.7001333236694336sec
65
56.0h 0.0min 12.529116153717041sec
70
60.0h 12.0min 53.34191632270813sec
75
64.0h 24.0min 41.240044593811035sec
80
68.0h 35.0min 41.21275043487549sec
85
72.0h 46.0min 20.452898263931274sec
90
77.0h 0.0min 37.19913458824158sec
95
81.0h 12.0min 3.402787923812866sec
100                                                                                                                                  
85.0h 23.0min 3.189783811569214sec                                                                                                   
Maximum iteration of 20 reached                                                                                                      
85.0h 23.0min 3.204815626144409sec
```
### OCT
```
FeatureModel input shape:  (None, 299, 299, 1)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 299, 299, 8)       80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 149, 149, 8)      0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 149, 149, 8)      32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 149, 149, 16)      1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 74, 74, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 74, 74, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 74, 74, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 37, 37, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 37, 37, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 37, 37, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 18, 18, 64)       0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 18, 18, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 18, 18, 128)       73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 9, 9, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 10368)             0         
                                                                 
 dense (Dense)               (None, 256)               2654464   
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 2,787,108
Trainable params: 2,786,612
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 299, 299, 1)
------------- START -------------
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates=self.state_updates,
2023-01-02 20:29:13.106840: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500
5
0.0h 13.0min 14.986412525177002sec
10
0.0h 25.0min 45.602110862731934sec
15
0.0h 38.0min 27.230398893356323sec
20
0.0h 50.0min 51.58543276786804sec 
25
1.0h 3.0min 26.57380247116089sec 
30
1.0h 16.0min 3.9078195095062256sec 
35
1.0h 28.0min 39.946377992630005sec 
40                                                                                                                                                                
1.0h 41.0min 9.735098838806152sec                                                                                                                                 
45                                                                                                                                                                
1.0h 53.0min 34.58841419219971sec                                                                                                                                 
50                                                                                                                                                                
2.0h 6.0min 8.351786851882935sec                                                                                                                                  
Maximum iteration of 10 reached                                                                                                                                   
2.0h 6.0min 8.352851629257202sec   
55
0.0h 12.0min 28.84254765510559sec 
60
0.0h 24.0min 50.93837761878967sec 
65
0.0h 36.0min 56.599833250045776sec 
70
0.0h 49.0min 7.660876035690308sec 
75
1.0h 1.0min 13.927350759506226sec 
80
1.0h 13.0min 24.770382404327393sec 
85
1.0h 25.0min 40.816601276397705sec 
90
1.0h 37.0min 56.93804168701172sec 
95
1.0h 50.0min 12.458970069885254sec 
100
2.0h 2.0min 34.97942328453064sec 
Maximum iteration of 20 reached
2.0h 2.0min 34.98037552833557sec                                                                                                                                
------------- FINISHED -------------                                                                                                                              
FeatureModel input shape:  (None, 224, 224, 3)                                                                                                                    
                                                                                                                                                                  
==============                                                                                                                                                    
Current Dataset: oct_cc                                                                                                                                           
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']                                                                                                             
Length of Train Data: 72937                                                                                                                                       
Length of Validation Data: 8091                                                                                                                                   
Length of Test Data: 2528                                                                                                                                         
==============                                                                                                                                                    
                                                                                                                                                                  
Initializing Image Generator ...                                                                                                                                  
Found 72937 images belonging to 4 classes.                                                                                                                        
Found 8091 images belonging to 4 classes.                                                                                                                         
Found 2528 images belonging to 4 classes.                                                                                                                         
------------- START -------------                                                                                                                                 
5                                                                                                                                                                 
0.0h 12.0min 51.683762311935425sec                                                                                                                                
10                                                                                                                                                                
0.0h 25.0min 30.225303649902344sec                                                                                                                                
15                                                                                                                                                                
0.0h 38.0min 17.791508197784424sec                                                                                                                                
20                                                                                                                                                                
0.0h 50.0min 42.77172660827637sec                                                                                                                                 
25                                                                                                                                                                
1.0h 3.0min 16.74456214904785sec                                                                                                                                  
30                                                                                                                                                                
1.0h 15.0min 48.5703558921814sec                                                                                                                                  
35                                                                                                                                                                
1.0h 28.0min 24.244770288467407sec                                                                                                                                
40                                                                                                                                                                
1.0h 40.0min 44.58174753189087sec                                                                                                                                 
45                                                                                                                                                                
1.0h 53.0min 7.3069682121276855sec                                                                                                                                
50                                                                                                                                                                
2.0h 5.0min 40.825642585754395sec                                                                                                                                 
Maximum iteration of 10 reached                                                                                                                                   
2.0h 5.0min 40.82664442062378sec     
55
0.0h 12.0min 18.38513445854187sec 
60
0.0h 24.0min 28.81975245475769sec 
65
0.0h 36.0min 24.99224305152893sec 
70
0.0h 48.0min 25.265398263931274sec 
75
1.0h 0.0min 18.789661407470703sec 
80
1.0h 12.0min 19.25826358795166sec 
85
1.0h 24.0min 23.131223917007446sec 
90
1.0h 36.0min 27.33821201324463sec 
95
1.0h 48.0min 29.032140970230103sec 
100
2.0h 0.0min 41.46690845489502sec 
Maximum iteration of 20 reached
2.0h 0.0min 41.467941999435425sec                                                                                              
------------- FINISHED -------------                                                                                                                              
FeatureModel input shape:  (None, 299, 299, 1)                                                                                                                    
                                                                                                                                                                  
==============                                                                                                                                                    
Current Dataset: oct_cc                                                                                                                                           
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']                                                                                                             
Length of Train Data: 72937                                                                                                                                       
Length of Validation Data: 8091                                                                                                                                   
Length of Test Data: 2528                                                                                                                                         
==============                                                                                                                                                    
                                                                                                                                                                  
Initializing Image Generator ...                                                                                                                                  
Found 72937 images belonging to 4 classes.                                                                                                                        
Found 8091 images belonging to 4 classes.                                                                                                                         
Found 2528 images belonging to 4 classes.                                                                                                                         
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...                                                          
Model: "sequential"
_________________________________________________________________                                                                                         
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 299, 299, 8)       80

 max_pooling2d (MaxPooling2D  (None, 149, 149, 8)      0
 )

 batch_normalization (BatchN  (None, 149, 149, 8)      32
 ormalization)

 conv2d_1 (Conv2D)           (None, 149, 149, 16)      1168

 max_pooling2d_1 (MaxPooling  (None, 74, 74, 16)       0
 2D)

 batch_normalization_1 (Batc  (None, 74, 74, 16)       64
 hNormalization)

 conv2d_2 (Conv2D)           (None, 74, 74, 32)        4640

 max_pooling2d_2 (MaxPooling  (None, 37, 37, 32)       0
 2D)

 batch_normalization_2 (Batc  (None, 37, 37, 32)       128
 hNormalization)

 conv2d_3 (Conv2D)           (None, 37, 37, 64)        18496

 max_pooling2d_3 (MaxPooling  (None, 18, 18, 64)       0
 2D)

 batch_normalization_3 (Batc  (None, 18, 18, 64)       256
 hNormalization)

 conv2d_4 (Conv2D)           (None, 18, 18, 128)       73856

 max_pooling2d_4 (MaxPooling  (None, 9, 9, 128)        0
 2D)

 flatten (Flatten)           (None, 10368)             0

 dense (Dense)               (None, 256)               2654464

 dropout (Dropout)           (None, 256)               0

 dense_1 (Dense)             (None, 128)               32896

 dropout_1 (Dropout)         (None, 128)               0

 batch_normalization_4 (Batc  (None, 128)              512
 hNormalization)

 activation (Activation)     (None, 128)               0

 dense_2 (Dense)             (None, 4)                 516

 activation_1 (Activation)   (None, 4)                 0

=================================================================
Total params: 2,787,108
Trainable params: 2,786,612
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 299, 299, 1)
------------- START -------------
5
2.0h 46.0min 24.80335021018982sec
10
5.0h 6.0min 35.98440718650818sec
15
7.0h 39.0min 51.44905710220337sec
20
9.0h 51.0min 52.918466091156006sec
25
12.0h 36.0min 23.95797038078308sec
________ pause here
30
2.0h 26.0min 45.20502424240112sec
35
4.0h 54.0min 26.137991905212402sec
40
7.0h 5.0min 49.84993052482605sec
45
9.0h 18.0min 8.248792171478271sec
50
11.0h 56.0min 2.2458243370056152sec
Maximum iteration of 10 reached
11.0h 56.0min 2.246870994567871sec
55
2.0h 19.0min 30.980053424835205sec
60
4.0h 44.0min 58.552266359329224sec
65
6.0h 48.0min 36.42590141296387sec
70
9.0h 9.0min 3.188098430633545sec
75
11.0h 21.0min 51.02318048477173sec
80
13.0h 43.0min 36.94641375541687sec
85
16.0h 18.0min 46.378464698791504sec
90
18.0h 47.0min 43.15345525741577sec
95
21.0h 15.0min 16.140029907226562sec
100
23.0h 56.0min 40.593724966049194sec
Maximum iteration of 20 reached
23.0h 56.0min 40.59469819068909sec

------------- FINISHED -------------
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
------------- START -------------
5
2.0h 36.0min 2.4487321376800537sec
10
5.0h 1.0min 58.42137050628662sec
15
7.0h 41.0min 51.74971055984497sec
20
9.0h 54.0min 45.32301163673401sec
— break
25
2.0h 30.0min 26.19158697128296sec 
30
4.0h 57.0min 18.09048318862915sec 
35
7.0h 33.0min 25.271342039108276sec 
40
9.0h 44.0min 25.52774930000305sec 
45
11.0h 56.0min 43.21330785751343sec 
50
14.0h 25.0min 8.846301317214966sec 
55
16.0h 45.0min 35.845542669296265sec 
60
19.0h 10.0min 22.70689845085144sec
65
21.0h 15.0min 39.72352623939514sec
70
23.0h 40.0min 1.459670066833496sec
75
25.0h 54.0min 14.868552684783936sec
80
28.0h 14.0min 7.281526565551758sec
85
30.0h 48.0min 43.022860288619995sec
90
2.0h 29.0min 22.72258734703064sec
95
4.0h 56.0min 15.438121318817139sec
100
7.0h 38.0min 41.460933685302734sec
Maximum iteration of 20 reached
7.0h 38.0min 41.475506067276sec

FeatureModel input shape:  (None, 299, 299, 1)                                                                                                                    
   
==============                                                                                                                                                    
Current Dataset: oct_cc                                                                                                                                           
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']                                                                                                             
Length of Train Data: 72937                                                                                                                                       
Length of Validation Data: 8091                                                                                                                                   
Length of Test Data: 2528                                                                                                                                         
==============                                                                                                                                                    
                                                                                                                                                                  
... newly loaded feature embeddings, which were not considered yet :  0                                                                                           
Initializing Image Generator ...                                                                                                                                  
Found 72937 images belonging to 4 classes.                                                                                                                        
Found 8091 images belonging to 4 classes.                                                                                                                         
Found 2528 images belonging to 4 classes.                                                                                                                         
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...                                                          
Model: "sequential"                                                                                                                                               
_________________________________________________________________                                                                                                 
 Layer (type)                Output Shape              Param #                                                                                                    
=================================================================                                                                                                 
 conv2d (Conv2D)             (None, 299, 299, 8)       80                                                                                                         
                                                                                                                                                                  
 max_pooling2d (MaxPooling2D  (None, 149, 149, 8)      0                                                                                                          
 )                                                                                                                                                                
                                                                                                                                                                  
 batch_normalization (BatchN  (None, 149, 149, 8)      32                                                                                                         
 ormalization)                                                                                                                                                    
                                                                                                                                                                  
 conv2d_1 (Conv2D)           (None, 149, 149, 16)      1168                                                                                                       
                                                                                                                                                                  
 max_pooling2d_1 (MaxPooling  (None, 74, 74, 16)       0                                                                                                          
 2D)                                                                                                                                                              
                                                                                                                                                                  
 batch_normalization_1 (Batc  (None, 74, 74, 16)       64                                                                                                         
 hNormalization)                                                                                                                                                  
                                                                                                                                                                  
 conv2d_2 (Conv2D)           (None, 74, 74, 32)        4640                                                                                                       
                                                                                                                                                                  
 max_pooling2d_2 (MaxPooling  (None, 37, 37, 32)       0                                                                                                          
 2D)                                                                                                                                                              
                                                                                                                                                                  
 batch_normalization_2 (Batc  (None, 37, 37, 32)       128                                                                                                        
 hNormalization)                                                                                                                                                  
                                                                                                                                                                  
 conv2d_3 (Conv2D)           (None, 37, 37, 64)        18496                                                                                                      
                                                                                                                                                                  
 max_pooling2d_3 (MaxPooling  (None, 18, 18, 64)       0                                                                                                          
 2D)                                                                                                                                                              
                                                                                                                                                                  
 batch_normalization_3 (Batc  (None, 18, 18, 64)       256                                                                                                        
 hNormalization)                                                                                                                                                  
                                                                                                                                                                  
 conv2d_4 (Conv2D)           (None, 18, 18, 128)       73856                                                                                                      
  
 max_pooling2d_4 (MaxPooling  (None, 9, 9, 128)        0                                                                                                          
 2D)                                                                                                                                                              
                                                                                                                                                                  
 flatten (Flatten)           (None, 10368)             0                                                                                                          
                                                                                                                                                                  
 dense (Dense)               (None, 256)               2654464                                                                                                    
                                                                                                                                                                  
 dropout (Dropout)           (None, 256)               0                                                                                                          
                                                                                                                                                                  
 dense_1 (Dense)             (None, 128)               32896                                                                                                      
                                                                                                                                                                  
 dropout_1 (Dropout)         (None, 128)               0                                                                                                          
                                                                                                                                                                  
 batch_normalization_4 (Batc  (None, 128)              512                                                                                                        
 hNormalization)                                                                                                                                                  
                                                                                                                                                                  
 activation (Activation)     (None, 128)               0                                                                                                          
                                                                                                                                                                  
 dense_2 (Dense)             (None, 4)                 516                                                                                                        
                                                                                                                                                                  
 activation_1 (Activation)   (None, 4)                 0                                                                                                          
                                                                                                                                                                  
=================================================================                                                                                                 
Total params: 2,787,108                                                                                                                                           
Trainable params: 2,786,612                                                                                                                                       
Non-trainable params: 496                                                                                                                                         
_________________________________________________________________                                                                                                 
None                                                                                                                                                              
Model input shape:  (None, 299, 299, 1)                                                                                                                           
------------- START -------------                                                                                                                                 
5                                                                                                                                                                 
0.0h 0.0min 57.18609404563904sec                                                                                                                                  
10                                                                                                                                                                
0.0h 1.0min 0.5633106231689453sec                                                                                                                                 
15                                                                                                                                                                
0.0h 1.0min 3.9573259353637695sec                                                                                                                                 
20                                                                                                                                                                
0.0h 1.0min 7.361639499664307sec                                                                                                                                  
25                                                                                                                                                                
0.0h 1.0min 10.76249361038208sec                                                                                                                                  
30                                                                                                                                                                
0.0h 1.0min 14.13526725769043sec                                                                                                                                  
35                                                                                                                                                                
0.0h 1.0min 17.55215096473694sec                                                                                                                                  
40                                                                                                                                                                
0.0h 1.0min 20.95653247833252sec                                                                                                                                  
45                                                                                                                                                                
0.0h 1.0min 24.373652935028076sec                                                                                                                                 
50                                                                                                                                                                
0.0h 1.0min 27.700730800628662sec                                                                                                                                 
Maximum iteration of 10 reached                                                                                                                                   
0.0h 1.0min 27.70147681236267sec   
55
0.0h 0.0min 44.08021521568298sec 
60
0.0h 0.0min 47.30610108375549sec 
65
0.0h 0.0min 50.534579038619995sec 
70
0.0h 0.0min 53.76398015022278sec 
75
0.0h 0.0min 56.98878312110901sec 
80
0.0h 1.0min 0.19993901252746582sec 
85
0.0h 1.0min 3.4200291633605957sec 
90
0.0h 1.0min 6.643134117126465sec 
95
0.0h 1.0min 9.869309902191162sec 
100
0.0h 1.0min 13.084669351577759sec 
Maximum iteration of 20 reached
0.0h 1.0min 13.085405826568604sec                                                                                              
------------- FINISHED -------------                                                                                                                              
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

... newly loaded feature embeddings, which were not considered yet :  0
Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
------------- START -------------
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.                                                                   
  updates=self.state_updates,
2023-01-03 18:46:48.778979: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500                                                         
5
0.0h 0.0min 55.28749871253967sec 
10
0.0h 1.0min 3.7524306774139404sec 
15
0.0h 1.0min 12.380831956863403sec 
20
0.0h 1.0min 21.02756929397583sec 
25
0.0h 1.0min 29.51663827896118sec 
30
0.0h 1.0min 38.02130889892578sec 
35
0.0h 1.0min 46.581068992614746sec 
40
0.0h 1.0min 55.10542845726013sec 
45
0.0h 2.0min 3.653848886489868sec 
50
0.0h 2.0min 12.198934078216553sec 
Maximum iteration of 10 reached
0.0h 2.0min 12.199664115905762sec 
55
0.0h 1.0min 4.381495475769043sec 
60
0.0h 1.0min 13.225658178329468sec 
65
0.0h 1.0min 21.967401027679443sec 
70
0.0h 1.0min 30.68554377555847sec 
75
0.0h 1.0min 39.253119707107544sec 
80
0.0h 1.0min 47.87934494018555sec 
85
0.0h 1.0min 56.510223627090454sec 
90
0.0h 2.0min 5.109288215637207sec 
95
0.0h 2.0min 13.707428216934204sec 
100
0.0h 2.0min 22.312018632888794sec 
Maximum iteration of 20 reached
0.0h 2.0min 22.31281042098999sec
------------- FINISHED -------------
WARNING:tensorflow:From /home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/layers/normalization/batch_normalization.py:562: _colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
FeatureModel input shape:  (None, 299, 299, 1)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 299, 299, 8)       80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 149, 149, 8)      0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 149, 149, 8)      32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 149, 149, 16)      1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 74, 74, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 74, 74, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 74, 74, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 37, 37, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 37, 37, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 37, 37, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 18, 18, 64)       0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 18, 18, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 18, 18, 128)       73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 9, 9, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 10368)             0         
                                                                 
 dense (Dense)               (None, 256)               2654464   
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 2,787,108
Trainable params: 2,786,612
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 299, 299, 1)
------------- START -------------
5
0.0h 9.0min 25.16838026046753sec 
10
0.0h 18.0min 26.57681918144226sec 
15
0.0h 27.0min 29.143061637878418sec 
20
0.0h 36.0min 18.957569122314453sec 
25
0.0h 45.0min 9.48112177848816sec 
30
0.0h 53.0min 58.45214653015137sec 
35
1.0h 2.0min 49.09746241569519sec 
40
1.0h 11.0min 42.36852979660034sec 
45
1.0h 20.0min 30.770551443099976sec 
50
1.0h 29.0min 21.766809940338135sec 
Maximum iteration of 10 reached
1.0h 29.0min 21.76793384552002sec 
55
0.0h 9.0min 5.450965881347656sec 
60
0.0h 18.0min 5.822131633758545sec 
65
0.0h 27.0min 3.8890199661254883sec 
70
0.0h 35.0min 58.96794819831848sec 
75
0.0h 44.0min 58.39135122299194sec 
80
0.0h 53.0min 54.261783599853516sec 
85
1.0h 2.0min 49.592029094696045sec 
90
1.0h 11.0min 52.615742206573486sec 
95
1.0h 20.0min 51.82223129272461sec 
100
1.0h 29.0min 53.90738010406494sec 
Maximum iteration of 20 reached
1.0h 29.0min 53.90841507911682sec 
------------- FINISHED -------------
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
------------- START -------------
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.                                                                   
  updates=self.state_updates,
2023-01-03 21:08:27.631987: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500 
5
0.0h 6.0min 34.57836937904358sec 
10
0.0h 12.0min 36.559364795684814sec 
15
0.0h 18.0min 27.43905019760132sec 
20
0.0h 24.0min 18.22929859161377sec 
25
0.0h 30.0min 8.830620288848877sec 
30
0.0h 36.0min 5.342203378677368sec 
35
0.0h 41.0min 58.38518714904785sec 
40
0.0h 47.0min 50.99643564224243sec 
45
0.0h 53.0min 44.04136896133423sec 
50
0.0h 59.0min 37.192744970321655sec 
Maximum iteration of 10 reached
0.0h 59.0min 37.19395351409912sec 
55
0.0h 6.0min 13.83845067024231sec 
60
0.0h 12.0min 10.26806092262268sec 
65
0.0h 18.0min 4.964406490325928sec 
70
0.0h 23.0min 59.63564348220825sec 
75
0.0h 29.0min 51.594656467437744sec 
80
0.0h 35.0min 42.15387010574341sec 
85
0.0h 41.0min 37.252336740493774sec 
90
0.0h 47.0min 27.046546697616577sec 
95
0.0h 53.0min 18.18203377723694sec 
100
0.0h 59.0min 6.374322891235352sec 
Maximum iteration of 20 reached
0.0h 59.0min 6.37563157081604sec 

------------- FINISHED -------------
FeatureModel input shape:  (None, 299, 299, 1)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
/home/zimmer/masterthesis/code_mh_main/static/models/model_history_oct_cc_cnn_seed3871.hdf5  loading ...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 299, 299, 8)       80        
                                                                 
 max_pooling2d (MaxPooling2D  (None, 149, 149, 8)      0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 149, 149, 8)      32        
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 149, 149, 16)      1168      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 74, 74, 16)       0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 74, 74, 16)       64        
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 74, 74, 32)        4640      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 37, 37, 32)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 37, 37, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 37, 37, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 18, 18, 64)       0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 18, 18, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 18, 18, 128)       73856     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 9, 9, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 10368)             0         
                                                                 
 dense (Dense)               (None, 256)               2654464   
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_4 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
 activation_1 (Activation)   (None, 4)                 0         
                                                                 
=================================================================
Total params: 2,787,108
Trainable params: 2,786,612
Non-trainable params: 496
_________________________________________________________________
None
Model input shape:  (None, 299, 299, 1)
------------- START -------------
/home/zimmer/masterthesis/venv/lib/python3.9/site-packages/keras/engine/training_v1.py:2356: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  updates=self.state_updates,
2023-01-03 22:11:58.213121: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500
5
0.0h 29.0min 7.526078462600708sec 
10
0.0h 57.0min 58.347084045410156sec 
15
1.0h 27.0min 13.737595558166504sec 
20
1.0h 56.0min 5.291808128356934sec 
25
2.0h 25.0min 18.81308150291443sec 
30
2.0h 54.0min 14.703808784484863sec 
35
3.0h 23.0min 15.033632755279541sec 
40
3.0h 52.0min 18.090206146240234sec 
45
4.0h 21.0min 34.022427797317505sec 
50
4.0h 50.0min 6.892533540725708sec 
Maximum iteration of 10 reached
4.0h 50.0min 6.893486976623535sec 
55
0.0h 28.0min 51.306076765060425sec 
60
0.0h 57.0min 39.017192125320435sec 
65
1.0h 26.0min 20.87941026687622sec
70
1.0h 54.0min 38.0036780834198sec 
75
2.0h 23.0min 3.9691710472106934sec 
80
2.0h 51.0min 29.277474641799927sec 
85
3.0h 19.0min 49.559306144714355sec 
90
3.0h 48.0min 25.06050443649292sec 
95
0.0h 32.0min 40.3086462020874sec
100
1.0h 1.0min 42.1988365650177sec
Maximum iteration of 20 reached
1.0h 1.0min 42.199836015701294sec
------------- FINISHED -------------
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
------------- START -------------
FeatureModel input shape:  (None, 224, 224, 3)

==============
Current Dataset: oct_cc
Available Classes: ['DRUSEN', 'DME', 'NORMAL', 'CNV']
Length of Train Data: 72937
Length of Validation Data: 8091
Length of Test Data: 2528
==============

Initializing Image Generator ...
Found 72937 images belonging to 4 classes.
Found 8091 images belonging to 4 classes.
Found 2528 images belonging to 4 classes.
------------- START -------------
5
0.0h 16.0min 58.3949294090271sec
10
0.0h 33.0min 13.502985715866089sec
15
0.0h 49.0min 44.047619581222534sec
20
1.0h 5.0min 44.9956328868866sec
25
1.0h 22.0min 10.221977233886719sec
30
1.0h 38.0min 21.76308584213257sec
35
1.0h 54.0min 28.848475456237793sec
40
2.0h 10.0min 25.84551739692688sec
45
2.0h 26.0min 22.82482385635376sec
50
2.0h 42.0min 16.022125244140625sec
Maximum iteration of 10 reached
2.0h 42.0min 16.02311420440674sec
55
0.0h 16.0min 56.52380990982056sec
60
0.0h 33.0min 47.855441093444824sec
65
0.0h 50.0min 29.724642515182495sec
70
1.0h 7.0min 29.608109951019287sec
75
1.0h 24.0min 18.418936729431152sec
80
1.0h 40.0min 39.2689995765686sec
85
1.0h 57.0min 11.160041093826294sec
90
2.0h 13.0min 50.780476331710815sec
95
2.0h 30.0min 29.089734315872192sec
100
2.0h 47.0min 2.883349895477295sec
Maximum iteration of 20 reached
2.0h 47.0min 2.884471893310547sec
------------- FINISHED -------------
```

## 8. Evaluation
Please see my master thesis for the evaluation and the jupyter notebooks in the _main_code_ folder.


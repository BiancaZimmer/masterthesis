# Retina-AI

### Dataset Collection:
	
    Collect a dataset and split into directories for matching categories
	with an all uppercase name inside each of 3 parent directories labeled
	'train', 'test', and 'val'.

### For Example:
	When downloading the included OCT dataset, you will notice it contains
	2 folders (train, test). First, you must move some images from the training or testing sets into a new folder called 'val'. Inside each are 4 folders with their
	corresponding category (CNV, DME, DRUSEN, NORMAL). The image path expected
	at command-line matches that of the folder containing the first 3 folders
	(train, test, val).

### Sample Usage:
```python retrain.py --images /path/to/images ```

### Generating ROC:
  Uncomment the last few lines in the main function of the retrain.py file. Change [LIST_OF_POS_IDX] with
  a list of indices of the positive categories (per the output_labels.txt file). Run the script.

### Occlusion:
```
        python occlusion.py
                --image_dir /path/to/image
                --graph /tmp/output_graph.pb
                --labels /tmp/output_labels.txt
```
----------- 
## Added by Bianca:

Changed the code so it will run in parallel (walking through paths is no done in
alphabetical/sorted order).

### Sample usage:
```
python retrain.py
    --images "/Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small3"
    --output_graph "/Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small3/results/retrained_graph_1.pb"
    --output_labels "/Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small3/results/output_labels.txt"
    --summaries_dir "/Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small3/results/retrain_logs1"
    --bottleneck_dir "/Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small3/bottleneck"
    --training_steps 200
```

On hydra:
```
python3 retrain.py
    --images "/home/zimmer/data/data_kermany_small3"
    --output_graph "/home/zimmer/data/data_kermany_small3/results/retrained_graph_small.pb"
    --output_labels "/home/zimmer/data/data_kermany_small3/results/output_labels.txt"
    --summaries_dir "/home/zimmer/data/data_kermany_small3/results/retrain_logs_small"
    --bottleneck_dir "/home/zimmer/data/data_kermany_small3/bottelneck/"
    --training_steps 200
```

To view Tensorboard go into the results folder and type the following into the terminal:

``` tensorboard --logdir retrain_logs1```

```python retrain.py --help``` </br>
usage: retrain.py [-h] [--images IMAGES] [--output_graph OUTPUT_GRAPH]
                  [--output_labels OUTPUT_LABELS]
                  [--summaries_dir SUMMARIES_DIR]
                  [--training_steps TRAINING_STEPS]
                  [--learning_rate LEARNING_RATE]
                  [--eval_frequency EVAL_FREQUENCY]
                  [--train_batch_size TRAIN_BATCH_SIZE]
                  [--test_batch_size TEST_BATCH_SIZE]
                  [--validation_batch_size VALIDATION_BATCH_SIZE]
                  [--model_dir MODEL_DIR] [--bottleneck_dir BOTTLENECK_DIR]
                  [--final_tensor_name FINAL_TENSOR_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --images IMAGES       Path to folder containing subdirectories of the
                        training categories (filesnames all CAPS)
  --output_graph OUTPUT_GRAPH
                        Output directory to save the trained graph.
  --output_labels OUTPUT_LABELS
                        Directory in which to save the labels.
  --summaries_dir SUMMARIES_DIR
                        Path to save summary logs for TensorBoard.
  --training_steps TRAINING_STEPS
                        How many training steps to run before ending.
  --learning_rate LEARNING_RATE
                        Set learning rate
  --eval_frequency EVAL_FREQUENCY
                        How often to evaluate the training results.
  --train_batch_size TRAIN_BATCH_SIZE
                        How many images to train on at a time.
  --test_batch_size TEST_BATCH_SIZE
                        Number of images from test set to test on. Value of -1
                        will cause entire directory to be used. Since it is
                        used only once, -1 will work in most cases.
  --validation_batch_size VALIDATION_BATCH_SIZE
                        Number of images from validation set to validate on.
                        Value of -1 will cause entire directory to be used.
                        Large batch sizes may slow down training size it is
                        performed frequently.
  --model_dir MODEL_DIR
                        Path to pretrained weights
  --bottleneck_dir BOTTLENECK_DIR
                        Path to store bottleneck layer values.
  --final_tensor_name FINAL_TENSOR_NAME
                        The name of the output classification layer in the
                        retrained graph.

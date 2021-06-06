# NOSES Project

## Main Research Question
Can we use machine learning on drone data to obtain estimates for the seal population?

## Data Storage
All data used for this project are stored in Cal Poly’s f35 servers under **/data2/noses_data**

## The Data
Images: We collected drone images of elephant seals from http://www.anonuevoresearch.com/data/elephant-seal

Bounding Box Data: We tagged each of the collected images with bounding boxes using a labeling tool from https://github.com/tzutalin/labelImg

The images and bounding box data are stored locally in **/images_data** and on the servers in **/data2/noses_data/images_data/**
- **DJI_xxxx.JPG** files are the images
- **DJI_xxxx.xml** files are the bounding box data

## The Code
#### split_and_label.py
- Takes in the images and the corresponding bounding box data as from *path_data*
- Splits the images into smaller sub images and generates its corresponding bounding box data
- Saves the sub images and bounding box data to the *path_images*.  Three different sets of data are stored, small, medium, and large:
  - Small - 1000 images: **images_small.npy** and **bb_data_small.npy**
  - Medium - 5000 images: **images_medium.npy** and bb_data_small.npy**
  - Large - all images: **images.npy** and **bb_data.npy**

* The small and medium datasets are used for testing the functionality of the code while making updates while the large dataset is used in an attempt to train high performing models.

#### funcs_train.py
Contains the functions used in train.py
Each function contains a short header that describes it’s input and output.

#### train.py
Performs a comprehensive grid search over different hyperparameters to find the best model for predicting the presence of a seal in an image.  We have two different modes of classificatio, binary and tertiary.  Please note that we encountered some errors while running tertiary classification.  There may exist bugs in the code that need fixing in order to get tertiary classification working properly.  The descriptions below give further context into how we are performing the two different methods of classification:
- Binary: classification for whether an image contains a seal or not.  Classification labels: 0 == “no seal”, 1 == “ful seal”
  - “No seal”: any image where the percentage of seal found in the image is 0
  - "Full seal": any image where the percentage of seal found in the image is greater than threshold_min
- Tertiary: classification for whether an image doesn’t contain a seal, contains a partial seal, or contains a full seal.  Classification labels: 0 == “no seal”, 1 == “partial seal”, 2 == “full seal”
  - “No seal”: any image where the percentage of seal found in the image is between 0 and threshold_min
  - “Partial seal”: any image where the percentage of seal found in the image is between threshold_min and threshold_max
  - “Full seal”: any image where the percentage of seal found in the image is greater than threshold_max

The “User Input” section at the top is where you can specify various parameters including:
- *path_data*: the path for the input training data and labels to the cnn
- *path_plots*: the path for the outputted bin plots
- *size*: the size of the dataset
- *binary_flag*: true for binary classification, false for tertiary classification 
- *full_grid_search*: true to run a full grid search, false to run an arbitrary model
- *threshold_min_grid*: grid for threshold_min, the minimum percentage to indicate the presence of a seal
- *threshold_max_grid*: grid for threshold_max, the maximum percentage to indicate the presence of a partial seal
- *cnn_blocks_grid*: grid for the number of Conv-Conv-Pool (CCP) blocks
- *dense_layers_grid*: grid for the number of dense layers
- *filter_mult_grid*: grid for the filter multiplier, a multiplier that determines the dimensionality of the output space for the convolutional layers
- *kernel_size_grid*: grid for the kernel size, which specifies the height and width of the convolution window
- *strides_grid*: grid for the strides, which specifies the strides of the convolution along the height and width
- *dropout_flag_grid*: grid for dropout flag, a boolean flag indicating whether to add a dropout layer 

## Training Output
Information about each model trained in the grid search and the test metrics are printed to stdout.


The bin plot of the (percentage of images predicted as a seal) vs (the percentage of the seal in the image) are added to **/cnn_plots**.  These plots are also desribed in **final_report.pdf**.

## Running the Docker

The bash commands shown below demonstrate how to build and run the docker container to start running **train.py**.

jdavi104@f35:~/noses $ ./build_docker.sh 

...

jdavi104@f35:~/noses $ ./start_docker.sh 

...

You are running this container as user with ID 3070646 and group 9999,
which should map to the ID and group for your user on the Docker host. Great!

tf-docker / > cd noses/

tf-docker /noses > python3 train.py 

## Future Work
Future research will involve making the transition from binary classification to tertiary classification.  Tertiary classification proves much more difficult than binary classification.  How is the network to distinguish between a seal that is partially in the image from a seal that is fully in the image?  How can we manipulate the threshold_min and threshold_max levels to understand to better understand and utilize the network’s decision-making process?  This is the next major step towards answering our research question - can we use machine learning on drone data to obtain estimates for the seal population?




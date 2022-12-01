from funcs_train import *
import argparse
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

#------------------- GETTING INPUT -----------------
parser = argparse.ArgumentParser(description = "Min and Max")
parser.add_argument("-l", required = True, help= "Minimum Threshold")
parser.add_argument("-u", required = True, help = "Maximum Threshold")
args = parser.parse_args()

#-------------------- USER INPUT --------------------

path_data = "/data2/noses_data/cnn_data/"
path_plots = "/noses/cnn_plots/"


# binary or tertiary classificaiton
binary_flag = True
full_grid_search = True

# grid search parameters
# note for threshold_min_grid: probably want to be bigger for binary and small for tertiary
if full_grid_search:
    threshold_min_grid = [float(args.l)]
    threshold_max_grid = [float(args.u)]
    cnn_blocks_grid = [5]
    dense_layers_grid = [7]
    filter_mult_grid = [0.5, 1]
    kernel_size_grid = [2, 3]
    strides_grid = [(2, 2), (3, 3)]
    dropout_flag_grid = [True]
else:
    threshold_min_grid = [float(args.l)]
    threshold_max_grid = [float(args.u)]
    cnn_blocks_grid = [3]
    dense_layers_grid = [2]
    filter_mult_grid = [.5]
    kernel_size_grid = [3]
    strides_grid = [(2, 2)]
    dropout_flag_grid = [True]


#-------------------- RETRIEVE DATA --------------------
imgs = np.load("/data2/noses_data/cnn_data/images.npy", allow_pickle=True)
bb_data = np.load("/data2/noses_data/cnn_data/bb_data.npy", allow_pickle=True)
print(bb_data[0])
#-------------------- GRID SEARCH --------------------
"""Parameters:
- Threshold Min: minimum percentage to indicate the presence of a seal
- Threshold Max: maximum percentage to indicate the presence of a partial seal
- Number of Conv-Conv-Pool (CCP) blocks
- Number of Dense layers
- Filter Multiplier: multiplier to determine the dimensionality of the output space
for the convolutional layers
- Kernel Size: specifies height and width of convolution window
- Strides: specifies the strides of the convolution along the height and width
- Dropout Flag: flag indicating whether to dropout
"""

if binary_flag:
    threshold_max_grid = ["Not Applicable"]

model_params_grid = list(itertools.product(threshold_min_grid, threshold_max_grid, cnn_blocks_grid,
                                        dense_layers_grid, filter_mult_grid,
                                        kernel_size_grid, strides_grid,
                                        dropout_flag_grid))
first_pass = True
prev_threshold_min = threshold_min_grid[0]
prev_threshold_max = threshold_max_grid[0]
model_num = 480
for model_params in model_params_grid:
    threshold_min = model_params[0]
    threshold_max = model_params[1]
    cnn_blocks = model_params[2]
    dense_layers = model_params[3]
    filter_mult = model_params[4]
    kernel_size = model_params[5]
    strides = model_params[6]
    dropout_flag = model_params[7]


    label_0 = []
    img_0 = []
    bb_data_0 = []
    label_1 = []
    img_1 = []
    bb_data_1 = []
    label_2 = []
    img_2 = []
    bb_data_2 = []
    

    labels = get_labels_binary(bb_data, threshold_min)
    #labels = get_labels_tertiary(bb_data, threshold_min, threshold_max)
    # Separate images based on label 
    for x in range(len(labels)):
       if labels[x] == 0:
          label_0.append(labels[x])
          img_0.append(imgs[x])
          bb_data_0.append(bb_data[x])
       elif labels[x] == 1:
          label_1.append(labels[x])
          img_1.append(imgs[x])
          bb_data_1.append(bb_data[x])
       elif labels[x] == 2:
          label_2.append(labels[x])
          img_2.append(imgs[x])
          bb_data_2.append(bb_data[x])


    used_labels = []
    used_imgs = []
    used_bb_data = []


    print("No Seals", len(label_0), len(label_0)/ len(labels))
    print("Partial Seals", len(label_1), len(label_1)/ len(labels))
    print("Full Seals", len(label_2), len(label_2)/ len(labels))

    # Randomly selecting 25,000 no seal images
    sample_no_seal = random.sample(range(len(label_0)), 15000)
    # Add sampled data to used labels, imgs, and bb_data
    for x in sample_no_seal:
       used_labels.append(label_0[x])
       used_imgs.append(img_0[x])
       used_bb_data.append(bb_data_0[x])
    used_img0 = len(used_imgs)
    print("no seals:", used_img0)
    '''
    # Randomly selecting 25,000 no seal images
    sample_no_seal = random.sample(range(len(label_1)), 4000)
    # Add sampled data to used labels, imgs, and bb_data
    for x in sample_no_seal:
       used_labels.append(label_1[x])
       used_imgs.append(img_1[x])
       used_bb_data.append(bb_data_1[x])
    
    # Randomly selecting 25,000 no seal images
    sample_no_seal = random.sample(range(len(label_2)), 4000)
    # Add sampled data to used labels, imgs, and bb_data
    for x in sample_no_seal:
       used_labels.append(label_2[x])
       used_imgs.append(img_2[x])
       used_bb_data.append(bb_data_2[x])
    '''
    # Add approx 12500 partial seal images
    for x in range(len(label_1)):
       current_img = img_1[x]
       used_imgs.append(current_img)
       used_labels.append(label_1[x])
       used_bb_data.append(bb_data_1[x])
       '''# Flip image
       flipped_img = tf.image.flip_left_right(current_img)
       used_imgs.append(flipped_img)
       used_labels.append(label_1[x])
       used_bb_data.append(bb_data_1[x])'''
       '''# Rotate 180
       rotated_img = tf.image.rot90(tf.image.rot90(current_img))
       used_imgs.append(rotated_img)
       used_labels.append(label_2[x])
       used_bb_data.append(bb_data_2[x])'''
    print("partial seals:", len(used_labels) - used_img0)

    # Add approx 12500 full seal images
    for x in range(len(label_2)):
       current_img = img_2[x]
       used_imgs.append(current_img)
       used_labels.append(label_2[x])
       used_bb_data.append(bb_data_2[x])
       '''# Flip image
       flipped_img = tf.image.flip_left_right(current_img)
       used_imgs.append(flipped_img)
       used_labels.append(label_2[x])
       used_bb_data.append(bb_data_2[x])
       '''


    print("Total labels")
    print(len(used_labels))
    print("Starting test train split")
    x_train, x_test, y_train, y_test = train_test_split(used_imgs, used_labels, test_size = .1, random_state=1) 
    ''' 
    #creating test train split
    total_0 = []
    total_1 = []
    total_2 = []
    
    #Sorting for test train
    for i in range(len(used_imgs)):
      if used_labels[i] == 0:
         total_0.append(i)
      elif used_labels[i] == 1:
         total_1.append(i)
      else:
         total_2.append(i)

    
    #test for no seals
    train_0, test_0 = train_test(total_0)
    #test for partial seals
    train_1, test_1 = train_test(total_1)
    #test for partial seals
    train_2, test_2 = train_test(total_2)


    test_indices = test_0 + test_1 + test_2
    train_indices = train_0 + train_1 + train_2
    '''
    train_test_dict = {}
    #train_test_dict["Xtrain"], train_test_dict["ytrain"] = preprocessing([used_imgs[i] for i in train_indices], [used_labels[i] for i in train_indices])
    #train_test_dict["Xtest"], train_test_dict["ytest"] = preprocessing([used_imgs[i] for i in test_indices], [used_labels[i] for i in test_indices])
    print("Preprocessing")
    train_test_dict["Xtrain"], train_test_dict["ytrain"] = preprocessing(x_train, y_train)
    train_test_dict["Xtest"], train_test_dict["ytest"] = preprocessing(x_test, y_test)
    '''
    # get a new train test split on first run through or if the thresholds change
    if first_pass == True or prev_threshold_min != threshold_min or prev_threshold_max != threshold_max:
        train_test_dict = train_test_split(used_imgs, used_bb_data, used_labels, threshold_min)
        first_pass = False
    '''
    print("Starting training")
    ypred = run_model(train_test_dict,
                        threshold_min, threshold_max, cnn_blocks, dense_layers, filter_mult, kernel_size,
                        strides, dropout_flag, model_num)
    print("Finished Training")
    ytest = train_test_dict["ytest"]
    #ytest_no_filter = train_test_dict["ytest_no_filter"]
    #seal_percents = train_test_dict["seal_percents"]
    #seal_percents_no_filter = train_test_dict["seal_percents_no_filter"]
    ypred_, ytest_ = convert_arrs(ypred, ytest, binary_flag)
    #ypred_no_filter_, ytest_no_filter_ = convert_arrs(ypred_no_filter, ytest_no_filter, binary_flag)
    print_metrics(ypred_, ytest_)
    print_confusion_matrix(ypred_, ytest_, binary_flag)
    #plot_buckets(ypred_no_filter_, ytest_no_filter_, seal_percents_no_filter, threshold_min, threshold_max, binary_flag, path_plots + str(model_num))
    model_num += 1


#-------------------- OLD CODE --------------------
"""
# Evaluation

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss over time')
plt.legend(['train','val'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('iteration')
plt.ylabel('acc')
plt.title('Accuracy over time')
plt.legend(['train','val'])
plt.show()
"""

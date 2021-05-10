from funcs_train import *

path_data = "/data2/noses_data/cnn_data/"
# path_plots = "/data2/noses_data/cnn_plots/"
path_plots = "/noses/cnn_plots/"

""" User Input """
# size of dataset - "small", "medium", or "large"
size = "small"

# binary or tertiary classificaiton
binary_flag = True

# grid search parameters
threshold_min_grid = [.3] # [0, .1, .2, .3, .4, .5]
cnn_blocks_grid = [3]
dense_layers_grid = [1]
filter_mult_grid  = [.5]
kernel_size_grid = [2]
strides_grid = [(5, 5)]
dense_size_grid = [128]
dropout_flag_grid = [True, False]
#dp_rate_grid = [.1, .2, .3, .4, .5]

# threshold_min_grid = [0, .1, .2, .3, .4, .5]
# cnn_blocks_grid = [1, 2, 3]
# dense_layers_grid = [1, 2, 3]
# filter_multiplier_grid  = [.5, 1, 2]
# kernel_size_grid = [2, 3, 4]
# strides_grid = [(1, 1), (2, 2), (3, 3)]
# dense_output_size_grid = [1024, 2048, 4096]

""" End of User Input """

""" Load in images and bounding box meta data """
if size == "small": f1 = "images_small.npy"; f2 = "bb_data_small.npy"
if size == "medium": f1 = "images_medium.npy"; f2 = "bb_data_medium.npy"
if size == "large": f1 = "images.npy"; f2 = "bb_data.npy"

imgs = np.load(path_data + f1, allow_pickle=True)
bb_data = np.load(path_data + f2, allow_pickle=True)

# print(imgs.shape)
"""  Extract labels """
if binary_flag == True:
    labels = get_labels_binary(bb_data)
else:
    labels = get_labels_tertiary(bb_data)

"""  Grid Search
Paramterers:
- Threshold Min: minimum percentage to indicate the presence of a seal
- Number of Conv-Conv-Pool (CCP) blocks
- Number of Dense layers
- Kernel Size: specifies height and width of convolution window
- Strides: specifies the strides of the convolution along the height and width
- Dense Ouput Size: size of output space for the dense layer(s)
"""

model_params_grid = list(itertools.product(threshold_min_grid, cnn_blocks_grid,
                                        dense_layers_grid, filter_mult_grid,
                                        kernel_size_grid, strides_grid,
                                        dense_size_grid, dropout_flag_grid))
first_pass = True
prev_threshold_min = threshold_min_grid[0]
model_num = 31
for model_params in model_params_grid:
    threshold_min = model_params[0]
    cnn_blocks = model_params[1]
    dense_layers = model_params[2]
    filter_mult = model_params[3]
    kernel_size = model_params[4]
    strides = model_params[5]
    dense_size = model_params[6]
    dropout_flag = model_params[7]

    if first_pass == True or prev_threshold_min != threshold_min:
        train_test_dict, seal_percents = train_test_split(imgs, bb_data, labels, threshold_min)
        first_pass = False

    ypred, ypred_no_filter = run_model(train_test_dict,
                        threshold_min, cnn_blocks, dense_layers, filter_mult, kernel_size,
                        strides, dense_size, dropout_flag, model_num)
    ytest = train_test_dict["ytest"]
    ytest_no_filter = train_test_dict["ytest_no_filter"]
    ypred_, ytest_ = print_metrics(ypred, ytest, False, binary_flag)
    print_metrics(ypred_no_filter, ytest_no_filter, True, binary_flag)
    print_confusion_matrix(ypred_, ytest_, binary_flag)
    plot_acc_buckets(ytest_, ypred_, seal_percents, path_plots + str(model_num))
    model_num += 1


"""## Evaluation"""

"""
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

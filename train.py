from funcs_train import *

#-------------------- USER INPUT --------------------

path_data = "/data2/noses_data/cnn_data/"
path_plots = "/noses/cnn_plots/"

# size of dataset - "small", "medium", or "large"
size = "large"

# binary or tertiary classificaiton
binary_flag = False
full_grid_search = True

# grid search parameters
# note for threshold_min_grid: probably want to be bigger for binary and small for tertiary
if full_grid_search:
    threshold_min_grid = [.3]
    threshold_max_grid = [.7]
    cnn_blocks_grid = [4]
    dense_layers_grid = [3, 4, 5]
    filter_mult_grid = [0.5, 1]
    kernel_size_grid = [2, 3]
    strides_grid = [(2, 2), (3, 3)]
    dropout_flag_grid = [True]
else:
    threshold_min_grid = [.3]
    threshold_max_grid = [.7]
    cnn_blocks_grid = [3]
    dense_layers_grid = [2]
    filter_mult_grid = [.5]
    kernel_size_grid = [3]
    strides_grid = [(2, 2)]
    dropout_flag_grid = [True]


#-------------------- RETRIEVE DATA --------------------
if size == "small": f1 = "images_small.npy"; f2 = "bb_data_small.npy"
if size == "medium": f1 = "images_medium.npy"; f2 = "bb_data_medium.npy"
if size == "large": f1 = "images.npy"; f2 = "bb_data.npy"

imgs = np.load(path_data + f1, allow_pickle=True)
bb_data = np.load(path_data + f2, allow_pickle=True)


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
    # print(model_num)
    threshold_min = model_params[0]
    threshold_max = model_params[1]
    cnn_blocks = model_params[2]
    dense_layers = model_params[3]
    filter_mult = model_params[4]
    kernel_size = model_params[5]
    strides = model_params[6]
    dropout_flag = model_params[7]

    if binary_flag == True:
        labels = get_labels_binary(bb_data, threshold_min)
    else:
        labels = get_labels_tertiary(bb_data, threshold_min, threshold_max)

    # get a new train test split on first run through or if the thresholds change
    if first_pass == True or prev_threshold_min != threshold_min or prev_threshold_max != threshold_max:
        train_test_dict = train_test_split(imgs, bb_data, labels, threshold_min)
        first_pass = False

    ypred, ypred_no_filter = run_model(train_test_dict,
                        threshold_min, threshold_max, cnn_blocks, dense_layers, filter_mult, kernel_size,
                        strides, dropout_flag, model_num)
    ytest = train_test_dict["ytest"]
    ytest_no_filter = train_test_dict["ytest_no_filter"]
    seal_percents = train_test_dict["seal_percents"]
    seal_percents_no_filter = train_test_dict["seal_percents_no_filter"]
    ypred_, ytest_ = convert_arrs(ypred, ytest, binary_flag)
    ypred_no_filter_, ytest_no_filter_ = convert_arrs(ypred_no_filter, ytest_no_filter, binary_flag)
    print_metrics(ypred_, ytest_)
    print_confusion_matrix(ypred_, ytest_, binary_flag)
    plot_buckets(ypred_no_filter_, ytest_no_filter_, seal_percents_no_filter, threshold_min, threshold_max,
                 binary_flag, path_plots + str(model_num))
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

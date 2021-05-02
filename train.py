from funcs_train import *

path_data = "/data2/noses_data/cnn_data/"

""" Load in images and bounding box meta data """

imgs = np.load(path_data + "images_medium.npy", allow_pickle=True)
bb_data = np.load(path_data + "bb_data_medium.npy", allow_pickle=True)
print("images shape:", imgs.shape)
print("bb_data shape:", bb_data.shape)

"""  Extract labels """
labels = get_labels_binary(bb_data)


"""## Grid Search
Paramterers:
- Number of Conv-Conv-Pool (CCP) blocks
- Number of Dense layers
- Kernel Size: specifies height and width of convolution window
- Strides: specifies the strides of the convolution along the height and width
- Dense Ouput Size: size of output space for the dense layer(s)
"""

# cnn_blocks_grid = [1, 2, 3]
# dense_layers_grid = [1, 2, 3]
# filter_multiplier_grid  = [.5, 1, 2]
# kernel_size_grid = [2, 3, 4]
# strides_grid = [(1, 1), (2, 2), (3, 3)]
# dense_output_size_grid = [1024, 2048, 4096]

cnn_blocks_grid = [3]
dense_layers_grid = [1]
filter_mult_grid  = [.5]
kernel_size_grid = [2]
strides_grid = [(5, 5)]
dense_size_grid = [128]
threshold_min_grid = [.1]

model_params_grid = list(itertools.product(cnn_blocks_grid, dense_layers_grid,
                                        filter_mult_grid, kernel_size_grid,
                                        strides_grid, dense_size_grid,
                                        threshold_min_grid))
for model_params in model_params_grid:
    cnn_blocks = model_params[0]; dense_layers = model_params[1]; filter_mult = model_params[2]
    kernel_size = model_params[3]; strides = model_params[4]; dense_size= model_params[5]
    threshold_min = model_params[6]
    Xtrain, ytrain, Xtest, ytest, seal_percents = train_test_split(imgs, bb_data, labels, threshold_min)
    ypred = run_model(Xtrain, ytrain, Xtest, ytest,
                        cnn_blocks, dense_layers, filter_mult, kernel_size,
                        strides, dense_size, threshold_min)
    ypred_, ytest_ = print_metrics(ypred, ytest)

fname = "acc_buckets_plot.png"
plot_acc_buckets(ytest_, ypred_, seal_percents, fname)

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

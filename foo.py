import pandas as pd
from matplotlib import pyplot as plt
import os
import sys
import itertools
from sklearn.metrics import confusion_matrix

cnn_blocks_grid = [1]
dense_layers_grid = [1]
filter_mult_grid  = [.5]
kernel_size_grid = [2]
strides_grid = [(5, 5)]
dense_size_grid = [128]
threshold_min_grid = [.1]

model_params_grid = list(itertools.product(threshold_min_grid, cnn_blocks_grid, dense_layers_grid,
                                        filter_mult_grid, kernel_size_grid,
                                        strides_grid, dense_size_grid
                                        ))

for model_params in model_params_grid:
    s = ""
    for param in model_params:
        s += str(param) + "-"
    print(s)

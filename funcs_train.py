# standard
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
import itertools

# cnn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# results
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# file system
import os

LINE_LEN = 26

""" Training """
lr = 3e-4
batch_size = 32
conv_dim_init = 64
epochs = 20


def create_model(cnn_blocks=1, dense_layers=1, filter_multiplier = 1, kernel_size=3,
                 strides=(1, 1), dense_output_size =1024, classification="binary"):
  model = keras.models.Sequential()
  for i in range(cnn_blocks):
    # conv_output_dim = int((conv_dim_init * filter_multiplier) * (i + 1))
    conv_output_dim = int((conv_dim_init * filter_multiplier) * (2 ** i))
    model.add(layers.Conv2D(filters=conv_output_dim, kernel_size=kernel_size,
                            strides=strides, activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=conv_output_dim, kernel_size=kernel_size,
                            strides=strides, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

  model.add(layers.Flatten())
  for i in range(dense_layers):
    model.add(layers.Dense(units=dense_output_size , activation='relu'))

  if classification == 'binary':
      model.add(layers.Dense(1, activation='sigmoid', name='z'))
  else:
      model.add(layers.Dense(3, activation='softmax',name='z'))

  opt = tf.keras.optimizers.Adam(lr=lr) #sgd
  if classification == 'binary':
      model.compile(loss='binary_crossentropy',optimizer=opt,metrics='accuracy')
  else:
      model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics='accuracy')
  return model


def run_model(Xtrain, ytrain, Xtest, ytest,
              cnn_blocks, dense_layers, filter_multiplier, kernel_size, strides,
              dense_output_size, threshold_min):
  model = create_model(cnn_blocks, dense_layers, filter_multiplier, kernel_size, strides, dense_output_size)
  model.fit(Xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=False)
  # print(model.summary())
  res = model.evaluate(Xtest, ytest); loss = res[0]; acc = res[1]
  ypred = model.predict(Xtest)
  print_model_params(cnn_blocks, dense_layers, filter_multiplier, kernel_size,
                        strides, dense_output_size, threshold_min)
  return ypred


def print_model_params(cnn_blocks, dense_layers, filter_multiplier, kernel_size,
                        strides, dense_output_size, threshold_min):
    line = " " + "-" * LINE_LEN; len_line = len(line)
    print(line)
    s = "| Model Parameters"; print(get_string(s, len_line))
    print(line)
    s = "| CCP Block(s) %8d" % cnn_blocks; print(get_string(s, len_line))
    s = "| Dense Layer(s) %6d" % dense_layers; print(get_string(s, len_line))
    s = "| Filter Multiplier %3dx" % filter_multiplier; print(get_string(s, len_line))
    s = "| Kernel Size %9d" % kernel_size; print(get_string(s, len_line))
    s = "| Strides %13d" % strides[0]; print(get_string(s, len_line))
    s = "| Dense Output Size %5d" % dense_output_size; print(get_string(s, len_line))
    s = "| Threshold Min" + "\t     " + '{:.2f}'.format(threshold_min).lstrip('0'); print(get_string(s, len_line))
    print(line)

def print_metrics(ypred, ytest):
  ypred_, ytest_ = convert_arrs(ypred, ytest)
  acc = accuracy_score(ytest_, ypred_)
  f1 = f1_score(np.array(ytest_), ypred_, labels=np.unique(ytest_), average="weighted")
  precision = precision_score(np.array(ytest_), ypred_, labels=np.unique(ypred_), average="weighted")
  recall = recall_score(np.array(ytest_), ypred_, labels=np.unique(ypred_), average="weighted")

  line = " " + "-" * LINE_LEN; len_line = len(line)
  print(line)
  s = "| Test Metrics"; print(get_string(s, len_line))
  print(line)
  s = "| Accuracy %14.4f" % acc; print(get_string(s, len_line))
  s = "| F1 Score %14.4f" % f1; print(get_string(s, len_line))
  s = "| Precision %13.4f" % precision; print(get_string(s, len_line))
  s = "| Recall %16.4f" % recall; print(get_string(s, len_line))
  print(line); print(line)
  s = "| Confusion Matrix"; print(get_string(s, len_line))
  cm = confusion_matrix(ytest_, ypred_)
  print_cm(cm, labels=["0", "1"])
  return ypred_, ytest_


""" Train-Test Split """
def train_test_split(imgs, bb_data, labels, threshold_min):
    train_indices, test_indices, seal_percents = get_indices_and_percents(bb_data, threshold_min)
    Xtrain, ytrain = preprocessing(imgs[train_indices], labels[train_indices])
    Xtest, ytest = preprocessing(imgs[test_indices], labels[test_indices])
    return Xtrain, ytrain, Xtest, ytest, seal_percents


def get_indices_and_percents(bb_data, threshold_min, test_frac = .1):
  # get indices for images w seals
  indices_w_seal = []
  seal_percents = []
  for i in range(len(bb_data)):
    df_subimg = bb_data[i]
    if not df_subimg is None:
      seal_percent = get_seal_percent(df_subimg)
      if seal_percent > threshold_min:
          indices_w_seal.append(i)
          seal_percents.append(seal_percent)

  num_w_seal = len(indices_w_seal)
  size_dataset = int(num_w_seal * 10/4) # size for dataset with 40% of images containing seals
  num_wo_seal = size_dataset - num_w_seal

  # get indices for images w/o seals
  all_indices_wo_seal = [x for x in list(range(len(bb_data))) if x not in indices_w_seal]
  indices_wo_seal = random.sample(all_indices_wo_seal, num_wo_seal)

  print("Number of sub images for dataset with 40% images containing seals", size_dataset)
  print("Number of sub images with a seal", num_w_seal)
  print("Number of sub images without a seal", num_wo_seal)
  print()

  num_train_w_seal = round((1 - test_frac) * num_w_seal)
  num_train_wo_seal = round((1 - test_frac) * num_wo_seal)
  num_test_wo_seal = size_dataset - num_w_seal - num_train_wo_seal

  seal_percents = np.array(seal_percents[num_train_w_seal:])
  seal_percents = np.pad(seal_percents, (0, num_test_wo_seal), 'constant')
  # train test split of indices, each with 40% of sub imgs containing seals
  train_indices = np.concatenate((np.array(indices_w_seal[:num_train_w_seal]),
                                  np.array(indices_wo_seal[:num_train_wo_seal])))
  test_indices = np.concatenate((np.array(indices_w_seal[num_train_w_seal:]),
                                  np.array(indices_wo_seal[num_train_wo_seal:])))
  return train_indices, test_indices, seal_percents


def plot_acc_buckets(ytest, ypred, percents, fname):
    acc_dict = {}
    buckets = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]
    for bucket in buckets:
        acc_dict[bucket] = {"total": 0, "sum": 0}
    for elem in zip(ytest, ypred, percents):
        label = elem[0]; prediction = elem[1]; percent = elem[2]
        if percent < .1: bucket = "0-1"
        elif percent < .2: bucket = "1-2"
        elif percent < .3: bucket = "2-3"
        elif percent < .4: bucket = "3-4"
        elif percent < .5: bucket = "4-5"
        elif percent < .6: bucket = "5-6"
        elif percent < .7: bucket = "6-7"
        elif percent < .8: bucket = "7-8"
        elif percent < .9: bucket = "8-9"
        else: bucket = "9-10"
        if prediction == label:
            acc_dict[bucket]["sum"] += 1
        acc_dict[bucket]["total"] += 1

    for bucket, dict in acc_dict.items():
        if dict['total'] == 0:
            acc = 0
        else:
            acc = dict['sum'] / dict['total']
        acc_dict[bucket] = acc

    buckets = list(acc_dict.keys())
    acc = list(acc_dict.values())
    plt.bar(range(len(acc_dict)), acc, tick_label=buckets)
    plt.savefig(fname)
    return acc_dict

def preprocessing(X_init, y_init):
  X_init = X_init[:] / 255.
  y_init = y_init[:] / 1.

  X = []
  y = []
  for Xi, yi in zip(X_init, y_init):
    im = tf.image.resize(Xi, (96, 96))
    X.append(im)
    y.append(yi)

  X = tf.stack(X)
  y = tf.stack(y)
  return X, y

# given the df for a subimg, returns the highest percentage seal found
def get_seal_percent(df_subimg):
    seal_percents = []
    for i in df_subimg.itertuples():
      seal_percents.append(i[3])
    return max(seal_percents)


# convert ypred back to predictions of 0, 1, 2
# convert ytest from tensor to list
def convert_arrs(ypred, ytest):
  ypred_ = []
  for i in range(len(ypred)):
    confidence_arr = list(ypred[i])
    ypred_.append(np.argmax(confidence_arr))
  ytest_ = []
  for i in range(len(ytest)):
    ytest_.append(int(ytest[i]))
  return ypred_, ytest_


""" Extract Labels """
def get_labels_tertiary(bb_data, threshold = 1):
  labels = []
  for df_subimg in bb_data:
    if not isinstance(df_subimg, type(None)):
      seal_percent = get_seal_percent(df_subimg)
      if seal_percent > threshold:
        val = 2
      else:
        val = 1
    else:
      val = 0
    labels.append(val)
  labels = np.array(labels)
  return labels


def get_labels_binary(bb_data):
    labels = []
    for df_subimg in bb_data:
      if not isinstance(df_subimg, type(None)):
          val = 1
      else:
          val = 0
      labels.append(val)
    labels = np.array(labels)
    return labels

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    line = " " + "-" * 26
    print(line)
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("|    " + fst_empty_cell, end=" ")
    # End CHANGES

    s_end = "\t   |"
    for label in labels:
        if label == labels[len(labels) - 1]:
            print("%{0}s".format(columnwidth) % label, end=s_end)
        else:
            print("%{0}s".format(columnwidth) % label, end=" ")


    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("|    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            if j == 1:
                print(cell, end=s_end)
            else:
                print(cell, end=" ")
        print()
    print(line)


def get_string(s, len_line):
    num_spaces = len_line - len(s)
    return s + ' ' * num_spaces + "|"

  # given X and y data, returns newly shuffled X and y data
  # def shuffle_xy(X, y):
  #   indices = tf.range(start=0, limit=tf.shape(Xtrain)[0], dtype=tf.int32)
  #   shuffled_indices = tf.random.shuffle(indices)
  #   X_shuffled = tf.gather(X, shuffled_indices)
  #   y_shuffled = tf.gather(y, shuffled_indices)
  #   return X_shuffled, y_shuffled

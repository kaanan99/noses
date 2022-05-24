#-------------------- IMPORTS --------------------
# standard
import random
import itertools
import math
import os

# data
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# cnn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# computing metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score,  \
                            recall_score, confusion_matrix

LINE_LEN = 26 # used for pretty printing

#-------------------- TRAINING PARAMS - NOT HYPERRAMATIZED --------------------
lr = 3e-4
batch_size = 32
conv_dim_init = 64
epochs = 25
dp_rate = .1


#-------------------- MODEL ARCHITECTURE --------------------
# Used for testing
def create_model_2():
   model = keras.models.Sequential()
   # Input layer
   #model.add(layers.Flatten())
   #model.add(layers.Dense(units = 150**2, activation = 'relu', input_shape = (150,150, 3)))
   #Convolutional Layer
   model.add(layers.Conv2D(filters=64, kernel_size= 3, input_shape = (150, 150, 3), activation = "relu"))
   model.add(layers.Conv2D(filters=32, kernel_size= 3, activation = "relu"))
   model.add(layers.MaxPooling2D((4, 4)))
   '''
   model.add(layers.Conv2D(filters=16, kernel_size= 3, activation = "relu"))
   model.add(layers.MaxPooling2D((2, 2)))
   '''
   # Flatten after Convolutional layers
   model.add(layers.Flatten())
   # 3 Dense Layers
   #model.add(layers.Dense(units = 1024, activation = 'relu'))
   #model.add(layers.Dense(units = 512, activation = 'relu'))
   model.add(layers.Dense(units = 256, activation = 'relu'))
   model.add(layers.Dense(units = 128, activation = 'relu'))
   model.add(layers.Dense(units = 64, activation = 'relu'))
   model.add(layers.Dense(units = 32, activation = 'relu'))
   #model.add(layers.Flatten())
   # Output layer
   model.add(layers.Dense(units = 3, activation = 'softmax'))
   # originally sparse_categorical_crossentropy
   model.compile(loss='categorical_crossentropy', metrics='accuracy')
   return model

# given the model hyperparameters
# returns the model to train
def create_model(cnn_blocks=1, dense_layers=1, filter_multiplier = 1, kernel_size=3,
                 strides=(1, 1), dropout = True, classification="binary"):
  model = keras.models.Sequential()
  for i in range(cnn_blocks):
    conv_output_dim = int((conv_dim_init * filter_multiplier) * (2 ** i))
    model.add(layers.Conv2D(filters=conv_output_dim, kernel_size=kernel_size,
                            strides=strides, activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=conv_output_dim, kernel_size=kernel_size,
                            strides=strides, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    if dropout == True: model.add(layers.Dropout(dp_rate))
  model.add(layers.Flatten())
  for i in range(dense_layers):
    if i == 0:
      model.add(layers.Dense(units=conv_output_dim * max(1, (math.ceil(150 / strides[0] ** (cnn_blocks * 3))) ** 2), activation='relu'))
    else:
      model.add(layers.Dense(units=conv_output_dim * max(1, round(150/ strides[0] ** (cnn_blocks*3))) / (2*i), activation='relu'))

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


#-------------------- MODEL TRAINING --------------------
# given the model hyperparameters
# runs the model
# returns the predictions for the test set and the test set without a filter
def run_model(train_test_dict,
              threshold_min, threshold_max, cnn_blocks, dense_layers, filter_multiplier,
              kernel_size, strides, dropout_flag, model_num):
  Xtrain = train_test_dict["Xtrain"]
  ytrain = train_test_dict["ytrain"]
  Xtest = train_test_dict["Xtest"]
  ytest = train_test_dict["ytest"]
  #Xtest_no_filter = train_test_dict["Xtest_no_filter"]
  #ytest_no_filter = train_test_dict["ytest_no_filter"]

  model = create_model_2()#create_model(cnn_blocks, dense_layers, filter_multiplier, kernel_size, strides, dropout_flag)
  print("model created")
  #Modified fit
  model.fit(Xtrain, pd.get_dummies(pd.Series(ytrain)), batch_size=None, epochs=1, verbose=1)
  print("finished fitting")
  res = model.evaluate(Xtest, pd.get_dummies(pd.Series(ytest))); loss = res[0]; acc = res[1]
  print("finished evaluation")
  ypred = model.predict(Xtest)
  print("finished predicting")
  #ypred_no_filter = model.predict(Xtest_no_filter)
  print_model_params(threshold_min, threshold_max, cnn_blocks, dense_layers, filter_multiplier,
                        kernel_size, strides, dropout_flag, model_num)

  #print(model.summary())
  return ypred#, ypred_no_filter


#-------------------- TRAIN TEST SPLIT --------------------
# given the images, the bounding box data, and the labels
# preprocceses the data and
# returns a dictionary containing the train test split
# and the seal percentages for the test set with a filter and without a filter
def train_test_split(imgs, bb_data, labels, threshold_min):
    train_test = {}
    indices_and_percents = get_indices_and_percents(bb_data, threshold_min)
    train_indices = indices_and_percents["train_indices"]
    test_indices = indices_and_percents["test_indices"]
    #test_indices_no_filter = indices_and_percents["test_indices_no_filter"]
    train_test["Xtrain"], train_test["ytrain"] = preprocessing(imgs[train_indices], labels[train_indices])
    train_test["Xtest"], train_test["ytest"] = preprocessing(imgs[test_indices], labels[test_indices])
    #train_test["Xtest_no_filter"], train_test["ytest_no_filter"] = preprocessing(imgs[test_indices_no_filter], labels[test_indices_no_filter])
    train_test["seal_percents"] = indices_and_percents["seal_percents"]
    #train_test["seal_percents_no_filter"] = indices_and_percents["seal_percents_no_filter"]
    return train_test


# given the bounding box data and the thresholds
# returns the indices for the train test split
# and the seal percentages for the test set with a filter and without a filter
def get_indices_and_percents(bb_data, threshold_min, test_frac = .1):
  # get indices for images w seals
  indices_w_seal = []; indices_w_seal_no_filter = []
  seal_percents = []; seal_percents_no_filter = []
  for i in range(len(bb_data)):
    df_subimg = bb_data[i]
    if not df_subimg is None:
      seal_percent = get_seal_percent(df_subimg)
      if seal_percent > threshold_min:
          indices_w_seal.append(i)
          seal_percents.append(seal_percent)
      indices_w_seal_no_filter.append(i)
      seal_percents_no_filter.append(seal_percent)

  # get num with seal, num without seal
  num_w_seal = len(indices_w_seal)
  size_dataset = int(num_w_seal * 10/4) # size for dataset with 40% of images containing seals
  #Original Code
  #num_wo_seal = size_dataset - num_w_seal
  # EDIT by Kaanan
  num_wo_seal = len(bb_data)- num_w_seal

  '''
  # get num with seal, num without seal (no filter)
  num_w_seal_no_filter = len(indices_w_seal_no_filter)
  # ORIGINAL CODE
  #size_dataset_no_filter = int(num_w_seal_no_filter * 10/4)
  #EDIT BY KAANAN
  size_dataset_no_filter = num_w_seal_no_filter 
  num_wo_seal_no_filter = size_dataset_no_filter - num_w_seal_no_filter
  '''

  # get indices for images w/o seals
  all_indices_wo_seal = [x for x in list(range(len(bb_data))) if x not in indices_w_seal]
  print(len(all_indices_wo_seal))
  print(num_wo_seal)
  indices_wo_seal = random.sample(all_indices_wo_seal, num_wo_seal)

  '''
  # get indices for images w/o seals (no filter)
  all_indices_wo_seal_no_filter = [x for x in list(range(len(bb_data))) if x not in indices_w_seal_no_filter]
  indices_wo_seal_no_filter = random.sample(all_indices_wo_seal_no_filter, num_wo_seal_no_filter)
  '''

  # numbers to get train test split
  num_train_w_seal = round((1 - test_frac) * num_w_seal)
  num_train_wo_seal = round((1 - test_frac) * num_wo_seal)
  # ORIGINAL CODE
  #num_test_wo_seal = size_dataset - num_w_seal - num_train_wo_seal
  # EDIT BY KAANAN
  num_test_wo_seal = len(bb_data) - num_w_seal - num_train_wo_seal


  # numbers to get train test split (no filter)
  num_train_w_seal_no_filter = round((1 - test_frac) * num_w_seal_no_filter)
  num_train_wo_seal_no_filter = round((1 - test_frac) * num_wo_seal_no_filter)
  num_test_wo_seal_no_filter = size_dataset_no_filter - num_w_seal_no_filter - num_train_wo_seal_no_filter

  # seal percents
  seal_percents = np.array(seal_percents[num_train_w_seal:])
  seal_percents = np.pad(seal_percents, (0, num_test_wo_seal), 'constant')

  seal_percents_no_filter = np.array(seal_percents_no_filter[num_train_w_seal_no_filter:])
  seal_percents_no_filter = np.pad(seal_percents_no_filter, (0, num_test_wo_seal_no_filter), 'constant')

  # train test split of indices, each with 40% of sub imgs containing seals
  train_indices = np.concatenate((np.array(indices_w_seal[:num_train_w_seal]),
                                  np.array(indices_wo_seal[:num_train_wo_seal])))
  test_indices = np.concatenate((np.array(indices_w_seal[num_train_w_seal:]),
                                  np.array(indices_wo_seal[num_train_wo_seal:])))
  test_indices_no_filter = np.concatenate((np.array(indices_w_seal_no_filter[num_train_w_seal_no_filter:]),
                                  np.array(indices_wo_seal_no_filter[num_train_wo_seal_no_filter:])))

  return_dict = {"train_indices": train_indices, "test_indices": test_indices,
                 "test_indices_no_filter": test_indices_no_filter,
                 "seal_percents": seal_percents, "seal_percents_no_filter": seal_percents_no_filter}

  return return_dict


# given the df for a subimg
# returns the highest percentage seal found within the image
def get_seal_percent(df_subimg):
    seal_percents = []
    for i in df_subimg.itertuples():
      seal_percents.append(i[3])
    return max(seal_percents)


# data preprocessing
def preprocessing(X_init, y_init):
  '''print("Normalizing")
  X_init = np.array(X_init)
  y_init = np.array(y_init)
  X_init = X_init[:] / 255. # normalizes image array
  y_init = y_init[:] / 1.
  '''
  X = []
  y = []
  '''for Xi, yi in zip(X_init, y_init):
    im = tf.image.resize(Xi, (150, 150))
    X.append(im)
    y.append(yi)'''
  print("Resizing")
  X = tf.image.resize(X_init, (150, 150))
  #X = tf.stack(X)
  #y = tf.stack(y)
  return X, y_init


#-------------------- CONVERTING PREDICTIONS --------------------
# convert ypred from confidence values to classifications
# (e.g. for binary: [.55 .45] -> [0 1])
# convert ytest from tensor to list
def convert_arrs(ypred, ytest, binary_flag=True):
  ypred_ = []
  if binary_flag == True:
      for i in range(len(ypred)):
          if ypred[i] > .5:
              ypred_.append(1)
          else:
              ypred_.append(0)
  else:
      for i in range(len(ypred)):
        confidence_arr = list(ypred[i])
        ypred_.append(np.argmax(confidence_arr))
  ytest_ = []
  for i in range(len(ytest)):
    ytest_.append(int(ytest[i]))
  return ypred_, ytest_


#-------------------- EXTRACTING LABELS --------------------
# given the bounding box data
# extracts and returns tertiary labels
# e.g. 0 == "no seal", 1 == "partial seal", 2 == "full seal"
def get_labels_tertiary(bb_data, min_threshold, max_threshold):
  labels = []
  for df_subimg in bb_data:
    if not isinstance(df_subimg, type(None)):
      seal_percent = get_seal_percent(df_subimg)
      if seal_percent > max_threshold:
        val = 2
      elif seal_percent > min_threshold:
        val = 1
      else:
        val = 0
    else:
      val = 0
    labels.append(val)
  labels = np.array(labels)
  return labels



# given the bounding box data
# extracts and returns tertiary labels
# e.g. -1 == "no seal", 0 == "partial seal", 1 == "full seal"
def modified_get_labels_tertiary(bb_data, min_threshold, max_threshold):
  labels = []
  for df_subimg in bb_data:
    if not isinstance(df_subimg, type(None)):
      seal_percent = get_seal_percent(df_subimg)
      if seal_percent > max_threshold:
        val = 1
      elif seal_percent > min_threshold:
        val = 0
      else:
        val = -1
    else:
      val = -1
    labels.append(val)
  labels = np.array(labels)
  return labels


# given the bounding box data
# extracts and returns binary labels
# e.g. 0 == "no seal", 1 == "seal"
def get_labels_binary(bb_data, min_threshold = 0.3):
    labels = []
    for df_subimg in bb_data:
      if not isinstance(df_subimg, type(None)):
        if get_seal_percent(df_subimg) > min_threshold:
          val = 1
        else:
          val = 0
      else:
        val = 0
      labels.append(val)
    labels = np.array(labels)
    return labels


#-------------------- PRETTY PRINTING --------------------
# given the model hyperparameters
# pretty prints the hyperparamaters
def print_model_params(threshold_min, threshold_max, cnn_blocks, dense_layers, filter_multiplier,
                        kernel_size, strides, dropout_flag, model_num):
    line = " " + "-" * LINE_LEN; len_line = len(line)
    print(line)
    s = "| Model Number: %d" % model_num; print(get_string(s, len_line))
    print(line)
    s = "| CCP Block(s) %8d" % cnn_blocks; print(get_string(s, len_line))
    s = "| Dense Layer(s) %6d" % dense_layers; print(get_string(s, len_line))
    s = "| Filter Multiplier %3dx" % filter_multiplier; print(get_string(s, len_line))
    s = "| Kernel Size %9d" % kernel_size; print(get_string(s, len_line))
    s = "| Strides %13d" % strides[0]; print(get_string(s, len_line))
    s = "| Dense Output Size %5d" % (int(int((conv_dim_init * filter_multiplier) * (2 ** (cnn_blocks-1))) *
                                         max(1, (math.ceil(150 / strides[0] ** (cnn_blocks * 3)))**2))); print(get_string(s, len_line))
    #s = "| Dense Output Size %5d" % dense_output_size; print(get_string(s, len_line))
    s = "| Threshold Min" + "\t     " + '{:.2f}'.format(threshold_min).lstrip('0'); print(get_string(s, len_line))
    if isinstance(threshold_max, str):
        s = "| Threshold Max" + "\t     " + "N/A"; print(get_string(s, len_line))
    else:
        s = "| Threshold Max" + "\t     " + '{:.2f}'.format(threshold_max).lstrip('0'); print(get_string(s, len_line))
    if dropout_flag == True: x = "Y"
    else: x = "N"
    s = "| Dropout? %12s" % x; print(get_string(s, len_line))
    print(line)


# given the predicitons and labels for the test set
# computes and prints the resulting test metrics
def print_metrics(ypred, ytest, no_filter=False):
  acc = accuracy_score(ytest, ypred)
  f1 = f1_score(np.array(ytest), ypred, labels=np.unique(ytest), average="weighted")
  precision = precision_score(np.array(ytest), ypred, labels=np.unique(ypred), average="weighted")
  recall = recall_score(np.array(ytest), ypred, labels=np.unique(ypred), average="weighted")

  line = " " + "-" * LINE_LEN; len_line = len(line)
  print(line)
  if no_filter == False: s = "| Test Metrics"; print(get_string(s, len_line))
  else: s = "| Test Metrics - No Filter"; print(get_string(s, len_line))
  print(line)
  s = "| Accuracy %14.4f" % acc; print(get_string(s, len_line))
  s = "| Precision %13.4f" % precision; print(get_string(s, len_line))
  s = "| Recall %16.4f" % recall; print(get_string(s, len_line))
  s = "| F1 Score %14.4f" % f1; print(get_string(s, len_line))
  print(line)


# given the predictions and labels for the test set with a filter
# pretty prints the confusion matrix
def print_confusion_matrix(ytest, ypred, binary_flag):
    line = " " + "-" * LINE_LEN; len_line = len(line)
    print(line)
    s = "| Confusion Matrix "; print(get_string(s, len_line))
    cm = confusion_matrix(ytest, ypred)
    if binary_flag == True: labels = ["0", "1"]
    else: labels = ["0", "1", "2"]

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
    hide_zeroes=False; hide_diagonal=False; hide_threshold=None
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
    # print_cm(cm, labels=cm_labels)


# helper function for various pretty printing functions
def get_string(s, len_line):
    num_spaces = len_line - len(s)
    return s + ' ' * num_spaces + "|"


#-------------------- MODEL PLOTTING --------------------
# given the predictions and labels for the test set with no filter
# saves a bar plot for the % predicted as a seal over different seal percentage buckets
def plot_buckets(ypred, ytest, percents, threshold_min, threshold_max, binary_flag, fname):
    d = {}
    buckets = ["0%", "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    for bucket in buckets:
        d[bucket] = {"sum": 0, "total": 0}
    for elem in zip(ytest, ypred, percents):
        label = elem[0]; prediction = elem[1]; percent = elem[2]
        if percent == 0: bucket = "0%"
        elif percent < .1: bucket = "0-10%"
        elif percent < .2: bucket = "10-20%"
        elif percent < .3: bucket = "20-30%"
        elif percent < .4: bucket = "30-40%"
        elif percent < .5: bucket = "40-50%"
        elif percent < .6: bucket = "50-60%"
        elif percent < .7: bucket = "60-70%"
        elif percent < .8: bucket = "70-80%"
        elif percent < .9: bucket = "80-90%"
        else: bucket = "90-100%"

        if prediction == 1:
            d[bucket]["sum"] += 1
        d[bucket]["total"] += 1

    new_d = {}
    for bucket, dict in d.items():
        if d[bucket]["total"] == 0:
            new_d[bucket] = 0
        else:
            new_d[bucket] = d[bucket]["sum"] / d[bucket]["total"]

    buckets = list(new_d.keys()); values = list(new_d.values())
    plt.figure(figsize=[6.4, 8])
    if binary_flag:
        plt.bar(range(1), values[0], color='r')
        plt.bar(range(1, int(math.floor(threshold_min*10)+1)), values[1:int(math.floor(threshold_min*10))+1], color='y')
        plt.bar(range(int(math.floor(threshold_min*10+1)), len(new_d)), values[int(math.floor(threshold_min*10)+1):],
                color='g')
    else:
        plt.bar(range(int(math.floor(threshold_min * 10) + 1)), values[:int(math.floor(threshold_min * 10) + 1)],
                color='r')
        plt.bar(range(int(math.floor(threshold_min * 10) + 1), int(math.floor(threshold_max * 10) + 1)),
                values[int(math.floor(threshold_min * 10) + 1):int(math.floor(threshold_max * 10)) + 1], color='y')
        plt.bar(range(int(math.floor(threshold_max * 10 + 1)), len(new_d)),
                values[int(math.floor(threshold_max * 10) + 1):], color='g')
    # plt.bar(range(len(new_d)), values, tick_label=buckets, color='r')
    plt.xticks(range(len(new_d)), labels=buckets, rotation=45)
    plt.xlabel("% Seal In Image")
    plt.ylabel("% Predicted as a Full Seal")
    plt.title("Predicted Full Seals")
    plt.ylim(0, 1)
    plt.legend(['No Seal', 'Partial Seal', 'Full Seal'], loc='upper left')
    plt.savefig(fname, edgecolor="r")
    plt.close()


#-------------------- OLD CODE --------------------
"""
# NO LONGER USED - REPLACED BY plot_buckets()
# given the predictions and labels for the test set with no filter
# saves a bar plot for several test metrics different seal percentage buckets
def plot_metric_buckets(ypred, ytest, percents, fname):
    d = {}
    buckets = ["0%", "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    for bucket in buckets:
        d[bucket] = {"ytrue": [], "ypred": []}
    for elem in zip(ytest, ypred, percents):
        label = elem[0]; prediction = elem[1]; percent = elem[2]
        if percent == 0: bucket = "0%"
        elif percent < .1: bucket = "0-10%"
        elif percent < .2: bucket = "10-20%"
        elif percent < .3: bucket = "20-30%"
        elif percent < .4: bucket = "30-40%"
        elif percent < .5: bucket = "40-50%"
        elif percent < .6: bucket = "50-60%"
        elif percent < .7: bucket = "60-70%"
        elif percent < .8: bucket = "70-80%"
        elif percent < .9: bucket = "80-90%"
        else: bucket = "90-100%"
        d[bucket]["ytrue"].append(label)
        d[bucket]["ypred"].append(prediction)

    acc_dict = {}; prec_dict = {}; rec_dict = {}; f1_dict = {}
    for bucket, dict in d.items():
        if len(d[bucket]["ytrue"]) == 0:
            acc_dict[bucket] = 0
            prec_dict[bucket] = 0
            rec_dict[bucket] = 0
            # f1_dict[bucket] = 0
        else:
            acc_dict[bucket] = accuracy_score(d[bucket]["ytrue"], d[bucket]["ypred"])
            prec_dict[bucket] = precision_score(d[bucket]["ytrue"], d[bucket]["ypred"], average='weighted', zero_division=0)
            rec_dict[bucket] = recall_score(d[bucket]["ytrue"], d[bucket]["ypred"], average='weighted', zero_division=0)
            # 1_dict[bucket] = f1_score(d[bucket]["ytrue"], d[bucket]["ypred"], average='weighted', zero_division=0)

    buckets = list(acc_dict.keys())
    acc = list(acc_dict.values())
    prec = list(prec_dict.values())
    rec = list(rec_dict.values())
    #f1 = list(f1_dict.values())

    plt.bar(range(len(acc_dict)), acc, tick_label=buckets, color='r')
    plt.xticks(rotation=45)
    plt.xlabel("% Seal")
    plt.title("Accuracy")
    plt.ylim(0, 1)
    plt.savefig(fname + "-a", edgecolor="r")
    plt.close()

    plt.bar(range(len(prec_dict)), prec, tick_label=buckets, color='g')
    plt.xticks(rotation=45)
    plt.xlabel("% Seal")
    plt.title("Precision")
    plt.ylim(0, 1)
    plt.savefig(fname + "-p")
    plt.close()

    plt.bar(range(len(rec_dict)), rec, tick_label=buckets, color='b')
    plt.xticks(rotation=45)
    plt.xlabel("% Seal")
    plt.title("Recall")
    plt.ylim(0, 1)
    plt.savefig(fname + "-r")
    plt.close()


# given X and y data, returns newly shuffled X and y data
def shuffle_xy(X, y):
    indices = tf.range(start=0, limit=tf.shape(Xtrain)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    X_shuffled = tf.gather(X, shuffled_indices)
    y_shuffled = tf.gather(y, shuffled_indices)
    return X_shuffled, y_shuffled
"""
#---------- Kaanans Functions -----------
def train_test(indices):
   test = random.sample(indices, int(len(indices) * .3 // 1))
   train = [i for i in indices if i not in test]
   return train, test

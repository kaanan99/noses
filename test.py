import pandas as pd
from matplotlib import pyplot as plt
import os
import sys
import itertools
from sklearn.metrics import confusion_matrix

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

    s_end = "\t  |"
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

ytrue = [0, 0, 0, 0, 0]
ypred = [1, 0, 1, 0, 1]

cm = confusion_matrix(ytrue, ypred)
print_cm(cm, ["0", "1"])

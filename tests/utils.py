# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Utility variables for the tests.

"""

# Standard packages
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_diabetes
from keras.datasets import mnist

# Own modules
from atom.utils import merge


# Variables ========================================================= >>

# Directory for storing all files created by the tests
FILE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/files/"

# Sklearn datasets for all three tasks as np.array
X_bin_array, y_bin_array = load_breast_cancer(return_X_y=True)
X_class_array, y_class_array = load_digits(return_X_y=True)
X_reg_array, y_reg_array = load_diabetes(return_X_y=True)

# Sklearn datasets for all three tasks as pd.DataFrame
X_bin, y_bin = load_breast_cancer(return_X_y=True, as_frame=True)
X_class, y_class = load_wine(return_X_y=True, as_frame=True)
X_class2, y_class2 = load_digits(return_X_y=True, as_frame=True)
X_reg, y_reg = load_diabetes(return_X_y=True, as_frame=True)

# Train and test sets for all three tasks
kwargs = dict(test_size=0.3, random_state=1)
bin_train, bin_test = train_test_split(merge(X_bin, y_bin), **kwargs)
class_train, class_test = train_test_split(merge(X_class, y_class), **kwargs)
reg_train, reg_test = train_test_split(merge(X_reg, y_reg), **kwargs)

# Small dimensional dataset
X10 = [
    [0.2, 2, 1],
    [0.2, 2, 1],
    [0.2, 2, 2],
    [0.24, 2, 1],
    [0.23, 2, 2],
    [0.19, 0, 1],
    [0.21, 3, 2],
    [0.2, 2, 1],
    [0.2, 2, 1],
    [0.2, 2, 0],
]

# Dataset with missing value
X10_nan = [
    [np.NaN, 2, 1],
    [0.2, 2, 1],
    [4, 2, 2],
    [3, 2, 1],
    [3, 2, 2],
    [1, 0, 1],
    [0, 3, 2],
    [4, np.NaN, 1],
    [5, 2, 1],
    [3, 2, 0],
]

# Dataset with categorical column
X10_str = [
    [2, 0, "a"],
    [2, 3, "a"],
    [5, 2, "b"],
    [1, 2, "a"],
    [1, 2, "c"],
    [2, 0, "d"],
    [2, 3, "d"],
    [5, 2, "d"],
    [1, 2, "a"],
    [1, 2, "d"],
]

# Dataset with categorical column (only two classes)
X10_str2 = [
    [2, 0, "a"],
    [2, 3, "a"],
    [5, 2, "b"],
    [1, 2, "a"],
    [1, 2, "a"],
    [2, 0, "a"],
    [2, 3, "b"],
    [5, 2, "b"],
    [1, 2, "a"],
    [1, 2, "a"],
]

# Dataset with missing value in categorical column
X10_sn = [
    [2, 0, np.NaN],
    [2, 3, "a"],
    [5, 2, "b"],
    [1, 2, "a"],
    [1, 2, "c"],
    [2, 0, "d"],
    [2, 3, "d"],
    [5, 2, "d"],
    [1, 2, "a"],
    [1, 2, "d"],
]

# Dataset with outliers
X20_out = [
    [2, 0, 2],
    [2, 3, 1],
    [3, 2, 2],
    [1, 2, 1],
    [1, 2, 1],
    [1, 2, 1],
    [1, 2, 1],
    [1, 2, 1],
    [1, 2, 1],
    [1, 2, 1],
    [1, 2, 1],
    [1, 2, 1],
    [1, 2, 1],
    [1, 2, 1],
    [1, 1e6, 2],
    [2, 0, 2],
    [2, 3, 2],
    [3, 2, 1],
    [1, 2, 2],
    [1e6, 2, 1],
]

# Target columns (int, missing, categorical and mixed)
y10 = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
y10_nan = [0, 1, 0, np.NaN, 1, 0, 1, 0, 1, 1]
y10_str = ["y", "n", "y", "y", "n", "y", "n", "y", "n", "n"]
y10_sn = ["y", "n", np.NaN, "y", "n", "y", "n", "y", "n", "n"]

# Download the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data to fit model
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

mnist = (X_train, y_train), (X_test, y_test)

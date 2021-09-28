# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Utility variables for the tests.

"""

# Standard packages
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_digits,
    load_diabetes,
)

# Own modules
from atom.utils import merge


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

# Image data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
mnist = (X_train, y_train), (X_test, y_test)

# Text data
X_text = [["I àm in ne'w york"], ["New york is nice"], ["hi new york"], ["yes sir 12"]]
y_text = [0, 1, 1, 0]

# Small dimensional dataset
X10 = [
    [0.2, 2, 1],
    [0.2, 2, 1],
    [0.2, 2, 2],
    [0.24, 2, 1],
    [0.23, 2, 2],
    [0.19, 0.01, 1],
    [0.21, 3, 2],
    [0.2, 2, 1],
    [0.2, 2, 1],
    [0.2, 2, 0.01],
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
    [2, 0, "a", True],
    [2, 3, "a", False],
    [5, 2, "b", True],
    [1, 2, "a", True],
    [1, 2, "a", False],
    [2, 0, "a", False],
    [2, 3, "b", False],
    [5, 2, "b", True],
    [1, 2, "a", True],
    [1, 2, "a", False],
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

# Dataset with dates
X10_dt = [
    [2, "21", "13/02/2021", 4],
    [2, "12", "31/3/2020", 22],
    [5, "06", "30/3/2020", 21],
    [1, "03", "31/5/2020", 2],
    [1, "202", np.NaN, 4],
    [2, "11", "06/6/2000", 6],
    [2, "01", "31/3/2020", 7],
    [5, "22", "9/12/2020", 6],
    [1, "24", "5/11/2020", 8],
    [1, "00", "22/03/2018", 9],
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

# Target columns (int, matgomissing, categorical and mixed)
y10 = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
y10_nan = [0, 1, 0, np.NaN, 1, 0, 1, 0, 1, 1]
y10_str = ["y", "n", "y", "y", "n", "y", "n", "y", "n", "n"]
y10_sn = ["y", "n", np.NaN, "y", "n", "y", "n", "y", "n", "n"]

# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the encode method of the ATOM class.

"""

# Import packages
import pytest
import numpy as np
from atom import ATOMClassifier


# << ====================== Variables ===================== >>

X_dim4 = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
y_dim4 = [0, 1, 0, 1]
X_dim10 = [[2, 0, 'a'], [2, 3, 'a'], [5, 2, 'b'], [1, 2, 'a'], [1, 2, 'c'],
           [2, 0, 'd'], [2, 3, 'd'], [5, 2, 'd'], [1, 2, 'a'], [1, 2, 'd']]
y_dim10 = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]


# << ======================= Tests ========================= >>

# << =================== Test parameters =================== >>

def test_max_onehot_parameter():
    """ Assert that the max_onehot parameter is set correctly """

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(ValueError, atom.encode, max_onehot=-2)


def test_encode_type_parameter():
    """ Assert that the encode_type parameter is set correctly """

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(ValueError, atom.encode, encode_type='unknown')


def test_frac_to_other_parameter():
    """ Assert that the frac_to_other parameter is set correctly """

    atom = ATOMClassifier(X_dim4, y_dim4)
    pytest.raises(ValueError, atom.encode, frac_to_other=2.2)


# << ================ Test frac_to_other ================= >>

def test_frac_to_other():
    """ Assert that the other values are created when encoding """

    atom = ATOMClassifier(X_dim10, y_dim10, test_size=0.2, random_state=1)
    atom.encode(max_onehot=5, frac_to_other=0.3)
    assert 'Feature 2_other' in atom.dataset.columns


# << ================ Test encoding types ================ >>

def test_raise_missing():
    """ Assert that an error is raised when missing values in data """

    X = [[np.NaN, 0, 1], ['y', 3, 4], ['n', 2, 7], ['n', 9, 10]]
    atom = ATOMClassifier(X, y_dim4, test_size=0.25, random_state=1)
    pytest.raises(ValueError, atom.encode, max_onehot=None)


def test_label_encoder():
    """ Assert that the label-encoder works as intended """

    X = [['y', 0, 1], ['y', 3, 4], ['n', 2, 7], ['n', 9, 10]]
    atom = ATOMClassifier(X, y_dim4, test_size=0.25, random_state=1)
    atom.encode(max_onehot=None)
    col = atom.dataset['Feature 0']
    assert np.all((col == 1) | (col == 2))


def test_one_hot_encoder():
    """ Assert that the one-hot-encoder works as intended """

    atom = ATOMClassifier(X_dim10, y_dim10, random_state=1)
    atom.encode(max_onehot=4)
    assert 'Feature 2_c' in atom.dataset.columns


def test_all_encoder_types():
    """ Assert that all encoder types work as intended """

    encode_types = ['BackwardDifference',
                     'BaseN',
                     'Binary',
                     'CatBoost',
                     # 'Hashing',
                     'Helmert',
                     'JamesStein',
                     'LeaveOneOut',
                     'MEstimate',
                     # 'OneHot',
                     'Ordinal',
                     'Polynomial',
                     'Sum',
                     'Target',
                     'WOE']

    for encoder in encode_types:
        atom = ATOMClassifier(X_dim10, y_dim10, random_state=1)
        atom.encode(max_onehot=None, encode_type=encoder)
        for col in atom.dataset:
            assert atom.dataset[col].dtype.kind in 'ifu'

def test_kwargs_parameters():
    """ Assert that the kwargs parameter works as intended """

    atom = ATOMClassifier(X_dim10, y_dim10, random_state=1)
    atom.encode(max_onehot=None, encode_type='LeaveOneOut', sigma=0.5)
    assert atom.encoder._encoders['Feature 2'].get_params()['sigma'] == 0.5

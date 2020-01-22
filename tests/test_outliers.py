# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the outliers method of the ATOM class.

'''

# Import packages
import pytest
from atom import ATOMClassifier


# << ====================== Variables ===================== >>

X_dim10 = [[0.2, 2, 1], [0.2, 2, 1], [0.2, 2, 2], [0.24, 2, 1], [0.23, 2, 2],
           [0.19, 0, 1], [0.21, 3, 2], [0.2, 2, 1], [0.2, 2, 1], [0.2, 2, 0]]
y_dim10 = [0, 1, 0, 1, 1, 0, 245, 0, 1, 1]


# << ======================= Tests ========================= >>

# << ================== Test parameters =================== >>

def test_max_sigma_parameter():
    ''' Assert that the max_sigma parameter is set correctly '''

    atom = ATOMClassifier(X_dim10, y_dim10)
    pytest.raises(TypeError, atom.outliers, max_sigma='test')
    pytest.raises(ValueError, atom.outliers, max_sigma=0)


def test_include_target_parameter():
    ''' Assert that the include_target parameter is set correctly '''

    atom = ATOMClassifier(X_dim10, y_dim10)
    pytest.raises(TypeError, atom.outliers, include_target=1.1)


# << ================= Test functionality ================= >>

def test_remove_outliers():
    ''' Assert that method works as intended '''

    atom = ATOMClassifier(X_dim10, y_dim10, test_size=0.1, random_state=1)
    length = len(atom.train)
    atom.outliers(max_sigma=2)
    assert len(atom.train) + 1 == length


def test_remove_outlier_in_target():
    ''' Assert that method works as intended for target columns as well '''

    atom = ATOMClassifier(X_dim10, y_dim10, test_size=0.1, random_state=1)
    length = len(atom.train)
    atom.outliers(max_sigma=2, include_target=True)
    assert len(atom.train) + 1 == length

# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the outliers method of the ATOM class.

"""

# Import packages
import pytest
from atom import ATOMClassifier


# << ====================== Variables ===================== >>

X_dim10 = [[0.2, 2, 1], [0.2, 2, 1], [0.2, 2, 2], [0.24, 2, 1], [0.23, 2, 2],
           [0.19, 0, 1], [0.21, 3, 2], [0.2, 2, 1], [0.2, 2, 1], [0.2, 2, 0]]
y_dim10 = [0, 1, 0, 1, 1, 0, 245, 0, 1, 1]


# << ======================= Tests ========================= >>

# << ================== Test parameters ===================== >>

def test_invalid_strategy_parameter():
    """ Assert that the strategy parameter is set correctly """

    atom = ATOMClassifier(X_dim10, y_dim10)
    pytest.raises(ValueError, atom.outliers, strategy='invalid')


def test_caps_strategy_parameter():
    """ Assert that the strategy parameter works with caps as well """

    atom = ATOMClassifier(X_dim10, y_dim10)
    atom.outliers(strategy='MIN_MAX')
    assert 1 == 1


def test_max_sigma_parameter():
    """ Assert that the max_sigma parameter is set correctly """

    atom = ATOMClassifier(X_dim10, y_dim10)
    pytest.raises(ValueError, atom.outliers, max_sigma=0)


# << ================= Test functionality =================== >>

def test_max_sigma_functionality():
    """ Assert that the max_sigma parameter works as intended """

    # Test 3 different values for sigma and number of rows they remove
    atom1 = ATOMClassifier(X_dim10, y_dim10, test_size=0.1, random_state=1)
    atom1.outliers(max_sigma=1)

    atom2 = ATOMClassifier(X_dim10, y_dim10, test_size=0.1, random_state=1)
    atom2.outliers(max_sigma=2)

    atom3 = ATOMClassifier(X_dim10, y_dim10, test_size=0.1, random_state=1)
    atom3.outliers(max_sigma=3)

    assert len(atom1.train) < len(atom2.train) < len(atom3.train)


def test_remove_outliers():
    """ Assert that the method works as intended when strategy='remove' """

    atom = ATOMClassifier(X_dim10, y_dim10, test_size=0.1, random_state=1)
    length = len(atom.train)
    atom.outliers(max_sigma=2)
    assert len(atom.train) + 2 == length


def test_min_max_outliers():
    """ Assert that the method works as intended when strategy='min_max' """

    atom = ATOMClassifier(X_dim10, y_dim10, test_size=0.1, random_state=1)
    atom.outliers(strategy='min_max', max_sigma=2)
    assert atom.train.iloc[2, 1] == 2
    assert atom.train.iloc[5, 0] == 0.23


def test_value_outliers():
    """ Assert that the method works as intended when strategy=value """

    atom = ATOMClassifier(X_dim10, y_dim10, test_size=0.1, random_state=1)
    atom.outliers(strategy=-99, max_sigma=2)
    assert atom.train.iloc[2, 1] == -99
    assert atom.train.iloc[5, 0] == -99


def test_remove_outlier_in_target():
    """ Assert that method works as intended for target columns as well """

    atom = ATOMClassifier(X_dim10, y_dim10, test_size=0.1, random_state=1)
    length = len(atom.train)
    atom.outliers(max_sigma=2, include_target=True)
    assert len(atom.train) + 2 == length

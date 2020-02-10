# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the balance method of the ATOM class.

'''

# Import packages
import pytest
from sklearn.datasets import load_breast_cancer, load_wine, load_boston
from atom import ATOMClassifier, ATOMRegressor


# << ======================= Tests ========================= >>

# << =================== Test parameters =================== >>

def test_not_classification_task():
    ''' Assert that error s raised when task == regression '''

    X, y = load_boston(return_X_y=True)
    atom = ATOMRegressor(X, y)
    pytest.raises(ValueError, atom.balance, undersample=0.8)


def test_oversample_parameter():
    ''' Assert that the oversample parameter is set correctly '''

    # Binary classification tasks
    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)
    pytest.raises(ValueError, atom.balance, oversample=-2.1)
    pytest.raises(ValueError, atom.balance, oversample='test')

    # Multiclass classification tasks
    X, y = load_wine(return_X_y=True)
    atom = ATOMClassifier(X, y)
    pytest.raises(TypeError, atom.balance, undersample=1.0)


def test_undersample_parameter():
    ''' Assert that the undersample parameter is set correctly '''

    # Binary classification tasks
    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)
    pytest.raises(ValueError, atom.balance, undersample=-3.)
    pytest.raises(ValueError, atom.balance, undersample='test')

    # Multiclass classification tasks
    X, y = load_wine(return_X_y=True)
    atom = ATOMClassifier(X, y)
    pytest.raises(TypeError, atom.balance, undersample=0.8)


def test_n_neighbors_parameter():
    ''' Assert that the n_neighbors parameter is set correctly '''

    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)
    pytest.raises(ValueError, atom.balance, n_neighbors=0)


def test_None_both_parameter():
    ''' Assert that error raises when over and undersample are both None '''

    X, y = load_breast_cancer(return_X_y=True)
    atom = ATOMClassifier(X, y)
    pytest.raises(ValueError, atom.balance)


# << ================== Test functionality ================== >>

def test_oversampling_method():
    ''' Assert that the oversampling method works as intended '''

    # Binary classification (1 is majority class)
    strats = [1.0, 0.9, 'minority', 'not majority', 'all']
    for strat in strats:
        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y, random_state=1)
        length = (atom.y_train == 0).sum()
        atom.balance(oversample=strat)
        assert (atom.y_train == 0).sum() != length

    # Multiclass classification
    strats = ['minority', 'not majority', 'all']
    for strat in strats:
        X, y = load_wine(return_X_y=True)
        atom = ATOMClassifier(X, y, random_state=1)
        length = (atom.y_train == 2).sum()
        atom.balance(oversample=strat)
        assert (atom.y_train == 2).sum() != length


def test_undersampling_method():
    ''' Assert that the undersampling method works as intended '''

    # Binary classification (1 is majority class)
    strats = [1.0, 0.7, 'majority', 'not minority', 'all']
    for strat in strats:
        X, y = load_breast_cancer(return_X_y=True)
        atom = ATOMClassifier(X, y, random_state=1)
        length = (atom.y_train == 1).sum()
        atom.balance(undersample=strat)
        assert (atom.y_train == 1).sum() != length

    # Multiclass classification
    strats = ['majority', 'not minority', 'all']
    for strat in strats:
        X, y = load_wine(return_X_y=True)
        atom = ATOMClassifier(X, y, random_state=1)
        length = (atom.y_train == 1).sum()
        atom.balance(undersample=strat)
        assert (atom.y_train == 1).sum() != length

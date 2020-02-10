# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the feature_insertion method of the ATOM class.

'''

# Import packages
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from atom import ATOMClassifier


# << ====================== Functions ====================== >>

def load_df(dataset):
    ''' Load dataset as pd.DataFrame '''

    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    data = pd.DataFrame(data, columns=columns)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y


# << ======================= Tests ========================= >>

# << =================== Test parameters =================== >>

def test_population_parameter():
    ''' Assert that the population parameter is set correctly '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    pytest.raises(ValueError, atom.feature_insertion, population=30)


def test_generations_parameter():
    ''' Assert that the generations parameter is set correctly '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    pytest.raises(ValueError, atom.feature_insertion, generations=0)


def test_n_features_parameter():
    ''' Assert that the n_features parameter is set correctly '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    with pytest.raises(ValueError, match=r"Invalid value for .*"):
        atom.feature_insertion(n_features=-2)
    with pytest.raises(ValueError, match=r".*be more than 1%.*"):
        atom.feature_insertion(n_features=23)


# << ================== Test functionality ================= >>

def test_attribute_genetic_algorithm():
    ''' Assert that the genetic_algorithm attribute is created '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    atom.feature_insertion(generations=3, population=200)
    assert hasattr(atom, 'genetic_algorithm')


def test_attribute_genetic_features():
    ''' Assert that the genetic_features attribute is created '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    atom.feature_insertion(generations=3, population=200)
    assert hasattr(atom, 'genetic_features')


def test_updated_dataset():
    ''' Assert that self.dataset has the new non-linear features '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    atom.feature_insertion(n_features=2, generations=3, population=200)
    assert len(atom.X.columns) == len(X.columns) + 2

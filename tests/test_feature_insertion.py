# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the feature_generator method of the ATOM class.

"""

# Import packages
import pytest
from sklearn.datasets import load_breast_cancer
from atom import ATOMClassifier


# << ====================== Variables ====================== >>

X_bin, y_bin = load_breast_cancer(return_X_y=True)


# << ======================= Tests ========================= >>

# << =================== Test parameters =================== >>

def test_population_parameter():
    """ Assert that the population parameter is set correctly """

    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(ValueError, atom.feature_generation, population=30)


def test_generations_parameter():
    """ Assert that the generations parameter is set correctly """

    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(ValueError, atom.feature_generation, generations=0)


def test_n_features_parameter():
    """ Assert that the n_features parameter is set correctly """

    atom = ATOMClassifier(X_bin, y_bin)
    with pytest.raises(ValueError, match=r".*value for the n_features.*"):
        atom.feature_generation(n_features=-2)
    with pytest.raises(ValueError, match=r".*should be <1%.*"):
        atom.feature_generation(n_features=23)


# << ================== Test functionality ================= >>

def test_attribute_genetic_features():
    """ Assert that the genetic_features attribute is created """

    atom = ATOMClassifier(X_bin, y_bin)
    atom.feature_generation(generations=3, population=200)
    assert hasattr(atom, 'genetic_features')


def test_updated_dataset():
    """ Assert that self.dataset has the new non-linear features """

    atom = ATOMClassifier(X_bin, y_bin)
    atom.feature_generation(n_features=2, generations=3, population=200)
    assert len(atom.X.columns) == X_bin.shape[1] + 2

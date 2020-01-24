# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the plot methods in the ATOM and BaseModel classes.

'''

# Import packages
import pytest
from sklearn.datasets import load_breast_cancer, load_wine, load_boston
from atom import ATOMClassifier, ATOMRegressor


# << ====================== Functions ====================== >>

X_bin, y_bin = load_breast_cancer(return_X_y=True)
X_class, y_class = load_wine(return_X_y=True)
X_reg, y_reg = load_boston(return_X_y=True)


# << ======================= Tests ========================= >>

def test_plot_correlation():
    ''' Assert that the plot_correlation method work as intended '''

    atom = ATOMClassifier(X_bin, y_bin)
    atom.plot_correlation(display=False)
    assert 1 == 1


def test_plot_PCA():
    ''' Assert that the plot_PCA method work as intended '''

    # When PCA is not called yet
    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(AttributeError, atom.plot_PCA)

    # When PCA has been called
    atom = ATOMClassifier(X_bin, y_bin)
    atom.feature_selection(strategy='pca', max_features=10)
    atom.plot_PCA(display=False)
    assert 1 == 1

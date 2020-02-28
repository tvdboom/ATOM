# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the models and pre-set metrics available.

'''

# Import packages
from sklearn.datasets import load_breast_cancer, load_wine, load_boston
from atom import ATOMClassifier, ATOMRegressor


# << ====================== Variables ====================== >>

# List of all the available models
model_list = ['BNB', 'GNB', 'MNB', 'GP', 'OLS', 'Ridge', 'Lasso', 'EN',
              'BR', 'LR', 'LDA', 'QDA', 'KNN', 'Tree', 'Bag', 'ET',
              'RF', 'AdaB', 'GBM', 'XGB', 'LGB', 'CatB', 'lSVM', 'kSVM',
              'PA', 'SGD', 'MLP']

# List of models that only work for regression/classification tasks
only_classification = ['BNB', 'GNB', 'MNB', 'LR', 'LDA', 'QDA']
only_regression = ['OLS', 'Lasso', 'EN', 'BR']

X_bin, y_bin = load_breast_cancer(return_X_y=True)
X_class, y_class = load_wine(return_X_y=True)
X_reg, y_reg = load_boston(return_X_y=True)


# << ======================= Tests ========================= >>

def test_models_attributes():
    ''' Assert that model subclasses have the right attributes '''

    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(models='lr', metric='f1', max_iter=0)
    assert atom.lr.name == 'LR'
    assert atom.lr.longname == 'Logistic Regression'


def test_models_binary():
    ''' Assert that the fit method works with all models for binary '''

    for model in [m for m in model_list if m not in only_regression]:
        atom = ATOMClassifier(X_bin, y_bin, random_state=1)
        atom.pipeline(models=model,
                      metric='f1',
                      max_iter=1,
                      init_points=1,
                      cv=1)
    assert 1 == 1  # Assert that all models ran wihtout errors


def test_models_multiclass():
    ''' Assert that the fit method works with all models for multiclass'''

    for model in [m for m in model_list if m not in only_regression]:
        atom = ATOMClassifier(X_class, y_class, random_state=1)
        atom.pipeline(models=model,
                      metric='f1_micro',
                      max_iter=1,
                      init_points=1,
                      cv=1)
    assert 3 == 3


def test_models_regression():
    ''' Assert that the fit method works with all models for regression '''

    for model in [m for m in model_list if m not in only_classification]:
        atom = ATOMRegressor(X_reg, y_reg, random_state=1)
        atom.pipeline(models=model,
                      metric='neg_mean_absolute_error',
                      max_iter=1,
                      init_points=1,
                      cv=1)
    assert 5 == 5

# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the models and pre-set metrics available.

'''

# Import packages
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, max_error
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

# List of pre-set binary classification metrics
mbin = ['tn', 'fp', 'fn', 'tp', 'ap']

# List of pre-set classification metrics
mclass = ['accuracy', 'auc', 'mcc', 'f1', 'hamming', 'jaccard', 'logloss',
          'precision', 'recall']

# List of pre-set regression metrics
# No MSLE cause can't handle neg inputs (when data is normalized to mean 0)
mreg = ['mae', 'max_error', 'mse', 'r2']


# << ====================== Functions ====================== >>

def load_df(dataset, nrows=250):
    ''' Load dataset as pd.DataFrame '''

    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    data = pd.DataFrame(data, columns=columns)
    data = data.sample(frac=1).reset_index(drop=True)
    X = data.drop('target', axis=1)
    y = data['target']
    return X.iloc[:nrows, :], y.iloc[:nrows]


# << ======================= Tests ========================= >>

def test_metrics_attributes():
    ''' Assert that self.metric has the right attributes '''

    # For classification
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    atom.fit(models='lr', metric='f1', max_iter=0)
    assert atom.metric.name == 'f1'
    assert atom.metric.longname == 'f1_score'
    assert atom.metric.gib
    assert not atom.metric.needs_proba
    assert not atom.metric.hamming.gib
    assert atom.metric.r2.gib
    assert not atom.metric.max_error.needs_proba

    # For regression
    X, y = load_df(load_boston())
    atom = ATOMRegressor(X, y)
    atom.fit(models='ols', metric='mse', max_iter=0)
    assert atom.metric.name == 'mse'
    assert atom.metric.longname == 'mean_squared_error'
    assert not atom.metric.gib
    assert not atom.metric.needs_proba


def test_models_attributes():
    ''' Assert that model subclasses have the right attributes '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    atom.fit(models='lr', metric='f1', max_iter=0)
    assert atom.lr.name == 'LR'
    assert atom.lr.longname == 'Logistic Regression'


def test_models_binary():
    ''' Assert that the fit method works with all models for binary '''

    X, y = load_df(load_breast_cancer())
    for model in [m for m in model_list if m not in only_regression]:
        atom = ATOMClassifier(X, y, random_state=1)
        atom.fit(models=model, metric='auc', max_iter=1, init_points=1, cv=1)
    assert 1 == 1  # Assert that all models ran wihtout errors


def test_metrics_binary():
    ''' Assert that the fit method works with all metrics for binary '''

    X, y = load_df(load_breast_cancer())
    for metric in mbin + mreg:
        atom = ATOMClassifier(X, y, random_state=1)
        atom.fit(models='tree', metric=metric, max_iter=0)
    assert 2 == 2  # Assert that all models ran wihtout errors


def test_models_multiclass():
    ''' Assert that the fit method works with all models for multiclass'''

    X, y = load_df(load_wine())
    for model in [m for m in model_list if m not in only_regression]:
        atom = ATOMClassifier(X, y, random_state=1)
        atom.fit(models=model, metric='f1', max_iter=1, init_points=1, cv=1)
    assert 3 == 3


def test_metrics_multiclass():
    ''' Assert that the fit method works with all metrics for multiclass'''

    X, y = load_df(load_wine())
    for metric in mclass + mreg + [f1_score]:
        atom = ATOMClassifier(X, y, random_state=1)
        atom.fit(models='pa', metric=metric, max_iter=0)
    assert 4 == 4


def test_models_regression():
    ''' Assert that the fit method works with all models for regression '''

    X, y = load_df(load_boston())
    for model in [m for m in model_list if m not in only_classification]:
        atom = ATOMRegressor(X, y, random_state=1)
        atom.fit(models=model, metric='mse', max_iter=1, init_points=1, cv=1)
    assert 5 == 5


def test_metrics_regression():
    ''' Assert that the fit method works with all metrics for regression '''

    X, y = load_df(load_boston())
    for metric in mreg + [max_error]:
        atom = ATOMRegressor(X, y, random_state=1)
        atom.fit(models='tree', metric=metric, max_iter=0)
    assert 6 == 6

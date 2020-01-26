# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the fit method of the ATOM class.

'''

# Import packages
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_boston
from atom import ATOMClassifier, ATOMRegressor
from atom.basemodel import BaseModel


# << ====================== Variables ====================== >>

X_dim4 = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
y_dim4_class = [0, 1, 1, 0]
y_dim4_reg = [1, 2, 4, 3]


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

# << ================ Test class variables ================= >>

def test_set_style():
    ''' Assert that the set_style classmethod works as intended '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    atom.set_style('white')
    assert ATOMClassifier.style == 'white'
    assert BaseModel.style == 'white'


def test_set_palette():
    ''' Assert that the set_palette classmethod works as intended '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    atom.set_palette('Blues')
    assert ATOMClassifier.palette == 'Blues'
    assert BaseModel.palette == 'Blues'


def test_set_title_fontsize():
    ''' Assert that the set_title_fontsize classmethod works as intended '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    atom.set_title_fontsize(21)
    assert ATOMClassifier.title_fs == 21
    assert BaseModel.title_fs == 21


def test_set_label_fontsize():
    ''' Assert that the set_label_fontsize classmethod works as intended '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    atom.set_label_fontsize(4)
    assert ATOMClassifier.label_fs == 4
    assert BaseModel.label_fs == 4


def test_set_tick_fontsize():
    ''' Assert that the set_tick_fontsize classmethod works as intended '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    atom.set_tick_fontsize(13)
    assert ATOMClassifier.tick_fs == 13
    assert BaseModel.tick_fs == 13


# << =================== Test parameters =================== >>

def test_models_parameter():
    ''' Assert that the models parameter is set correctly '''

    # When wrong type
    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, models=42, metric='f1')

    # Removes unknown or wrong models
    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(ValueError, atom.fit, models='test', metric='auc')
    pytest.raises(ValueError, atom.fit, models='OLS', metric='recall')

    atom = ATOMRegressor(X_dim4, y_dim4_reg)
    pytest.raises(ValueError, atom.fit, models='lda', metric='r2')

    # Makes it a list
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y)
    atom.fit('lr', 'precision', max_iter=0)
    assert isinstance(atom.models, list)


def test_metric_parameter():
    ''' Assert that the metric parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, models='lda', metric=42)
    pytest.raises(ValueError, atom.fit, models='lda', metric='test')
    pytest.raises(ValueError, atom.fit, models='lda', metric='ap')

    X, y = load_df(load_wine())
    atom = ATOMClassifier(X, y)
    pytest.raises(ValueError, atom.fit, models='lda', metric='fp')

    atom = ATOMRegressor(X_dim4, y_dim4_reg)
    pytest.raises(ValueError, atom.fit, models='sgd', metric='auc')


def test_greater_is_better_parameter():
    ''' Assert that the greater_is_better parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', greater_is_better=42)


def test_needs_proba_parameter():
    ''' Assert that the needs_proba parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', needs_proba='test')


def test_successive_halving_parameter():
    ''' Assert that the successive_halving parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', successive_halving='test')


def test_skip_iter_parameter():
    ''' Assert that the skip_iter parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', skip_iter=[0, 1])
    pytest.raises(ValueError, atom.fit, 'lda', 'f1', skip_iter=-2)


def test_max_iter_parameter():
    ''' Assert that the max_iter parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', max_iter=42.)
    pytest.raises(ValueError, atom.fit, 'lda', 'f1', max_iter=-2)


def test_max_time_parameter():
    ''' Assert that the max_time parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', max_time=[0, 1])
    pytest.raises(ValueError, atom.fit, 'lda', 'f1', max_time=-2)


def test_eps_parameter():
    ''' Assert that the eps parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', eps=[0, 1])
    pytest.raises(ValueError, atom.fit, 'lda', 'f1', eps=-2)


def test_batch_size_parameter():
    ''' Assert that the batch_size parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', batch_size=[0, 1])
    pytest.raises(ValueError, atom.fit, 'lda', 'f1', batch_size=-2)


def test_init_points_parameter():
    ''' Assert that the init_points parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', init_points=42.0)
    pytest.raises(ValueError, atom.fit, 'lda', 'f1', init_points=-2)


def test_plot_bo_parameter():
    ''' Assert that the plot_bo parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', plot_bo='test')


def test_cv_parameter():
    ''' Assert that the cv parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', cv=42.0)
    pytest.raises(ValueError, atom.fit, 'lda', 'f1', cv=-2)


def test_bagging_parameter():
    ''' Assert that the bagging parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(TypeError, atom.fit, 'lda', 'f1', bagging=42.0)
    pytest.raises(ValueError, atom.fit, 'lda', 'f1', bagging=-2)


# << ================== Test functionality ================= >>

def test_data_preparation():
    ''' Assert that the data_preparation function works as intended '''

    # For scaling and non-scaling models
    X, y = load_df(load_breast_cancer())
    for model in ['tree', 'lgb']:
        atom = ATOMClassifier(X, y, random_state=1)
        atom.fit(models=model, metric='f1', max_iter=0)
        assert isinstance(atom.data, dict)
        for set_ in ['X', 'X_train', 'X_test']:
            assert isinstance(atom.data[set_], pd.DataFrame)
        for set_ in ['y', 'y_train', 'y_test']:
            assert isinstance(atom.data[set_], pd.Series)

        if model == 'lgb':
            assert atom.data['X_train_scaled'].iloc[:, 1].mean() < 0.05
            assert atom.data['X_test_scaled'].iloc[:, 0].std() < 1.25


def test_successive_halving_results():
    ''' Assert that self.results is correctly created '''

    # Without bagging
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.fit(models=['tree', 'rf', 'xgb', 'lgb'],
             metric='f1',
             successive_halving=True,
             max_iter=0,
             bagging=0)
    assert isinstance(atom.results, list)
    assert len(atom.results) == 3

    # With bagging
    atom = ATOMClassifier(X, y, random_state=1)
    atom.fit(models=['tree', 'rf', 'xgb', 'lgb'],
             metric='f1',
             successive_halving=True,
             max_iter=0,
             bagging=5)
    assert isinstance(atom.results, list)
    assert len(atom.results) == 3


def test_skip_iter_results():
    ''' Assert that self.results is correctly created when skip_iter > 0 '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.fit(models=['tree', 'rf', 'xgb', 'lgb'],
             metric='f1',
             successive_halving=True,
             skip_iter=1,
             max_iter=0)
    assert isinstance(atom.results, list)
    assert len(atom.results) == 2


def test_errors_in_models():
    ''' Assert that errors when running models are handled correctly '''

    X, y = load_df(load_boston())
    X.iloc[2, 3] = np.NaN  # Make it fail
    atom = ATOMRegressor(X, y, random_state=1)
    atom.fit(models=['Tree', 'XGB'],
             metric='msle',  # This metric will cause an error for the LGB
             max_iter=0)
    assert 'Tree' in atom.errors.keys()
    assert 'Tree' not in atom.models


def test_lower_case_model_attribute():
    ''' Assert that model attributes can be called with lowercase as well '''

    atom = ATOMClassifier(X_dim4, y_dim4_class,
                          random_state=1, verbose=1)  # vb=1 for coverage :D
    atom.fit(models='tree', metric='f1', max_iter=0)
    assert atom.Tree == atom.tree


#def test_plot_bo():
#    ''' Assert that plot_bo works as intended '''
#
#    X, y = load_df(load_breast_cancer())
#    atom = ATOMClassifier(X, y, random_state=1)
#    atom.fit(models='tree', metric='f1',
#             max_iter=15, init_points=10, plot_bo=True)
#    assert 1 == 1  # Test it ran correctly


def test_different_cv_values():
    ''' Assert that changing the cv parameter works as intended '''

    # For classification
    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.fit(models='pa', metric='auc', max_iter=5, cv=3)
    assert 1 == 1

    # For regression
    X, y = load_df(load_boston())
    atom = ATOMRegressor(X, y, random_state=1)
    atom.fit(models='tree', metric='r2', max_iter=5, cv=3)
    assert 2 == 2


def test_model_attributes():
    ''' Assert that the model subclass has all attributes set '''

    X, y = load_df(load_breast_cancer())
    atom = ATOMClassifier(X, y, random_state=1)
    atom.fit(models=['tree', 'lgb'], metric='f1', max_iter=5, cv=1)
    assert 'params' in atom.Tree.BO.keys()
    assert 'score' in atom.Tree.BO.keys()
    assert hasattr(atom.Tree, 'best_params')
    assert hasattr(atom.lgb, 'best_model')
    assert hasattr(atom.Tree, 'best_model_fit')
    assert hasattr(atom.Tree, 'predict_train')
    assert hasattr(atom.lgb, 'predict_test')
    assert hasattr(atom.Tree, 'predict_proba_train')
    assert hasattr(atom.Tree, 'predict_proba_test')
    assert hasattr(atom.Tree, 'score_train')
    assert hasattr(atom.lgb, 'score_test')


def test_bagging():
    ''' Assert that bagging workas as intended '''

    X, y = load_df(load_breast_cancer())

    # For metric needs proba
    atom = ATOMClassifier(X, y, random_state=1)
    atom.fit(models=['tree', 'lgb'], metric='auc', max_iter=1, cv=1, bagging=5)
    assert hasattr(atom.Tree, 'bagging_scores')

    # For metric needs proba but hasn't attr
    atom = ATOMClassifier(X, y, random_state=1)
    atom.fit(models='PA', metric='auc', max_iter=1, cv=1, bagging=5)
    assert hasattr(atom.PA, 'bagging_scores')

    # For metric does not needs proba
    atom = ATOMClassifier(X, y, random_state=1)
    atom.fit(models='tree', metric='f1', max_iter=1, cv=1, bagging=5)
    assert hasattr(atom.Tree, 'bagging_scores')

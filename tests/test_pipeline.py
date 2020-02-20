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
from sklearn.metrics import get_scorer, f1_score
from sklearn.datasets import load_breast_cancer, load_wine, load_boston
from atom import ATOMClassifier, ATOMRegressor


# << ====================== Variables ====================== >>

X_dim4 = [[2, 0, 1], [2, 3, 4], [5, 2, 7], [8, 9, 10]]
y_dim4_class = [0, 1, 1, 0]
y_dim4_reg = [1, 2, 4, 3]
X_bin, y_bin = load_breast_cancer(return_X_y=True)
X_class, y_class = load_wine(return_X_y=True)
X_reg, y_reg = load_boston(return_X_y=True)


# << ======================= Tests ========================= >>

# << =================== Test parameters =================== >>

def test_models_parameter():
    ''' Assert that the models parameter is set correctly '''

    # Raises error when unknown, wrong or duplicate models
    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(ValueError, atom.pipeline, models='test')
    pytest.raises(ValueError, atom.pipeline, models='OLS')
    pytest.raises(ValueError, atom.pipeline, models=['lda', 'lda'])

    atom = ATOMRegressor(X_dim4, y_dim4_reg)
    pytest.raises(ValueError, atom.pipeline, models='lda', metric='r2')

    # Makes it a list
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline('lr', 'precision', max_iter=0)
    assert isinstance(atom.models, list)


def test_metric_parameter():
    ''' Assert that the metric parameter is set correctly '''

    # Test default metrics
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline('lr', max_iter=0)
    assert atom.metric.name == 'f1'

    atom = ATOMClassifier(X_class, y_class)
    atom.pipeline('lr', max_iter=0)
    assert atom.metric.name == 'f1_weighted'

    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline('ols', max_iter=0)
    assert atom.metric.name == 'r2'

    # Test unknown metric
    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(ValueError, atom.pipeline, models='lda', metric='unknown')

    # Test custom metric
    def metric_func(x, y):
        return x, y
    pytest.raises(ValueError, atom.pipeline, models='lda', metric=metric_func)

    atom.pipeline('lr', metric=f1_score, max_iter=0)
    assert 1 == 1

    # Test scoring metric
    atom = ATOMRegressor(X_dim4, y_dim4_reg)
    scorer = get_scorer('neg_mean_squared_error')
    atom.pipeline('ols', metric=scorer, max_iter=0)
    assert 2 == 2


def test_skip_iter_parameter():
    ''' Assert that the skip_iter parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(ValueError, atom.pipeline, 'lda', 'f1', skip_iter=-2)


def test_max_iter_parameter():
    ''' Assert that the max_iter parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(ValueError, atom.pipeline, 'lda', 'f1', max_iter=-2)
    pytest.raises(ValueError, atom.pipeline, 'lda', 'f1', max_iter=[2, 2])


def test_max_time_parameter():
    ''' Assert that the max_time parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(ValueError, atom.pipeline, 'lda', 'f1', max_time=-2)
    pytest.raises(ValueError, atom.pipeline, 'lda', 'f1', max_time=[2, 2])


def test_init_points_parameter():
    ''' Assert that the init_points parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(ValueError, atom.pipeline, 'lda', 'f1', init_points=-2)
    pytest.raises(ValueError, atom.pipeline, 'lda', 'f1', init_points=[2, 2])


def test_cv_parameter():
    ''' Assert that the cv parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(ValueError, atom.pipeline, 'lda', 'f1', cv=-2)
    pytest.raises(ValueError, atom.pipeline, 'lda', 'f1', cv=[2, 2])


def test_bagging_parameter():
    ''' Assert that the bagging parameter is set correctly '''

    atom = ATOMClassifier(X_dim4, y_dim4_class)
    pytest.raises(ValueError, atom.pipeline, 'lda', 'f1', bagging=-2)


# << ================== Test functionality ================= >>

def test_data_preparation():
    ''' Assert that the data_preparation function works as intended '''

    # For scaling and non-scaling models
    for model in ['tree', 'lgb']:
        atom = ATOMClassifier(X_bin, y_bin, random_state=1)
        atom.pipeline(models=model, metric='f1', max_iter=0)
        assert isinstance(atom.data, dict)
        for set_ in ['X', 'X_train', 'X_test']:
            assert isinstance(atom.data[set_], pd.DataFrame)
        for set_ in ['y', 'y_train', 'y_test']:
            assert isinstance(atom.data[set_], pd.Series)

        if model == 'lgb':
            assert atom.data['X_train_scaled'].iloc[:, 1].mean() < 0.05
            assert atom.data['X_test_scaled'].iloc[:, 0].std() < 1.25


def test_successive_halving_scores():
    ''' Assert that self.scores is correctly created '''

    # Without bagging
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(models=['tree', 'rf', 'xgb', 'lgb'],
                  metric='f1',
                  successive_halving=True,
                  max_iter=0,
                  bagging=0)
    assert isinstance(atom.scores, list)
    assert len(atom.scores) == 3

    # With bagging
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(models=['tree', 'rf', 'xgb', 'lgb'],
                  metric='f1',
                  successive_halving=True,
                  max_iter=0,
                  bagging=5)
    assert isinstance(atom.scores, list)
    assert len(atom.scores) == 3


def test_skip_iter_scores():
    ''' Assert that self.scores is correctly created when skip_iter > 0 '''

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(models=['tree', 'rf', 'xgb', 'lgb'],
                  metric='f1',
                  successive_halving=True,
                  skip_iter=1,
                  max_iter=0)
    assert isinstance(atom.scores, list)
    assert len(atom.scores) == 2


def test_errors_in_models():
    ''' Assert that errors when running models are handled correctly '''

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.X.iloc[2, 3] = np.NaN  # Make it fail
    atom.update('X')
    atom.pipeline(models=['Tree', 'XGB'],
                  metric='neg_mean_squared_log_error',
                  max_iter=0)
    assert 'Tree' in atom.errors.keys()
    assert 'Tree' not in atom.models


def test_lower_case_model_attribute():
    ''' Assert that model attributes can be called with lowercase as well '''

    atom = ATOMClassifier(X_dim4, y_dim4_class,
                          random_state=1, verbose=1)  # vb=1 for coverage :D
    atom.pipeline(models='tree', metric='f1', max_iter=0)
    assert atom.Tree == atom.tree


def test_plot_bo():
    ''' Assert that plot_bo works as intended '''

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline('tree', 'f1',  max_iter=15, init_points=10, plot_bo=True)
    assert 1 == 1


def test_different_cv_values():
    ''' Assert that changing the cv parameter works as intended '''

    # For classification
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(models='pa', metric='roc_auc', max_iter=5, cv=3)
    assert 1 == 1

    # For regression
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline(models='tree', metric='r2', max_iter=5, cv=3)
    assert 2 == 2


def test_model_attributes():
    ''' Assert that the model subclass has all attributes set '''

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(models=['tree', 'lgb'], metric='f1', max_iter=5, cv=1)
    assert 'params' in atom.Tree.BO.keys()
    assert 'score' in atom.Tree.BO.keys()
    assert 'time' in atom.Tree.BO.keys()
    assert 'total_time' in atom.Tree.BO.keys()
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

    # For metric needs proba
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(models=['tree', 'lgb'], max_iter=1, cv=1, bagging=5)
    assert hasattr(atom.Tree, 'bagging_scores')

    # For metric needs proba but hasn't attr
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(models='PA', metric='roc_auc', max_iter=1, cv=1, bagging=5)
    assert hasattr(atom.PA, 'bagging_scores')

    # For metric does not needs proba
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(models='tree', metric='f1', max_iter=1, cv=1, bagging=5)
    assert hasattr(atom.Tree, 'bagging_scores')


def test_winner_attribute():
    ''' Assert that the best model is attached to the winner attribute '''

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['lr', 'tree', 'lgb'], 'f1', max_iter=0)
    assert atom.winner.name == 'LGB'

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline(['br', 'ols', 'tree'], 'max_error', max_iter=0)
    assert atom.winner.name == 'Tree'

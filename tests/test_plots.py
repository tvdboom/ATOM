# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the plot methods in the ATOM and BaseModel classes.

'''

# Import packages
import pytest
from sklearn.metrics import f1_score, recall_score, get_scorer
from sklearn.datasets import load_breast_cancer, load_wine, load_boston
from atom import ATOMClassifier, ATOMRegressor


# << ====================== Variables ====================== >>

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

    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(AttributeError, atom.plot_PCA)  # When no PCA attribute
    atom.feature_selection(strategy='pca', max_features=10)

    # When show is invalid value
    pytest.raises(ValueError, atom.plot_PCA, -2)

    # When correct
    atom.plot_PCA(display=False)
    assert 1 == 1


def test_plot_bagging():
    ''' Assert that the plot_bagging method work as intended '''

    # When fit is not called yet
    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(AttributeError, atom.plot_bagging)

    # When fit is called without bagging
    atom.pipeline('tree', 'f1', max_iter=2, bagging=0)
    pytest.raises(AttributeError, atom.plot_bagging)

    # When model is unknown
    atom.pipeline('tree', 'f1', max_iter=2, bagging=3)
    pytest.raises(ValueError, atom.plot_bagging, models='unknown')

    # Without successive_halving
    atom.plot_bagging(display=False)
    atom.tree.plot_bagging(display=False)
    assert 1 == 1


def test_plot_successive_halving():
    ''' Assert that the plot_successive_halving method work as intended '''

    # When fit is not called yet
    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(AttributeError, atom.plot_successive_halving)

    # When fit is called without successive_halving
    atom.pipeline(['tree', 'lgb'], 'f1', successive_halving=False, max_iter=0)
    pytest.raises(AttributeError, atom.plot_successive_halving)

    # When model is unknown
    atom.pipeline(['tree', 'lgb'], 'f1', successive_halving=True, max_iter=0)
    pytest.raises(ValueError, atom.plot_successive_halving, models='unknown')

    # When correct
    atom.pipeline(['tree', 'lgb'], 'f1', successive_halving=True, max_iter=3)
    atom.plot_successive_halving(display=False)
    atom.tree.plot_successive_halving(display=False)
    assert 1 == 1


def test_plot_ROC():
    ''' Assert that the plot_ROC method work as intended '''

    # When task is not binary
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline('tree', 'r2', max_iter=0)
    pytest.raises(AttributeError, atom.plot_ROC)

    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(AttributeError, atom.plot_ROC)  # When fit is not called yet
    atom.pipeline('tree', 'r2', max_iter=0)

    # When model is unknown
    pytest.raises(ValueError, atom.plot_ROC, 'unknown')

    # When correct
    atom.plot_ROC(display=False)
    atom.tree.plot_ROC(display=False)
    assert 1 == 1


def test_plot_PRC():
    ''' Assert that the plot_PRC method work as intended '''

    # When task is not binary
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline('tree', 'r2', max_iter=0)
    pytest.raises(AttributeError, atom.plot_PRC)

    atom = ATOMClassifier(X_bin, y_bin)
    pytest.raises(AttributeError, atom.plot_PRC)  # When fit is not called yet
    atom.pipeline('tree', 'r2', max_iter=0)

    # When model is unknown
    pytest.raises(ValueError, atom.plot_PRC, 'unknown')

    # When correct
    atom.plot_PRC(display=False)
    atom.tree.plot_PRC(display=False)
    assert 1 == 1


def test_plot_permutation_importance():
    ''' Assert that the plot_permutation_importance method work as intended '''

    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(['tree', 'lr'], 'f1', max_iter=0)

    # When show is invalid value
    pytest.raises(ValueError, atom.plot_permutation_importance, show=-2)

    # When n_repeats is invalid value
    pytest.raises(ValueError, atom.plot_permutation_importance, n_repeats=-2)

    # When model is unknown
    pytest.raises(ValueError, atom.plot_permutation_importance, 'unknown')

    # When correct
    atom.plot_permutation_importance(display=False)
    atom.tree.plot_permutation_importance(display=False)
    assert 1 == 1


def test_plot_feature_importance():
    ''' Assert that the plot_feature_importance method work as intended '''

    # When model not a tree-based model
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline('pa', 'r2', max_iter=0)
    pytest.raises(AttributeError, atom.pa.plot_feature_importance)

    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(['tree', 'bag'], 'f1', max_iter=0)

    # When show is invalid value
    pytest.raises(ValueError, atom.plot_feature_importance, show=-2)

    # When model is unknown
    pytest.raises(ValueError, atom.plot_feature_importance, 'unknown')

    # When correct
    atom.plot_feature_importance(display=False)
    atom.bag.plot_feature_importance(display=False)
    assert 1 == 1


def test_plot_confusion_matrix():
    ''' Assert that the plot_confusion_matrix method work as intended '''

    # When task is not classification
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline('ols', 'r2', max_iter=2, init_points=2)
    pytest.raises(AttributeError, atom.ols.plot_confusion_matrix)

    # For binary classification tasks
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(['lda', 'et'], max_iter=2, init_points=2)

    # When model is unknown
    pytest.raises(ValueError, atom.plot_confusion_matrix, 'unknown')

    # When correct
    atom.plot_confusion_matrix(normalize=True, display=False)
    atom.lda.plot_confusion_matrix(normalize=False, display=False)
    assert 1 == 1

    # For multiclass classification tasks
    atom = ATOMClassifier(X_class, y_class)
    atom.pipeline(['lda', 'et'], max_iter=2, init_points=2)

    # Multiclass and multiple models not supported
    pytest.raises(NotImplementedError, atom.plot_confusion_matrix)
    atom.lda.plot_confusion_matrix(normalize=True, display=False)
    assert 1 == 1


def test_plot_threshold():
    ''' Assert that the plot_threshold method work as intended '''

    # When task is not binary
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline('tree', 'r2', max_iter=0)
    pytest.raises(AttributeError, atom.tree.plot_threshold)

    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(['lda', 'et'], 'f1', max_iter=0)

    # When invalid model or metric
    pytest.raises(ValueError, atom.plot_threshold, 'unknown')
    pytest.raises(ValueError, atom.plot_threshold, 'lda', 'unknown')

    # For metric is None, functions or scorers
    scorer = get_scorer('f1_macro')
    atom.lda.plot_threshold(display=False)
    atom.plot_threshold(metric=[f1_score, recall_score, 'r2'], display=False)
    atom.lda.plot_threshold([scorer, 'precision'], display=False)
    assert 1 == 1


def test_plot_probabilities():
    ''' Assert that the plot_probabilities method work as intended '''

    # When task is not classification
    atom = ATOMRegressor(X_reg, y_reg)
    atom.pipeline('tree', 'r2', max_iter=0)
    pytest.raises(AttributeError, atom.tree.plot_probabilities)

    # When model hasn't the predict_proba method
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline('pa', 'r2', max_iter=0)
    pytest.raises(ValueError, atom.pa.plot_probabilities)

    # When invalid model
    pytest.raises(ValueError, atom.plot_probabilities, models='unknown')

    # For target is string
    y = ['a' if i == 0 else 'b' for i in y_bin]
    atom = ATOMClassifier(X_bin, y)
    atom.pipeline(['tree', 'lda'], 'f1', max_iter=0)
    atom.tree.plot_probabilities(target='a', display=False)
    atom.plot_probabilities(target='a', display=False)
    assert 1 == 1

    # For target is numerical
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(['tree', 'lda'], 'f1', max_iter=0)
    atom.tree.plot_probabilities(target=0, display=False)
    atom.plot_probabilities(target=1, display=False)
    assert 2 == 2

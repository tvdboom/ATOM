# coding: utf-8

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for the plot methods in the ATOM and BaseModel classes.

'''

# Import packages
import pytest
from sklearn.metrics import f1_score, recall_score
from sklearn.datasets import load_breast_cancer, load_wine, load_boston
from atom import ATOMClassifier, ATOMRegressor


# << ====================== Variables ====================== >>

X_bin, y_bin = load_breast_cancer(return_X_y=True)
X_class, y_class = load_wine(return_X_y=True)
X_reg, y_reg = load_boston(return_X_y=True)


# << ======================= Tests ========================= >>

# << ======================= ATOM ========================== >>

#def test_plot_correlation():
#    ''' Assert that the plot_correlation method work as intended '''
#
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.plot_correlation(display=False)
#    assert 1 == 1
#
#
#def test_plot_PCA():
#    ''' Assert that the plot_PCA method work as intended '''
#
#    # When PCA is not called yet
#    atom = ATOMClassifier(X_bin, y_bin)
#    pytest.raises(AttributeError, atom.plot_PCA)
#
#    # When PCA has been called
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.feature_selection(strategy='pca', max_features=10)
#    atom.plot_PCA(display=False)
#    assert 1 == 1
#
#
#def test_plot_bagging():
#    ''' Assert that the plot_bagging method work as intended '''
#
#    # When fit is not called yet
#    atom = ATOMClassifier(X_bin, y_bin)
#    pytest.raises(AttributeError, atom.plot_bagging)
#
#    # When fit is called without bagging
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit('tree', 'f1', max_iter=2, bagging=0)
#    pytest.raises(AttributeError, atom.plot_bagging)
#
#    # Without successive_halving
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit('tree', 'f1', max_iter=2, bagging=5)
#    atom.plot_bagging(display=False)
#    assert 1 == 1
#
#    # With successive_halving
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit(['tree', 'lgb'], 'f1', successive_halving=True, bagging=5)
#    atom.plot_bagging(display=False)
#    assert 2 == 2
#
#
#def test_plot_successive_halving():
#    ''' Assert that the plot_successive_halving method work as intended '''
#
#    # When fit is not called yet
#    atom = ATOMClassifier(X_bin, y_bin)
#    pytest.raises(AttributeError, atom.plot_successive_halving)
#
#    # When fit is called without successive_halving
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit(['tree', 'lgb'], 'f1', successive_halving=False, max_iter=0)
#    pytest.raises(AttributeError, atom.plot_successive_halving)
#
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit(['tree', 'lgb'], 'f1', successive_halving=True, max_iter=3)
#    atom.plot_successive_halving(display=False)
#    assert 1 == 1
#
#
#def test_plot_ROC():
#    ''' Assert that the plot_ROC method work as intended '''
#
#    # When task is not binary
#    atom = ATOMRegressor(X_reg, y_reg)
#    pytest.raises(AttributeError, atom.plot_ROC)
#
#    # When fit is not called yet
#    atom = ATOMClassifier(X_bin, y_bin)
#    pytest.raises(AttributeError, atom.plot_ROC)
#
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit(['tree', 'lgb'], 'f1', max_iter=0)
#    atom.plot_ROC(display=False)
#    assert 1 == 1
#
#
#def test_plot_PRC():
#    ''' Assert that the plot_PRC method work as intended '''
#
#    # When task is not binary
#    atom = ATOMRegressor(X_reg, y_reg)
#    pytest.raises(AttributeError, atom.plot_PRC)
#
#    # When fit is not called yet
#    atom = ATOMClassifier(X_bin, y_bin)
#    pytest.raises(AttributeError, atom.plot_PRC)
#
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit(['tree', 'lgb'], 'f1', max_iter=0)
#    atom.plot_PRC(display=False)
#    assert 1 == 1
#
#
## << ===================== BaseModel ====================== >>
#
#def test_plot_threshold():
#    ''' Assert that the plot_threshold method work as intended '''
#
#    # When task is not binary
#    atom = ATOMRegressor(X_reg, y_reg)
#    atom.fit('tree', 'r2', max_iter=0)
#    pytest.raises(AttributeError, atom.tree.plot_threshold)
#
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit('tree', 'f1', max_iter=0)
#
#    # For metric is None
#    atom.tree.plot_threshold(display=False)
#    assert 1 == 1
#
#    # For metric is list with functions
#    atom.tree.plot_threshold([f1_score, recall_score, 'fp'], display=False)
#    assert 2 == 2
#
#
#def test_plot_probabilities():
#    ''' Assert that the plot_probabilities method work as intended '''
#
#    # When task is not classification
#    atom = ATOMRegressor(X_reg, y_reg)
#    atom.fit('tree', 'r2', max_iter=0)
#    pytest.raises(AttributeError, atom.tree.plot_probabilities)
#
#    # For target is string
#    y = ['a' if i == 0 else 'b' for i in y_bin]
#    atom = ATOMClassifier(X_bin, y)
#    atom.fit('tree', 'f1', max_iter=0)
#    atom.tree.plot_probabilities(target='a', display=False)
#    assert 1 == 1
#
#    # For target is numerical
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit('tree', 'f1', max_iter=0)
#    atom.tree.plot_probabilities(target=0, display=False)
#    assert 2 == 2
#
#
#def test_plot_permutation_importance():
#    ''' Assert that the plot_permutation_importance method work as intended '''
#
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit('tree', 'f1', max_iter=0)
#    atom.tree.plot_permutation_importance(display=False)
#    assert 1 == 1
#
#
#def test_plot_feature_importance():
#    ''' Assert that the plot_feature_importance method work as intended '''
#
#    # When model not a tree-based model
#    atom = ATOMRegressor(X_reg, y_reg)
#    atom.fit('pa', 'r2', max_iter=0)
#    pytest.raises(AttributeError, atom.pa.plot_probabilities)
#
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit(['tree', 'bag'], 'f1', max_iter=0)
#    atom.tree.plot_feature_importance(display=False)
#    atom.bag.plot_feature_importance(display=False)
#    assert 1 == 1
#
#
#def test_plot_ROC_BaseModel():
#    ''' Assert that the plot_ROC method work as intended '''
#
#    # When task is not binary
#    atom = ATOMRegressor(X_reg, y_reg)
#    atom.fit('tree', 'r2', max_iter=0)
#    pytest.raises(AttributeError, atom.tree.plot_ROC)
#
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit('tree', 'f1', max_iter=0)
#    atom.tree.plot_ROC(display=False)
#    assert 1 == 1
#
#
#def test_plot_PRC_BaseModel():
#    ''' Assert that the plot_PRC method work as intended '''
#
#    # When task is not binary
#    atom = ATOMRegressor(X_reg, y_reg)
#    atom.fit('tree', 'r2', max_iter=0)
#    pytest.raises(AttributeError, atom.tree.plot_PRC)
#
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit('tree', 'f1', max_iter=0)
#    atom.tree.plot_PRC(display=False)
#    assert 1 == 1
#
#
#def test_plot_confusion_matrix():
#    ''' Assert that the plot_confusion_matrix method work as intended '''
#
#    # When task is not classification
#    atom = ATOMRegressor(X_reg, y_reg)
#    atom.fit('tree', 'r2', max_iter=0)
#    pytest.raises(AttributeError, atom.tree.plot_confusion_matrix)
#
#    # For binary classification tasks
#    atom = ATOMClassifier(X_bin, y_bin)
#    atom.fit('tree', 'f1', max_iter=0)
#    atom.tree.plot_confusion_matrix(display=False)
#    assert 1 == 1
#
#    # For multiclass classification tasks
#    atom = ATOMClassifier(X_class, y_class)
#    atom.fit('tree', 'f1', max_iter=0)
#    atom.tree.plot_confusion_matrix(normalize=False, display=False)
#    assert 1 == 1
#
#
##def test_save():
##    ''' Assert that the save method work as intended '''
##
##    atom = ATOMClassifier(X_class, y_class)
##    atom.fit('tree', 'f1', max_iter=0)
##    atom.tree.save()

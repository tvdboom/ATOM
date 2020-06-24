# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for plots.py

"""

# Import packages
import glob
import pytest
from sklearn.metrics import f1_score, recall_score, get_scorer

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from .utils import (
    FILE_DIR, X_bin, y_bin, X_class, y_class, X_reg, y_reg
    )


# << ======================= Tests ========================= >>

def test_plot_correlation():
    """Assert that the plot_correlation method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_correlation(filename=FILE_DIR + 'correlation', display=False)
    assert glob.glob(FILE_DIR + 'correlation.png')


def test_plot_pca():
    """Assert that the plot_pca method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # When no PCA attribute
    pytest.raises(AttributeError, atom.plot_pca)

    # When correct
    atom.feature_selection(strategy='PCA', n_features=10)
    atom.plot_pca(filename=FILE_DIR + 'pca', display=False)
    assert glob.glob(FILE_DIR + 'pca.png')


def test_plot_components():
    """Assert that the plot_components method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # When no PCA attribute
    pytest.raises(AttributeError, atom.plot_components)
    atom.feature_selection(strategy='PCA', n_features=10)

    # When show is invalid value
    pytest.raises(ValueError, atom.plot_components, -2)

    # When correct (test if show is converted to max components)
    atom.plot_components(show=100,
                         filename=FILE_DIR + 'components',
                         display=False)
    assert glob.glob(FILE_DIR + 'components.png')


def test_plot_rfecv():
    """Assert that the plot_rfecv method work as intended """
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # When no RFECV attribute
    pytest.raises(AttributeError, atom.plot_rfecv)

    # When scoring is unspecified
    atom.feature_selection(strategy='rfecv', solver='lgb', n_features=27)
    atom.plot_rfecv(filename=FILE_DIR + 'rfecv1', display=False)
    assert glob.glob(FILE_DIR + 'rfecv1.png')

    # When scoring is specified
    atom.feature_selection(strategy='rfecv',
                           solver='lgb',
                           scoring='recall',
                           n_features=27)
    atom.plot_rfecv(filename=FILE_DIR + 'rfecv2', display=False)
    assert glob.glob(FILE_DIR + 'rfecv2.png')


def test_plot_bagging():
    """Assert that the plot_bagging method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # When fit is not called yet
    pytest.raises(AttributeError, atom.plot_bagging)

    # When fit is called without bagging
    atom.pipeline('Tree', 'f1', n_calls=2, n_random_starts=1)
    pytest.raises(AttributeError, atom.plot_bagging)

    # When model is unknown
    atom.pipeline('Tree', 'f1', n_calls=2, n_random_starts=1, bagging=5)
    pytest.raises(ValueError, atom.plot_bagging, models='unknown')

    # Without successive_halving
    atom.plot_bagging(filename=FILE_DIR + 'bagging1', display=False)
    atom.tree.plot_bagging(filename=FILE_DIR + 'bagging2', display=False)
    assert glob.glob(FILE_DIR + 'bagging1.png')
    assert glob.glob(FILE_DIR + 'bagging2.png')


def test_plot_successive_halving():
    """Assert that the plot_successive_halving method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # When fit is not called yet
    pytest.raises(AttributeError, atom.plot_successive_halving)

    # When the pipeline didn't run with successive_halving
    atom.pipeline('Tree', 'f1', n_calls=2, n_random_starts=1)
    pytest.raises(AttributeError, atom.plot_successive_halving)

    # When model is unknown
    atom.successive_halving(['Tree', 'LGB'], n_calls=2, n_random_starts=1)
    pytest.raises(ValueError, atom.plot_successive_halving, models='unknown')

    # When correct (without bagging)
    atom.successive_halving(['Tree', 'LGB'], n_calls=2, n_random_starts=1)
    atom.plot_successive_halving(filename=FILE_DIR + 'sh1', display=False)
    atom.tree.plot_successive_halving(filename=FILE_DIR + 'sh2', display=False)
    assert glob.glob(FILE_DIR + 'sh1.png')
    assert glob.glob(FILE_DIR + 'sh2.png')

    # When correct (with bagging)
    atom.successive_halving(['tree', 'lgb'], bagging=3)
    atom.plot_successive_halving(filename=FILE_DIR + 'sh3', display=False)
    atom.tree.plot_successive_halving(filename=FILE_DIR + 'sh4', display=False)
    assert glob.glob(FILE_DIR + 'sh3.png')
    assert glob.glob(FILE_DIR + 'sh4.png')


def test_plot_learning_curve():
    """Assert that the plot_learning_curve method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # When fit is not called yet
    pytest.raises(AttributeError, atom.plot_learning_curve)

    # When the pipeline didn't run with train_sizing
    atom.pipeline('Tree')
    pytest.raises(AttributeError, atom.plot_learning_curve)

    # When model is unknown
    atom.train_sizing(['Tree', 'LGB'], 'f1')
    pytest.raises(ValueError, atom.plot_learning_curve, models='unknown')

    # When correct (without bagging)
    atom.train_sizing(['Tree', 'LGB'], 'f1', n_calls=2, n_random_starts=1)
    atom.plot_learning_curve(filename=FILE_DIR + 'lc1', display=False)
    atom.tree.plot_learning_curve(filename=FILE_DIR + 'lc2', display=False)
    assert glob.glob(FILE_DIR + 'lc1.png')
    assert glob.glob(FILE_DIR + 'lc2.png')

    # When correct (with bagging)
    atom.train_sizing(['Tree', 'LGB'], n_calls=2, n_random_starts=1, bagging=3)
    atom.plot_learning_curve(filename=FILE_DIR + 'lc3', display=False)
    atom.tree.plot_learning_curve(filename=FILE_DIR + 'lc4', display=False)
    assert glob.glob(FILE_DIR + 'lc3.png')
    assert glob.glob(FILE_DIR + 'lc4.png')


def test_plot_roc():
    """Assert that the plot_roc method work as intended."""
    # When task is not binary
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline('Tree', 'r2')
    pytest.raises(AttributeError, atom.plot_roc)

    # When task is binary
    atom = ATOMClassifier(X_bin, y_bin)

    # When fit is not called yet
    pytest.raises(AttributeError, atom.plot_roc)

    # When model is unknown
    atom.pipeline(['Tree', 'LDA'], 'r2')
    pytest.raises(ValueError, atom.plot_roc, 'unknown')

    # When correct
    atom.plot_roc(filename=FILE_DIR + 'roc1', display=False)
    atom.tree.plot_roc(filename=FILE_DIR + 'roc2', display=False)
    assert glob.glob(FILE_DIR + 'roc1.png')
    assert glob.glob(FILE_DIR + 'roc2.png')


def test_plot_prc():
    """Assert that the plot_prc method work as intended."""
    # When task is not binary
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline('Tree', 'r2')
    pytest.raises(AttributeError, atom.plot_prc)

    # When task is binary
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(AttributeError, atom.plot_prc)  # When fit is not called yet
    atom.pipeline(['Tree', 'LDA'], 'r2')

    # When model is unknown
    pytest.raises(ValueError, atom.plot_prc, 'unknown')

    # When correct
    atom.plot_prc(filename=FILE_DIR + 'prc1', display=False)
    atom.tree.plot_prc(filename=FILE_DIR + 'prc2', display=False)
    assert glob.glob(FILE_DIR + 'prc1.png')
    assert glob.glob(FILE_DIR + 'prc2.png')


def test_plot_permutation_importance():
    """Assert that the plot_permutation_importance method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['Tree', 'LR'], 'f1')

    # When show is invalid value
    pytest.raises(ValueError, atom.plot_permutation_importance, show=-2)

    # When n_repeats is invalid value
    pytest.raises(ValueError, atom.plot_permutation_importance, n_repeats=-2)

    # When model is unknown
    pytest.raises(ValueError, atom.plot_permutation_importance, 'unknown')

    # When correct
    atom.plot_permutation_importance(filename=FILE_DIR + 'permutation1',
                                     display=False)
    atom.tree.plot_permutation_importance(filename=FILE_DIR + 'permutation2',
                                          display=False)
    assert glob.glob(FILE_DIR + 'permutation1.png')
    assert glob.glob(FILE_DIR + 'permutation2.png')


def test_plot_feature_importance():
    """Assert that the plot_feature_importance method work as intended."""
    # When model not a tree-based model
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline('PA', 'r2')
    pytest.raises(AttributeError, atom.pa.plot_feature_importance)

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['Tree', 'Bag'], 'f1')

    # When show is invalid value
    pytest.raises(ValueError, atom.plot_feature_importance, show=-2)

    # When model is unknown
    pytest.raises(ValueError, atom.plot_feature_importance, 'unknown')

    # When correct
    atom.plot_feature_importance(filename=FILE_DIR + 'feature1',
                                 display=False)
    atom.Bag.plot_feature_importance(filename=FILE_DIR + 'feature2',
                                     display=False)
    assert glob.glob(FILE_DIR + 'feature1.png')
    assert glob.glob(FILE_DIR + 'feature2.png')


def test_plot_confusion_matrix():
    """Assert that the plot_confusion_matrix method work as intended."""
    # When task is not classification
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline('OLS', 'r2')
    pytest.raises(AttributeError, atom.OLS.plot_confusion_matrix)

    # For binary classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['LDA', 'ET'])

    # When model is unknown
    pytest.raises(ValueError, atom.plot_confusion_matrix, 'unknown')

    # When correct
    atom.plot_confusion_matrix(normalize=True,
                               filename=FILE_DIR + 'cm1',
                               display=False)
    atom.LDA.plot_confusion_matrix(normalize=False,
                                   filename=FILE_DIR + 'cm2',
                                   display=False)
    assert glob.glob(FILE_DIR + 'cm1.png')
    assert glob.glob(FILE_DIR + 'cm2.png')

    # For multiclass classification tasks
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.pipeline(['LDA', 'ET'])

    # Multiclass and multiple models not supported
    pytest.raises(NotImplementedError, atom.plot_confusion_matrix)
    atom.LDA.plot_confusion_matrix(normalize=True,
                                   filename=FILE_DIR + 'cm3',
                                   display=False)
    assert glob.glob(FILE_DIR + 'cm3.png')


def test_plot_threshold():
    """Assert that the plot_threshold method work as intended."""
    # When task is not binary
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline('Tree', 'r2')
    pytest.raises(AttributeError, atom.Tree.plot_threshold)

    # When task is binary
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['LDA', 'ET'], 'f1')

    # When invalid model or metric
    pytest.raises(ValueError, atom.plot_threshold, 'unknown')
    pytest.raises(ValueError, atom.plot_threshold, 'LDA', 'unknown')

    # For metric is None, functions or scorers
    scorer = get_scorer('f1_macro')
    atom.LDA.plot_threshold(display=False)
    atom.plot_threshold(metric=[f1_score, recall_score, 'r2'],
                        filename=FILE_DIR + 'threshold1',
                        display=False)
    atom.LDA.plot_threshold([scorer, 'precision'],
                            filename=FILE_DIR + 'threshold2',
                            display=False)
    assert glob.glob(FILE_DIR + 'threshold1.png')
    assert glob.glob(FILE_DIR + 'threshold2.png')


def test_plot_probabilities():
    """Assert that the plot_probabilities method work as intended."""
    # When task is not classification
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline('Tree', 'r2')
    pytest.raises(AttributeError, atom.Tree.plot_probabilities)

    # When model hasn't the predict_proba method
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline('PA', 'r2')
    pytest.raises(ValueError, atom.PA.plot_probabilities)

    # When invalid model
    pytest.raises(ValueError, atom.plot_probabilities, models='unknown')

    # For target is string
    y = ['a' if i == 0 else 'b' for i in y_bin]
    atom = ATOMClassifier(X_bin, y)
    atom.pipeline(['LDA', 'QDA'], 'f1')
    atom.LDA.plot_probabilities(target='a',
                                filename=FILE_DIR + 'probabilities1',
                                display=False)
    atom.plot_probabilities(target='b',
                            filename=FILE_DIR + 'probabilities2',
                            display=False)
    assert glob.glob(FILE_DIR + 'probabilities1.png')
    assert glob.glob(FILE_DIR + 'probabilities2.png')

    # For target is numerical
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['LDA', 'QDA'], 'f1')
    atom.LDA.plot_probabilities(target=0,
                                filename=FILE_DIR + 'probabilities3',
                                display=False)
    atom.plot_probabilities(target=1,
                            filename=FILE_DIR + 'probabilities4',
                            display=False)
    assert glob.glob(FILE_DIR + 'probabilities3.png')
    assert glob.glob(FILE_DIR + 'probabilities4.png')


def test_plot_calibration():
    """Assert that the plot_calibration method work as intended."""
    # When task is not binary
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline('Tree', 'r2')
    pytest.raises(AttributeError, atom.Tree.plot_calibration)

    # When task is binary
    atom = ATOMClassifier(X_bin, y_bin)
    atom.pipeline(['Tree', 'pa'], 'f1')

    # When invalid model
    pytest.raises(ValueError, atom.plot_calibration, models='unknown')

    # When invalid bins
    pytest.raises(ValueError, atom.plot_calibration, n_bins=3)

    # When correct
    atom.Tree.plot_calibration(filename=FILE_DIR + 'calibration1',
                               display=False)
    atom.plot_calibration(filename=FILE_DIR + 'calibration2',
                          display=False)
    assert glob.glob(FILE_DIR + 'calibration1.png')
    assert glob.glob(FILE_DIR + 'calibration2.png')


def test_plot_gains():
    """Assert that the plot_gains method work as intended."""
    # When task is not binary
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline('Tree', 'r2')
    pytest.raises(AttributeError, atom.Tree.plot_gains)

    # When task is binary
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['Tree', 'LGB', 'PA'], 'f1')

    # When invalid model
    pytest.raises(ValueError, atom.plot_gains, models='unknown')

    # When model with no predict_proba method
    pytest.raises(ValueError, atom.PA.plot_gains)

    # When correct
    atom.Tree.plot_gains(filename=FILE_DIR + 'gains1', display=False)
    atom.plot_gains(['Tree', 'LGB'],
                    filename=FILE_DIR + 'gains2',
                    display=False)
    assert glob.glob(FILE_DIR + 'gains1.png')
    assert glob.glob(FILE_DIR + 'gains2.png')


def test_plot_lift():
    """Assert that the plot_lift method work as intended."""
    # When task is not binary
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.pipeline('Tree', 'r2')
    pytest.raises(AttributeError, atom.Tree.plot_lift)

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.pipeline(['Tree', 'LGB', 'PA'], 'f1')

    # When invalid model
    pytest.raises(ValueError, atom.plot_lift, models='unknown')

    # When model with no predict_proba method
    pytest.raises(ValueError, atom.PA.plot_lift)

    # When correct
    atom.Tree.plot_lift(filename=FILE_DIR + 'lift1', display=False)
    atom.plot_lift(['Tree', 'LGB'], filename=FILE_DIR + 'lift2', display=False)
    assert glob.glob(FILE_DIR + 'lift1.png')
    assert glob.glob(FILE_DIR + 'lift2.png')


# << ================== Test BasePlotter ================== >>

def test_set_style():
    """Assert that the set_style classmethod works as intended."""
    style = 'white'
    atom = ATOMClassifier(X_bin, y_bin)
    atom.set_style(style)
    assert ATOMClassifier.style == style


def test_set_palette():
    """Assert that the set_palette classmethod works as intended."""
    palette = 'Blues'
    atom = ATOMClassifier(X_bin, y_bin)
    atom.set_palette(palette)
    assert ATOMClassifier.palette == palette


def test_set_title_fontsize():
    """Assert that the set_title_fontsize classmethod works as intended."""
    title_fontsize = 21
    atom = ATOMClassifier(X_bin, y_bin)
    atom.set_title_fontsize(title_fontsize)
    assert ATOMClassifier.title_fontsize == title_fontsize


def test_set_label_fontsize():
    """Assert that the set_label_fontsize classmethod works as intended."""
    label_fontsize = 4
    atom = ATOMClassifier(X_bin, y_bin)
    atom.set_label_fontsize(label_fontsize)
    assert ATOMClassifier.label_fontsize == label_fontsize


def test_set_tick_fontsize():
    """Assert that the set_tick_fontsize classmethod works as intended."""
    tick_fontsize = 13
    atom = ATOMClassifier(X_bin, y_bin)
    atom.set_tick_fontsize(tick_fontsize)
    assert ATOMClassifier.tick_fontsize == tick_fontsize
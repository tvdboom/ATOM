# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for plots.py

"""

# Import packages
import glob
import pytest
from sklearn.metrics import f1_score, get_scorer

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.feature_engineering import FeatureSelector
from atom.training import (
    TrainerClassifier, TrainerRegressor,
    SuccessiveHalvingRegressor, TrainSizingRegressor
)
from atom.plots import BasePlotter
from atom.utils import NotFittedError
from .utils import (
    FILE_DIR, X_bin, y_bin, X_class, y_class, X_reg, y_reg, bin_train, bin_test,
    class_train, class_test, reg_train, reg_test
)


# Test BasePlotter ========================================================== >>

def test_aesthetics_property():
    """Assert that aesthetics returns the classes aesthetics as dict."""
    assert isinstance(BasePlotter().aesthetics, dict)


def test_aesthetics_setter():
    """Assert that the aesthetics setter works as intended."""
    base = BasePlotter()
    base.aesthetics = {'palette': 'Blues'}
    assert base.palette == 'Blues'


def test_style_property():
    """Assert that style returns the classes aesthetics as dict."""
    assert isinstance(BasePlotter().style, str)


def test_style_setter():
    """Assert that the style setter works as intended."""
    with pytest.raises(ValueError):
        BasePlotter().style = 'unknown'


def test_palette_property():
    """Assert that palette returns the classes aesthetics as dict."""
    assert isinstance(BasePlotter().palette, str)


def test_palette_setter():
    """Assert that the palette setter works as intended."""
    with pytest.raises(ValueError):
        BasePlotter().palette = 'unknown'


def test_title_fontsize_property():
    """Assert that title_fontsize returns the classes aesthetics as dict."""
    assert isinstance(BasePlotter().title_fontsize, int)


def test_title_fontsize_setter():
    """Assert that the title_fontsize setter works as intended."""
    with pytest.raises(ValueError):
        BasePlotter().title_fontsize = 0


def test_label_fontsize_property():
    """Assert that label_fontsize returns the classes aesthetics as dict."""
    assert isinstance(BasePlotter().label_fontsize, int)


def test_label_fontsize_setter():
    """Assert that the label_fontsize setter works as intended."""
    with pytest.raises(ValueError):
        BasePlotter().label_fontsize = 0


def test_tick_fontsize_property():
    """Assert that tick_fontsize returns the classes aesthetics as dict."""
    assert isinstance(BasePlotter().tick_fontsize, int)


def test_tick_fontsize_setter():
    """Assert that the tick_fontsize setter works as intended."""
    with pytest.raises(ValueError):
        BasePlotter().tick_fontsize = 0


# Test plots ================================================================ >>

def test_plot_correlation():
    """Assert that the plot_correlation method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_correlation(filename=FILE_DIR + 'correlation', display=False)
    assert glob.glob(FILE_DIR + 'correlation.png')


def test_plot_pipeline():
    """Assert that the plot_pipeline method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.impute()
    atom.outliers()
    atom.successive_halving(['Tree', 'AdaB'])
    atom.plot_pipeline(True, filename=FILE_DIR + 'pipeline1', display=False)
    atom.plot_pipeline(False, filename=FILE_DIR + 'pipeline2', display=False)
    assert glob.glob(FILE_DIR + 'pipeline1.png')
    assert glob.glob(FILE_DIR + 'pipeline2.png')


def test_plot_pca():
    """Assert that the plot_pca method work as intended."""
    # When no PCA was run
    fs = FeatureSelector()
    pytest.raises(PermissionError, fs.plot_pca)

    # When correct
    fs = FeatureSelector('PCA', n_features=12)
    fs.fit_transform(X_bin, y_bin)
    fs.plot_pca(filename=FILE_DIR + 'pca1', display=False)
    assert glob.glob(FILE_DIR + 'pca1.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='PCA', n_features=10)
    atom.plot_pca(filename=FILE_DIR + 'pca2', display=False)
    assert glob.glob(FILE_DIR + 'pca2.png')


def test_plot_components():
    """Assert that the plot_components method work as intended."""
    # When no PCA was run
    fs = FeatureSelector()
    pytest.raises(PermissionError, fs.plot_components)

    # When show is invalid
    fs = FeatureSelector('PCA', n_features=12)
    fs.fit_transform(X_bin, y_bin)
    pytest.raises(ValueError, fs.plot_components, show=0)

    # When correct (test if show is converted to max components)
    fs.plot_components(show=100, filename=FILE_DIR + 'components1', display=False)
    assert glob.glob(FILE_DIR + 'components1.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy='PCA', n_features=10)
    atom.plot_components(filename=FILE_DIR + 'components2', display=False)
    assert glob.glob(FILE_DIR + 'components2.png')


def test_plot_rfecv():
    """Assert that the plot_rfecv method work as intended """
    # When no RFECV was run
    fs = FeatureSelector()
    pytest.raises(PermissionError, fs.plot_rfecv)

    # When correct
    fs = FeatureSelector('RFECV', solver='lgb_class', n_features=12, scoring='f1')
    fs.fit_transform(X_bin, y_bin)
    fs.plot_rfecv(filename=FILE_DIR + 'rfecv1', display=False)
    assert glob.glob(FILE_DIR + 'rfecv1.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lr', metric='precision')
    atom.feature_selection(strategy='RFECV', n_features=10)
    atom.plot_rfecv(filename=FILE_DIR + 'rfecv2', display=False)
    assert glob.glob(FILE_DIR + 'rfecv2.png')


def test_plot_successive_halving():
    """Assert that the plot_successive_halving method work as intended."""
    # When its not a SuccessiveHalving instance
    trainer = TrainerRegressor(['ols', 'ridge'], metric='max_error')
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.ols.plot_successive_halving)

    # When its not fitted
    sh = SuccessiveHalvingRegressor(['ols', 'ridge'], metric='max_error', bagging=4)
    pytest.raises(NotFittedError, sh.plot_successive_halving)

    # When model is unknown or not in pipeline
    sh.run(reg_train, reg_test)
    pytest.raises(ValueError, sh.plot_successive_halving, models='unknown')
    pytest.raises(ValueError, sh.plot_successive_halving, models='BR')

    # When metric_ is invalid, unknown or not in pipeline
    pytest.raises(ValueError, sh.plot_successive_halving, metric='unknown')
    pytest.raises(ValueError, sh.plot_successive_halving, metric=-1)
    pytest.raises(ValueError, sh.plot_successive_halving, metric=1)
    pytest.raises(ValueError, sh.plot_successive_halving, metric='roc_auc')

    # When correct
    sh.plot_successive_halving(models=['OLS', 'Ridge'],
                               metric='me',
                               filename=FILE_DIR + 'sh1',
                               display=False)
    sh.ols.plot_successive_halving(filename=FILE_DIR + 'sh2', display=False)
    assert glob.glob(FILE_DIR + 'sh1.png')
    assert glob.glob(FILE_DIR + 'sh2.png')

    # From ATOM
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.successive_halving('ols', metric='max_error')
    atom.plot_successive_halving(filename=FILE_DIR + 'sh3', display=False)
    atom.ols.plot_successive_halving(filename=FILE_DIR + 'sh4', display=False)
    assert glob.glob(FILE_DIR + 'sh3.png')
    assert glob.glob(FILE_DIR + 'sh4.png')


def test_plot_learning_curve():
    """Assert that the plot_learning_curve method work as intended."""
    # When its not a TrainSizing instance
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.ols.plot_learning_curve)

    ts = TrainSizingRegressor(['ols', 'ridge'], metric='r2', bagging=4)
    pytest.raises(NotFittedError, ts.plot_learning_curve)
    ts.run(reg_train, reg_test)
    ts.plot_learning_curve(filename=FILE_DIR + 'ts1', display=False)
    ts.ols.plot_learning_curve(filename=FILE_DIR + 'ts2', display=False)
    assert glob.glob(FILE_DIR + 'ts1.png')
    assert glob.glob(FILE_DIR + 'ts2.png')

    # From ATOM
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.train_sizing('ols', metric='max_error')
    atom.plot_learning_curve(filename=FILE_DIR + 'ts3', display=False)
    atom.ols.plot_learning_curve(filename=FILE_DIR + 'ts4', display=False)
    assert glob.glob(FILE_DIR + 'ts3.png')
    assert glob.glob(FILE_DIR + 'ts4.png')


def test_plot_bagging():
    """Assert that the plot_bagging method work as intended."""
    # When its not fitted
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2', bagging=0)
    pytest.raises(NotFittedError, trainer.plot_bagging)

    # When called without bagging
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.plot_bagging, models='OLS')

    # When correct
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2', bagging=3)
    trainer.run(reg_train, reg_test)
    trainer.plot_bagging(filename=FILE_DIR + 'bagging1', display=False)
    trainer.ols.plot_bagging(filename=FILE_DIR + 'bagging2', display=False)
    assert glob.glob(FILE_DIR + 'bagging1.png')
    assert glob.glob(FILE_DIR + 'bagging2.png')

    # From ATOM
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('ols', metric=['me', 'r2'], bagging=3)
    atom.plot_bagging(metric='me', filename=FILE_DIR + 'bagging3', display=False)
    atom.ols.plot_bagging(filename=FILE_DIR + 'bagging4', display=False)
    assert glob.glob(FILE_DIR + 'bagging3.png')
    assert glob.glob(FILE_DIR + 'bagging4.png')


def test_plot_bo():
    """Assert that the plot_bo method work as intended."""
    # When its not fitted
    trainer = TrainerRegressor(['lasso', 'ridge'], metric='r2', bagging=0)
    pytest.raises(NotFittedError, trainer.plot_bo)

    # When called without running the bayesian optimization
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.plot_bo)

    # When correct
    trainer = TrainerRegressor(['lasso', 'ridge'], metric='r2', n_calls=10)
    trainer.run(reg_train, reg_test)
    trainer.plot_bo(filename=FILE_DIR + 'bagging1', display=False)
    trainer.lasso.plot_bo(filename=FILE_DIR + 'bagging2', display=False)
    assert glob.glob(FILE_DIR + 'bagging1.png')
    assert glob.glob(FILE_DIR + 'bagging2.png')

    # From ATOM
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('lasso', metric='max_error', n_calls=10)
    atom.plot_bo(filename=FILE_DIR + 'bagging3', display=False)
    atom.lasso.plot_bo(filename=FILE_DIR + 'bagging4', display=False)
    assert glob.glob(FILE_DIR + 'bagging3.png')
    assert glob.glob(FILE_DIR + 'bagging4.png')


def test_plot_evals():
    """Assert that the plot_evals method work as intended."""
    trainer = TrainerClassifier(['LR', 'XGB'], metric='f1')
    trainer.run(bin_train, bin_test)

    # When more than one model
    pytest.raises(ValueError, trainer.plot_evals)

    # When model has no evals attr
    pytest.raises(AttributeError, trainer.LR.plot_evals)

    trainer = TrainerClassifier('LGB', metric='f1')
    pytest.raises(NotFittedError, trainer.plot_evals, models='LDA')
    trainer.run(bin_train, bin_test)
    trainer.plot_evals(filename=FILE_DIR + 'evals1', display=False)
    trainer.lgb.plot_evals(filename=FILE_DIR + 'evals2', display=False)
    assert glob.glob(FILE_DIR + 'evals1.png')
    assert glob.glob(FILE_DIR + 'evals2.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('XGB', metric='f1')
    atom.plot_evals(filename=FILE_DIR + 'evals3', display=False)
    atom.xgb.plot_evals(filename=FILE_DIR + 'evals4', display=False)
    assert glob.glob(FILE_DIR + 'evals3.png')
    assert glob.glob(FILE_DIR + 'evals4.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_roc(dataset):
    """Assert that the plot_roc method work as intended."""
    # When task is not binary
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.plot_roc)

    # When correct
    trainer = TrainerClassifier(['RF', 'LGB'], metric='f1')
    pytest.raises(NotFittedError, trainer.plot_roc, models='LDA')
    trainer.run(bin_train, bin_test)
    pytest.raises(ValueError, trainer.plot_roc, models='LDA', dataset='invalid')
    trainer.plot_roc(
        dataset=dataset,
        filename=FILE_DIR + f'roc1_{dataset}',
        display=False
    )
    trainer.rf.plot_roc(
        dataset=dataset,
        filename=FILE_DIR + f'roc2_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'roc1_{dataset}.png')
    assert glob.glob(FILE_DIR + f'roc2_{dataset}.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lgb', metric='f1')
    atom.plot_roc(
        dataset=dataset,
        filename=FILE_DIR + f'roc3_{dataset}',
        display=False
    )
    atom.lgb.plot_roc(
        dataset=dataset,
        filename=FILE_DIR + f'roc4_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'roc3_{dataset}.png')
    assert glob.glob(FILE_DIR + f'roc4_{dataset}.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_prc(dataset):
    """Assert that the plot_prc method work as intended."""
    # When task is not binary
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.plot_prc)

    # When correct
    trainer = TrainerClassifier(['RF', 'LGB'], metric='f1')
    pytest.raises(NotFittedError, trainer.plot_prc, models='LDA')
    trainer.run(bin_train, bin_test)
    trainer.plot_prc(
        dataset=dataset,
        filename=FILE_DIR + f'prc1_{dataset}',
        display=False
    )
    trainer.rf.plot_prc(
        dataset=dataset,
        filename=FILE_DIR + f'prc2_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'prc1_{dataset}.png')
    assert glob.glob(FILE_DIR + f'prc2_{dataset}.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lgb', metric='f1')
    atom.plot_prc(
        dataset=dataset,
        filename=FILE_DIR + f'prc3_{dataset}',
        display=False
    )
    atom.lgb.plot_prc(
        dataset=dataset,
        filename=FILE_DIR + f'prc4_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'prc3_{dataset}.png')
    assert glob.glob(FILE_DIR + f'prc4_{dataset}.png')


def test_plot_permutation_importance():
    """Assert that the plot_permutation_importance method work as intended."""
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    pytest.raises(NotFittedError, trainer.plot_permutation_importance)

    # When invalid parameters
    trainer.run(reg_train, reg_test)
    pytest.raises(ValueError, trainer.plot_permutation_importance, show=0)
    pytest.raises(ValueError, trainer.plot_permutation_importance, n_repeats=0)

    # When correct
    trainer.plot_permutation_importance(filename=FILE_DIR + 'p1', display=False)
    trainer.ols.plot_permutation_importance(filename=FILE_DIR + 'p2', display=False)
    assert glob.glob(FILE_DIR + 'p1.png')
    assert glob.glob(FILE_DIR + 'p2.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('lda', metric='f1')
    atom.plot_permutation_importance(filename=FILE_DIR + 'p3', display=False)
    atom.lda.plot_permutation_importance(filename=FILE_DIR + 'p4', display=False)
    assert glob.glob(FILE_DIR + 'p3.png')
    assert glob.glob(FILE_DIR + 'p4.png')


def test_plot_feature_importance():
    """Assert that the plot_feature_importance method work as intended."""
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    pytest.raises(NotFittedError, trainer.plot_feature_importance)

    # When model not a tree-based model
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.ols.plot_feature_importance)

    # When invalid parameters
    trainer = TrainerRegressor(['Bag', 'lgb'], metric='r2')
    trainer.run(reg_train, reg_test)
    pytest.raises(ValueError, trainer.plot_feature_importance, show=0)

    # When correct
    trainer.plot_feature_importance(filename=FILE_DIR + 'f1', display=False)
    trainer.bag.plot_feature_importance(filename=FILE_DIR + 'f2', display=False)
    assert glob.glob(FILE_DIR + 'f1.png')
    assert glob.glob(FILE_DIR + 'f2.png')

    # From ATOM
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run('tree', metric='f1_micro')
    atom.plot_feature_importance(filename=FILE_DIR + 'f3', display=False)
    atom.tree.plot_feature_importance(filename=FILE_DIR + 'f4', display=False)
    assert glob.glob(FILE_DIR + 'f2.png')
    assert glob.glob(FILE_DIR + 'f4.png')


def test_plot_partial_dependence():
    """Assert that the plot_partial_dependence method work as intended."""
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    pytest.raises(NotFittedError, trainer.plot_partial_dependence)

    # When invalid parameters
    trainer.run(reg_train, reg_test)

    # More than 3 features
    with pytest.raises(ValueError, match=r".*Maximum 3 allowed.*"):
        trainer.plot_partial_dependence(features=[0, 1, 2, 3])

    # Triple feature
    with pytest.raises(ValueError, match=r".*should be single or in pairs.*"):
        trainer.ols.plot_partial_dependence(features=[(0, 1, 2), 2])

    # Pair for multi-model
    with pytest.raises(ValueError, match=r".*when plotting multiple models.*"):
        trainer.plot_partial_dependence(features=[(0, 2), 2])

    # Unknown feature
    with pytest.raises(ValueError, match=r".*Unknown column.*"):
        trainer.plot_partial_dependence(features=['test', 2])

    # Invalid index
    with pytest.raises(ValueError, match=r".*got index.*"):
        trainer.plot_partial_dependence(features=[120, 2])

    # When correct
    trainer.plot_permutation_importance(display=False)  # Assign best_features
    trainer.plot_partial_dependence(
        features=2,
        filename=FILE_DIR + 'pd1',
        display=False
    )
    trainer.ols.plot_partial_dependence(filename=FILE_DIR + 'pd2', display=False)
    assert glob.glob(FILE_DIR + 'pd1.png')
    assert glob.glob(FILE_DIR + 'pd2.png')

    # From ATOM (test multiclass)
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run('lda', metric='f1_macro')
    atom.plot_partial_dependence(target=0, filename=FILE_DIR + 'pd3', display=False)
    atom.lda.plot_partial_dependence(
        features=[('alcohol', 'ash')],
        target=2,
        filename=FILE_DIR + 'pd4',
        display=False
    )
    assert glob.glob(FILE_DIR + 'pd3.png')
    assert glob.glob(FILE_DIR + 'pd4.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_errors(dataset):
    """Assert that the plot_errors method work as intended."""
    # When task is not regression
    trainer = TrainerClassifier('Tree', metric='f1')
    trainer.run(bin_train, bin_test)
    pytest.raises(PermissionError, trainer.plot_errors)

    trainer = TrainerRegressor(['RF', 'LGB'], metric='MAE')
    pytest.raises(NotFittedError, trainer.plot_errors, models='OLS')
    trainer.run(bin_train, bin_test)
    trainer.plot_errors(
        dataset=dataset,
        filename=FILE_DIR + f'errors1_{dataset}',
        display=False
    )
    trainer.rf.plot_errors(
        dataset=dataset,
        filename=FILE_DIR + f'errors2_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'errors1_{dataset}.png')
    assert glob.glob(FILE_DIR + f'errors2_{dataset}.png')

    # From ATOM
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(['Lasso', 'BR'], metric='MSE')
    atom.plot_errors(
        dataset=dataset,
        filename=FILE_DIR + f'errors3_{dataset}',
        display=False
    )
    atom.br.plot_errors(
        dataset=dataset,
        filename=FILE_DIR + f'errors4_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'errors3_{dataset}.png')
    assert glob.glob(FILE_DIR + f'errors4_{dataset}.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_residuals(dataset):
    """Assert that the plot_residuals method work as intended."""
    # When task is not regression
    trainer = TrainerClassifier('Tree', metric='f1')
    trainer.run(bin_train, bin_test)
    pytest.raises(PermissionError, trainer.plot_residuals)

    trainer = TrainerRegressor(['rf', 'LGB'], metric='MAE')
    pytest.raises(NotFittedError, trainer.plot_residuals, models='OLS')
    trainer.run(bin_train, bin_test)
    trainer.plot_residuals(
        dataset=dataset,
        filename=FILE_DIR + f'residuals1_{dataset}',
        display=False
    )
    trainer.rf.plot_residuals(
        dataset=dataset,
        filename=FILE_DIR + f'residuals2_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'residuals1_{dataset}.png')
    assert glob.glob(FILE_DIR + f'residuals2_{dataset}.png')

    # From ATOM
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(['Lasso', 'LGB'], metric='MSE')
    atom.plot_residuals(
        dataset=dataset,
        filename=FILE_DIR + f'residuals3_{dataset}',
        display=False
    )
    atom.lgb.plot_residuals(
        dataset=dataset,
        filename=FILE_DIR + f'residuals4_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'residuals3_{dataset}.png')
    assert glob.glob(FILE_DIR + f'residuals4_{dataset}.png')


@pytest.mark.parametrize('dataset', ['train', 'test'])
@pytest.mark.parametrize('normalize', [True, False])
def test_plot_confusion_matrix(dataset, normalize):
    """Assert that the plot_confusion_matrix method work as intended."""
    # When task is not classification
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.plot_confusion_matrix)

    # For binary classification tasks
    trainer = TrainerClassifier(['RF', 'LGB'], metric='f1')
    pytest.raises(NotFittedError, trainer.plot_confusion_matrix, models='LDA')
    trainer.run(bin_train, bin_test)
    trainer.plot_confusion_matrix(
        dataset=dataset,
        normalize=normalize,
        filename=FILE_DIR + f'cm1_{dataset}_{normalize}',
        display=False
    )
    trainer.lgb.plot_confusion_matrix(
        dataset=dataset,
        normalize=normalize,
        filename=FILE_DIR + f'cm2_{dataset}_{normalize}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'cm1_{dataset}_{normalize}.png')
    assert glob.glob(FILE_DIR + f'cm2_{dataset}_{normalize}.png')

    # For multiclass classification tasks
    trainer = TrainerClassifier(['RF', 'LGB'], metric='f1_macro')
    pytest.raises(NotFittedError, trainer.plot_confusion_matrix, models='LDA')
    trainer.run(class_train, class_test)
    pytest.raises(NotImplementedError, trainer.plot_confusion_matrix)
    trainer.lgb.plot_confusion_matrix(
        dataset=dataset,
        normalize=normalize,
        filename=FILE_DIR + f'cm3_{dataset}_{normalize}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'cm3_{dataset}_{normalize}.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LGB', metric='f1')
    atom.plot_confusion_matrix(
        dataset=dataset,
        normalize=normalize,
        filename=FILE_DIR + f'cm4_{dataset}_{normalize}',
        display=False
    )
    atom.lgb.plot_confusion_matrix(
        dataset=dataset,
        normalize=normalize,
        filename=FILE_DIR + f'cm5_{dataset}_{normalize}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'cm4_{dataset}_{normalize}.png')
    assert glob.glob(FILE_DIR + f'cm5_{dataset}_{normalize}.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_threshold(dataset):
    """Assert that the plot_threshold method work as intended."""
    # When task is not binary
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.plot_threshold)

    # When model has no predict_proba method
    trainer = TrainerClassifier('PA', metric='f1')
    trainer.run(bin_train, bin_test)
    pytest.raises(AttributeError, trainer.plot_threshold)

    trainer = TrainerClassifier(['RF', 'LGB'], metric='f1')
    pytest.raises(NotFittedError, trainer.plot_threshold, models='LDA')
    trainer.run(bin_train, bin_test)
    pytest.raises(ValueError, trainer.plot_threshold, metric='unknown')
    trainer.plot_threshold(
        dataset=dataset,
        filename=FILE_DIR + f'threshold1_{dataset}',
        display=False
    )
    metrics = [f1_score, get_scorer('average_precision'), 'precision', 'auc']
    trainer.rf.plot_threshold(
        metric=metrics,
        dataset=dataset,
        filename=FILE_DIR + f'threshold2_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'threshold1_{dataset}.png')
    assert glob.glob(FILE_DIR + f'threshold2_{dataset}.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LGB', metric='f1')
    atom.plot_threshold(
        dataset=dataset,
        filename=FILE_DIR + f'threshold3_{dataset}',
        display=False
    )
    atom.lgb.plot_threshold(
        dataset=dataset,
        filename=FILE_DIR + f'threshold4_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'threshold3_{dataset}.png')
    assert glob.glob(FILE_DIR + f'threshold4_{dataset}.png')


@pytest.mark.parametrize('dataset', ['train', 'test'])
def test_plot_probabilities(dataset):
    """Assert that the plot_probabilities method work as intended."""
    # When task is not classification
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.plot_probabilities)

    # When model has no predict_proba method
    trainer = TrainerClassifier('PA', metric='f1')
    trainer.run(bin_train, bin_test)
    pytest.raises(AttributeError, trainer.plot_probabilities)

    trainer = TrainerClassifier(['RF', 'LGB'], metric='f1')
    pytest.raises(NotFittedError, trainer.plot_probabilities, models='LDA')
    trainer.run(bin_train, bin_test)
    trainer.plot_probabilities(
        dataset=dataset,
        filename=FILE_DIR + f'prob1_{dataset}',
        display=False
    )
    trainer.rf.plot_probabilities(
        dataset=dataset,
        filename=FILE_DIR + f'prob2_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'prob1_{dataset}.png')
    assert glob.glob(FILE_DIR + f'prob2_{dataset}.png')

    # From ATOM (target is str)
    y = ['a' if i == 0 else 'b' for i in y_bin]
    atom = ATOMClassifier(X_bin, y, random_state=1)
    atom.run('LGB', metric='f1')
    atom.plot_probabilities(
        dataset=dataset,
        target='b',
        filename=FILE_DIR + f'prob3_{dataset}',
        display=False
    )
    atom.lgb.plot_probabilities(
        dataset=dataset,
        filename=FILE_DIR + f'prob4_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'prob3_{dataset}.png')
    assert glob.glob(FILE_DIR + f'prob4_{dataset}.png')


def test_plot_calibration():
    """Assert that the plot_calibration method work as intended."""
    # When task is not binary
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.plot_calibration)

    trainer = TrainerClassifier(['LDA', 'kSVM'], metric='f1')
    pytest.raises(NotFittedError, trainer.plot_calibration)
    trainer.run(bin_train, bin_test)
    pytest.raises(ValueError, trainer.plot_calibration, n_bins=4)
    trainer.plot_calibration(filename=FILE_DIR + 'calibration1', display=False)
    trainer.lda.plot_calibration(filename=FILE_DIR + 'calibration2', display=False)
    assert glob.glob(FILE_DIR + 'calibration1.png')
    assert glob.glob(FILE_DIR + 'calibration2.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['tree', 'kSVM'], metric='f1')
    atom.plot_calibration(filename=FILE_DIR + 'calibration3', display=False)
    atom.tree.plot_calibration(filename=FILE_DIR + 'calibration4', display=False)
    assert glob.glob(FILE_DIR + 'calibration3.png')
    assert glob.glob(FILE_DIR + 'calibration4.png')


@pytest.mark.parametrize('dataset', ['train', 'test'])
def test_plot_gains(dataset):
    """Assert that the plot_gains method work as intended."""
    # When task is not binary
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.plot_gains)

    # When model has no predict_proba method
    trainer = TrainerClassifier('PA', metric='f1')
    trainer.run(bin_train, bin_test)
    pytest.raises(AttributeError, trainer.plot_gains)

    trainer = TrainerClassifier(['RF', 'LGB'], metric='f1')
    pytest.raises(NotFittedError, trainer.plot_gains, models='LDA')
    trainer.run(bin_train, bin_test)
    trainer.plot_gains(
        dataset=dataset,
        filename=FILE_DIR + f'gains1_{dataset}',
        display=False
    )
    trainer.rf.plot_gains(
        dataset=dataset,
        filename=FILE_DIR + f'gains2_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'gains1_{dataset}.png')
    assert glob.glob(FILE_DIR + f'gains2_{dataset}.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LGB', metric='f1')
    atom.plot_gains(
        dataset=dataset,
        filename=FILE_DIR + f'gains3_{dataset}',
        display=False
    )
    atom.lgb.plot_gains(
        dataset=dataset,
        filename=FILE_DIR + f'gains4_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'gains3_{dataset}.png')
    assert glob.glob(FILE_DIR + f'gains4_{dataset}.png')


@pytest.mark.parametrize('dataset', ['train', 'test'])
def test_plot_lift(dataset):
    """Assert that the plot_lift method work as intended."""
    # When task is not binary
    trainer = TrainerRegressor(['ols', 'ridge'], metric='r2')
    trainer.run(reg_train, reg_test)
    pytest.raises(PermissionError, trainer.plot_lift)

    # When model has no predict_proba method
    trainer = TrainerClassifier('PA', metric='f1')
    trainer.run(bin_train, bin_test)
    pytest.raises(AttributeError, trainer.plot_lift)

    trainer = TrainerClassifier(['RF', 'LGB'], metric='f1')
    pytest.raises(NotFittedError, trainer.plot_lift, models='LDA')
    trainer.run(bin_train, bin_test)
    trainer.plot_lift(
        dataset=dataset,
        filename=FILE_DIR + f'lift1_{dataset}',
        display=False
    )
    trainer.rf.plot_lift(
        dataset=dataset,
        filename=FILE_DIR + f'lift2_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'lift1_{dataset}.png')
    assert glob.glob(FILE_DIR + f'lift2_{dataset}.png')

    # From ATOM
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LGB', metric='f1')
    atom.plot_lift(
        dataset=dataset,
        filename=FILE_DIR + f'lift3_{dataset}',
        display=False
    )
    atom.lgb.plot_lift(
        dataset=dataset,
        filename=FILE_DIR + f'lift4_{dataset}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'lift3_{dataset}.png')
    assert glob.glob(FILE_DIR + f'lift4_{dataset}.png')

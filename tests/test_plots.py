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
from atom.training import TrainerClassifier
from atom.plots import BasePlotter
from atom.utils import NotFittedError
from .utils import (
    FILE_DIR, X_bin, y_bin, X_class, y_class, X_reg, y_reg, bin_train, bin_test
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


@pytest.mark.parametrize('show_params', [True, False])
def test_plot_pipeline(show_params):
    """Assert that the plot_pipeline method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.impute()
    atom.outliers()
    atom.feature_selection('PCA', n_features=10)
    atom.successive_halving(['Tree', 'AdaB'])
    atom.plot_pipeline(
        show_params=show_params,
        filename=FILE_DIR + f'pipeline_{show_params}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'pipeline_{show_params}.png')


def test_plot_pca():
    """Assert that the plot_pca method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(PermissionError, atom.plot_pca)  # No PCA in pipeline
    atom.feature_selection(strategy='PCA', n_features=10)
    atom.plot_pca(filename=FILE_DIR + 'pca', display=False)
    assert glob.glob(FILE_DIR + 'pca.png')


def test_plot_components():
    """Assert that the plot_components method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(PermissionError, atom.plot_components)  # No PCA in pipeline
    atom.feature_selection(strategy='PCA', n_features=10)
    pytest.raises(ValueError, atom.plot_components, show=0)  # Show is invalid
    atom.plot_components(show=100, filename=FILE_DIR + 'components', display=False)
    assert glob.glob(FILE_DIR + 'components.png')


def test_plot_rfecv():
    """Assert that the plot_rfecv method work as intended """
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(PermissionError, atom.plot_rfecv)  # No RFECV in pipeline
    atom.run('lr', metric='precision')
    atom.feature_selection(strategy='RFECV', n_features=10)
    atom.plot_rfecv(filename=FILE_DIR + 'rfecv', display=False)
    assert glob.glob(FILE_DIR + 'rfecv.png')


def test_plot_successive_halving():
    """Assert that the plot_successive_halving method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_successive_halving)
    atom.run('LGB')
    pytest.raises(PermissionError, atom.plot_successive_halving)
    atom.successive_halving(['LGB', 'Tree'], metric='max_error')
    pytest.raises(ValueError, atom.plot_successive_halving, models='unknown')
    pytest.raises(ValueError, atom.plot_successive_halving, models='BR')
    pytest.raises(ValueError, atom.plot_successive_halving, metric='unknown')
    pytest.raises(ValueError, atom.plot_successive_halving, metric=-1)
    pytest.raises(ValueError, atom.plot_successive_halving, metric=1)
    pytest.raises(ValueError, atom.plot_successive_halving, metric='roc_auc')
    atom.plot_successive_halving(
        filename=FILE_DIR + 'successive_halving_1',
        display=False
    )
    atom.lgb.plot_successive_halving(
        filename=FILE_DIR + 'successive_halving_2',
        display=False
    )
    assert glob.glob(FILE_DIR + 'successive_halving_1.png')
    assert glob.glob(FILE_DIR + 'successive_halving_2.png')


def test_plot_learning_curve():
    """Assert that the plot_learning_curve method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_learning_curve)
    atom.run('LGB')
    pytest.raises(PermissionError, atom.plot_learning_curve)  # No train_sizing
    atom.train_sizing(['Tree', 'LGB'], metric='max_error')
    atom.plot_learning_curve(filename=FILE_DIR + 'train_sizing_1', display=False)
    atom.lgb.plot_learning_curve(filename=FILE_DIR + 'train_sizing_2', display=False)
    assert glob.glob(FILE_DIR + 'train_sizing_1.png')
    assert glob.glob(FILE_DIR + 'train_sizing_2.png')


def test_plot_bagging():
    """Assert that the plot_bagging method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_bagging)
    atom.run('Tree', metric=['me', 'r2'], bagging=0)
    pytest.raises(PermissionError, atom.plot_bagging, models='Tree')  # No bagging
    atom.run('Tree', metric=['me', 'r2'], bagging=3)
    atom.plot_bagging(metric='me', filename=FILE_DIR + 'bagging_1', display=False)
    atom.tree.plot_bagging(filename=FILE_DIR + 'bagging_2', display=False)
    assert glob.glob(FILE_DIR + 'bagging_1.png')
    assert glob.glob(FILE_DIR + 'bagging_2.png')


def test_plot_bo():
    """Assert that the plot_bo method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_bo)
    atom.run('lasso', metric='max_error', n_calls=0)
    pytest.raises(PermissionError, atom.plot_bo)  # No BO in pipeline
    atom.run('lasso', metric='max_error', n_calls=10)
    atom.plot_bo(filename=FILE_DIR + 'bagging_1', display=False)
    atom.lasso.plot_bo(filename=FILE_DIR + 'bagging_2', display=False)
    assert glob.glob(FILE_DIR + 'bagging_1.png')
    assert glob.glob(FILE_DIR + 'bagging_2.png')


def test_plot_evals():
    """Assert that the plot_evals method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(['LR', 'LGB'], metric='f1')
    pytest.raises(ValueError, atom.plot_evals)  # More than 1 model
    pytest.raises(AttributeError, atom.LR.plot_evals)   # LR has no in-training eval
    atom.plot_evals(models='LGB', filename=FILE_DIR + 'evals_1', display=False)
    atom.lgb.plot_evals(filename=FILE_DIR + 'evals_2', display=False)
    assert glob.glob(FILE_DIR + 'evals_1.png')
    assert glob.glob(FILE_DIR + 'evals_2.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_roc(dataset):
    """Assert that the plot_roc method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('LGB')
    pytest.raises(PermissionError, atom.plot_roc)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_roc)
    atom.run('LGB', metric='f1')
    pytest.raises(ValueError, atom.lgb.plot_roc, dataset='invalid')
    atom.plot_roc(
        dataset=dataset,
        filename=FILE_DIR + f'roc_{dataset}_1',
        display=False
    )
    atom.lgb.plot_roc(
        dataset=dataset,
        filename=FILE_DIR + f'roc_{dataset}_2',
        display=False
    )
    assert glob.glob(FILE_DIR + f'roc_{dataset}_1.png')
    assert glob.glob(FILE_DIR + f'roc_{dataset}_2.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_prc(dataset):
    """Assert that the plot_prc method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('LGB')
    pytest.raises(PermissionError, atom.plot_prc)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_prc)
    atom.run('LGB', metric='f1')
    atom.plot_prc(
        dataset=dataset,
        filename=FILE_DIR + f'prc_{dataset}_1',
        display=False
    )
    atom.lgb.plot_prc(
        dataset=dataset,
        filename=FILE_DIR + f'prc_{dataset}_2',
        display=False
    )
    assert glob.glob(FILE_DIR + f'prc_{dataset}_1.png')
    assert glob.glob(FILE_DIR + f'prc_{dataset}_2.png')


def test_plot_permutation_importance():
    """Assert that the plot_permutation_importance method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_permutation_importance)
    atom.run('LGB', metric='f1')
    pytest.raises(ValueError, atom.plot_permutation_importance, show=0)
    pytest.raises(ValueError, atom.plot_permutation_importance, n_repeats=0)
    atom.plot_permutation_importance(
        filename=FILE_DIR + 'permutation_importance_1',
        display=False
    )
    atom.lgb.plot_permutation_importance(
        filename=FILE_DIR + 'permutation_importance_2',
        display=False
    )
    assert glob.glob(FILE_DIR + 'permutation_importance_1.png')
    assert glob.glob(FILE_DIR + 'permutation_importance_2.png')


def test_plot_feature_importance():
    """Assert that the plot_feature_importance method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_feature_importance)
    atom.run(['KNN', 'Tree', 'Bag'], metric='f1_micro')
    pytest.raises(PermissionError, atom.knn.plot_feature_importance)
    pytest.raises(ValueError, atom.tree.plot_feature_importance, show=0)
    atom.plot_feature_importance(
        models=['Tree', 'Bag'],
        filename=FILE_DIR + 'feature_importance_1',
        display=False
    )
    atom.tree.plot_feature_importance(
        filename=FILE_DIR + 'feature_importance_2',
        display=False
    )
    assert glob.glob(FILE_DIR + 'feature_importance_1.png')
    assert glob.glob(FILE_DIR + 'feature_importance_2.png')


def test_plot_partial_dependence():
    """Assert that the plot_partial_dependence method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_partial_dependence)
    atom.run(['Tree', 'LGB'], metric='f1_macro')

    # More than 3 features
    with pytest.raises(ValueError, match=r".*Maximum 3 allowed.*"):
        atom.plot_partial_dependence(features=[0, 1, 2, 3])

    # Triple feature
    with pytest.raises(ValueError, match=r".*should be single or in pairs.*"):
        atom.lgb.plot_partial_dependence(features=[(0, 1, 2), 2])

    # Pair for multi-model
    with pytest.raises(ValueError, match=r".*when plotting multiple models.*"):
        atom.plot_partial_dependence(features=[(0, 2), 2])

    # Unknown feature
    with pytest.raises(ValueError, match=r".*Unknown column.*"):
        atom.plot_partial_dependence(features=['test', 2])

    # Invalid index
    with pytest.raises(ValueError, match=r".*got index.*"):
        atom.plot_partial_dependence(features=[120, 2])

    atom.plot_partial_dependence(
        target=0,
        filename=FILE_DIR + 'partial_dependence_1',
        display=False
    )
    atom.lgb.plot_partial_dependence(
        features=[('alcohol', 'ash')],
        target=2,
        filename=FILE_DIR + 'partial_dependence_2',
        display=False
    )
    assert glob.glob(FILE_DIR + 'partial_dependence_1.png')
    assert glob.glob(FILE_DIR + 'partial_dependence_2.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_errors(dataset):
    """Assert that the plot_errors method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LR')
    pytest.raises(PermissionError, atom.plot_errors)  # Task is not regression

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_errors)
    atom.run(['Tree', 'Bag'], metric='MSE')
    atom.plot_errors(
        dataset=dataset,
        filename=FILE_DIR + f'errors_{dataset}_1',
        display=False
    )
    atom.tree.plot_errors(
        dataset=dataset,
        filename=FILE_DIR + f'errors_{dataset}_2',
        display=False
    )
    assert glob.glob(FILE_DIR + f'errors_{dataset}_1.png')
    assert glob.glob(FILE_DIR + f'errors_{dataset}_2.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_residuals(dataset):
    """Assert that the plot_residuals method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run('LR')
    pytest.raises(PermissionError, atom.plot_residuals)  # Task is not regression

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_residuals)
    atom.run(['Tree', 'Bag'], metric='MSE')
    atom.plot_residuals(
        dataset=dataset,
        filename=FILE_DIR + f'residuals_{dataset}_1',
        display=False
    )
    atom.tree.plot_residuals(
        dataset=dataset,
        filename=FILE_DIR + f'residuals_{dataset}_2',
        display=False
    )
    assert glob.glob(FILE_DIR + f'residuals_{dataset}_1.png')
    assert glob.glob(FILE_DIR + f'residuals_{dataset}_2.png')


@pytest.mark.parametrize('dataset', ['train', 'test'])
@pytest.mark.parametrize('normalize', [True, False])
def test_plot_confusion_matrix(dataset, normalize):
    """Assert that the plot_confusion_matrix method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(['Ridge'])
    pytest.raises(PermissionError, atom.plot_confusion_matrix)  # Task is not classif

    # For binary classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_confusion_matrix)
    atom.run(['RF', 'LGB'])
    atom.plot_confusion_matrix(
        dataset=dataset,
        normalize=normalize,
        filename=FILE_DIR + f'confusion_matrix_{dataset}_{normalize}_1',
        display=False
    )
    atom.lgb.plot_confusion_matrix(
        dataset=dataset,
        normalize=normalize,
        filename=FILE_DIR + f'confusion_matrix_{dataset}_{normalize}_2',
        display=False
    )
    assert glob.glob(FILE_DIR + f'confusion_matrix_{dataset}_{normalize}_1.png')
    assert glob.glob(FILE_DIR + f'confusion_matrix_{dataset}_{normalize}_2.png')

    # For multiclass classification tasks
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_confusion_matrix)
    atom.run(['RF', 'LGB'])
    pytest.raises(NotImplementedError, atom.plot_confusion_matrix)
    atom.lgb.plot_confusion_matrix(
        dataset=dataset,
        normalize=normalize,
        filename=FILE_DIR + f'confusion_matrix_{dataset}_{normalize}_3',
        display=False
    )
    assert glob.glob(FILE_DIR + f'confusion_matrix_{dataset}_{normalize}_3.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_threshold(dataset):
    """Assert that the plot_threshold method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('Ridge')
    pytest.raises(PermissionError, atom.plot_threshold)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_threshold)
    atom.run(['Tree', 'LGB', 'PA'], metric='f1')
    pytest.raises(AttributeError, atom.pa.plot_threshold)  # No predict_proba
    pytest.raises(ValueError, atom.tree.plot_threshold, metric='unknown')
    atom.plot_threshold(
        models=['Tree', 'LGB'],
        dataset=dataset,
        filename=FILE_DIR + f'threshold_{dataset}_1',
        display=False
    )
    atom.lgb.plot_threshold(
        metric=[f1_score, get_scorer('average_precision'), 'precision', 'auc'],
        dataset=dataset,
        filename=FILE_DIR + f'threshold_{dataset}_2',
        display=False
    )
    assert glob.glob(FILE_DIR + f'threshold_{dataset}_1.png')
    assert glob.glob(FILE_DIR + f'threshold_{dataset}_2.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_probabilities(dataset):
    """Assert that the plot_probabilities method work as intended."""
    trainer = TrainerClassifier(['RF', 'LGB'], metric='f1')
    trainer.run(bin_train, bin_test)
    trainer.plot_probabilities(
        dataset=dataset,
        filename=FILE_DIR + f'probabilities_{dataset}_1',
        display=False
    )

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('Ridge')
    pytest.raises(PermissionError, atom.plot_probabilities)  # Task is not classif

    y = ['a' if i == 0 else 'b' for i in y_bin]
    atom = ATOMClassifier(X_bin, y, random_state=1)
    pytest.raises(NotFittedError, atom.plot_probabilities)
    atom.run(['Tree', 'LGB', 'PA'], metric='f1')
    pytest.raises(AttributeError, atom.pa.plot_probabilities)  # No predict_proba
    atom.plot_probabilities(
        models=['Tree', 'LGB'],
        dataset=dataset,
        target='a',
        filename=FILE_DIR + f'probabilities_{dataset}_2',
        display=False
    )
    atom.lgb.plot_probabilities(
        dataset=dataset,
        target='b',
        filename=FILE_DIR + f'probabilities_{dataset}_3',
        display=False
    )
    assert glob.glob(FILE_DIR + f'probabilities_{dataset}_1.png')
    assert glob.glob(FILE_DIR + f'probabilities_{dataset}_2.png')
    assert glob.glob(FILE_DIR + f'probabilities_{dataset}_3.png')


def test_plot_calibration():
    """Assert that the plot_calibration method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('Ridge')
    pytest.raises(PermissionError, atom.plot_calibration)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_calibration)
    atom.run(['Tree', 'kSVM'], metric='f1')
    pytest.raises(ValueError, atom.plot_calibration, n_bins=4)
    atom.plot_calibration(filename=FILE_DIR + 'calibration_1', display=False)
    atom.tree.plot_calibration(filename=FILE_DIR + 'calibration_2', display=False)
    assert glob.glob(FILE_DIR + 'calibration_1.png')
    assert glob.glob(FILE_DIR + 'calibration_2.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_gains(dataset):
    """Assert that the plot_gains method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('Ridge')
    pytest.raises(PermissionError, atom.plot_gains)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_gains)
    atom.run(['Tree', 'LGB', 'PA'], metric='f1')
    pytest.raises(AttributeError, atom.pa.plot_gains)  # No predict_proba
    atom.plot_gains(
        models=['Tree', 'LGB'],
        dataset=dataset,
        filename=FILE_DIR + f'gains_{dataset}_1',
        display=False
    )
    atom.lgb.plot_gains(
        dataset=dataset,
        filename=FILE_DIR + f'gains_{dataset}_2',
        display=False
    )
    assert glob.glob(FILE_DIR + f'gains_{dataset}_1.png')
    assert glob.glob(FILE_DIR + f'gains_{dataset}_2.png')


@pytest.mark.parametrize('dataset', ['train', 'test', 'both'])
def test_plot_lift(dataset):
    """Assert that the plot_lift method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run('Ridge')
    pytest.raises(PermissionError, atom.plot_lift)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_lift)
    atom.run(['Tree', 'LGB', 'PA'], metric='f1')
    pytest.raises(AttributeError, atom.pa.plot_lift)  # No predict_proba
    atom.plot_lift(
        models=['Tree', 'LGB'],
        dataset=dataset,
        filename=FILE_DIR + f'lift_{dataset}_1',
        display=False
    )
    atom.lgb.plot_lift(
        dataset=dataset,
        filename=FILE_DIR + f'lift_{dataset}_2',
        display=False
    )
    assert glob.glob(FILE_DIR + f'lift_{dataset}_1.png')
    assert glob.glob(FILE_DIR + f'lift_{dataset}_2.png')


@pytest.mark.parametrize('model', ['OLS', 'Tree', 'KNN', 'XGB', 'LGB', 'CatB'])
@pytest.mark.parametrize('index', [(12, (430, 432)), (-5, None)])
def test_force_plot(model, index):
    """Assert that the force_plot method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.force_plot)
    atom.run(model, metric='MSE')
    atom.force_plot(
        models=model,
        index=index[0],
        matplotlib=True,
        filename=FILE_DIR + f'force_{model}_{index}_1',
        display=False
    )
    atom.force_plot(
        models=model,
        index=index[1],
        matplotlib=False,
        filename=FILE_DIR + f'force_{model}_{index}_2',
        display=False
    )
    assert glob.glob(FILE_DIR + f'force_{model}_{index}_1.png')
    assert glob.glob(FILE_DIR + f'force_{model}_{index}_2.html')


@pytest.mark.parametrize('ind', [4, 'mean texture', 'rank(3)'])
def test_dependence_plot(ind):
    """Assert that the dependence_plot method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.dependence_plot)
    atom.run(['LR', 'Tree'], metric='AP')
    atom.lr.dependence_plot(
        ind=ind,
        filename=FILE_DIR + f'dependence_{ind}',
        display=False
    )
    assert glob.glob(FILE_DIR + f'dependence_{ind}.png')


def test_summary_plot():
    """Assert that the summary_plot method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.summary_plot)
    atom.run(['LR', 'Tree'], metric='AP')
    pytest.raises(ValueError, atom.tree.summary_plot, show=0)
    atom.lr.summary_plot(filename=FILE_DIR + f'summary', display=False)
    assert glob.glob(FILE_DIR + f'summary.png')


@pytest.mark.parametrize('index', [12, (430, 432), -5, None])
def test_decision_plot_binary(index):
    """Assert that the decision_plot method work as intended for binary tasks."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.decision_plot)
    atom.run(['LR', 'Tree'], metric='AP')
    atom.lr.decision_plot(
        index=index,
        filename=FILE_DIR + f'decision_{index}_1',
        display=False
    )
    assert glob.glob(FILE_DIR + f'decision_{index}_1.png')


# def test_decision_plot_multiclass():
#     """Assert that the decision_plot method work as intended for multiclass tasks."""
#     atom = ATOMClassifier(X_class, y_class, random_state=1)
#     atom.run(['LR', 'Tree'], metric='f1_macro')
#     pytest.raises(ValueError, atom.tree.decision_plot, index=(20, 24))
#     atom.lr.decision_plot(index=20, filename=FILE_DIR + f'decision_2', display=False)
#     assert glob.glob(FILE_DIR + f'decision_2.png')

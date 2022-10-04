# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Unit tests for plots.py

"""

import glob
from unittest.mock import patch

import pytest
from sklearn.metrics import f1_score, get_scorer

from atom import ATOMClassifier, ATOMRegressor
from atom.plots import BasePlot
from atom.utils import NotFittedError

from .conftest import (
    X10, X10_str, X_bin, X_class, X_reg, X_sparse, X_text, y10, y10_str, y_bin,
    y_class, y_reg,
)


# Test BasePlot ================================================= >>

def test_aesthetics_property():
    """Assert that aesthetics returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().aesthetics, dict)


def test_aesthetics_setter():
    """Assert that the aesthetics setter works as intended."""
    base = BasePlot()
    base.aesthetics = {"palette": "Blues"}
    assert base.palette == "Blues"


def test_style_property():
    """Assert that style returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().style, str)


def test_style_setter():
    """Assert that the style setter works as intended."""
    with pytest.raises(ValueError, match=".*the style parameter.*"):
        BasePlot().style = "unknown"


def test_palette_property():
    """Assert that palette returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().palette, str)


def test_palette_setter():
    """Assert that the palette setter works as intended."""
    with pytest.raises(ValueError):
        BasePlot().palette = "unknown"


def test_title_fontsize_property():
    """Assert that title_fontsize returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().title_fontsize, int)


def test_title_fontsize_setter():
    """Assert that the title_fontsize setter works as intended."""
    with pytest.raises(ValueError, match=".*the title_fontsize parameter.*"):
        BasePlot().title_fontsize = 0


def test_label_fontsize_property():
    """Assert that label_fontsize returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().label_fontsize, int)


def test_label_fontsize_setter():
    """Assert that the label_fontsize setter works as intended."""
    with pytest.raises(ValueError, match=".*the label_fontsize parameter.*"):
        BasePlot().label_fontsize = 0


def test_tick_fontsize_property():
    """Assert that tick_fontsize returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().tick_fontsize, int)


def test_tick_fontsize_setter():
    """Assert that the tick_fontsize setter works as intended."""
    with pytest.raises(ValueError, match=".*the tick_fontsize parameter.*"):
        BasePlot().tick_fontsize = 0


def test_reset_aesthetics():
    """Assert that the reset_aesthetics method set values to default."""
    plotter = BasePlot()
    plotter.style = "white"
    plotter.reset_aesthetics()
    assert plotter.style == "darkgrid"


def test_canvas():
    """Assert that the canvas works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    with atom.canvas(1, 2, title="Title", display=False) as fig:
        atom.plot_residuals(title="Residuals plot")
        atom.plot_feature_importance(title="Feature importance plot")
    assert fig.__class__.__name__ == "Figure"


def test_canvas_too_many_plots():
    """Assert that the canvas works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    with atom.canvas(1, 2, display=False):
        atom.plot_prc()
        atom.plot_roc()
        with pytest.raises(ValueError, match=".*number of plots.*"):
            atom.plot_prc()


@patch("mlflow.tracking.MlflowClient.log_figure")
def test_figure_to_mlflow(mlflow):
    """Assert that the figure is logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run(["Tree", "LGB"])
    atom.log_plots = True
    atom.tree.plot_results(display=False)
    atom.lgb.plot_roc(display=False)
    atom.plot_prc(display=False)
    assert mlflow.call_count == 4


@patch("atom.plots.plt.Figure.savefig")
def test_figure_is_saved(func):
    """Assert that the figure is saved if a filename is provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_correlation(filename="auto", display=False)
    func.assert_called_with("plot_correlation")


@patch("atom.plots.plt.Figure.savefig")
def test_figure_is_saved_canvas(func):
    """Assert that the figure is only saved after finishing the canvas."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    with atom.canvas(1, 2, filename="canvas", display=False):
        atom.plot_prc()
        func.assert_not_called()
        atom.plot_roc()
        func.assert_not_called()
    func.assert_called_with("canvas")  # Only at the end it is saved


def test_figure_is_returned():
    """Assert that the method returns the figure for display=None."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    fig = atom.plot_correlation(display=None)
    assert fig.__class__.__name__ == "Figure"


# Test FeatureSelectorPlot ========================================= >>

@pytest.mark.parametrize("show", [10, None])
def test_plot_components(show):
    """Assert that the plot_components method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="pca", n_features=10)

    # Invalid show parameter
    with pytest.raises(ValueError, match=".*Value should be >0.*"):
        atom.plot_components(show=0, display=False)

    atom.plot_components(show=show, display=False)


@pytest.mark.parametrize("X", [X10, X_sparse])
def test_plot_pca(X):
    """Assert that the plot_pca method work as intended."""
    atom = ATOMClassifier(X, y10, random_state=1)
    atom.feature_selection(strategy="pca", n_features=2)
    atom.plot_pca(display=False)


@pytest.mark.parametrize("scoring", [None, "auc"])
def test_plot_rfecv(scoring):
    """Assert that the plot_rfecv method work as intended """
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(
        strategy="rfecv",
        solver="lr",
        n_features=20,
        scoring=scoring,
    )
    atom.plot_rfecv(display=False)


# Test DataPlot ==================================================== >>

def test_plot_correlation():
    """Assert that the plot_correlation method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # Invalid method
    with pytest.raises(ValueError, match=".*the method parameter.*"):
        atom.plot_correlation(method="invalid", display=False)

    atom.plot_correlation(display=False)


@pytest.mark.parametrize("columns", [2, "x0", [0, 1]])
def test_plot_distribution(columns):
    """Assert that the plot_distribution method work as intended."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)

    # Invalid show parameter
    with pytest.raises(ValueError, match=".*the show parameter.*"):
        atom.plot_distribution(columns=2, show=-1, display=False)

    atom.plot_distribution(columns=columns, distributions="pearson3", display=False)


@pytest.mark.parametrize("ngram", [1, 2, 3, 4])
def test_plot_ngrams(ngram):
    """Assert that the plot_ngrams method work as intended."""
    atom = ATOMClassifier(X_text, y10, random_state=1)

    # Invalid ngram parameter
    with pytest.raises(ValueError, match=".*the ngram parameter.*"):
        atom.plot_ngrams(ngram=6, display=False)

    atom.plot_ngrams(ngram=ngram, display=False)  # When corpus is str
    atom.tokenize()
    atom.plot_ngrams(ngram=ngram, display=False)  # When corpus are tokens


def test_plot_qq():
    """Assert that the plot_qq method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_qq(columns=[0, 1], distributions="pearson3", display=False)


def test_plot_scatter_matrix():
    """Assert that the plot_scatter_matrix method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # Can't be called from a canvas
    with atom.canvas(display=False):
        with pytest.raises(PermissionError, match=".*from a canvas.*"):
            atom.plot_scatter_matrix()

    atom.plot_scatter_matrix(columns=[0, 1, 2], display=False)


def test_plot_wordcloud():
    """Assert that the plot_wordcloud method work as intended."""
    atom = ATOMClassifier(X_text, y10, random_state=1)
    atom.plot_wordcloud(display=False)  # When corpus is str
    atom.tokenize()
    atom.plot_wordcloud(display=False)  # When corpus are tokens


# Test ModelPlot =================================================== >>

def test_plot_calibration():
    """Assert that the plot_calibration method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_calibration)
    atom.run(["Tree", "SVM"], metric="f1")

    # Invalid n_bins parameter
    with pytest.raises(ValueError, match=".*the n_bins parameter.*"):
        atom.plot_calibration(n_bins=4, display=False)

    atom.plot_calibration(display=False)
    atom.tree.plot_calibration(display=False)


def test_plot_confusion_matrix():
    """Assert that the plot_confusion_matrix method work as intended."""
    # For binary classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_confusion_matrix)
    atom.run(["RF", "LGB"])

    # Invalid dataset parameter
    with pytest.raises(ValueError, match=".*from: train, test or holdout.*"):
        atom.plot_confusion_matrix(dataset="invalid", display=False)

    # No holdout data set
    with pytest.raises(ValueError, match=".*No holdout.*"):
        atom.plot_confusion_matrix(dataset="holdout", display=False)

    atom.plot_confusion_matrix(display=False)
    atom.lgb.plot_confusion_matrix(normalize=True, display=False)

    # For multiclass classification tasks
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(["RF", "LGB"])

    # Not available for multiclass
    with pytest.raises(NotImplementedError, match=".*not support the comparison.*"):
        atom.plot_confusion_matrix(display=False)

    atom.lgb.plot_confusion_matrix(normalize=True, display=False)


def test_plot_det():
    """Assert that the plot_det method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_det)
    atom.run(["LGB", "SVM"])

    # No holdout data set
    with pytest.raises(ValueError, match=".*No holdout data set.*"):
        atom.lgb.plot_det(dataset="holdout", display=False)

    # Invalid dataset parameter
    with pytest.raises(ValueError, match=".*from: train, test, both.*"):
        atom.lgb.plot_det(dataset="invalid", display=False)

    atom.plot_det(display=False)
    atom.lgb.plot_det(display=False)


def test_plot_evals():
    """Assert that the plot_evals method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB", "MLP"], metric="f1")

    # No holdout allowed in dataset
    with pytest.raises(ValueError, match=".*Choose from: train, test or both.*"):
        atom.lgb.plot_evals(dataset="holdout")

    # No in-training validation
    with pytest.raises(ValueError, match=".*no in-training validation.*"):
        atom.lr.plot_evals(display=False)

    atom.plot_evals(models=["LGB", "MLP"], display=False)
    atom.lgb.plot_evals(display=False)


def test_plot_errors():
    """Assert that the plot_errors method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_errors)
    atom.run(["Tree", "Bag"], metric="MSE")
    atom.plot_errors(display=False)
    atom.tree.plot_errors(display=False)


def test_plot_feature_importance():
    """Assert that the plot_feature_importance method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_feature_importance)
    atom.run(["KNN", "Tree", "Bag"])

    # Show is invalid
    with pytest.raises(ValueError, match=".*value for the show parameter.*"):
        atom.tree.plot_feature_importance(show=-1, display=False)

    # Model has no feature importance values
    with pytest.raises(ValueError, match=".*has no feature.*"):
        atom.knn.plot_feature_importance(display=False)

    atom.plot_feature_importance(models=["Tree", "Bag"], display=False)
    atom.tree.plot_feature_importance(show=5, display=False)


def test_plot_gains():
    """Assert that the plot_gains method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_gains)
    atom.run(["LGB", "SVM"], metric="f1")
    atom.plot_gains(display=False)
    atom.lgb.plot_gains(display=False)


@pytest.mark.parametrize("metric", ["r2", ["r2", "max_error"]])
def test_plot_learning_curve(metric):
    """Assert that the plot_learning_curve method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_learning_curve)
    atom.train_sizing(["Tree", "LGB"], metric=metric, n_bootstrap=4)

    # Invalid metric parameter
    with pytest.raises(ValueError, match=".*for the metric parameter.*"):
        atom.plot_learning_curve(metric=10, display=False)

    atom.plot_learning_curve(display=False)
    atom.train_sizing(["Tree", "LGB"], metric=metric)
    atom.plot_learning_curve(display=False)


def test_plot_lift():
    """Assert that the plot_lift method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_lift)
    atom.run(["LGB", "SVM"], metric="f1")
    atom.plot_lift(display=False)
    atom.lgb.plot_lift(display=False)


def test_plot_parshap():
    """Assert that the plot_parshap method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance("smote")  # To get samples over 500
    pytest.raises(NotFittedError, atom.plot_parshap)
    atom.run(["LR", "LGB"])
    atom.plot_parshap(display=False)


def test_plot_partial_dependence_binary():
    """Assert that the plot_partial_dependence method work for binary tasks."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_partial_dependence)
    atom.run(["KNN", "LGB"], metric="f1")

    # Invalid kind parameter
    with pytest.raises(ValueError, match=".*for the kind parameter.*"):
        atom.plot_partial_dependence(kind="invalid", display=False)

    # More than 3 features
    with pytest.raises(ValueError, match=".*Maximum 3 allowed.*"):
        atom.plot_partial_dependence(columns=[0, 1, 2, 3], display=False)

    # Triple feature
    with pytest.raises(ValueError, match=".*should be single or in pairs.*"):
        atom.lgb.plot_partial_dependence(columns=[(0, 1, 2), 2], display=False)

    # Pair for multimodel
    with pytest.raises(ValueError, match=".*when plotting multiple models.*"):
        atom.plot_partial_dependence(columns=[(0, 2), 2], display=False)

    # Unknown feature
    with pytest.raises(ValueError, match=".*not find any column.*"):
        atom.plot_partial_dependence(columns=["invalid", 2], display=False)

    # Different features for multiple models
    atom.branch = "b2"
    atom.feature_selection(strategy="pca", n_features=5)
    atom.run(["Tree"])
    with pytest.raises(ValueError, match=".*models use the same features.*"):
        atom.plot_partial_dependence(columns=(0, 1), display=False)

    del atom.branch
    atom.plot_partial_dependence(columns=[0, 1, 2], kind="both", display=False)
    atom.knn.plot_partial_dependence(display=False)  # No feature_importance
    atom.lgb.plot_partial_dependence(display=False)  # Has feature_importance


@pytest.mark.parametrize("columns", [(("ash", "alcohol"), 2, "ash"), ("ash", 2), 2])
def test_plot_partial_dependence_multiclass(columns):
    """Assert that the plot_partial_dependence method work for multiclass tasks."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(["Tree", "LGB"], metric="f1_macro")

    # Invalid target int
    with pytest.raises(ValueError, match=".*classes, got .*"):
        atom.plot_partial_dependence(target=5, display=False)

    # Invalid target str
    with pytest.raises(ValueError, match=".*not found in the mapping.*"):
        atom.plot_partial_dependence(target="Yes", display=False)

    atom.lgb.plot_partial_dependence(columns, target=2, title="title", display=False)


def test_plot_permutation_importance():
    """Assert that the plot_permutation_importance method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_permutation_importance)
    atom.run(["Tree", "LGB"], metric="f1")

    # Invalid n_repeats parameter
    with pytest.raises(ValueError, match=".*the n_repeats parameter.*"):
        atom.plot_permutation_importance(n_repeats=0, display=False)

    atom.plot_permutation_importance(display=False)
    atom.lgb.plot_permutation_importance(display=False)


def test_plot_pipeline():
    """Assert that the plot_pipeline method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB")
    atom.plot_pipeline(display=False)  # No transformers

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.scale()
    atom.plot_pipeline(display=False)  # No model

    atom.run("Tree", n_trials=2)
    atom.tree.plot_pipeline(display=False)  # Only one branch

    atom.branch = "b2"
    atom.prune()
    atom.run(["OLS", "EN"])
    atom.voting()
    atom.plot_pipeline(title="Pipeline plot", display=False)  # Multiple branches


def test_plot_prc():
    """Assert that the plot_prc method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_prc)
    atom.run(["LGB", "SVM"], metric="f1")
    atom.plot_prc(display=False)
    atom.lgb.plot_prc(display=False)


def test_plot_probabilities():
    """Assert that the plot_probabilities method work as intended."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    atom.clean()  # Encode the target column
    pytest.raises(NotFittedError, atom.plot_probabilities)
    atom.run(["Tree", "LGB", "SVM"], metric="f1")

    # Model has no predict_proba attribute
    with pytest.raises(AttributeError, match=".*with a predict_proba method.*"):
        atom.svm.plot_probabilities(display=False)

    atom.plot_probabilities(models=["Tree", "LGB"], target="y", display=False)
    atom.lgb.plot_probabilities(target="n", display=False)


def test_plot_residuals():
    """Assert that the plot_residuals method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_residuals)
    atom.run(["Tree", "Bag"], metric="MSE")
    atom.plot_residuals(title="plot", display=False)
    atom.tree.plot_residuals(display=False)


@pytest.mark.parametrize("metric", ["me", ["me", "r2"]])
def test_plot_results_metric(metric):
    """Assert that the plot_results method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_results)

    # Without bootstrap
    atom.run(["Tree", "LGB"], metric=metric, n_bootstrap=0)
    atom.voting()
    atom.plot_results(metric="me", display=False)
    atom.tree.plot_results(display=False)

    # With bootstrap
    atom.run("Tree", metric=metric, n_bootstrap=3)
    atom.plot_results(metric="me", display=False)
    atom.tree.plot_results(display=False)


@pytest.mark.parametrize("metric", ["time_ht", "time_fit", "time"])
def test_plot_results_time(metric):
    """Assert that the plot_results method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["Tree", "LGB"], metric="r2", n_trials=1)

    # No bootstrap available
    with pytest.raises(ValueError, match=".*doesn't have metric.*"):
        atom.plot_results(metric="time_bootstrap", display=False)

    atom.plot_results(metric=metric, display=False)
    atom.tree.plot_results(metric=metric, display=False)


@pytest.mark.parametrize("dataset", ["train", "test", "both", "holdout"])
def test_plot_roc(dataset):
    """Assert that the plot_roc method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run(["LGB", "SVM"], metric="f1")
    atom.plot_roc(dataset=dataset, display=False)
    atom.lgb.plot_roc(dataset=dataset, display=False)


@pytest.mark.parametrize("metric", ["f1", ["f1", "recall"]])
def test_plot_successive_halving(metric):
    """Assert that the plot_successive_halving method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_successive_halving)
    atom.successive_halving(["Tree", "Bag", "RF", "LGB"], metric=metric, n_bootstrap=4)
    atom.plot_successive_halving(display=False)
    atom.successive_halving(["Tree", "Bag", "RF", "LGB"], metric=metric)
    atom.plot_successive_halving(display=False)


@pytest.mark.parametrize("metric", [f1_score, get_scorer("f1"), "precision", "auc"])
def test_plot_threshold(metric):
    """Assert that the plot_threshold method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_threshold)
    atom.run(["Tree", "LGB", "SVM"], metric="f1")
    atom.plot_threshold(models=["Tree", "LGB"], display=False)
    atom.lgb.plot_threshold(metric=metric, display=False)


def test_plot_trials():
    """Assert that the plot_bo method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_trials)
    atom.run("lasso", metric="max_error", n_trials=0)

    # Model didn't run hyperparameter tuning
    with pytest.raises(PermissionError, match=".*ran hyperparameter tuning.*"):
        atom.plot_trials(display=False)

    atom.run(["lasso", "ridge"], metric="max_error", n_trials=1)
    atom.plot_trials(display=False)
    atom.lasso.plot_trials(display=False)


# Test ShapPlot ==================================================== >>

def test_plot_shap_bar():
    """Assert that the plot_shap_bar method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_bar)
    atom.run(["LR", "Tree"], metric="f1_macro")

    with pytest.raises(ValueError, match=".*only accepts one model.*"):
        atom.plot_shap_bar(display=False)

    atom.lr.plot_shap_bar(display=False)


def test_plot_shap_beeswarm():
    """Assert that the plot_shap_beeswarm method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_beeswarm)
    atom.run("LR", metric="f1_macro")
    atom.plot_shap_beeswarm(display=False)


def test_plot_shap_decision():
    """Assert that the plot_shap_decision method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_decision)
    atom.run("LR", metric="f1_macro")
    atom.lr.plot_shap_decision(display=False)


def test_plot_shap_force():
    """Assert that the plot_shap_force method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_force)
    atom.run(["LR", "MLP"], metric="MSE")

    # Can't be called from a canvas
    with atom.canvas(display=False):
        with pytest.raises(PermissionError, match=".*called from a canvas.*"):
            atom.plot_shap_force(matplotlib=True)

    # Expected value from Explainer
    atom.lr.plot_shap_force(index=100, matplotlib=True, display=False)

    # Own calculation of expected value
    atom.mlp.plot_shap_force(index=100, matplotlib=True, display=False)

    atom.lr.plot_shap_force(matplotlib=False, filename="force", display=True)
    assert glob.glob("force.html")


def test_plot_shap_heatmap():
    """Assert that the plot_shap_heatmap method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_heatmap)
    atom.run("LR", metric="f1_macro")
    atom.plot_shap_heatmap(display=False)


@pytest.mark.parametrize("feature", [0, -1, "mean texture"])
def test_plot_shap_scatter(feature):
    """Assert that the plot_shap_scatter method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_scatter)
    atom.run("LR", metric="f1")
    atom.plot_shap_scatter(display=False)


def test_plot_shap_waterfall():
    """Assert that the plot_shap_waterfall method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_waterfall)
    atom.run("Tree", metric="f1_macro")
    atom.plot_shap_waterfall(display=False)

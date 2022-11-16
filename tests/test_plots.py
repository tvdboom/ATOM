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
from atom.plots import BaseFigure, BasePlot
from atom.utils import NotFittedError

from .conftest import (
    X10, X10_str, X_bin, X_class, X_reg, X_sparse, X_text, y10, y10_str, y_bin,
    y_class, y_reg,
)


# Test BaseFigure ================================================== >>

def test_invalid_horizontal_spacing():
    """Assert that an error is raised when horizontal_spacing is invalid."""
    with pytest.raises(ValueError, match=".*horizontal_spacing parameter.*"):
        BaseFigure(horizontal_spacing=2)


def test_invalid_vertical_spacing():
    """Assert that an error is raised when vertical_spacing is invalid."""
    with pytest.raises(ValueError, match=".*vertical_spacing parameter.*"):
        BaseFigure(vertical_spacing=0)


def test_get_colors():
    """Assert that markers are assigned correctly."""
    base = BaseFigure()
    assert base.get_color() == "rgb(95, 70, 144)"
    assert base.get_color("x") == "rgb(29, 105, 150)"
    assert base.get_color("x") == "rgb(29, 105, 150)"


def test_get_marker():
    """Assert that markers are assigned correctly."""
    base = BaseFigure()
    assert base.get_marker() == "circle"
    assert base.get_marker("x") == "x"
    assert base.get_marker("x") == "x"


def test_get_dashes():
    """Assert that dashes are assigned correctly."""
    base = BaseFigure()
    assert base.get_dashes() is None
    assert base.get_dashes("x") == "dashdot"
    assert base.get_dashes("x") == "dashdot"


def test_get_shapes():
    """Assert that shapes are assigned correctly."""
    base = BaseFigure()
    assert base.get_shapes() == ""
    assert base.get_shapes("x") == "/"
    assert base.get_shapes("x") == "/"


# Test BasePlot ==================================================== >>

def test_aesthetics_property():
    """Assert that aesthetics returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().aesthetics, dict)


def test_aesthetics_setter():
    """Assert that the aesthetics setter works."""
    base = BasePlot()
    base.aesthetics = {"palette": "Prism"}
    assert base.palette == "Prism"


def test_palette_property():
    """Assert that palette returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().palette, dict)


def test_palette_setter():
    """Assert that the palette setter works."""
    with pytest.raises(ValueError, match=".*the palette parameter.*"):
        BasePlot().palette = "unknown"


def test_title_fontsize_property():
    """Assert that title_fontsize returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().title_fontsize, int)


def test_title_fontsize_setter():
    """Assert that the title_fontsize setter works."""
    with pytest.raises(ValueError, match=".*the title_fontsize parameter.*"):
        BasePlot().title_fontsize = 0


def test_label_fontsize_property():
    """Assert that label_fontsize returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().label_fontsize, int)


def test_label_fontsize_setter():
    """Assert that the label_fontsize setter works."""
    with pytest.raises(ValueError, match=".*the label_fontsize parameter.*"):
        BasePlot().label_fontsize = 0


def test_tick_fontsize_property():
    """Assert that tick_fontsize returns the classes aesthetics as dict."""
    assert isinstance(BasePlot().tick_fontsize, int)


def test_tick_fontsize_setter():
    """Assert that the tick_fontsize setter works."""
    with pytest.raises(ValueError, match=".*the tick_fontsize parameter.*"):
        BasePlot().tick_fontsize = 0


def test_reset_aesthetics():
    """Assert that the reset_aesthetics method set values to default."""
    plotter = BasePlot()
    plotter.tick_fontsize = 30
    assert plotter.tick_fontsize == 30
    plotter.reset_aesthetics()
    assert plotter.tick_fontsize == 12


def test_update_layout():
    """Assert that the update_layout method set default layout values."""
    plotter = BasePlot()
    plotter.update_layout(template="plotly-dark")
    plotter._custom_layout["template"] = "plotly-dark"


def test_custom_palette():
    """Assert that a custom palette can be defined."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.palette = ["red", "green", "blue"]
    atom.plot_correlation(columns=[0, 1, 2], display=False)


def test_get_subclass_max_one():
    """Assert that an error is raised with more than one model."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LGB"])
    with pytest.raises(ValueError, match=".*only accepts one model.*"):
        atom._get_subclass(models=["Tree", "LGB"], max_one=True)


def test_get_metric_None():
    """Assert that all metrics are returned when None."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric=["f1", "recall"])
    assert atom._get_metric(metric=None, max_one=False) == [0, 1]


def test_get_metric_time():
    """Assert that time metrics are accepted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom._get_metric(metric="TIME", max_one=True) == "time"


def test_get_metric_time_invalid():
    """Assert that an error is raised for invalid time metrics."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*the metric parameter.*"):
        atom._get_metric(metric="time+invalid", max_one=False)


def test_get_metric_multiple():
    """Assert that time metrics are accepted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric=["f1", "recall"])
    assert atom._get_metric(metric="f1+recall", max_one=False) == [0, 1]


def test_get_metric_invalid_name():
    """Assert that an error is raised for an invalid metric name."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric="recall")
    with pytest.raises(ValueError, match=".*wasn't used to fit the models.*"):
        atom._get_metric(metric="precision", max_one=True)


def test_get_metric_max_one():
    """Assert that an error is raised when multiple metrics are selected."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric=["f1", "recall"])
    with pytest.raises(ValueError, match=".*Only one metric is allowed.*"):
        atom._get_metric(metric="f1+recall", max_one=True)


def test_get_metric_by_int():
    """Assert that a metric can be selected by position."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric=["f1", "recall"])
    assert atom._get_metric(metric=1, max_one=True) == 1


def test_get_metric_invalid_int():
    """Assert that an error is raised when the value is out of range."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric=["f1", "recall"])
    with pytest.raises(ValueError, match=".*out of range.*"):
        atom._get_metric(metric=3, max_one=True)


def test_get_set():
    """Assert that data sets can be selected."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    assert atom._get_set(dataset="Train+Test", max_one=False) == ["train", "test"]
    assert atom._get_set(dataset=["Train", "Test"], max_one=False) == ["train", "test"]


def test_get_set_no_holdout():
    """Assert that an error is raised when there's no holdout data set."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*No holdout data set.*"):
        atom._get_set(dataset="holdout", max_one=False)


def test_get_set_no_holdout_allowed():
    """Assert that an error is raised when holdout isn't allowed."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    with pytest.raises(ValueError, match=".*Choose from: train or test.*"):
        atom._get_set(dataset="holdout", max_one=False, allow_holdout=False)


def test_get_set_invalid():
    """Assert that an error is raised when the set is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*Choose from: train, test.*"):
        atom._get_set(dataset="invalid", max_one=False)


def test_get_set_multiple():
    """Assert that an error is raised when more than one set is selected."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*Only one data set is allowed.*"):
        atom._get_set(dataset="train+test", max_one=True)


def test_get_target():
    """Assert that the target can be retrieved from the name."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    atom.clean()
    assert atom._get_target(target="y") == 1
    assert atom._get_target(target=0) == 0


def test_get_target_invalid():
    """Assert that an error is raised when the target is invalid."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    with pytest.raises(ValueError, match=".*not found in the mapping.*"):
        atom._get_target(target="invalid")


def test_get_target_int_invalid():
    """Assert that an error is raised when the value is invalid."""
    atom = ATOMClassifier(X10, y10_str, random_state=1)
    with pytest.raises(ValueError, match=".*There are 2 classes.*"):
        atom._get_target(target=3)


def test_get_show():
    """Assert that the show returns max the number of features."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    assert atom._get_show(show=80, model=atom.tree) == X_bin.shape[1]


def test_get_show_invalid():
    """Assert that an error is raised when the value is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    with pytest.raises(ValueError, match=".*should be >0.*"):
        atom._get_show(show=0, model=atom.tree)


@patch("atom.plots.go.Figure.show")
def test_custom_title_and_legend(func):
    """Assert that title and legend can be customized."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    atom.plot_roc(title=dict(text="test", x=0), legend=dict(font_color="red"))
    func.assert_called_once()


@pytest.mark.parametrize("legend", [
    "upper left",
    "lower left",
    "upper right",
    "lower right",
    "upper center",
    "lower center",
    "center left",
    "center right",
    "center",
    "out",
])
def test_custom_legend_position(legend):
    """Assert that the legend position can be specified."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    atom.plot_roc(legend=legend, display=False)


def test_custom_legend_position_invalid():
    """Assert that an error is raised when the legend position is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    with pytest.raises(ValueError, match=".*the legend parameter.*"):
        atom.plot_roc(legend="invalid", display=False)


@patch("mlflow.tracking.MlflowClient.log_figure")
def test_figure_to_mlflow(mlflow):
    """Assert that the figure is logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run(["Tree", "LGB"])
    atom.log_plots = True
    atom.plot_results(display=False)
    atom.lgb.plot_shap_scatter(display=False)
    assert mlflow.call_count == 3


@patch("atom.plots.go.Figure.write_html")
def test_figure_is_saved_html(func):
    """Assert that the figure is saved as .html by default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_correlation(filename="auto", display=False)
    func.assert_called_with("plot_correlation.html")


@patch("atom.plots.go.Figure.write_image")
def test_figure_is_saved_png(func):
    """Assert that the figure is saved as .png if specified."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_correlation(filename="corr.png", display=False)
    func.assert_called_with("corr.png")


@patch("atom.plots.plt.Figure.savefig")
def test_figure_is_saved_png_plt(func):
    """Assert that the figure is saved as .png if specified."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.scale()
    atom.plot_pipeline(filename="pipeline", display=False)
    func.assert_called_with("pipeline")


def test_figure_is_returned():
    """Assert that the method returns the figure for display=None."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    fig = atom.plot_correlation(display=None)
    assert fig.__class__.__name__ == "Figure"
    assert fig.__class__.__module__.startswith("plotly")

    fig = atom.plot_shap_bar(display=None)
    assert fig.__class__.__name__ == "Figure"
    assert fig.__class__.__module__.startswith("matplotlib")


def test_canvas():
    """Assert that the canvas works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    with atom.canvas(1, 2, title="Title", display=False) as fig:
        atom.plot_residuals(title=dict(text="Residuals plot", x=0))
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


@patch("atom.plots.go.Figure.write_html")
def test_figure_is_saved_canvas(func):
    """Assert that the figure is only saved after finishing the canvas."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    with atom.canvas(1, 2, filename="canvas", display=False):
        atom.plot_prc()
        func.assert_not_called()
        atom.plot_roc()
        func.assert_not_called()
    func.assert_called_with("canvas.html")  # Only at the end it is saved


# Test FeatureSelectorPlot ========================================= >>

@pytest.mark.parametrize("show", [10, None])
def test_plot_components(show):
    """Assert that the plot_components method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection(strategy="pca", n_features=10)

    # Invalid show parameter
    with pytest.raises(ValueError, match=".*Value should be >0.*"):
        atom.plot_components(show=0, display=False)

    atom.plot_components(show=show, display=False)


@pytest.mark.parametrize("X", [X10, X_sparse])
def test_plot_pca(X):
    """Assert that the plot_pca method works."""
    atom = ATOMClassifier(X, y10, random_state=1)
    atom.feature_selection(strategy="pca", n_features=2)
    atom.plot_pca(display=False)


@pytest.mark.parametrize("scoring", [None, "auc"])
def test_plot_rfecv(scoring):
    """Assert that the plot_rfecv method works """
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.feature_selection("rfecv", solver="lr", n_features=20, scoring=scoring)
    atom.plot_rfecv(display=False)


# Test DataPlot ==================================================== >>

def test_plot_correlation():
    """Assert that the plot_correlation method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)

    # Invalid method
    with pytest.raises(ValueError, match=".*the method parameter.*"):
        atom.plot_correlation(method="invalid", display=False)

    atom.plot_correlation(display=False)


def test_plot_distribution():
    """Assert that the plot_distribution method works."""
    atom = ATOMClassifier(X10_str, y10, random_state=1)

    # Invalid show parameter
    with pytest.raises(ValueError, match=".*the show parameter.*"):
        atom.plot_distribution(columns=2, show=-1, display=False)

    atom.plot_distribution(columns=2, distributions=None, display=False)
    atom.plot_distribution(columns="x0", distributions=None, display=False)
    atom.plot_distribution(columns=[0, 1], distributions="pearson3", display=False)


@pytest.mark.parametrize("ngram", [1, 2, 3, 4])
def test_plot_ngrams(ngram):
    """Assert that the plot_ngrams method works."""
    atom = ATOMClassifier(X_text, y10, random_state=1)

    # Invalid ngram parameter
    with pytest.raises(ValueError, match=".*the ngram parameter.*"):
        atom.plot_ngrams(ngram=6, display=False)

    atom.plot_ngrams(ngram=ngram, display=False)  # When corpus is str
    atom.tokenize()
    atom.plot_ngrams(ngram=ngram, display=False)  # When corpus are tokens


def test_plot_qq():
    """Assert that the plot_qq method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_qq(columns=[0, 1], distributions="pearson3", display=False)


def test_plot_relationships():
    """Assert that the plot_relationships method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_relationships(display=False)


def test_plot_wordcloud():
    """Assert that the plot_wordcloud method works."""
    atom = ATOMClassifier(X_text, y10, random_state=1)
    atom.plot_wordcloud(display=False)  # When corpus is str
    atom.tokenize()
    atom.plot_wordcloud(display=False)  # When corpus are tokens


# Test HTPlot =========================================================== >>

def test_plot_edf():
    """Assert that the plot_edf method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_trials)
    atom.run(["lasso", "ridge"], n_trials=(5, 0))

    # Model didn't ran hyperparameter tuning
    with pytest.raises(ValueError, match=".*ran hyperparameter tuning.*"):
        atom.ridge.plot_edf(display=False)

    atom.lasso.plot_edf(display=False)


def test_plot_hyperparameter_importance():
    """Assert that the plot_hyperparameter_importance method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_trials)
    atom.run(["lasso", "ridge"], n_trials=(5, 0))

    # Invalid show parameter
    with pytest.raises(ValueError, match=".*the show parameter.*"):
        atom.lasso.plot_hyperparameter_importance(show=-1, display=False)

    # Model didn't ran hyperparameter tuning
    with pytest.raises(ValueError, match=".*ran hyperparameter tuning.*"):
        atom.ridge.plot_hyperparameter_importance(display=False)

    atom.lasso.plot_hyperparameter_importance(display=False)


def test_plot_hyperparameters():
    """Assert that the plot_hyperparameters method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_trials)
    atom.run(["lasso", "ridge"], n_trials=(5, 0))

    # Model didn't ran hyperparameter tuning
    with pytest.raises(ValueError, match=".*ran hyperparameter tuning.*"):
        atom.ridge.plot_hyperparameters(display=False)

    # Only one hyperparameter
    with pytest.raises(ValueError, match=".*minimum of two parameters.*"):
        atom.lasso.plot_hyperparameters(params=[0], display=False)

    atom.lasso.plot_hyperparameters(display=False)


def test_plot_parallel_coordinate():
    """Assert that the plot_parallel_coordinate method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_trials)
    atom.run(["tree", "ridge"], n_trials=(5, 0))

    # Model didn't ran hyperparameter tuning
    with pytest.raises(ValueError, match=".*ran hyperparameter tuning.*"):
        atom.ridge.plot_parallel_coordinate(display=False)

    atom.tree.plot_parallel_coordinate(display=False)


def test_plot_trials():
    """Assert that the plot_bo method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_trials)
    atom.run("lasso", metric="max_error", n_trials=0)

    # Model didn't ran hyperparameter tuning
    with pytest.raises(ValueError, match=".*ran hyperparameter tuning.*"):
        atom.plot_trials(display=False)

    atom.run(["lasso", "ridge"], metric="max_error", n_trials=1)
    atom.plot_trials(display=False)


# Test PredictionPlot =================================================== >>

def test_plot_calibration():
    """Assert that the plot_calibration method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_calibration)
    atom.run(["Tree", "SVM"], metric="f1")

    # Invalid n_bins parameter
    with pytest.raises(ValueError, match=".*the n_bins parameter.*"):
        atom.plot_calibration(n_bins=4, display=False)

    atom.plot_calibration(display=False)


def test_plot_confusion_matrix():
    """Assert that the plot_confusion_matrix method works."""
    # For binary classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_confusion_matrix)
    atom.run(["RF", "LGB"])
    atom.plot_confusion_matrix(display=False)

    # For multiclass classification tasks
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(["RF", "LGB"])

    # Not available for multiclass
    with pytest.raises(NotImplementedError, match=".*not support the comparison.*"):
        atom.plot_confusion_matrix(display=False)

    atom.lgb.plot_confusion_matrix(display=False)


def test_plot_det():
    """Assert that the plot_det method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_det)
    atom.run(["LGB", "SVM"])
    atom.plot_det(display=False)


def test_plot_errors():
    """Assert that the plot_errors method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_errors)
    atom.run("Tree")
    atom.plot_errors(display=False)


def test_plot_evals():
    """Assert that the plot_evals method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB", "MLP"], metric="f1")

    # No in-training validation
    with pytest.raises(ValueError, match=".*no in-training validation.*"):
        atom.lr.plot_evals(display=False)

    atom.plot_evals(models=["LGB", "MLP"], display=False)


def test_plot_feature_importance():
    """Assert that the plot_feature_importance method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_feature_importance)
    atom.run(["KNN", "Tree", "Bag"])

    # Model has no feature importance values
    with pytest.raises(ValueError, match=".*has no feature.*"):
        atom.knn.plot_feature_importance(display=False)

    atom.plot_feature_importance(models=["Tree", "Bag"], display=False)
    atom.tree.plot_feature_importance(show=5, display=False)


def test_plot_gains():
    """Assert that the plot_gains method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_gains)
    atom.run(["LGB", "SVM"])
    atom.plot_gains(display=False)


def test_plot_learning_curve():
    """Assert that the plot_learning_curve method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_learning_curve)
    atom.train_sizing(["Tree", "LGB"], n_bootstrap=4)
    atom.plot_learning_curve(display=False)


def test_plot_lift():
    """Assert that the plot_lift method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_lift)
    atom.run(["LGB", "SVM"], metric="f1")
    atom.plot_lift(display=False)
    atom.lgb.plot_lift(display=False)


def test_plot_parshap():
    """Assert that the plot_parshap method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance("smote")  # To get samples over 500
    pytest.raises(NotFittedError, atom.plot_parshap)
    atom.run(["GNB", "LR"])
    atom.plot_parshap(display=False)


def test_plot_partial_dependence():
    """Assert that the plot_partial_dependence method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_partial_dependence)
    atom.run(["KNN", "LGB"], metric="f1")

    # Invalid kind parameter
    with pytest.raises(ValueError, match=".*for the kind parameter.*"):
        atom.plot_partial_dependence(kind="invalid", display=False)

    # Pair for multimodel
    with pytest.raises(ValueError, match=".*when plotting multiple models.*"):
        atom.plot_partial_dependence(columns=2, pair=3, display=False)

    # Different features for multiple models
    atom.branch = "b2"
    atom.feature_selection(strategy="pca", n_features=5)
    atom.run(["Tree"])
    with pytest.raises(ValueError, match=".*models use the same features.*"):
        atom.plot_partial_dependence(columns=(0, 1), display=False)

    del atom.branch
    atom.plot_partial_dependence(columns=[0, 1], kind="average+individual", display=False)
    atom.lgb.plot_partial_dependence(columns=[0, 1], pair=2, display=False)


def test_plot_permutation_importance():
    """Assert that the plot_permutation_importance method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_permutation_importance)
    atom.run(["Tree", "LGB"], metric="f1")

    # Invalid n_repeats parameter
    with pytest.raises(ValueError, match=".*the n_repeats parameter.*"):
        atom.plot_permutation_importance(n_repeats=0, display=False)

    atom.plot_permutation_importance(display=False)
    atom.lgb.plot_permutation_importance(display=False)


def test_plot_pipeline():
    """Assert that the plot_pipeline method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LGB")
    atom.plot_pipeline(display=False)  # No transformers

    # Called from a canvas
    with pytest.raises(PermissionError, match=".*called from a canvas.*"):
        with atom.canvas(2, 1, display=False):
            atom.plot_pipeline(display=False)
            atom.plot_pipeline(display=False)

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
    """Assert that the plot_prc method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_prc)
    atom.run(["LGB", "SVM"], metric="f1")
    atom.plot_prc(display=False)


def test_plot_probabilities():
    """Assert that the plot_probabilities method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_probabilities)
    atom.run(["Tree", "LGB", "SVM"], metric="f1")

    # Model has no predict_proba attribute
    with pytest.raises(AttributeError, match=".*with a predict_proba method.*"):
        atom.svm.plot_probabilities(display=False)

    atom.plot_probabilities(["Tree", "LGB"], display=False)
    atom.lgb.plot_probabilities(display=False)


def test_plot_residuals():
    """Assert that the plot_residuals method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_residuals)
    atom.run(["Tree", "Bag"], metric="MSE")
    atom.plot_residuals(title="plot", display=False)


@pytest.mark.parametrize("metric", ["me", ["me", "r2"]])
def test_plot_results_metric(metric):
    """Assert that the plot_results method works."""
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
    """Assert that the plot_results method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["Tree", "LGB"], metric="r2", n_trials=1)
    atom.plot_results(metric=metric, display=False)


def test_plot_roc():
    """Assert that the plot_roc method works."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run(["LGB", "SVM"], metric="f1")
    atom.plot_roc(display=False)


def test_plot_successive_halving():
    """Assert that the plot_successive_halving method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_successive_halving)
    atom.successive_halving(["Tree", "Bag", "RF", "LGB"], n_bootstrap=4)
    atom.plot_successive_halving(display=False)


@pytest.mark.parametrize("metric", [f1_score, get_scorer("f1"), "precision", "auc"])
def test_plot_threshold(metric):
    """Assert that the plot_threshold method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_threshold)
    atom.run(["Tree", "LGB", "SVM"], metric="f1")
    atom.plot_threshold(models=["Tree", "LGB"], display=False)
    atom.lgb.plot_threshold(metric=metric, display=False)


# Test ShapPlot ==================================================== >>

def test_plot_shap_bar():
    """Assert that the plot_shap_bar method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_bar)
    atom.run(["LR", "Tree"], metric="f1_macro")
    atom.lr.plot_shap_bar(display=False)


def test_plot_shap_beeswarm():
    """Assert that the plot_shap_beeswarm method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_beeswarm)
    atom.run("LR", metric="f1_macro")
    atom.plot_shap_beeswarm(display=False)


def test_plot_shap_decision():
    """Assert that the plot_shap_decision method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_decision)
    atom.run("LR", metric="f1_macro")
    atom.lr.plot_shap_decision(display=False)


def test_plot_shap_force():
    """Assert that the plot_shap_force method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_force)
    atom.run(["LR", "MLP"], metric="MSE")

    # Expected value from Explainer
    atom.lr.plot_shap_force(index=100, matplotlib=True, display=False)

    # Own calculation of expected value
    atom.mlp.plot_shap_force(index=100, matplotlib=True, display=False)

    atom.lr.plot_shap_force(matplotlib=False, filename="force", display=True)
    assert glob.glob("force.html")


def test_plot_shap_heatmap():
    """Assert that the plot_shap_heatmap method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_heatmap)
    atom.run("LR", metric="f1_macro")
    atom.plot_shap_heatmap(display=False)


@pytest.mark.parametrize("feature", [0, -1, "mean texture"])
def test_plot_shap_scatter(feature):
    """Assert that the plot_shap_scatter method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_scatter)
    atom.run("LR", metric="f1")
    atom.plot_shap_scatter(display=False)


def test_plot_shap_waterfall():
    """Assert that the plot_shap_waterfall method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_shap_waterfall)
    atom.run("Tree", metric="f1_macro")
    atom.plot_shap_waterfall(display=False)

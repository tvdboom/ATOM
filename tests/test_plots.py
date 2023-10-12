# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Unit tests for the plots module.

"""

import glob
from unittest.mock import patch

import pytest
from sklearn.metrics import f1_score, get_scorer

from atom import ATOMClassifier, ATOMForecaster, ATOMRegressor
from atom.plots.baseplot import Aesthetics, BaseFigure, BasePlot
from atom.utils.types import Legend
from atom.utils.utils import NotFittedError

from .conftest import (
    X10, X10_str, X_bin, X_class, X_label, X_reg, X_sparse, X_text, y10, y_bin,
    y_class, y_fc, y_label, y_multiclass, y_reg,
)


# Test BaseFigure ================================================== >>

def test_get_elem():
    """Assert that elements are assigned correctly."""
    base = BaseFigure()
    assert base.get_elem() == "rgb(95, 70, 144)"
    assert base.get_elem("x") == "rgb(95, 70, 144)"
    assert base.get_elem("x") == "rgb(95, 70, 144)"


# Test BasePlot ==================================================== >>

def test_aesthetics():
    """Assert that the aesthetics getter works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(atom.aesthetics, Aesthetics)
    assert isinstance(atom.palette, list)
    assert isinstance(atom.title_fontsize, int)
    assert isinstance(atom.label_fontsize, int)
    assert isinstance(atom.tick_fontsize, int)
    assert isinstance(atom.line_width, int)
    assert isinstance(atom.marker_size, int)


def test_aesthetics_setter():
    """Assert that the aesthetics setter works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.aesthetics = {"line_width": 3}
    assert atom.line_width == 3


def test_palette_setter():
    """Assert that the palette setter works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.palette = ["red", "rgb(255, 34, 20)", "#0044ff"]
    fig = atom.plot_distribution(columns=[0, 1], display=None)
    assert "rgb(255, 34, 20)" in str(fig._data_objs[2])


def test_palette_setter_invalid_name():
    """Assert that an error is raised when an invalid palette is used."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*the palette parameter.*"):
        atom.palette = "unknown"


def test_get_plot_index():
    """Assert that indices can be converted to timestamps."""
    atom = ATOMForecaster(y_fc, random_state=1)
    atom.run("ES")
    print(atom._get_plot_index(atom.dataset), type(atom._get_plot_index(atom.dataset)))
    assert atom._get_plot_index(atom.dataset) == X_bin.shape[1]


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


def test_get_hyperparams():
    """Assert that hyperparameters can be retrieved."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=3)
    assert len(atom._get_hyperparams(params=None, model=atom.tree)) == 7
    assert len(atom._get_hyperparams(params=slice(1, 4), model=atom.tree)) == 3
    assert len(atom._get_hyperparams(params=[0, 1], model=atom.tree)) == 2
    assert len(atom._get_hyperparams(params=["criterion"], model=atom.tree)) == 1
    assert len(atom._get_hyperparams(params="criterion+splitter", model=atom.tree)) == 2


def test_get_hyperparams_invalid_name():
    """Assert that an error is raised when a hyperparameter is invalid."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=3)
    with pytest.raises(ValueError, match=".*value for the params parameter.*"):
        atom._get_hyperparams(params="invalid", model=atom.tree)


def test_get_hyperparams_empty():
    """Assert that an error is raised when no hyperparameters are selected."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", n_trials=3)
    with pytest.raises(ValueError, match=".*Didn't find any hyperparameters.*"):
        atom._get_hyperparams(params=[], model=atom.tree)


def test_get_metric():
    """Assert that metrics can be selected."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric=["f1", "recall"])
    assert atom._get_metric(metric=None, max_one=False) == [0, 1]
    assert atom._get_metric(metric=["f1", "recall"], max_one=False) == [0, 1]
    assert atom._get_metric(metric="f1+recall", max_one=False) == [0, 1]


def test_get_metric_time():
    """Assert that time metrics are accepted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    assert atom._get_metric(metric="TIME", max_one=True) == "time"


def test_get_metric_time_invalid():
    """Assert that an error is raised for invalid time metrics."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*the metric parameter.*"):
        atom._get_metric(metric="time+invalid", max_one=False)


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
    with pytest.raises(ValueError, match=".*Choose from: train, test.*"):
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


@patch("atom.plots.base.go.Figure.show")
def test_custom_title_and_legend(func):
    """Assert that title and legend can be customized."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    atom.plot_roc(title=dict(text="test", x=0), legend=dict(font_color="red"))
    func.assert_called_once()


@pytest.mark.parametrize("legend", Legend.__args__)
def test_custom_legend_position(legend):
    """Assert that the legend position can be specified."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    atom.plot_roc(legend=legend, display=False)


@patch("mlflow.tracking.MlflowClient.log_figure")
def test_figure_to_mlflow(mlflow):
    """Assert that the figure is logged to mlflow."""
    atom = ATOMClassifier(X_bin, y_bin, experiment="test", random_state=1)
    atom.run(["Tree", "LGB"])
    atom.log_plots = True
    atom.plot_results(display=False)
    atom.lgb.plot_shap_scatter(display=False)
    assert mlflow.call_count == 3


@patch("atom.plots.base.go.Figure.write_html")
def test_figure_is_saved_html(func):
    """Assert that the figure is saved as .html by default."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_correlation(filename="auto", display=False)
    func.assert_called_with("plot_correlation.html")


@patch("atom.plots.base.go.Figure.write_image")
def test_figure_is_saved_png(func):
    """Assert that the figure is saved as .png if specified."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_correlation(filename="corr.png", display=False)
    func.assert_called_with("corr.png")


@patch("atom.plots.base.plt.Figure.savefig")
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


@patch("atom.plots.base.go.Figure.write_html")
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


def test_update_traces():
    """Assert that the update_traces method set default trace values."""
    plotter = BasePlot()
    plotter.update_traces(mode="lines+markers")
    plotter._custom_traces["mode"] = "lines+markers"


def test_plot_not_fitted():
    """Assert that an error is raised when atom is not fitted."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_calibration)


def test_plot_from_model():
    """Assert that plots can be called from a model class."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    atom.tree.plot_roc(display=False)
    atom.plot_roc("Tree", display=False)


def test_plot_only_one_model():
    """Assert that an error is raised when multiple models are provided."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "LDA"])
    with pytest.raises(ValueError, match=".*only accepts one model.*"):
        atom.plot_shap_beeswarm(display=False)


# Test FeatureSelectionPlot ========================================= >>

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
    atom.plot_distribution(columns="x0", distributions="kde", display=False)
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


# Test HyperparameterTuningPlot ==================================== >>

def test_plot_edf():
    """Assert that the plot_edf method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["lasso", "ridge"], n_trials=(3, 0))

    # Model didn't ran hyperparameter tuning
    with pytest.raises(ValueError, match=".*ran hyperparameter tuning.*"):
        atom.ridge.plot_edf(display=False)

    atom.lasso.plot_edf(display=False)


def test_plot_hyperparameter_importance():
    """Assert that the plot_hyperparameter_importance method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("lasso", n_trials=3)

    # Invalid show parameter
    with pytest.raises(ValueError, match=".*the show parameter.*"):
        atom.plot_hyperparameter_importance(show=-1, display=False)

    atom.plot_hyperparameter_importance(display=False)


def test_plot_hyperparameters():
    """Assert that the plot_hyperparameters method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", n_trials=3)

    # Only one hyperparameter
    with pytest.raises(ValueError, match=".*minimum of two parameters.*"):
        atom.plot_hyperparameters(params=[0], display=False)

    atom.plot_hyperparameters(params=(0, 1, 2), display=False)


def test_plot_parallel_coordinate():
    """Assert that the plot_parallel_coordinate method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("tree", n_trials=3)
    atom.plot_parallel_coordinate(display=False)


def test_plot_pareto_front():
    """Assert that the plot_pareto_front method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("tree", metric=["mae", "mse", "rmse"], n_trials=3)

    # Only one metric
    with pytest.raises(ValueError, match=".*minimum of two metrics.*"):
        atom.plot_pareto_front(metric=[0], display=False)

    atom.plot_pareto_front(display=False)


def test_plot_slice():
    """Assert that the plot_slice method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("lr", metric=["f1", "recall"], n_trials=3)
    atom.plot_slice(display=False)


def test_plot_terminator_improvements():
    """Assert that the plot_terminator_improvement method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("tree", n_trials=1)

    # No cross-validation
    with pytest.raises(ValueError, match=".*using cross-validation.*"):
        atom.plot_terminator_improvement()

    atom.run("tree", n_trials=1, ht_params={"cv": 2})
    atom.plot_terminator_improvement(display=False)


def test_plot_timeline():
    """Assert that the plot_timeline method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("tree", n_trials=1)
    atom.plot_timeline(display=False)


def test_plot_trials():
    """Assert that the plot_bo method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("lasso", n_trials=3)
    atom.plot_trials(display=False)


# Test PredictionPlot =================================================== >>

def test_plot_calibration():
    """Assert that the plot_calibration method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Dummy", "Tree"], metric="f1")

    # Invalid n_bins parameter
    with pytest.raises(ValueError, match=".*the n_bins parameter.*"):
        atom.plot_calibration(n_bins=4, display=False)

    atom.plot_calibration(display=False)


def test_plot_confusion_matrix():
    """Assert that the plot_confusion_matrix method works."""
    # For binary classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["RF", "LGB"])
    atom.plot_confusion_matrix(threshold=0.2, display=False)

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
    atom.run(["LGB", "SVM"])
    atom.plot_det(display=False)


def test_plot_errors():
    """Assert that the plot_errors method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
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
    atom.run(["KNN", "Tree", "Bag"])

    # Model has no feature importance values
    with pytest.raises(ValueError, match=".*has no feature.*"):
        atom.knn.plot_feature_importance(display=False)

    atom.plot_feature_importance(models=["Tree", "Bag"], display=False)
    atom.tree.plot_feature_importance(show=5, display=False)


def test_plot_gains():
    """Assert that the plot_gains method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    atom.plot_gains(display=False)


def test_plot_learning_curve():
    """Assert that the plot_learning_curve method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.train_sizing(["Tree", "LGB"], n_bootstrap=4)
    atom.plot_learning_curve(display=False)


def test_plot_lift():
    """Assert that the plot_lift method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric="f1")
    atom.plot_lift(display=False)


def test_plot_parshap():
    """Assert that the plot_parshap method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.balance("smote")  # To get samples over 500
    atom.run(["GNB", "LR"])
    atom.plot_parshap(display=False)  # With colorbar
    atom.gnb.plot_parshap(display=False)  # Without colorbar


def test_plot_partial_dependence():
    """Assert that the plot_partial_dependence method works."""
    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run("Tree")
    with pytest.raises(PermissionError, match=".*not available for multilabel.*"):
        atom.plot_partial_dependence(kind="invalid", display=False)

    atom = ATOMClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
    atom.run(["KNN", "LGB"])

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

    atom = ATOMClassifier(X_class, y_class, n_jobs=-1, random_state=1)
    atom.run(["Tree", "LDA"])
    atom.plot_partial_dependence(columns=[0, 1], kind="average+individual", display=False)
    atom.tree.plot_partial_dependence(columns=[0, 1], pair=2, display=False)


def test_plot_permutation_importance():
    """Assert that the plot_permutation_importance method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree", metric="f1")

    # Invalid n_repeats parameter
    with pytest.raises(ValueError, match=".*the n_repeats parameter.*"):
        atom.plot_permutation_importance(n_repeats=0, display=False)

    atom.plot_permutation_importance(display=False)


def test_plot_pipeline():
    """Assert that the plot_pipeline method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("KNN", errors="raise")
    atom.plot_pipeline(display=False)  # No transformers

    # Called from a canvas
    with pytest.raises(PermissionError, match=".*called from a canvas.*"):
        with atom.canvas(2, 1, display=False):
            atom.plot_results(display=False)
            atom.plot_pipeline(display=False)

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.scale()
    atom.plot_pipeline(display=False)  # No model

    atom.run("Tree", n_trials=2)
    atom.tree.plot_pipeline(display=False)  # Only one branch

    atom.branch = "b2"
    atom.prune()
    atom.run(["OLS", "EN"])
    atom.voting(["OLS", "EN"])
    atom.plot_pipeline(title="Pipeline plot", display=False)  # Multiple branches


def test_plot_prc():
    """Assert that the plot_prc method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    atom.plot_prc(display=False)


def test_plot_probabilities():
    """Assert that the plot_probabilities method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["Tree", "SVM"])

    # Model has no predict_proba attribute
    with pytest.raises(AttributeError, match=".*with a predict_proba method.*"):
        atom.svm.plot_probabilities(display=False)

    atom.plot_probabilities("Tree", display=False)


def test_plot_probabilities_multilabel():
    """Assert that the plot_probabilities method works for multioutput tasks."""
    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run("LR")
    atom.plot_probabilities(display=False)

    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    atom.run("LR")
    atom.plot_probabilities(display=False)


def test_plot_residuals():
    """Assert that the plot_residuals method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    atom.plot_residuals(display=False)


@pytest.mark.parametrize("metric", ["me", ["me", "r2"]])
def test_plot_results_metric(metric):
    """Assert that the plot_results method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["OLS", "Tree", "Tree_2"], metric=metric, n_bootstrap=(3, 3, 0))
    atom.voting()
    atom.plot_results(metric="me", display=False)  # Mixed bootstrap
    atom.plot_results(models=["OLS", "Tree"], metric="me", display=False)  # All bootstrap
    atom.plot_results(models="Tree_2", metric="me", display=False)  # No bootstrap


@pytest.mark.parametrize("metric", ["time_ht", "time_fit", "time"])
def test_plot_results_time(metric):
    """Assert that the plot_results method works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree", n_trials=1)
    atom.plot_results(metric=metric, display=False)


def test_plot_roc():
    """Assert that the plot_roc method works."""
    atom = ATOMClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    atom.run("Tree")
    atom.plot_roc(display=False)


def test_plot_successive_halving():
    """Assert that the plot_successive_halving method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.successive_halving(["Tree", "Bag", "RF", "LGB"], n_bootstrap=4)
    atom.plot_successive_halving(display=False)


@pytest.mark.parametrize("metric", [f1_score, get_scorer("f1"), "precision", "auc"])
def test_plot_threshold(metric):
    """Assert that the plot_threshold method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    atom.plot_threshold(metric=metric, display=False)


def test_plot_threshold_multilabel():
    """Assert that the plot_threshold method works for multilabel tasks."""
    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run("Tree")
    atom.plot_threshold(display=False)


# Test ShapPlot ==================================================== >>

def test_plot_shap_fail():
    """Assert that an error is raised when the explainer can't be created."""
    atom = ATOMClassifier(X_class, y=y_multiclass, random_state=1)
    atom.run("LDA")
    with pytest.raises(ValueError, match=".*Failed to get shap's explainer.*"):
        atom.plot_shap_beeswarm(display=False)


def test_plot_shap_multioutput():
    """Assert that the shap plots work with multioutput tasks."""
    atom = ATOMClassifier(X_label, y=y_label, stratify=False, random_state=1)
    atom.run(["LR", "Tree"])
    atom.lr.plot_shap_bar(display=False)  # Non-native multioutput
    atom.tree.plot_shap_bar(display=False)  # Native multioutput


def test_plot_shap_bar():
    """Assert that the plot_shap_bar method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(["LR", "Tree"], metric="f1_macro")
    atom.lr.plot_shap_bar(display=False)


def test_plot_shap_beeswarm():
    """Assert that the plot_shap_beeswarm method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("LR", metric="f1_macro")
    atom.plot_shap_beeswarm(display=False)


def test_plot_shap_decision():
    """Assert that the plot_shap_decision method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("LR", metric="f1_macro")
    atom.lr.plot_shap_decision(display=False)


def test_plot_shap_force():
    """Assert that the plot_shap_force method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
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
    atom.run("LR", metric="f1_macro")
    atom.plot_shap_heatmap(display=False)


@pytest.mark.parametrize("feature", [0, -1, "mean texture"])
def test_plot_shap_scatter(feature):
    """Assert that the plot_shap_scatter method works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    atom.plot_shap_scatter(display=False)


def test_plot_shap_waterfall():
    """Assert that the plot_shap_waterfall method works."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run("Tree")
    atom.plot_shap_waterfall(display=False)

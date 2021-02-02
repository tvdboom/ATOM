# coding: utf-8

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Unit tests for plots.py

"""

# Standard packages
import glob
import pytest
from sklearn.metrics import f1_score, get_scorer

# Own modules
from atom import ATOMClassifier, ATOMRegressor
from atom.plots import BasePlotter
from atom.utils import NotFittedError
from .utils import FILE_DIR, X_bin, y_bin, X_class, y_class, X_reg, y_reg


# Test BasePlotter ================================================= >>

def test_aesthetics_property():
    """Assert that aesthetics returns the classes aesthetics as dict."""
    assert isinstance(BasePlotter().aesthetics, dict)


def test_aesthetics_setter():
    """Assert that the aesthetics setter works as intended."""
    base = BasePlotter()
    base.aesthetics = {"palette": "Blues"}
    assert base.palette == "Blues"


def test_style_property():
    """Assert that style returns the classes aesthetics as dict."""
    assert isinstance(BasePlotter().style, str)


def test_style_setter():
    """Assert that the style setter works as intended."""
    with pytest.raises(ValueError):
        BasePlotter().style = "unknown"


def test_palette_property():
    """Assert that palette returns the classes aesthetics as dict."""
    assert isinstance(BasePlotter().palette, str)


def test_palette_setter():
    """Assert that the palette setter works as intended."""
    with pytest.raises(ValueError):
        BasePlotter().palette = "unknown"


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


def test_reset_aesthetics():
    """Assert that the reset_aesthetics method set values to default."""
    plotter = BasePlotter()
    plotter.style = "white"
    plotter.reset_aesthetics()
    assert plotter.style == "darkgrid"


def test_canvas():
    """Assert that the canvas works."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Tree")
    with atom.canvas(1, 2, title="Title", filename=FILE_DIR + "canvas", display=False):
        atom.plot_residuals()
        atom.plot_feature_importance()
    assert glob.glob(FILE_DIR + "canvas.png")


def test_canvas_too_many_plots():
    """Assert that the canvas works."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("Tree")
    with atom.canvas(1, 2, filename=FILE_DIR + "canvas", display=False):
        atom.plot_prc()
        atom.plot_roc()
        pytest.raises(RuntimeError, atom.plot_prc)


# Test plots ======================================================= >>

def test_plot_correlation():
    """Assert that the plot_correlation method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.plot_correlation(filename=FILE_DIR + "correlation", display=False)
    assert glob.glob(FILE_DIR + "correlation.png")


@pytest.mark.parametrize("show_params", [True, False])
def test_plot_pipeline(show_params):
    """Assert that the plot_pipeline method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.impute()
    atom.outliers()
    atom.feature_selection("univariate", n_features=10)
    atom.successive_halving(["Tree", "AdaB"])
    pytest.raises(ValueError, atom.plot_pipeline, branch="invalid")
    atom.plot_pipeline(
        show_params=show_params,
        filename=FILE_DIR + f"pipeline_{show_params}",
        display=False,
    )
    assert glob.glob(FILE_DIR + f"pipeline_{show_params}.png")


def test_plot_pca():
    """Assert that the plot_pca method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(PermissionError, atom.plot_pca)  # No PCA in pipeline
    atom.feature_selection(strategy="PCA", n_features=10)
    atom.plot_pca(filename=FILE_DIR + "pca", display=False)
    assert glob.glob(FILE_DIR + "pca.png")


@pytest.mark.parametrize("show", [10, 200, None])
def test_plot_components(show):
    """Assert that the plot_components method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(PermissionError, atom.plot_components)  # No PCA in pipeline
    atom.feature_selection(strategy="PCA", n_features=10)
    pytest.raises(ValueError, atom.plot_components, show=0)  # Show is invalid
    atom.plot_components(show=show, filename=FILE_DIR + "components", display=False)
    assert glob.glob(FILE_DIR + "components.png")


@pytest.mark.parametrize("scoring", [None, "auc"])
def test_plot_rfecv(scoring):
    """Assert that the plot_rfecv method work as intended """
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(PermissionError, atom.plot_rfecv)  # No RFECV in pipeline
    atom.run("lr", metric="precision")
    atom.feature_selection(strategy="RFECV", n_features=10, scoring=scoring)
    atom.plot_rfecv(filename=FILE_DIR + "rfecv", display=False)
    assert glob.glob(FILE_DIR + "rfecv.png")


def test_plot_successive_halving():
    """Assert that the plot_successive_halving method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_successive_halving)
    atom.run("LGB")
    atom.delete()  # Clear the pipeline to allow sh
    atom.successive_halving(models=["Tree", "Bag", "RF", "LGB"], bagging=4)
    pytest.raises(ValueError, atom.plot_successive_halving, models="unknown")
    pytest.raises(ValueError, atom.plot_successive_halving, models="BR")
    pytest.raises(ValueError, atom.plot_successive_halving, metric="unknown")
    pytest.raises(ValueError, atom.plot_successive_halving, metric=-1)
    pytest.raises(ValueError, atom.plot_successive_halving, metric=1)
    pytest.raises(ValueError, atom.plot_successive_halving, metric="roc_auc")
    atom.plot_successive_halving(
        filename=FILE_DIR + "successive_halving_1", display=False
    )
    atom.successive_halving(models=["Tree", "Bag", "RF", "LGB"])
    atom.plot_successive_halving(
        filename=FILE_DIR + "successive_halving_2", display=False
    )
    assert glob.glob(FILE_DIR + "successive_halving_1.png")
    assert glob.glob(FILE_DIR + "successive_halving_2.png")


def test_plot_learning_curve():
    """Assert that the plot_learning_curve method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_learning_curve)
    atom.run("LGB")
    atom.delete()  # Clear the pipeline to allow ts
    atom.train_sizing(["Tree", "LGB"], metric="max_error", bagging=4)
    atom.plot_learning_curve(filename=FILE_DIR + "train_sizing_1", display=False)
    atom.train_sizing(["Tree", "LGB"], metric="max_error")
    atom.plot_learning_curve(filename=FILE_DIR + "train_sizing_2", display=False)
    assert glob.glob(FILE_DIR + "train_sizing_1.png")
    assert glob.glob(FILE_DIR + "train_sizing_2.png")


@pytest.mark.parametrize("metric", ["me", ["me", "r2"]])
def test_plot_results(metric):
    """Assert that the plot_results method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_results)

    # Without bagging
    atom.run(["Tree", "LGB"], metric=metric, bagging=0)
    atom.voting()
    atom.plot_results(metric="me", filename=FILE_DIR + "bagging_1", display=False)
    atom.tree.plot_results(filename=FILE_DIR + "bagging_2", display=False)
    assert glob.glob(FILE_DIR + "bagging_1.png")
    assert glob.glob(FILE_DIR + "bagging_2.png")

    # With bagging
    atom.run("Tree", metric=metric, bagging=3)
    atom.plot_results(metric="me", filename=FILE_DIR + "bagging_3", display=False)
    atom.tree.plot_results(filename=FILE_DIR + "bagging_4", display=False)
    assert glob.glob(FILE_DIR + "bagging_3.png")
    assert glob.glob(FILE_DIR + "bagging_4.png")


def test_plot_bo():
    """Assert that the plot_bo method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_bo)
    atom.run("lasso", metric="max_error", n_calls=0)
    pytest.raises(PermissionError, atom.plot_bo)  # No BO in pipeline
    atom.run(["lasso", "ridge"], metric="max_error", n_calls=10)
    atom.plot_bo(filename=FILE_DIR + "bagging_1", display=False)
    atom.lasso.plot_bo(filename=FILE_DIR + "bagging_2", display=False)
    assert glob.glob(FILE_DIR + "bagging_1.png")
    assert glob.glob(FILE_DIR + "bagging_2.png")


def test_plot_evals():
    """Assert that the plot_evals method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run(["LR", "LGB"], metric="f1")
    pytest.raises(ValueError, atom.plot_evals)  # More than 1 model
    pytest.raises(AttributeError, atom.LR.plot_evals)  # LR has no in-training eval
    atom.plot_evals(models="LGB", filename=FILE_DIR + "evals_1", display=False)
    atom.lgb.plot_evals(filename=FILE_DIR + "evals_2", display=False)
    assert glob.glob(FILE_DIR + "evals_1.png")
    assert glob.glob(FILE_DIR + "evals_2.png")


@pytest.mark.parametrize("dataset", ["train", "test", "both"])
def test_plot_roc(dataset):
    """Assert that the plot_roc method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("LGB")
    pytest.raises(PermissionError, atom.plot_roc)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_roc)
    atom.run(["XGB", "LGB"], metric="f1")
    pytest.raises(ValueError, atom.lgb.plot_roc, dataset="invalid")
    atom.plot_roc(
        dataset=dataset, filename=FILE_DIR + f"roc_{dataset}_1", display=False
    )
    atom.lgb.plot_roc(
        dataset=dataset, filename=FILE_DIR + f"roc_{dataset}_2", display=False
    )
    assert glob.glob(FILE_DIR + f"roc_{dataset}_1.png")
    assert glob.glob(FILE_DIR + f"roc_{dataset}_2.png")


def test_plot_prc():
    """Assert that the plot_prc method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("LGB")
    pytest.raises(PermissionError, atom.plot_prc)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_prc)
    atom.run(["XGB", "LGB"], metric="f1")
    atom.plot_prc(filename=FILE_DIR + "prc_1", display=False)
    atom.lgb.plot_prc(filename=FILE_DIR + "prc_2", display=False)
    assert glob.glob(FILE_DIR + "prc_1.png")
    assert glob.glob(FILE_DIR + "prc_2.png")


def test_plot_permutation_importance():
    """Assert that the plot_permutation_importance method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_permutation_importance)
    atom.run(["Tree", "LGB"], metric="f1")
    pytest.raises(ValueError, atom.plot_permutation_importance, show=0)
    pytest.raises(ValueError, atom.plot_permutation_importance, n_repeats=0)
    atom.plot_permutation_importance(
        filename=FILE_DIR + "permutation_importance_1", display=False
    )
    atom.lgb.plot_permutation_importance(
        filename=FILE_DIR + "permutation_importance_2", display=False
    )
    assert glob.glob(FILE_DIR + "permutation_importance_1.png")
    assert glob.glob(FILE_DIR + "permutation_importance_2.png")


def test_plot_feature_importance():
    """Assert that the plot_feature_importance method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.plot_feature_importance)
    atom.run(["KNN", "Tree", "Bag"], metric="f1_micro")
    pytest.raises(PermissionError, atom.knn.plot_feature_importance)
    atom.plot_feature_importance(
        models=["Tree", "Bag"],
        filename=FILE_DIR + "feature_importance_1",
        display=False,
    )
    atom.tree.plot_feature_importance(
        filename=FILE_DIR + "feature_importance_2", display=False
    )
    assert glob.glob(FILE_DIR + "feature_importance_1.png")
    assert glob.glob(FILE_DIR + "feature_importance_2.png")


@pytest.mark.parametrize("features", [(("ash", "alcohol"), 2), "ash", 2])
def test_plot_partial_dependence(features):
    """Assert that the plot_partial_dependence method work as intended."""
    # For binary classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_partial_dependence)
    atom.run(["Tree", "LGB"], metric="f1")

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
        atom.plot_partial_dependence(features=["test", 2])

    # Invalid index
    with pytest.raises(ValueError, match=r".*got index.*"):
        atom.plot_partial_dependence(features=[120, 2])

    # Different features for multiple models
    atom.branch = "branch_2"
    atom.feature_selection(strategy="pca", n_features=5)
    atom.run(["tree2"])
    with pytest.raises(ValueError, match=r".*models use the same features.*"):
        atom.plot_partial_dependence(features=(0, 1))

    atom.delete("Tree2")  # Drop model created for test
    atom.branch.delete()  # Drop branch created for test
    atom.plot_partial_dependence(
        filename=FILE_DIR + "partial_dependence_1", display=False
    )
    atom.lgb.plot_feature_importance(show=5, display=False)
    atom.lgb.plot_partial_dependence(
        filename=FILE_DIR + "partial_dependence_2", display=False
    )
    assert glob.glob(FILE_DIR + "partial_dependence_1.png")
    assert glob.glob(FILE_DIR + "partial_dependence_2.png")

    # For multiclass classification tasks
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(["Tree", "LGB"], metric="f1_macro")

    # Invalid target int
    with pytest.raises(ValueError, match=r".*classes, got .*"):
        atom.plot_partial_dependence(target=5)

    # Invalid target str
    with pytest.raises(ValueError, match=r".*not found in the mapping.*"):
        atom.plot_partial_dependence(target="Yes")

    atom.lgb.plot_partial_dependence(
        features=features,
        target=2,
        filename=FILE_DIR + "partial_dependence_3",
        display=False,
    )
    assert glob.glob(FILE_DIR + "partial_dependence_3.png")


def test_plot_errors():
    """Assert that the plot_errors method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    pytest.raises(PermissionError, atom.plot_errors)  # Task is not regression

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_errors)
    atom.run(["Tree", "Bag"], metric="MSE")
    atom.plot_errors(filename=FILE_DIR + "errors_1", display=False)
    atom.tree.plot_errors(filename=FILE_DIR + "errors_2", display=False)
    assert glob.glob(FILE_DIR + "errors_1.png")
    assert glob.glob(FILE_DIR + "errors_2.png")


def test_plot_residuals():
    """Assert that the plot_residuals method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    atom.run("LR")
    pytest.raises(PermissionError, atom.plot_residuals)  # Task is not regression

    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.plot_residuals)
    atom.run(["Tree", "Bag"], metric="MSE")
    atom.plot_residuals(filename=FILE_DIR + "residuals_1", display=False)
    atom.tree.plot_residuals(filename=FILE_DIR + "residuals_2", display=False)
    assert glob.glob(FILE_DIR + f"residuals_1.png")
    assert glob.glob(FILE_DIR + f"residuals_2.png")


def test_plot_confusion_matrix():
    """Assert that the plot_confusion_matrix method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run(["Ridge"])
    pytest.raises(PermissionError, atom.plot_confusion_matrix)  # Task is not classif

    # For binary classification tasks
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_confusion_matrix)
    atom.run(["RF", "LGB"])
    atom.plot_confusion_matrix(filename=FILE_DIR + "confusion_matrix_1", display=False)
    atom.lgb.plot_confusion_matrix(
        normalize=True,
        filename=FILE_DIR + f"confusion_matrix_2",
        display=False,
    )
    assert glob.glob(FILE_DIR + "confusion_matrix_1.png")
    assert glob.glob(FILE_DIR + "confusion_matrix_2.png")

    # For multiclass classification tasks
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    atom.run(["RF", "LGB"])
    pytest.raises(NotImplementedError, atom.plot_confusion_matrix)
    atom.lgb.plot_confusion_matrix(
        normalize=True,
        filename=FILE_DIR + "confusion_matrix_3",
        display=False,
    )
    assert glob.glob(FILE_DIR + "confusion_matrix_3.png")


def test_plot_threshold():
    """Assert that the plot_threshold method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Ridge")
    pytest.raises(PermissionError, atom.plot_threshold)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_threshold)
    atom.run(["Tree", "LGB", "PA"], metric="f1")
    pytest.raises(AttributeError, atom.pa.plot_threshold)  # No predict_proba
    pytest.raises(ValueError, atom.tree.plot_threshold, metric="unknown")
    atom.plot_threshold(
        models=["Tree", "LGB"],
        filename=FILE_DIR + "threshold_1",
        display=False,
    )
    atom.lgb.plot_threshold(
        metric=[f1_score, get_scorer("average_precision"), "precision", "auc"],
        filename=FILE_DIR + "threshold_2",
        display=False,
    )
    assert glob.glob(FILE_DIR + "threshold_1.png")
    assert glob.glob(FILE_DIR + "threshold_2.png")


def test_plot_probabilities():
    """Assert that the plot_probabilities method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Ridge")
    pytest.raises(PermissionError, atom.plot_probabilities)  # Task is not classif

    y = ["a" if i == 0 else "b" for i in y_bin]
    atom = ATOMClassifier(X_bin, y, random_state=1)
    atom.clean()  # Encode the target column
    pytest.raises(NotFittedError, atom.plot_probabilities)
    atom.run(["Tree", "LGB", "PA"], metric="f1")
    pytest.raises(AttributeError, atom.pa.plot_probabilities)  # No predict_proba
    atom.plot_probabilities(
        models=["Tree", "LGB"],
        target="a",
        filename=FILE_DIR + "probabilities_1",
        display=False,
    )
    atom.lgb.plot_probabilities(
        target="b",
        filename=FILE_DIR + "probabilities_2",
        display=False,
    )
    assert glob.glob(FILE_DIR + "probabilities_1.png")
    assert glob.glob(FILE_DIR + "probabilities_2.png")


def test_plot_calibration():
    """Assert that the plot_calibration method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Ridge")
    pytest.raises(PermissionError, atom.plot_calibration)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_calibration)
    atom.run(["Tree", "kSVM"], metric="f1")
    pytest.raises(ValueError, atom.plot_calibration, n_bins=4)
    atom.plot_calibration(filename=FILE_DIR + "calibration_1", display=False)
    atom.tree.plot_calibration(filename=FILE_DIR + "calibration_2", display=False)
    assert glob.glob(FILE_DIR + "calibration_1.png")
    assert glob.glob(FILE_DIR + "calibration_2.png")


def test_plot_gains():
    """Assert that the plot_gains method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Ridge")
    pytest.raises(PermissionError, atom.plot_gains)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_gains)
    atom.run(["RNN", "LGB", "PA"], metric="f1")
    pytest.raises(AttributeError, atom.pa.plot_gains)  # No predict_proba
    atom.plot_gains(["RNN", "LGB"], filename=FILE_DIR + f"gains_1", display=False)
    atom.lgb.plot_gains(filename=FILE_DIR + "gains_2", display=False)
    assert glob.glob(FILE_DIR + "gains_1.png")
    assert glob.glob(FILE_DIR + "gains_2.png")


def test_plot_lift():
    """Assert that the plot_lift method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    atom.run("Ridge")
    pytest.raises(PermissionError, atom.plot_lift)  # Task is not binary

    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.plot_lift)
    atom.run(["Tree", "LGB", "PA"], metric="f1")
    pytest.raises(AttributeError, atom.pa.plot_lift)  # No predict_proba
    atom.plot_lift(["Tree", "LGB"], filename=FILE_DIR + "lift_1", display=False)
    atom.lgb.plot_lift(filename=FILE_DIR + "lift_2", display=False)
    assert glob.glob(FILE_DIR + f"lift_1.png")
    assert glob.glob(FILE_DIR + f"lift_2.png")


@pytest.mark.parametrize("index", [None, 100, (100, 200)])
def test_bar_plot(index):
    """Assert that the bar_plot method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.bar_plot)
    atom.run("Tree", metric="f1_macro")
    atom.bar_plot(index=index, filename=FILE_DIR + f"bar_{index}", display=False)
    assert glob.glob(FILE_DIR + f"bar_{index}.png")


def test_beeswarm_plot():
    """Assert that the beeswarm_plot method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.beeswarm_plot)
    atom.run("Tree", metric="f1_macro")
    pytest.raises(ValueError, atom.force_plot, index=(996, 998))  # Index not in dataset
    atom.beeswarm_plot(filename=FILE_DIR + "beeswarm", display=False)
    assert glob.glob(FILE_DIR + "beeswarm.png")


def test_decision_plot():
    """Assert that the decision_plot method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.decision_plot)
    atom.run("LR", metric="f1_macro")
    atom.lr.decision_plot(filename=FILE_DIR + "decision", display=False)
    assert glob.glob(FILE_DIR + "decision.png")


def test_force_plot():
    """Assert that the force_plot method work as intended."""
    atom = ATOMRegressor(X_reg, y_reg, random_state=1)
    pytest.raises(NotFittedError, atom.force_plot)
    atom.run("Tree", metric="MSE")
    with atom.canvas():
        pytest.raises(PermissionError, atom.force_plot, matplotlib=True)
    atom.force_plot(
        index=100,
        matplotlib=True,
        filename=FILE_DIR + "force_1",
        display=False,
    )
    atom.force_plot(matplotlib=False, filename=FILE_DIR + "force_2", display=True)
    assert glob.glob(FILE_DIR + "force_1.png")
    assert glob.glob(FILE_DIR + "force_2.html")


def test_heatmap_plot():
    """Assert that the heatmap_plot method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.heatmap_plot)
    atom.run("Tree", metric="f1_macro")
    atom.heatmap_plot(filename=FILE_DIR + "heatmap", display=False)
    assert glob.glob(FILE_DIR + "heatmap.png")


@pytest.mark.parametrize("feature", [0, -1, "mean texture"])
def test_scatter_plot(feature):
    """Assert that the scatter_plot method work as intended."""
    atom = ATOMClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, atom.scatter_plot)
    atom.run("Tree", metric="f1")
    atom.tree.scatter_plot(filename=FILE_DIR + f"scatter_{feature}", display=False)
    assert glob.glob(FILE_DIR + f"scatter_{feature}.png")


def test_waterfall_plot():
    """Assert that the waterfall_plot method work as intended."""
    atom = ATOMClassifier(X_class, y_class, random_state=1)
    pytest.raises(NotFittedError, atom.waterfall_plot)
    atom.run("LR", metric="f1_macro")
    atom.lr.waterfall_plot(filename=FILE_DIR + "waterfall", display=False)
    assert glob.glob(FILE_DIR + "waterfall.png")

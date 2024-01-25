"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the PredictionPlot class.

"""

from __future__ import annotations

from abc import ABCMeta
from collections import defaultdict
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from beartype import beartype
from joblib import Parallel, delayed
from numpy.random import default_rng
from plotly.colors import unconvert_from_RGB_255, unlabel_rgb
from scipy import stats
from scipy.stats.mstats import mquantiles
from sklearn.calibration import calibration_curve
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import (
    confusion_matrix, det_curve, precision_recall_curve, roc_curve,
)
from sklearn.utils import _safe_indexing
from sklearn.utils.metaestimators import available_if
from sktime.forecasting.base import ForecastingHorizon

from atom.plots.baseplot import BasePlot
from atom.utils.constants import PALETTE
from atom.utils.types import (
    Bool, ColumnSelector, FloatZeroToOneExc, Int, IntLargerEqualZero,
    IntLargerFour, IntLargerZero, Kind, Legend, MetricConstructor,
    MetricSelector, ModelsSelector, RowSelector, Sequence, TargetSelector,
    TargetsSelector, XSelector, index_t,
)
from atom.utils.utils import (
    Task, bk, check_canvas, check_dependency, check_empty, check_predict_proba,
    crash, divide, get_custom_scorer, has_task, lst, rnd,
)


@beartype
class PredictionPlot(BasePlot, metaclass=ABCMeta):
    """Prediction plots.

    Plots that use the model's predictions. These plots are accessible
    from the runners or from the models. If called from a runner, the
    `models` parameter has to be specified (if None, uses all models).
    If called from a model, that model is used and the `models` parameter
    becomes unavailable.

    """

    @crash
    def plot_bootstrap(
        self,
        models: ModelsSelector = None,
        metric: MetricSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the [bootstrapping][] scores.

        If all models applied bootstrap, it shows a boxplot of the
        results. If only some models applied bootstrap, the plot is
        a barplot, where the standard deviation of the bootstrapped
        results is shown as a black line on top of the bar. Models
        are ordered based on their score from the top down.

        !!! tip
            Use the [plot_results][] method to compare the model's
            scores on any metric.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        metric: int, str, sequence or None, default=None
            Metric to plot. Use a sequence or add `+` between options
            to select more than one. If None, the metric used to run
            the pipeline is selected.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of models.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_learning_curve
        atom.plots:PredictionPlot.plot_results
        atom.plots:PredictionPlot.plot_threshold

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["GNB", "LR"], metric=["f1", "recall"], n_bootstrap=5)
        atom.plot_bootstrap()

        # Add another model without bootstrap
        atom.run("LDA")
        atom.plot_bootstrap()
        ```

        """
        models_c = self._get_plot_models(models)
        metric_c = self._get_metric(metric)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        if all(m._bootstrap is None for m in models_c):
            raise ValueError(
                "Invalid value for the models parameter. None of the selected models "
                f"have bootstrap scores, got {models_c}. Run bootstrap through the "
                "run method,e.g., atom.run(models=['LR', 'LDA'], n_bootstrap=5)."
            )

        for met in metric_c:
            if any(m._bootstrap is None for m in models_c):
                fig.add_bar(
                    x=[m._best_score(met) for m in models_c],
                    y=[m.name for m in models_c],
                    error_x={
                        "type": "data",
                        "array": [
                            0 if m._bootstrap is None else m.bootstrap.loc[:, met].std()
                            for m in models_c
                        ],
                    },
                    orientation="h",
                    marker={
                        "color": f"rgba({BasePlot._fig.get_elem(met)[4:-1]}, 0.2)",
                        "line": {"width": 2, "color": BasePlot._fig.get_elem(met)},
                    },
                    hovertemplate="%{x}<extra></extra>",
                    name=met,
                    legendgroup=met,
                    showlegend=BasePlot._fig.showlegend(met, legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            else:
                fig.add_box(
                    x=np.ravel([m.bootstrap.loc[:, met] for m in models_c]),
                    y=np.ravel([[m.name] * len(m.bootstrap) for m in models_c]),
                    marker_color=BasePlot._fig.get_elem(met),
                    boxpoints="outliers",
                    orientation="h",
                    name=met,
                    legendgroup=met,
                    showlegend=BasePlot._fig.showlegend(met, legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}": {"categoryorder": "total ascending"},
                "bargroupgap": 0.05,
                "boxmode": "group",
            }
        )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="score",
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + len(models_c) * 50),
            plotname="plot_bootstrap",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @crash
    def plot_calibration(
        self,
        models: ModelsSelector = None,
        rows: str | Sequence[str] | dict[str, RowSelector] = "test",
        n_bins: IntLargerFour = 10,
        target: TargetSelector = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 900),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the calibration curve for a binary classifier.

        Well-calibrated classifiers are probabilistic classifiers for
        which the output of the `predict_proba` method can be directly
        interpreted as a confidence level. For instance, a calibrated
        (binary) classifier should classify the samples such that among
        the samples to which it gave a `predict_proba` value close to
        0.8, approx. 80% actually belong to the positive class. Read
        more in sklearn's [documentation][calibration].

        This figure shows two plots: the calibration curve, where the
        x-axis represents the average predicted probability in each bin
        and the y-axis is the fraction of positives, i.e., the proportion
        of samples whose class is the positive class (in each bin); and
        a distribution of all predicted probabilities of the classifier.
        This plot is available only for models with a `predict_proba`
        method in a binary or [multilabel][] classification task.

        !!! tip
            Use the [calibrate][adaboost-calibrate] method to calibrate
            the winning model.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        rows: str, sequence or dict, default="test"
            Selection of rows on which to calculate the metric.

            - If str: Name of the data set to plot.
            - If sequence: Names of the data sets to plot.
            - If dict: Names of the sets with corresponding
              [selection of rows][row-and-column-selection] as values.

        target: int or str, default=0
            Target column to look at. Only for [multilabel][] tasks.

        n_bins: int, default=10
            Number of bins used for calibration. Minimum of 5 required.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="upper left"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 900)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_lift
        atom.plots:PredictionPlot.plot_prc
        atom.plots:PredictionPlot.plot_roc

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["RF", "LGB"])
        atom.plot_calibration()
        ```

        """
        models_c = self._get_plot_models(models)
        check_predict_proba(models_c, "plot_calibration")

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(y=(0.31, 1.0))
        xaxis2, yaxis2 = BasePlot._fig.get_axes(y=(0.0, 0.29))

        for m in models_c:
            for child, ds in self._get_set(rows):
                y_true, y_pred = m._get_pred(ds, target, method="predict_proba")

                # Get calibration (frac of positives and predicted values)
                frac_pos, pred = calibration_curve(y_true, y_pred, n_bins=n_bins)

                self._draw_line(
                    x=pred,
                    y=frac_pos,
                    parent=m.name,
                    child=child,
                    mode="lines+markers",
                    marker_symbol="circle",
                    legend=legend,
                    xaxis=xaxis2,
                    yaxis=yaxis,
                )

                fig.add_histogram(
                    x=y_pred,
                    xbins={"start": 0, "end": 1, "size": 1.0 / n_bins},
                    marker={
                        "color": f"rgba({BasePlot._fig.get_elem(m.name)[4:-1]}, 0.2)",
                        "line": {"width": 2, "color": BasePlot._fig.get_elem(m.name)},
                    },
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=False,
                    xaxis=xaxis2,
                    yaxis=yaxis2,
                )

        self._draw_straight_line((pred, frac_pos), y="diagonal", xaxis=xaxis2, yaxis=yaxis)

        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}_anchor": f"x{xaxis2[1:]}",
                f"xaxis{xaxis2[1:]}_showgrid": True,
                "barmode": "overlay",
            }
        )

        self._plot(
            ax=(f"xaxis{xaxis2[1:]}", f"yaxis{yaxis2[1:]}"),
            xlabel="Predicted value",
            ylabel="Count",
            xlim=(0, 1),
        )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            ylabel="Fraction of positives",
            ylim=(-0.05, 1.05),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_calibration",
            filename=filename,
            display=display,
        )

    @available_if(has_task("classification"))
    @crash
    def plot_confusion_matrix(
        self,
        models: ModelsSelector = None,
        rows: RowSelector = "test",
        target: TargetSelector = 0,
        threshold: FloatZeroToOneExc = 0.5,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot a model's confusion matrix.

        For one model, the plot shows a heatmap. For multiple models,
        it compares TP, FP, FN and TN in a barplot (not implemented
        for multiclass classification tasks). This plot is available
        only for classification tasks.

        !!! tip
            Fill the `threshold` parameter with the result from the
            model's `get_best_threshold` method to optimize the results.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        rows: hashable, segment or sequence, default="test"
            [Selection of rows][row-and-column-selection] on which to
            calculate the confusion matrix.

        target: int or str, default=0
            Target column to look at. Only for [multioutput tasks][].

        threshold: float, default=0.5
            Threshold between 0 and 1 to convert predicted probabilities
            to class labels. Only for binary classification tasks.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="upper right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the plot's type.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_calibration
        atom.plots:PredictionPlot.plot_threshold

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, test_size=0.4)
        atom.run(["LR", "RF"])
        atom.lr.plot_confusion_matrix()  # For one model
        atom.plot_confusion_matrix()  # For multiple models
        ```

        """
        models_c = self._get_plot_models(models)
        target_c = self.branch._get_target(target, only_columns=True)

        if self.task.is_multiclass and len(models_c) > 1:
            raise NotImplementedError(
                "The plot_confusion_matrix method does not support "
                "the comparison of multiple models for multiclass "
                "or multiclass-multioutput classification tasks."
            )

        labels = np.array(
            (("True negatives", "False positives"), ("False negatives", "True positives"))
        )

        fig = self._get_figure()
        if len(models_c) == 1:
            xaxis, yaxis = BasePlot._fig.get_axes(
                x=(0, 0.87),
                coloraxis={
                    "colorscale": "Blues",
                    "cmin": 0,
                    "cmax": 100,
                    "title": "Percentage of samples",
                    "font_size": self.label_fontsize,
                },
            )
        else:
            xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            y_true, y_pred = m._get_pred(rows, target_c, method="predict")
            if threshold != 0.5:
                y_pred = (y_pred > threshold).astype(int)

            cm = confusion_matrix(y_true, y_pred)
            if len(models_c) == 1:  # Create matrix heatmap
                # Get mapping from branch or use unique values
                ticks = m.branch.mapping.get(
                    target_c, np.unique(m.branch.dataset[target_c]).astype(str)
                )

                fig.add_heatmap(
                    x=ticks,
                    y=ticks,
                    z=100.0 * cm / cm.sum(axis=1)[:, np.newaxis],
                    coloraxis=f"coloraxis{xaxis[1:]}",
                    text=cm,
                    customdata=labels,
                    texttemplate="%{text}<br>(%{z:.2f}%)",
                    textfont={"size": self.label_fontsize},
                    hovertemplate=(
                        "%{customdata}<extra></extra>"
                        if self.task.is_binary
                        else ""
                        "Predicted label:%{x}<br>True label:%{y}<br>Percentage:%{z}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

                fig.update_layout(
                    {
                        "template": "plotly_white",
                        f"yaxis{yaxis[1:]}_autorange": "reversed",
                        f"xaxis{xaxis[1:]}_showgrid": False,
                        f"yaxis{yaxis[1:]}_showgrid": False,
                    }
                )

            else:
                fig.add_bar(
                    x=cm.ravel(),
                    y=labels.ravel(),
                    orientation="h",
                    marker={
                        "color": f"rgba({BasePlot._fig.get_elem(m.name)[4:-1]}, 0.2)",
                        "line": {"width": 2, "color": BasePlot._fig.get_elem(m.name)},
                    },
                    hovertemplate="%{x}<extra></extra>",
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=BasePlot._fig.showlegend(m.name, legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

                fig.update_layout(bargroupgap=0.05)

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Predicted label" if len(models_c) == 1 else "Count",
            ylabel="True label" if len(models_c) == 1 else None,
            title=title,
            legend=legend,
            figsize=figsize or ((800, 800) if len(models_c) == 1 else (900, 600)),
            plotname="plot_confusion_matrix",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @crash
    def plot_det(
        self,
        models: ModelsSelector = None,
        rows: str | Sequence[str] | dict[str, RowSelector] = "test",
        target: TargetSelector = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ):
        """Plot the Detection Error Tradeoff curve.

        Read more about [DET][] in sklearn's documentation. Only
        available for binary classification tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        rows: str, sequence or dict, default="test"
            Selection of rows on which to calculate the metric.

            - If str: Name of the data set to plot.
            - If sequence: Names of the data sets to plot.
            - If dict: Names of the sets with corresponding
              [selection of rows][row-and-column-selection] as values.

        target: int or str, default=0
            Target column to look at. Only for [multilabel][] tasks.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="upper right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_gains
        atom.plots:PredictionPlot.plot_roc
        atom.plots:PredictionPlot.plot_prc

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["LR", "RF"])
        atom.plot_det()
        ```

        """
        models_c = self._get_plot_models(models)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            for child, ds in self._get_set(rows):
                # Get fpr-fnr pairs for different thresholds
                fpr, fnr, _ = det_curve(
                    *m._get_pred(ds, target, method=("decision_function", "predict_proba"))
                )

                self._draw_line(
                    x=fpr,
                    y=fnr,
                    mode="lines",
                    parent=m.name,
                    child=child,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="FPR",
            ylabel="FNR",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_det",
            filename=filename,
            display=display,
        )

    @available_if(has_task("!classification"))
    @crash
    def plot_errors(
        self,
        models: ModelsSelector = None,
        rows: str | Sequence[str] | dict[str, RowSelector] = "test",
        target: TargetSelector = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot a model's prediction errors.

        Plot the actual targets from a set against the predicted values
        generated by the regressor. A linear fit is made on the data.
        The gray, intersected line shows the identity line. This plot
        can be useful to detect noise or heteroscedasticity along a
        range of the target domain. This plot is unavailable for
        classification tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        rows: str, sequence or dict, default="test"
            Selection of rows on which to calculate the metric.

            - If str: Name of the data set to plot.
            - If sequence: Names of the data sets to plot.
            - If dict: Names of the sets with corresponding
              [selection of rows][row-and-column-selection] as values.

        target: int or str, default=0
            Target column to look at. Only for [multioutput tasks][].

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_residuals

        Examples
        --------
        ```pycon
        from atom import ATOMRegressor
        from sklearn.datasets import load_diabetes

        X, y = load_diabetes(return_X_y=True, as_frame=True)

        atom = ATOMRegressor(X, y)
        atom.run(["OLS", "LGB"])
        atom.plot_errors()
        ```

        """
        models_c = self._get_plot_models(models)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            for child, ds in self._get_set(rows):
                y_true, y_pred = m._get_pred(ds, target)

                self._draw_line(
                    x=y_true,
                    y=y_pred,
                    mode="markers",
                    parent=m.name,
                    child=child,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

                # Fit the points using linear regression
                from atom.models import OrdinaryLeastSquares

                model = OrdinaryLeastSquares(goal=self._goal)
                estimator = model._get_est({}).fit(bk.DataFrame(y_true), y_pred)

                self._draw_line(
                    x=(x := np.linspace(y_true.min(), y_true.max(), 100)),
                    y=estimator.predict(x[:, np.newaxis]),
                    mode="lines",
                    hovertemplate="(%{x}, %{y})<extra></extra>",
                    parent=m.name,
                    legend=None,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        self._draw_straight_line((y_true, y_pred), y="diagonal", xaxis=xaxis, yaxis=yaxis)

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            xlabel="True value",
            title=title,
            legend=legend,
            ylabel="Predicted value",
            figsize=figsize,
            plotname="plot_errors",
            filename=filename,
            display=display,
        )

    @crash
    def plot_evals(
        self,
        models: ModelsSelector = None,
        dataset: Literal["train", "test", "train+test", "test+train"] = "test",
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot evaluation curves.

        The evaluation curves are the main metric scores achieved by the
        models at every iteration of the training process. This plot is
        available only for models that allow [in-training validation][].

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str, default="test"
            Data set for which to plot the evaluation curves. Use `+`
            between options to select more than one. Choose from: "train",
            "test".

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:HyperparameterTuningPlot.plot_trials

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["XGB", "LGB"])
        atom.plot_evals()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            if not m.evals:
                raise ValueError(
                    "Invalid value for the models parameter. Model "
                    f"{m.name} has no in-training validation."
                )

            for ds in dataset.split("+"):
                self._draw_line(
                    x=list(range(len(m.evals[f"{self._metric[0].name}_{ds}"]))),
                    y=m.evals[f"{self._metric[0].name}_{ds}"],
                    marker_symbol="circle",
                    parent=m.name,
                    child=ds,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Iterations",
            ylabel=self._metric[0].name,
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_evals",
            filename=filename,
            display=display,
        )

    @crash
    def plot_feature_importance(
        self,
        models: ModelsSelector = None,
        show: IntLargerZero | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot a model's feature importance.

        The sum of importances for all features (per model) is 1.
        This plot is available only for models whose estimator has
        a `scores_`, `feature_importances_` or `coef` attribute.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_parshap
        atom.plots:PredictionPlot.plot_partial_dependence
        atom.plots:PredictionPlot.plot_permutation_importance

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["LR", "RF"])
        atom.plot_feature_importance(show=10)
        ```

        """
        models_c = self._get_plot_models(models)
        show_c = self._get_show(show, max(m.branch.n_features for m in models_c))

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            try:
                fi = m.feature_importance
            except AttributeError:
                raise ValueError(
                    "Invalid value for the models parameter. Estimator "
                    f"{m._est_class.__name__} has no scores_, feature_importances_ "
                    "nor coef_ attribute."
                ) from None

            fig.add_bar(
                x=fi,
                y=fi.index,
                orientation="h",
                marker={
                    "color": f"rgba({BasePlot._fig.get_elem(m.name)[4:-1]}, 0.2)",
                    "line": {"width": 2, "color": BasePlot._fig.get_elem(m.name)},
                },
                hovertemplate="%{x}<extra></extra>",
                name=m.name,
                legendgroup=m.name,
                showlegend=BasePlot._fig.showlegend(m.name, legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )

        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}": {"categoryorder": "total ascending"},
                "bargroupgap": 0.05,
            }
        )

        # Unique number of features over all branches
        n_fxs = len({fx for m in models_c for fx in m.branch.features})

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Normalized feature importance",
            ylim=(n_fxs - show_c - 0.5, n_fxs - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_feature_importance",
            filename=filename,
            display=display,
        )

    @available_if(has_task("forecast"))
    @crash
    def plot_forecast(
        self,
        models: ModelsSelector = None,
        fh: RowSelector | ForecastingHorizon = "dataset",
        X: XSelector | None = None,
        target: TargetSelector = 0,
        *,
        plot_insample: Bool = False,
        plot_interval: Bool = True,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 900),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot model forecasts for the target time series.

        This figure shows two plots: the upper plot shows the predicted
        values, where the gray, intersected line shows the target time
        series; and the lower plot, that shows the prediction residuals.
        This plot is only available for [forecast][time-series] tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        fh: hashable, segment, sequence, dataframe or [ForecastingHorizon][], default="dataset"
            The [forecasting horizon][row-and-column-selection] for
            which to plot the predictions.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to `fh`. This parameter
            is ignored if `fh` is part of the dataset. The data is
            transformed through the model's pipeline before using it
            for predictions.

        target: int or str, default=0
            Target column to look at. Only for [multivariate][] tasks.

        plot_insample: bool, default=False
            Whether to draw in-sample predictions (predictions on the training
            set). Models that do not support this feature are silently skipped.

        plot_interval: bool, default=True
            Whether to plot prediction intervals together with the exact
            predicted values. Models wihtout a `predict_interval` method
            are skipped silently.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="upper left"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 900)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_distribution
        atom.plots:DataPlot.plot_series
        atom.plots:PredictionPlot.plot_errors

        Examples
        --------
        ```pycon
        from atom import ATOMForecaster
        from sktime.datasets import load_airline

        y = load_airline()

        atom = ATOMForecaster(y, random_state=1)
        atom.run(
            models="arima",
            est_params={"order": (1, 1, 0), "seasonal_order": (0, 1, 0, 12)},
        )
        atom.plot_forecast()
        atom.plot_forecast(fh="train+test", plot_interval=False)

        # Forecast the next 4 years starting from the test set
        atom.plot_forecast(fh=range(len(atom.test), len(atom.test) + 48))
        ```

        """
        models_c = self._get_plot_models(models)
        target_c = self.branch._get_target(target, only_columns=True)

        if not isinstance(fh, ForecastingHorizon):
            fh = self.branch._get_rows(fh).index

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(y=(0.31, 1.0))
        xaxis2, yaxis2 = BasePlot._fig.get_axes(y=(0.0, 0.29))

        for m in models_c:
            if X is not None:
                X = m.transform(X)
            elif isinstance(fh, index_t):
                X = m.branch._all.loc[fh]

            # Draw predictions and interval
            y_pred = m.predict(fh=fh, X=check_empty(X))
            if self.task.is_multioutput:
                y_pred = y_pred[target_c]

            if not plot_insample:
                idx = y_pred.index.intersection(m.branch.train.index)
                y_pred.loc[idx] = np.NaN  # type: ignore[index]

            y_true = m.branch._all.loc[y_pred.index, target_c]

            self._draw_line(
                x=(x := self._get_plot_index(y_pred)),
                y=y_pred,
                mode="lines+markers",
                parent=m.name,
                legend=legend,
                xaxis=xaxis2,
                yaxis=yaxis,
            )

            # Draw residuals
            self._draw_line(
                x=x,
                y=np.subtract(y_true, y_pred),
                mode="lines+markers",
                parent=m.name,
                legend=legend,
                showlegend=False,
                xaxis=xaxis2,
                yaxis=yaxis2,
            )

            if plot_interval:
                try:
                    y_interval = m.predict_interval(fh=fh, X=X)
                except (AttributeError, NotImplementedError):
                    continue  # Fails for some models like ES

                if self.task.is_multioutput:
                    # Select interval of target column for multivariate
                    y = y_interval.iloc[:, y_interval.columns.get_loc(target_c)]
                else:
                    y = y_interval  # Univariate

                if not plot_insample:
                    y.loc[y.index.intersection(m.branch.train.index)] = np.NaN

                fig.add_traces(
                    [
                        go.Scatter(
                            x=x,
                            y=y.iloc[:, 1],
                            mode="lines",
                            line={"width": 1, "color": BasePlot._fig.get_elem(m.name)},
                            hovertemplate=f"%{{y}}<extra>{m.name} - upper bound</extra>",
                            legendgroup=m.name,
                            showlegend=False,
                            xaxis=xaxis2,
                            yaxis=yaxis,
                        ),
                        go.Scatter(
                            x=x,
                            y=y.iloc[:, 0],
                            mode="lines",
                            line={"width": 1, "color": BasePlot._fig.get_elem(m.name)},
                            fill="tonexty",
                            fillcolor=f"rgba{BasePlot._fig.get_elem(m.name)[3:-1]}, 0.2)",
                            hovertemplate=f"%{{y}}<extra>{m.name} - lower bound</extra>",
                            legendgroup=m.name,
                            showlegend=False,
                            xaxis=xaxis2,
                            yaxis=yaxis,
                        ),
                    ]
                )

        # Draw original time series
        fig.add_scatter(
            x=x,
            y=y_true,
            mode="lines+markers",
            line={"width": 1, "color": "black", "dash": "dash"},
            opacity=0.6,
            showlegend=False,
            xaxis=xaxis2,
            yaxis=yaxis,
        )

        # Draw horizontal reference line for residuals
        self._draw_straight_line((x, y_true), y=0, xaxis=xaxis2, yaxis=yaxis2)

        fig.update_layout({f"yaxis{yaxis[1:]}_anchor": f"x{xaxis2[1:]}"})

        self._plot(
            ax=(f"xaxis{xaxis2[1:]}", f"yaxis{yaxis2[1:]}"),
            xlabel=self.branch.dataset.index.name or "index",
            ylabel="Residuals",
        )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            ylabel=target_c,
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_forecast",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @crash
    def plot_gains(
        self,
        models: ModelsSelector = None,
        rows: str | Sequence[str] | dict[str, RowSelector] = "test",
        target: TargetSelector = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the cumulative gains curve.

        This plot is available only for binary and [multilabel][]
        classification tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        rows: str, sequence or dict, default="test"
            Selection of rows on which to calculate the metric.

            - If str: Name of the data set to plot.
            - If sequence: Names of the data sets to plot.
            - If dict: Names of the sets with corresponding
              [selection of rows][row-and-column-selection] as values.

        target: int or str, default=0
            Target column to look at. Only for [multilabel][] tasks.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_det
        atom.plots:PredictionPlot.plot_lift
        atom.plots:PredictionPlot.plot_roc

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["LR", "RF"])
        atom.plot_gains()
        ```

        """
        models_c = self._get_plot_models(models)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            for child, ds in self._get_set(rows):
                y_true, y_pred = m._get_pred(
                    ds, target, method=("decision_function", "predict_proba")
                )

                self._draw_line(
                    x=(x := np.arange(start=1, stop=len(y_true) + 1) / len(y_true)),
                    y=(y := np.cumsum(y_true.iloc[np.argsort(y_pred)[::-1]]) / y_true.sum()),
                    mode="lines",
                    parent=m.name,
                    child=child,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        self._draw_straight_line((x, y), y="diagonal", xaxis=xaxis, yaxis=yaxis)

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Fraction of sample",
            ylabel="Gain",
            xlim=(0, 1),
            ylim=(0, 1.02),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_gains",
            filename=filename,
            display=display,
        )

    @crash
    def plot_learning_curve(
        self,
        models: ModelsSelector = None,
        metric: MetricSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the learning curve: score vs number of training samples.

        This plot is available only for models fitted using
        [train sizing][]. [Ensembles][] are ignored.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        metric: int, str, sequence or None, default=None
            Metric to plot (only for multi-metric runs). Use a sequence
            or add `+` between options to select more than one. If None,
            the metric used to run the pipeline is selected.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_results
        atom.plots:PredictionPlot.plot_successive_halving

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.train_sizing(["LR", "RF"], n_bootstrap=5)
        atom.plot_learning_curve()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)
        metric_c = self._get_metric(metric)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for met in metric_c:
            x, y, std = defaultdict(list), defaultdict(list), defaultdict(list)
            for m in models_c:
                x[m._group].append(m._train_idx)
                y[m._group].append(m._best_score(met))
                if m.bootstrap is not None:
                    std[m._group].append(m.bootstrap.loc[:, met].std())

            for group in x:
                self._draw_line(
                    x=x[group],
                    y=y[group],
                    mode="lines+markers",
                    marker_symbol="circle",
                    error_y={"type": "data", "array": std[group], "visible": True},
                    parent=group,
                    child=self._metric[met].name,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

                # Add error bands
                if m.bootstrap is not None:
                    fillcolor = f"rgba{BasePlot._fig.get_elem(group)[3:-1]}, 0.2)"
                    fig.add_traces(
                        [
                            go.Scatter(
                                x=x[group],
                                y=np.add(y[group], std[group]),
                                mode="lines",
                                line={"width": 1, "color": BasePlot._fig.get_elem(group)},
                                hovertemplate="%{y}<extra>upper bound</extra>",
                                legendgroup=group,
                                showlegend=False,
                                xaxis=xaxis,
                                yaxis=yaxis,
                            ),
                            go.Scatter(
                                x=x[group],
                                y=np.subtract(y[group], std[group]),
                                mode="lines",
                                line={"width": 1, "color": BasePlot._fig.get_elem(group)},
                                fill="tonexty",
                                fillcolor=fillcolor,
                                hovertemplate="%{y}<extra>lower bound</extra>",
                                legendgroup=group,
                                showlegend=False,
                                xaxis=xaxis,
                                yaxis=yaxis,
                            ),
                        ]
                    )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            title=title,
            legend=legend,
            xlabel="Number of training samples",
            ylabel="Score",
            figsize=figsize,
            plotname="plot_learning_curve",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @crash
    def plot_lift(
        self,
        models: ModelsSelector = None,
        rows: str | Sequence[str] | dict[str, RowSelector] = "test",
        target: TargetSelector = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the lift curve.

        Only available for binary classification tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        rows: str, sequence or dict, default="test"
            Selection of rows on which to calculate the metric.

            - If str: Name of the data set to plot.
            - If sequence: Names of the data sets to plot.
            - If dict: Names of the sets with corresponding
              [selection of rows][row-and-column-selection] as values.

        target: int or str, default=0
            Target column to look at. Only for [multilabel][] tasks.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="upper right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_det
        atom.plots:PredictionPlot.plot_gains
        atom.plots:PredictionPlot.plot_prc

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["LR", "RF"])
        atom.plot_lift()
        ```

        """
        models_c = self._get_plot_models(models)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            for child, ds in self._get_set(rows):
                y_true, y_pred = m._get_pred(
                    ds, target, method=("decision_function", "predict_proba")
                )

                self._draw_line(
                    x=(x := np.arange(start=1, stop=len(y_true) + 1) / len(y_true)),
                    y=(y := np.cumsum(y_true.iloc[np.argsort(y_pred)[::-1]]) / y_true.sum() / x),
                    mode="lines",
                    parent=m.name,
                    child=child,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        self._draw_straight_line((x, y), y=1, xaxis=xaxis, yaxis=yaxis)

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Fraction of sample",
            ylabel="Lift",
            xlim=(0, 1),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_lift",
            filename=filename,
            display=display,
        )

    @available_if(has_task("!forecast"))
    @crash
    def plot_parshap(
        self,
        models: ModelsSelector = None,
        columns: ColumnSelector | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the partial correlation of shap values.

        Plots the train and test correlation between the shap value of
        every feature with its target value, after removing the effect
        of all other features (partial correlation). This plot is
        useful to identify the features that are contributing most to
        overfitting. Features that lie below the bisector (diagonal
        line) performed worse on the test set than on the training set.
        If the estimator has a `scores_`, `feature_importances_` or
        `coef_` attribute, its normalized values are shown in a color
        map. This plot is not available for [forecast][time-series] tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        columns: int, str, segment, sequence or None, default=None
            XSelector to plot. If None, it plots all features.

        target: int, str or tuple, default=1
            Class in the target column to target. For multioutput tasks,
            the value should be a tuple of the form (column, class).
            Note that for binary and multilabel tasks, the selected
            class is always the positive one.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="upper left"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_feature_importance
        atom.plots:PredictionPlot.plot_partial_dependence
        atom.plots:PredictionPlot.plot_permutation_importance

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["GNB", "RF"])
        atom.rf.plot_parshap(legend=None)
        atom.plot_parshap(columns=slice(5, 10))
        ```

        """
        models_c = self._get_plot_models(models)
        target_c = self.branch._get_target(target)

        fig = self._get_figure()

        # Colorbar is only needed when a model has feature_importance
        if any(hasattr(m, "feature_importance") for m in models_c):
            xaxis, yaxis = BasePlot._fig.get_axes()
        else:
            xaxis, yaxis = BasePlot._fig.get_axes(
                x=(0, 0.87),
                coloraxis={
                    "colorscale": "Reds",
                    "title": "Normalized feature importance",
                    "font_size": self.label_fontsize,
                },
            )

        for m in models_c:
            parshap = {}
            fxs = m.branch._get_columns(columns, include_target=False)

            for ds in ("train", "test"):
                # Calculating shap values is computationally expensive,
                # therefore, select a random subsample for large data sets
                if len(data := getattr(m, f"X_{ds}")) > 500:
                    data = data.sample(500, random_state=self.random_state)

                explanation = m._shap.get_explanation(data, target_c)
                shap = bk.DataFrame(explanation.values, columns=m.branch.features)

                parshap[ds] = pd.Series(index=fxs, dtype=float)
                for fx in fxs:
                    # Compute covariance (other variables are covariates)
                    V = shap[[c for c in shap if c != fx]].cov()

                    # Inverse covariance matrix
                    Vi = np.linalg.pinv(V, hermitian=True)
                    diag = Vi.diagonal()

                    D = np.diag(np.sqrt(1 / diag))

                    # Partial correlation matrix
                    partial_corr = -1 * (D @ Vi @ D)  # @ is matrix multiplication

                    # Semi-partial correlation matrix
                    with np.errstate(divide="ignore"):
                        V_sqrt = np.sqrt(np.diag(V))[..., None]
                        Vi_sqrt = np.sqrt(np.abs(diag - Vi**2 / diag[..., None])).T
                        semi_partial_correlation = partial_corr / V_sqrt / Vi_sqrt

                    # X covariates are removed
                    parshap[ds][fx] = semi_partial_correlation[1, 0]

            # Get the feature importance or coefficients
            color: str | pd.Series
            if hasattr(m, "feature_importance"):
                color = m.feature_importance.loc[fxs]
            else:
                color = BasePlot._fig.get_elem("parshap")

            fig.add_scatter(
                x=(x := parshap["train"]),
                y=(y := parshap["test"]),
                mode="markers+text",
                marker={
                    "color": color,
                    "size": self.marker_size,
                    "coloraxis": f"coloraxis{xaxis[1:]}",
                    "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                },
                text=m.branch.features,
                textposition="top center",
                customdata=(data := None if isinstance(color, str) else list(color)),
                hovertemplate=(
                    f"%{{text}}<br>(%{{x}}, %{{y}})"
                    f"{'<br>Feature importance: %{customdata:.4f}' if data else ''}"
                    f"<extra>{m.name}</extra>"
                ),
                name=m.name,
                legendgroup=m.name,
                showlegend=BasePlot._fig.showlegend(m.name, legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )

        self._draw_straight_line((x, y), y="diagonal", xaxis=xaxis, yaxis=yaxis)

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Training set",
            ylabel="Test set",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_parshap",
            filename=filename,
            display=display,
        )

    @crash
    def plot_partial_dependence(
        self,
        models: ModelsSelector = None,
        columns: ColumnSelector = (0, 1, 2),
        kind: Kind | Sequence[Kind] = "average",
        pair: IntLargerEqualZero | str | None = None,
        target: TargetSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the partial dependence of features.

        The partial dependence of a feature (or a set of features)
        corresponds to the response of the model for each possible
        value of the feature. The plot can take two forms:

        - If `pair` is None: Single feature partial dependence lines.
          The deciles of the feature values are shown with tick marks
          on the bottom.
        - If `pair` is defined: Two-way partial dependence plots are
          plotted as contour plots (only allowed for a single model).

        Read more about partial dependence on sklearn's
        [documentation][partial_dependence]. This plot is not available
        for multilabel nor multiclass-multioutput classification tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        columns: int, str, segment, sequence, dataframe, default=(0, 1, 2)
            [XSelector][row-and-column-selection] to get the partial
            dependence from.

        kind: str or sequence, default="average"
            Kind of dependence to plot. Use a sequence or add `+` between
            options to select more than one. Choose from:

            - "average": Partial dependence averaged across all samples
              in the dataset.
            - "individual": Partial dependence for up to 50 random
              samples (Individual Conditional Expectation).

            This parameter is ignored when plotting feature pairs.

        pair: int, str or None, default=None
            Feature with which to pair the features selected by
            `columns`. If specified, the resulting figure displays
            contour plots. Only allowed when plotting a single model.
            If None, the plots show the partial dependence of single
            features.

        target: int or str, default=1
            Class in the target column to look at (only for multiclass
            classification tasks).

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_feature_importance
        atom.plots:PredictionPlot.plot_parshap
        atom.plots:PredictionPlot.plot_permutation_importance

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["LR", "RF"])
        atom.plot_partial_dependence(kind="average+individual", legend="upper left")
        atom.rf.plot_partial_dependence(columns=(3, 4), pair=2)
        ```

        """
        if self.task.is_classification and self.task.is_multioutput:
            raise PermissionError(
                "The plot_partial_dependence method is not available for multilabel "
                f"nor multiclass-multioutput classification tasks, got {self.task}."
            )
        elif self.task.is_multiclass:
            _, target_c = self.branch._get_target(target)
        else:
            target_c = 0

        models_c = self._get_plot_models(models)

        axes: list[tuple[str, str]] = []
        names: list[str] = []
        fig = self._get_figure()

        for m in models_c:
            color = BasePlot._fig.get_elem(m.name)

            # Since every model can have different fxs, select them
            # every time and make sure the models use the same fxs
            columns_c = m.branch._get_columns(columns, include_target=False)

            if not names:
                names = columns_c
            elif names != columns_c:
                raise ValueError(
                    "Invalid value for the columns parameter. Not all "
                    f"models use the same features, got {names} and {columns_c}."
                )

            cols: list[tuple[str, ...]]
            if pair is not None:
                if len(models_c) > 1:
                    raise ValueError(
                        f"Invalid value for the pair parameter, got {pair}. "
                        "The value must be None when plotting multiple models"
                    )
                else:
                    pair_c = m.branch._get_columns(pair, include_target=False)
                    cols = [(c, pair_c[0]) for c in columns_c]
            else:
                cols = [(c,) for c in columns_c]

            # Create new axes
            if not axes:
                for i in range(len(cols)):
                    # Calculate the distance between subplots
                    offset = divide(0.025, len(cols) - 1)

                    # Calculate the size of the subplot
                    size = (1 - ((offset * 2) * (len(cols) - 1))) / len(cols)

                    # Determine the position for the axes
                    x_pos = i % len(cols) * (size + 2 * offset)

                    xaxis, yaxis = BasePlot._fig.get_axes(x=(x_pos, rnd(x_pos + size)))
                    axes.append((xaxis, yaxis))

            # Compute averaged predictions
            predictions = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(partial_dependence)(
                    estimator=m.estimator,
                    X=m.branch.X_test,
                    features=col,
                    kind="both" if "individual" in kind else "average",
                )
                for col in cols
            )

            # Compute deciles for ticks (only if line plots)
            if len(cols[0]) == 1:
                deciles = {}
                for fx in chain.from_iterable(cols):
                    if fx not in deciles:  # Skip if the feature is repeated
                        X_col = _safe_indexing(m.branch.X_test, fx, axis=1)
                        deciles[fx] = mquantiles(X_col, prob=np.arange(0.1, 1.0, 0.1))

            for i, (ax, fxs, pred) in enumerate(zip(axes, cols, predictions)):  # noqa: B905
                # Draw line or contour plot
                if len(pred["values"]) == 1:
                    # For both average and individual: draw ticks on the horizontal axis
                    for line in deciles[fxs[0]]:
                        fig.add_shape(
                            type="line",
                            x0=line,
                            x1=line,
                            xref=ax[0],
                            y0=0,
                            y1=0.05,
                            yref=f"{axes[0][1]} domain",
                            line={"width": 1, "color": BasePlot._fig.get_elem(m.name)},
                            opacity=0.6,
                            layer="below",
                        )

                    # Draw the mean of the individual lines
                    if "average" in kind:
                        fig.add_scatter(
                            x=pred["values"][0],
                            y=pred["average"][target_c].ravel(),
                            mode="lines",
                            line={"width": 2, "color": color},
                            name=m.name,
                            legendgroup=m.name,
                            showlegend=BasePlot._fig.showlegend(m.name, legend),
                            xaxis=ax[0],
                            yaxis=axes[0][1],
                        )

                    # Draw all individual (per sample) lines (ICE)
                    if "individual" in kind:
                        # Select up to 50 random samples to plot
                        idx = default_rng().choice(
                            list(range(len(pred["individual"][target_c]))),
                            size=min(len(pred["individual"][target_c]), 50),
                            replace=False,
                        )
                        for sample in pred["individual"][target_c, idx, :]:
                            fig.add_scatter(
                                x=pred["values"][0],
                                y=sample,
                                mode="lines",
                                line={"width": 0.5, "color": color},
                                name=m.name,
                                legendgroup=m.name,
                                showlegend=BasePlot._fig.showlegend(m.name, legend),
                                xaxis=ax[0],
                                yaxis=axes[0][1],
                            )

                else:
                    colorscale = PALETTE.get(BasePlot._fig.get_elem(m.name), "Teal")
                    fig.add_contour(
                        x=pred["values"][0],
                        y=pred["values"][1],
                        z=pred["average"][target_c],
                        contours={
                            "showlabels": True,
                            "labelfont": {
                                "size": self.tick_fontsize,
                                "color": "white",
                            },
                        },
                        hovertemplate="x:%{x}<br>y:%{y}<br>z:%{z}<extra></extra>",
                        hoverongaps=False,
                        colorscale=colorscale,
                        showscale=False,
                        showlegend=False,
                        xaxis=ax[0],
                        yaxis=axes[0][1],
                    )

                self._plot(
                    ax=(f"xaxis{ax[0][1:]}", f"yaxis{ax[1][1:]}"),
                    xlabel=fxs[0],
                    ylabel=(fxs[1] if len(fxs) > 1 else "Score") if i == 0 else None,
                )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            groupclick="togglegroup",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_partial_dependence",
            filename=filename,
            display=display,
        )

    @crash
    def plot_permutation_importance(
        self,
        models: ModelsSelector = None,
        show: Int | None = None,
        n_repeats: IntLargerZero = 10,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the feature permutation importance of models.

        !!! warning
            This method can be slow. Results are cached to fasten
            repeated calls.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

        n_repeats: int, default=10
            Number of times to permute each feature.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_feature_importance
        atom.plots:PredictionPlot.plot_partial_dependence
        atom.plots:PredictionPlot.plot_parshap

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["LR", "RF"])
        atom.plot_permutation_importance(show=10, n_repeats=7)
        ```

        """
        models_c = self._get_plot_models(models)
        show_c = self._get_show(show, max(m.branch.n_features for m in models_c))

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            # Permutation importances returns Bunch object
            permutations = self._memory.cache(permutation_importance)(
                estimator=m.estimator,
                X=m.branch.X_test,
                y=m.branch.y_test,
                scoring=self._metric[0],
                n_repeats=n_repeats,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

            fig.add_box(
                x=permutations["importances"].ravel(),
                y=list(np.ravel([[fx] * n_repeats for fx in m.branch.features])),
                marker_color=BasePlot._fig.get_elem(m.name),
                boxpoints="outliers",
                orientation="h",
                name=m.name,
                legendgroup=m.name,
                showlegend=BasePlot._fig.showlegend(m.name, legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )

        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}": {"categoryorder": "total ascending"},
                "boxmode": "group",
            }
        )

        # Unique number of features over all branches
        n_fxs = len({fx for m in models_c for fx in m.branch.features})

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Score",
            ylim=(n_fxs - show_c - 0.5, n_fxs - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_permutation_importance",
            filename=filename,
            display=display,
        )

    @crash
    def plot_pipeline(
        self,
        models: ModelsSelector = None,
        *,
        draw_hyperparameter_tuning: bool = True,
        color_branches: bool | None = None,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plot a diagram of the pipeline.

        !!! warning
            This plot uses the [schemdraw][] package, which is
            incompatible with [plotly][]. The returned plot is
            therefore a [matplotlib figure][pltfigure].

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models for which to draw the pipeline. If None, all
            pipelines are plotted.

        draw_hyperparameter_tuning: bool, default=True
            Whether to draw if the models used Hyperparameter Tuning.

        color_branches: bool or None, default=None
            Whether to draw every branch in a different color. If None,
            branches are colored when there is more than one.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Do nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the pipeline drawn.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as png. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_wordcloud

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["GNB", "RNN", "SGD", "MLP"])
        atom.voting(models=atom.winners[:2])
        atom.plot_pipeline()

        atom = ATOMClassifier(X, y, random_state=1)
        atom.scale()
        atom.prune()
        atom.run("RF", n_trials=30)

        atom.branch = "undersample"
        atom.balance("nearmiss")
        atom.run("RF_undersample")

        atom.branch = "oversample_from_main"
        atom.balance("smote")
        atom.run("RF_oversample")

        atom.plot_pipeline()
        ```

        """

        def get_length(pl, i):
            """Get the maximum length of the name of a block."""
            if len(pl) > i:
                return max(len(pl[i].__class__.__name__) * 0.5, 7)
            else:
                return 0

        def check_y(xy):
            """Return y unless there is something right, then jump."""
            while any(pos[0] > xy[0] and pos[1] == xy[1] for pos in positions.values()):
                xy = Point((xy[0], xy[1] + height))

            return xy[1]

        def add_wire(x, y):
            """Draw a connecting wire between two estimators."""
            d.add(
                Wire(shape="z", k=(x - d.here[0]) / (length + 1), arrow="->")
                .to((x, y))
                .color(branch["color"])
            )

            # Update arrowhead manually
            d.elements[-1].segments[-1].arrowwidth = 0.3
            d.elements[-1].segments[-1].arrowlength = 0.5

        check_dependency("schemdraw")
        from schemdraw import Drawing
        from schemdraw.flow import Data, RoundBox, Subroutine, Wire
        from schemdraw.util import Point

        # Allow selection with no models
        try:
            models_c = self._get_plot_models(models, check_fitted=False)
        except ValueError as ex:
            if "No models were selected" in str(ex):
                models_c = []
            else:
                raise ex from None

        fig = self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_pipeline")

        # Define branches to plot (if called from a model, it's only one)
        branches = []
        for branch in getattr(self, "_branches", [self.branch]):
            draw_models, draw_ensembles = [], []
            for m in models_c:
                if m.branch is branch:
                    if m.acronym not in ("Stack", "Vote"):
                        draw_models.append(m)
                    else:
                        draw_ensembles.append(m)

                        # Additionally, add all dependent models (if not already there)
                        draw_models.extend([i for i in m._models if i not in draw_models])

            if not models_c or draw_models:
                branches.append(
                    {
                        "name": branch.name,
                        "pipeline": list(branch.pipeline),
                        "models": draw_models,
                        "ensembles": draw_ensembles,
                    }
                )

        # Define colors per branch
        for branch in branches:
            if color_branches or (color_branches is None and len(branches) > 1):
                color = next(BasePlot._fig.palette)

                # Convert back to format accepted by matplotlib
                branch["color"] = unconvert_from_RGB_255(unlabel_rgb(color))
            else:
                branch["color"] = "black"

        # Create schematic drawing
        d = Drawing(unit=1, backend="matplotlib")
        d.config(fontsize=float(self.tick_fontsize))
        d.add(Subroutine(w=8, s=0.7).label("Raw data"))

        height = 3  # Height of every block
        length = 5  # Minimum arrow length

        # Define the x-position for every block
        x_pos = [d.here[0] + length]
        for i in range(max(len(b["pipeline"]) for b in branches)):
            len_block = reduce(max, [get_length(b["pipeline"], i) for b in branches])
            x_pos.append(x_pos[-1] + length + len_block)

        # Add positions for scaling, hyperparameter tuning and models
        x_pos.extend([x_pos[-1], x_pos[-1]])
        if any(m.scaler for m in models_c):
            x_pos[-1] = x_pos[-2] = x_pos[-3] + length + 7
        if draw_hyperparameter_tuning and any(m._study is not None for m in models_c):
            x_pos[-1] = x_pos[-2] + length + 11

        positions = {0: d.here}  # Contains the position of every element
        for branch in branches:
            d.here = positions[0]

            for i, est in enumerate(branch["pipeline"]):
                # If the estimator has already been seen, don't draw
                if id(est) in positions:
                    # Change location to estimator's end
                    d.here = positions[id(est)]
                    continue

                # Draw transformer
                add_wire(x_pos[i], check_y(d.here))
                d.add(
                    RoundBox(w=max(len(est.__class__.__name__) * 0.5, 7))
                    .label(est.__class__.__name__, color="k")
                    .color(branch["color"])
                    .anchor("W")
                    .drop("E")
                )

                positions[id(est)] = d.here

            for model in branch["models"]:
                # Position at last transformer or at start
                if branch["pipeline"]:
                    d.here = positions[id(est)]
                else:
                    d.here = positions[0]

                # For a single branch, center models
                if len(branches) == 1:
                    offset = height * (len(branch["models"]) - 1) / 2
                else:
                    offset = 0

                # Draw automated feature scaling
                if model.scaler:
                    add_wire(x_pos[-3], check_y((d.here[0], d.here[1] - offset)))
                    d.add(
                        RoundBox(w=7).label("Scaler", color="k").color(branch["color"]).drop("E")
                    )
                    offset = 0

                # Draw hyperparameter tuning
                if draw_hyperparameter_tuning and model._study is not None:
                    add_wire(x_pos[-2], check_y((d.here[0], d.here[1] - offset)))
                    d.add(
                        Data(w=11)
                        .label("Hyperparameter\nTuning", color="k")
                        .color(branch["color"])
                        .drop("E")
                    )
                    offset = 0

                # Remove classifier/regressor from model's name
                name = model.estimator.__class__.__name__
                if name.lower().endswith("classifier"):
                    name = name[:-10]
                elif name.lower().endswith("regressor"):
                    name = name[:-9]

                # Draw model
                add_wire(x_pos[-1], check_y((d.here[0], d.here[1] - offset)))
                d.add(
                    Data(w=max(len(name) * 0.5, 7))
                    .label(name, color="k")
                    .color(branch["color"])
                    .anchor("W")
                    .drop("E")
                )

                positions[id(model)] = d.here

        # Draw ensembles
        max_pos = max(pos[0] for pos in positions.values())  # Max length model names
        for branch in branches:
            for model in branch["ensembles"]:
                # Determine y-position of the ensemble
                y_pos = [positions[id(m)][1] for m in model._models]
                offset = height / 2 * (len(branch["ensembles"]) - 1)
                y = min(y_pos) + (max(y_pos) - min(y_pos)) * 0.5 - offset
                y = check_y((max_pos + length, max(min(y_pos), y)))

                d.here = (max_pos + length, y)

                d.add(
                    Data(w=max(len(model.fullname) * 0.5, 7))
                    .label(model.fullname, color="k")
                    .color(branch["color"])
                    .anchor("W")
                    .drop("E")
                )

                positions[id(model)] = d.here

                # Draw a wire from every model to the ensemble
                for m in model._models:
                    d.here = positions[id(m)]
                    add_wire(max_pos + length, y)

        if not figsize:
            dpi, bbox = fig.get_dpi(), d.get_bbox()
            figsize = (dpi * bbox.xmax // 4, (dpi / 2) * (bbox.ymax - bbox.ymin))

        d.draw(canvas=plt.gca(), showframe=False, show=False)
        plt.axis("off")

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=plt.gca(),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_pipeline",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @crash
    def plot_prc(
        self,
        models: ModelsSelector = None,
        rows: str | Sequence[str] | dict[str, RowSelector] = "test",
        target: TargetSelector = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the precision-recall curve.

        Read more about [PRC][] in sklearn's documentation. Only
        available for binary classification tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        rows: str, sequence or dict, default="test"
            Selection of rows on which to calculate the metric.

            - If str: Name of the data set to plot.
            - If sequence: Names of the data sets to plot.
            - If dict: Names of the sets with corresponding
              [selection of rows][row-and-column-selection] as values.

        target: int or str, default=0
            Target column to look at. Only for [multilabel][] tasks.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower left"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_det
        atom.plots:PredictionPlot.plot_lift
        atom.plots:PredictionPlot.plot_roc

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["LR", "RF"])
        atom.plot_prc()
        ```

        """
        models_c = self._get_plot_models(models)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            for child, ds in self._get_set(rows):
                y_true, y_pred = m._get_pred(
                    ds, target, method=("decision_function", "predict_proba")
                )

                # Get precision-recall pairs for different thresholds
                prec, rec, _ = precision_recall_curve(y_true, y_pred)

                self._draw_line(
                    x=rec,
                    y=prec,
                    mode="lines",
                    parent=m.name,
                    child=child,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        self._draw_straight_line(
            values=(rec, prec),
            y=sum(m.branch.y_test) / len(m.branch.y_test),
            xaxis=xaxis,
            yaxis=yaxis,
        )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Recall",
            ylabel="Precision",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_prc",
            filename=filename,
            display=display,
        )

    @available_if(has_task("classification"))
    @crash
    def plot_probabilities(
        self,
        models: ModelsSelector = None,
        rows: RowSelector = "test",
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the probability distribution of the target classes.

        This plot is available only for models with a `predict_proba`
        method in classification tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        rows: hashable, segment or sequence, default="test"
            [Selection of rows][row-and-column-selection] on which to
            calculate the metric.

        target: int, str or tuple, default=1
            Probability of being that class in the target column. For
            multioutput tasks, the value should be a tuple of the form
            (column, class).

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="upper right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_confusion_matrix
        atom.plots:PredictionPlot.plot_results
        atom.plots:PredictionPlot.plot_threshold

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["LR", "RF"])
        atom.plot_probabilities()
        ```

        """
        models_c = self._get_plot_models(models)
        check_predict_proba(models_c, "plot_probabilities")

        col, cls = self.branch._get_target(target)
        col = lst(self.branch.target)[col]

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            X, y_true = m.branch._get_rows(rows, return_X_y=True)
            y_pred = m.predict_proba(X.index)

            for v in np.unique(m.dataset[col]):
                # Get indices per class
                if self.task.is_multioutput:
                    if self.task is Task.multilabel_classification:
                        hist = y_pred.loc[y_true[col] == v, col]
                    else:
                        hist = y_pred.loc[cls, col].loc[y_true[col] == v]
                else:
                    hist = y_pred.loc[y_true == v, str(cls)]

                fig.add_sactter(
                    x=(x := np.linspace(0, 1, 100)),
                    y=stats.gaussian_kde(hist)(x),
                    mode="lines",
                    line={
                        "width": 2,
                        "color": BasePlot._fig.get_elem(m.name),
                        "dash": BasePlot._fig.get_elem(str(v), "dash"),
                    },
                    fill="tonexty",
                    fillcolor=f"rgba{BasePlot._fig.get_elem(m.name)[3:-1]}, 0.2)",
                    fillpattern={"shape": BasePlot._fig.get_elem(str(v), "shape")},
                    name=f"{col}={v}",
                    legendgroup=m.name,
                    legendgrouptitle={
                        "text": m.name,
                        "font_size": self.label_fontsize,
                    },
                    showlegend=BasePlot._fig.showlegend(f"{m.name}-{v}", legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="toggleitem",
            xlabel="Probability",
            ylabel="Probability density",
            xlim=(0, 1),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_probabilities",
            filename=filename,
            display=display,
        )

    @available_if(has_task("!classification"))
    @crash
    def plot_residuals(
        self,
        models: ModelsSelector = None,
        rows: str | Sequence[str] | dict[str, RowSelector] = "test",
        target: TargetSelector = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot a model's residuals.

        The plot shows the residuals (difference between the predicted
        and the true value) on the vertical axis and the independent
        variable on the horizontal axis. This plot can be useful to
        analyze the variance of the regressor's errors. If the points
        are randomly dispersed around the horizontal axis, a linear
        regression model is appropriate for the data; otherwise, a
        non-linear model is more appropriate. This plot is unavailable
        for classification tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        rows: str, sequence or dict, default="test"
            Selection of rows on which to calculate the metric.

            - If str: Name of the data set to plot.
            - If sequence: Names of the data sets to plot.
            - If dict: Names of the sets with corresponding
              [selection of rows][row-and-column-selection] as values.

        target: int or str, default=0
            Target column to look at. Only for [multioutput tasks][].

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="upper left"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_errors

        Examples
        --------
        ```pycon
        from atom import ATOMRegressor
        from sklearn.datasets import load_diabetes

        X, y = load_diabetes(return_X_y=True, as_frame=True)

        atom = ATOMRegressor(X, y)
        atom.run(["OLS", "LGB"])
        atom.plot_residuals()
        ```

        """
        models_c = self._get_plot_models(models)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(x=(0, 0.69))
        xaxis2, yaxis2 = BasePlot._fig.get_axes(x=(0.71, 1.0))

        for m in models_c:
            for child, ds in self._get_set(rows):
                y_true, y_pred = m._get_pred(ds, target)

                self._draw_line(
                    x=y_true,
                    y=(res := np.subtract(y_true, y_pred)),
                    mode="markers",
                    parent=m.name,
                    child=child,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

                fig.add_histogram(
                    y=res,
                    bingroup="residuals",
                    marker={
                        "color": f"rgba({BasePlot._fig.get_elem(m.name)[4:-1]}, 0.2)",
                        "line": {"width": 2, "color": BasePlot._fig.get_elem(m.name)},
                    },
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=False,
                    xaxis=xaxis2,
                    yaxis=yaxis,
                )

        self._draw_straight_line((y_true, res), y=0, xaxis=xaxis, yaxis=yaxis)

        fig.update_layout({f"yaxis{xaxis[1:]}_showgrid": True, "barmode": "overlay"})

        self._plot(
            ax=(f"xaxis{xaxis2[1:]}", f"yaxis{yaxis2[1:]}"),
            xlabel="Distribution",
            title=title,
        )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            ylabel="Residuals",
            xlabel="True value",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_residuals",
            filename=filename,
            display=display,
        )

    @crash
    def plot_results(
        self,
        models: ModelsSelector = None,
        metric: MetricSelector = None,
        rows: RowSelector = "test",
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Compare metric results of the models.

        Shows a barplot of the metric scores. Models are ordered based
        on their score from the top down.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        metric: int, str, sequence or None, default=None
            Metric to plot. Choose from any of sklearn's scorers, a
            function with signature `metric(y_true, y_pred, **kwargs)`
            or a scorer object. Use a sequence or add `+` between
            options to select more than one. If None, the metric used
            to run the pipeline is selected. Other available options
            are: "time_bo", "time_fit", "time_bootstrap", "time".

        rows: hashable, segment, sequence or dataframe, default="test"
            [Selection of rows][row-and-column-selection] on which to
            calculate the metric. This parameter is ignored if `metric`
            is a time metric.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of models.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_bootstrap
        atom.plots:PredictionPlot.plot_probabilities
        atom.plots:PredictionPlot.plot_threshold

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["GNB", "LR"], metric=["f1", "recall"])
        atom.plot_results()

        # Plot the time it took to fit the models
        atom.plot_results(metric="time_fit+time")

        # Plot a different metric
        atom.plot_results(metric="accuracy")

        # Plot the results on the training set
        atom.plot_results(metric="f1", rows="train")
        ```

        """
        models_c = self._get_plot_models(models)

        if metric is None:
            metric_c = list(self._metric.values())
        else:
            metric_c = []
            for m in lst(metric):
                if isinstance(m, str):
                    metric_c.extend(m.split("+"))
                else:
                    metric_c.append(m)
            metric_c = [m if "time" in m else get_custom_scorer(m) for m in metric_c]

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for met in metric_c:
            if isinstance(met, str):
                if any(not isinstance(m, str) for m in metric_c):
                    raise ValueError(
                        "Invalid value for the metric parameter. Time metrics "
                        f"can't be mixed with non-time metrics, got {metric_c}."
                    )

                fig.add_bar(
                    x=[m.results[met] for m in models_c],
                    y=[m.name for m in models_c],
                    orientation="h",
                    marker={
                        "color": f"rgba({BasePlot._fig.get_elem(met)[4:-1]}, 0.2)",
                        "line": {"width": 2, "color": BasePlot._fig.get_elem(met)},
                    },
                    hovertemplate=f"%{{x}}<extra>{met}</extra>",
                    name=met,
                    legendgroup=met,
                    showlegend=BasePlot._fig.showlegend(met, legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            else:
                fig.add_bar(
                    x=[m._get_score(met, rows) for m in models_c],
                    y=[m.name for m in models_c],
                    orientation="h",
                    marker={
                        "color": f"rgba({BasePlot._fig.get_elem(met.name)[4:-1]}, 0.2)",
                        "line": {"width": 2, "color": BasePlot._fig.get_elem(met.name)},
                    },
                    hovertemplate="%{x}<extra></extra>",
                    name=met.name,
                    legendgroup=met.name,
                    showlegend=BasePlot._fig.showlegend(met, legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}": {"categoryorder": "total ascending"},
                "bargroupgap": 0.05,
            }
        )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="time (s)" if isinstance(met, str) else "score",
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + len(models_c) * 50),
            plotname="plot_results",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @crash
    def plot_roc(
        self,
        models: ModelsSelector = None,
        rows: str | Sequence[str] | dict[str, RowSelector] = "test",
        target: TargetSelector = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the Receiver Operating Characteristics curve.

        Read more about [ROC][] in sklearn's documentation. Only
        available for classification tasks.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        rows: str, sequence or dict, default="test"
            Selection of rows on which to calculate the metric.

            - If str: Name of the data set to plot.
            - If sequence: Names of the data sets to plot.
            - If dict: Names of the sets with corresponding
              [selection of rows][row-and-column-selection] as values.

        target: int or str, default=0
            Target column to look at. Only for [multilabel][] tasks.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_gains
        atom.plots:PredictionPlot.plot_lift
        atom.plots:PredictionPlot.plot_prc

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["LR", "RF"])
        atom.plot_roc()
        ```

        """
        models_c = self._get_plot_models(models)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            for child, ds in self._get_set(rows):
                # Get False (True) Positive Rate as arrays
                fpr, tpr, _ = roc_curve(
                    *m._get_pred(ds, target, method=("decision_function", "predict_proba"))
                )

                self._draw_line(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    parent=m.name,
                    child=child,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        self._draw_straight_line((fpr, tpr), y="diagonal", xaxis=xaxis, yaxis=yaxis)

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.03),
            xlabel="FPR",
            ylabel="TPR",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_roc",
            filename=filename,
            display=display,
        )

    @crash
    def plot_successive_halving(
        self,
        models: ModelsSelector = None,
        metric: MetricSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot scores per iteration of the successive halving.

        Only use with models fitted using [successive halving][].
        [Ensembles][] are ignored.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        metric: int, str, sequence or None, default=None
            Metric to plot (only for multi-metric runs). Use a sequence
            or add `+` between options to select more than one. If None,
            the metric used to run the pipeline is selected.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_learning_curve
        atom.plots:PredictionPlot.plot_results

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.successive_halving(["Tree", "Bag", "RF", "LGB"], n_bootstrap=5)
        atom.plot_successive_halving()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)
        metric_c = self._get_metric(metric)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for met in metric_c:
            x, y, std = defaultdict(list), defaultdict(list), defaultdict(list)
            for m in models_c:
                x[m._group].append(len(m.branch._data.train_idx) // m._train_idx)
                y[m._group].append(m._best_score(met))
                if m.bootstrap is not None:
                    std[m._group].append(m.bootstrap.loc[:, met].std())

            for group in x:
                self._draw_line(
                    x=x[group],
                    y=y[group],
                    mode="lines+markers",
                    marker_symbol="circle",
                    error_y={"type": "data", "array": std[group], "visible": True},
                    parent=group,
                    child=self._metric[met].name,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

                # Add error bands
                if m.bootstrap is not None:
                    fig.add_traces(
                        [
                            go.Scatter(
                                x=x[group],
                                y=np.add(y[group], std[group]),
                                mode="lines",
                                line={"width": 1, "color": BasePlot._fig.get_elem(group)},
                                hovertemplate="%{y}<extra>upper bound</extra>",
                                legendgroup=group,
                                showlegend=False,
                                xaxis=xaxis,
                                yaxis=yaxis,
                            ),
                            go.Scatter(
                                x=x[group],
                                y=np.subtract(y[group], std[group]),
                                mode="lines",
                                line={"width": 1, "color": BasePlot._fig.get_elem(group)},
                                fill="tonexty",
                                fillcolor=f"rgba{BasePlot._fig.get_elem(group)[3:-1]}, 0.2)",
                                hovertemplate="%{y}<extra>lower bound</extra>",
                                legendgroup=group,
                                showlegend=False,
                                xaxis=xaxis,
                                yaxis=yaxis,
                            ),
                        ]
                    )

        fig.update_layout({f"xaxis{yaxis[1:]}": {"dtick": 1, "autorange": "reversed"}})

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            title=title,
            legend=legend,
            xlabel="n_models",
            ylabel="Score",
            figsize=figsize,
            plotname="plot_successive_halving",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @crash
    def plot_threshold(
        self,
        models: ModelsSelector = None,
        metric: MetricConstructor = None,
        rows: RowSelector = "test",
        target: TargetSelector = 0,
        steps: IntLargerZero = 100,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot metric performances against threshold values.

        This plot is available only for models with a `predict_proba`
        method in a binary or [multilabel][] classification task.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models are selected.

        metric: str, func, scorer, sequence or None, default=None
            Metric to plot. Choose from any of sklearn's scorers, a
            function with signature `metric(y_true, y_pred, **kwargs)`
            or a scorer object. Use a sequence or add `+` between
            options to select more than one. If None, the metric used
            to run the pipeline is selected.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Selection of rows][row-and-column-selection] on which to
            calculate the metric.

        target: int or str, default=0
            Target column to look at. Only for [multilabel][] tasks.

        steps: int, default=100
            Number of thresholds measured.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower left"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_calibration
        atom.plots:PredictionPlot.plot_confusion_matrix
        atom.plots:PredictionPlot.plot_probabilities

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["LR", "RF"])
        atom.plot_threshold()
        ```

        """
        models_c = self._get_plot_models(models)
        check_predict_proba(models_c, "plot_threshold")

        # Get all metric functions from the input
        if metric is None:
            metric_c = [m._score_func for m in self._metric]
        else:
            metric_c = []
            for m in lst(metric):
                if isinstance(m, str):
                    metric_c.extend(m.split("+"))
                else:
                    metric_c.append(m)
            metric_c = [get_custom_scorer(m)._score_func for m in metric_c]

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            y_true, y_pred = m._get_pred(rows, target, method="predict_proba")
            for met in metric_c:
                self._draw_line(
                    x=(x := np.linspace(0, 1, steps)),
                    y=[met(y_true, y_pred >= step) for step in x],
                    parent=m.name,
                    child=met.__name__,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Threshold",
            ylabel="Score",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_threshold",
            filename=filename,
            display=display,
        )

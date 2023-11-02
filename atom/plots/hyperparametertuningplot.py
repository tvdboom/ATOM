# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the HyperparameterTuningPlot class.

"""

from __future__ import annotations

from abc import ABCMeta
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from beartype.typing import Any, Callable, Sequence
from optuna.importance import FanovaImportanceEvaluator
from optuna.trial import TrialState
from optuna.visualization._parallel_coordinate import (
    _get_dims_from_info, _get_parallel_coordinate_info,
)
from optuna.visualization._terminator_improvement import _get_improvement_info
from optuna.visualization._utils import _is_log_scale
from sklearn.utils._bunch import Bunch

from atom.plots.baseplot import BasePlot
from atom.utils.constants import PALETTE
from atom.utils.types import (
    Bool, Int, IntLargerEqualZero, IntLargerZero, IntTypes, Legend,
    MetricSelector, Model, ModelSelector, ModelsSelector, ParamsSelector,
    Scalar, SegmentTypes,
)
from atom.utils.utils import (
    bk, check_dependency, crash, divide, get_segment, it, lst, rnd,
)


class HyperparameterTuningPlot(BasePlot, metaclass=ABCMeta):
    """Hyperparameter tuning plots.

    Plots that help interpret the model's study and corresponding
    trials. These plots are accessible from the runners or from the
    models. If called from a runner, the `models` parameter has to be
    specified (if None, uses all models). If called from a model, that
    model is used and the `models` parameter becomes unavailable.

    """

    @staticmethod
    def _check_hyperparams(models: list[Model]) -> list[Model]:
        """Filter for models that ran hyperparameter tuning.

        If none of the provided models ran hyperparameter tuning,
        raise an exception.

        Parameters
        ----------
        models: list of Model
            Models to check.

        Returns
        -------
        list of Model
            Models that ran hyperparameter tuning.

        """
        if not (models_c := [m for m in models if m._study is not None]):
            raise PermissionError(
                "This plot method is only available for "
                "models that ran hyperparameter tuning."
            )

        return models_c

    @staticmethod
    def _get_hyperparams(params: ParamsSelector | None, model: Model) -> list[str]:
        """Check and return a model's hyperparameters.

        Parameters
        ----------
        params: str, segment, sequence or None
            Hyperparameters to get. Use a sequence or add `+` between
            options to select more than one. If None, all the model's
            hyperparameters are selected.

        model: Model
            Get the params from this model.

        Returns
        -------
        list of str
            Selected hyperparameters.

        """
        if params is None:
            params_c = list(model._ht["distributions"])
        elif isinstance(params, SegmentTypes):
            params_c = get_segment(list(model._ht["distributions"]), params)
        else:
            params_c = []
            for param in lst(params):
                if isinstance(param, IntTypes):
                    params_c.append(list(model._ht["distributions"])[param])
                elif isinstance(param, str):
                    for p in param.split("+"):
                        if p not in model._ht["distributions"]:
                            raise ValueError(
                                "Invalid value for the params parameter. "
                                f"Hyperparameter {p} was not used during the "
                                f"optimization of model {model.name}."
                            )
                        else:
                            params_c.append(p)

        if not params_c:
            raise ValueError(f"Didn't find any hyperparameters for model {model.name}.")

        return params_c

    def _optuna_target(self, metric: str) -> Callable | None:
        """Value for the target parameter in optuna's classes.

        Parameters
        ----------
        metric: str
            Name of the metric to get.

        Returns
        -------
        lambda or None
            Returns None for single-metric runs and a lambda that
            returns the provided metric for multi-metric runs.

        """
        if len(self._metric) == 1:
            return None
        else:
            return lambda x: x.values[self._metric.index(metric)]

    @crash
    def plot_edf(
        self,
        models: ModelsSelector = None,
        metric: MetricSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the Empirical Distribution Function of a study.

        Use this plot to analyze and improve hyperparameter search
        spaces. The EDF assumes that the value of the objective
        function is in accordance with the uniform distribution over
        the objective space. This plot is only available for models
        that ran [hyperparameter tuning][].

        !!! note
            Only complete trials are considered when plotting the EDF.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models that used hyperparameter
            tuning are selected.

        metric: int, str, sequence or None, default=None
            Metric to plot (only for multi-metric runs). If str, add `+`
            between options to select more than one. If None, the metric
            used to run the pipeline is selected.

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
        atom.plots:HyperparameterTuningPlot.plot_hyperparameters
        atom.plots:HyperparameterTuningPlot.plot_trials

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from optuna.distributions import IntDistribution
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)

        # Run three models with different search spaces
        atom.run(
            models="RF_1",
            n_trials=20,
            ht_params={"distributions": {"n_estimators": IntDistribution(6, 10)}},
        )
        atom.run(
            models="RF_2",
            n_trials=20,
            ht_params={"distributions": {"n_estimators": IntDistribution(11, 15)}},
        )
        atom.run(
            models="RF_3",
            n_trials=20,
            ht_params={"distributions": {"n_estimators": IntDistribution(16, 20)}},
        )

        atom.plot_edf()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)
        models_c = self._check_hyperparams(models_c)
        metric_c = self._get_metric(metric)

        x_min = bk.concat([m.trials[metric_c] for m in models_c]).min(axis=None)
        x_max = bk.concat([m.trials[metric_c] for m in models_c]).max(axis=None)
        x = np.linspace(x_min, x_max, 100)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()
        for m in models_c:
            for met in metric_c:
                y = np.sum(m.trials[met].values[:, np.newaxis] <= x, axis=0)
                fig.add_trace(
                    self._draw_line(
                        x=x,
                        y=y / len(m.trials),
                        parent=m.name,
                        child=met,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            ylim=(0, 1),
            xlabel="Score",
            ylabel="Cumulative Probability",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_edf",
            filename=filename,
            display=display,
        )

    @crash
    def plot_hyperparameter_importance(
        self,
        models: ModelsSelector = None,
        metric: IntLargerEqualZero | str = 0,
        show: IntLargerZero | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot a model's hyperparameter importance.

        The hyperparameter importances are calculated using the
        [fANOVA][] importance evaluator. The sum of all importances for
        all parameters (per model) is 1. This plot is only available
        for models that ran [hyperparameter tuning][].

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models that used hyperparameter
            tuning are selected.

        metric: int or str, default=0
            Metric to plot (only for multi-metric runs).

        show: int or None, default=None
            Number of hyperparameters (ordered by importance) to show.
            None to show all.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of hyperparameters shown.

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
        atom.plots:HyperparameterTuningPlot.plot_hyperparameters
        atom.plots:HyperparameterTuningPlot.plot_trials

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["ET", "RF"], n_trials=10)
        atom.plot_hyperparameter_importance()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)
        models_c = self._check_hyperparams(models_c)
        metric_c = self._get_metric(metric, max_one=True)[0]
        params_c = len(set([k for m in models_c for k in m._ht["distributions"]]))
        show_c = self._get_show(show, params_c)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()
        for m in models_c:
            fanova = FanovaImportanceEvaluator(seed=self.random_state)
            importances = fanova.evaluate(m.study, target=self._optuna_target(metric_c))

            fig.add_trace(
                go.Bar(
                    x=np.array(list(importances.values())) / sum(importances.values()),
                    y=list(importances),
                    orientation="h",
                    marker=dict(
                        color=f"rgba({BasePlot._fig.get_elem(m.name)[4:-1]}, 0.2)",
                        line=dict(width=2, color=BasePlot._fig.get_elem(m.name)),
                    ),
                    hovertemplate="%{x}<extra></extra>",
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=BasePlot._fig.showlegend(m.name, legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}": dict(categoryorder="total ascending"),
                "bargroupgap": 0.05,
            }
        )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Normalized hyperparameter importance",
            ylim=(params_c - show_c - 0.5, params_c - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_hyperparameter_importance",
            filename=filename,
            display=display,
        )

    @crash
    def plot_hyperparameters(
        self,
        models: ModelSelector | None = None,
        params: ParamsSelector = (0, 1),
        metric: IntLargerEqualZero | str = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot hyperparameter relationships in a study.

        A model's hyperparameters are plotted against each other. The
        corresponding metric scores are displayed in a contour plot.
        The markers are the trials in the study. This plot is only
        available for models that ran [hyperparameter tuning][].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_hyperparameters()`.

        params: str, segment or sequence, default=(0, 1)
            Hyperparameters to plot. Use a sequence or add `+` between
            options to select more than one.

        metric: int or str, default=0
            Metric to plot (only for multi-metric runs).

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of hyperparameters shown.

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
        atom.plots:HyperparameterTuningPlot.plot_hyperparameter_importance
        atom.plots:HyperparameterTuningPlot.plot_parallel_coordinate
        atom.plots:HyperparameterTuningPlot.plot_trials

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run("LR", n_trials=15)
        atom.plot_hyperparameters(params=(0, 1, 2))
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False, max_one=True)
        model = self._check_hyperparams(models_c)[0]
        metric_c = self._get_metric(metric, max_one=True)[0]

        if len(params_c := self._get_hyperparams(params, model)) < 2:
            raise ValueError(
                "Invalid value for the hyperparameters parameter. A minimum "
                f"of two parameters is required, got {len(params_c)}."
            )

        fig = self._get_figure()
        for i in range((length := len(params_c) - 1) ** 2):
            x, y = i // length, i % length

            if y <= x:
                # Calculate the size of the subplot
                size = 1 / length

                # Determine the position for the axes
                x_pos = y * size
                y_pos = (length - x - 1) * size

                xaxis, yaxis = BasePlot._fig.get_axes(
                    x=(x_pos, rnd(x_pos + size)),
                    y=(y_pos, rnd(y_pos + size)),
                    coloraxis=dict(
                        axes="99",
                        colorscale=PALETTE.get(
                            BasePlot._fig.get_elem(model.name), "Blues"
                        ),
                        cmin=model.trials[metric_c].min(),
                        cmax=model.trials[metric_c].max(),
                        showscale=False,
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=model.trials[params_c[y]],
                        y=model.trials[params_c[x + 1]],
                        mode="markers",
                        marker=dict(
                            size=self.marker_size,
                            color=BasePlot._fig.get_elem(model.name),
                            line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                        ),
                        customdata=list(
                            zip(list(model.trials.index), model.trials[metric_c])
                        ),
                        hovertemplate=(
                            f"{params_c[y]}:%{{x}}<br>"
                            f"{params_c[x + 1]}:%{{y}}<br>"
                            f"{metric_c}:%{{customdata[1]:.4f}}"
                            "<extra>Trial %{customdata[0]}</extra>"
                        ),
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                fig.add_trace(
                    go.Contour(
                        x=model.trials[params_c[y]],
                        y=model.trials[params_c[x + 1]],
                        z=model.trials[metric_c],
                        contours=dict(
                            showlabels=True,
                            labelfont=dict(size=self.tick_fontsize, color="white")
                        ),
                        coloraxis="coloraxis99",
                        hoverinfo="skip",
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                if _is_log_scale(model.study.trials, params_c[y]):
                    fig.update_layout({f"xaxis{xaxis[1:]}_type": "log"})
                if _is_log_scale(model.study.trials, params_c[x + 1]):
                    fig.update_layout({f"yaxis{xaxis[1:]}_type": "log"})

                if x < length - 1:
                    fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
                if y > 0:
                    fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

                fig.update_layout(
                    {
                        "template": "plotly_white",
                        f"xaxis{xaxis[1:]}_showgrid": False,
                        f"yaxis{yaxis[1:]}_showgrid": False,
                        f"xaxis{yaxis[1:]}_zeroline": False,
                        f"yaxis{yaxis[1:]}_zeroline": False,
                    }
                )

                self._plot(
                    ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                    xlabel=params_c[y] if x == length - 1 else None,
                    ylabel=params_c[x + 1] if y == 0 else None,
                )

        BasePlot._fig.used_models.append(model)
        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (800 + 100 * length, 500 + 100 * length),
            plotname="plot_hyperparameters",
            filename=filename,
            display=display,
        )

    @crash
    def plot_parallel_coordinate(
        self,
        models: ModelSelector | None = None,
        params: ParamsSelector | None = None,
        metric: IntLargerEqualZero | str = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot high-dimensional parameter relationships in a study.

        Every line of the plot represents one trial. This plot is only
        available for models that ran [hyperparameter tuning][].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_parallel_coordinate()`.

        params: str, segment, sequence or None, default=None
            Hyperparameters to plot. Use a sequence or add `+` between
            options to select more than one. If None, all the model's
            hyperparameters are selected.

        metric: int or str, default=0
            Metric to plot (only for multi-metric runs).

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of hyperparameters shown.

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
        atom.plots:HyperparameterTuningPlot.plot_edf
        atom.plots:HyperparameterTuningPlot.plot_hyperparameter_importance
        atom.plots:HyperparameterTuningPlot.plot_hyperparameters

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run("RF", n_trials=15)
        atom.plot_parallel_coordinate(params=slice(1, 5))
        ```

        """

        def sort_mixed_types(values: list[str]) -> list[str]:
            """Sort a sequence of numbers and strings.

            Numbers are converted and take precedence over strings.

            Parameters
            ----------
            values: list
                Values to sort.

            Returns
            -------
            list of str
                Sorted values.

            """
            numbers: list[Scalar] = []
            categorical: list[str] = []
            for elem in values:
                try:
                    numbers.append(it(float(elem)))
                except (TypeError, ValueError):
                    categorical.append(str(elem))

            return list(map(str, sorted(numbers))) + sorted(categorical)

        models_c = self._get_plot_models(models, max_one=True, ensembles=False)
        model = self._check_hyperparams(models_c)[0]
        params_c = self._get_hyperparams(params, model)
        metric_c = self._get_metric(metric, max_one=True)[0]

        dims = _get_dims_from_info(
            _get_parallel_coordinate_info(
                study=model.study,
                params=params_c,
                target=self._optuna_target(metric_c),
                target_name=metric_c,
            )
        )

        # Clean and sort dimensions for nicer view
        dims = [dims[0]] + sorted(dims[1:], key=lambda x: params_c.index(x["label"]))
        for d in dims:
            if "ticktext" in d:
                # Skip processing for logarithmic params
                if all(isinstance(i, IntTypes) for i in d["values"]):
                    # Order categorical values
                    mapping = [d["ticktext"][i] for i in d["values"]]
                    d["ticktext"] = sort_mixed_types(d["ticktext"])
                    d["values"] = [d["ticktext"].index(v) for v in mapping]
            else:
                # Round numerical values
                d["tickvals"] = [
                    rnd(v) for v in np.linspace(min(d["values"]), max(d["values"]), 5)
                ]

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(
            coloraxis=dict(
                colorscale=PALETTE.get(BasePlot._fig.get_elem(model.name), "Blues"),
                cmin=min(dims[0]["values"]),
                cmax=max(dims[0]["values"]),
                title=metric_c,
                font_size=self.label_fontsize,
            )
        )

        fig.add_trace(
            go.Parcoords(
                dimensions=dims,
                line=dict(
                    color=dims[0]["values"],
                    coloraxis=f"coloraxis{xaxis[1:]}",
                ),
                unselected=dict(line=dict(color="gray", opacity=0.5)),
                labelside="bottom",
                labelfont=dict(size=self.label_fontsize),
            )
        )

        BasePlot._fig.used_models.append(model)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            figsize=figsize or (700 + len(params_c) * 50, 600),
            plotname="plot_parallel_coordinate",
            filename=filename,
            display=display,
        )

    @crash
    def plot_pareto_front(
        self,
        models: ModelSelector | None = None,
        metric: MetricSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the Pareto front of a study.

        Shows the trial scores plotted against each other. The marker's
        colors indicate the trial number. This plot is only available
        for models with [multi-metric runs][] and [hyperparameter tuning][].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_pareto_front()`.

        metric: str, sequence or None, default=None
            Metrics to plot.  Use a sequence or add `+` between options
            to select more than one. If None, the metrics used to run
            the pipeline are selected.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of metrics shown.

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
        atom.plots:HyperparameterTuningPlot.plot_edf
        atom.plots:HyperparameterTuningPlot.plot_slice
        atom.plots:HyperparameterTuningPlot.plot_trials

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(
            models="RF",
            metric=["f1", "accuracy", "recall"],
            n_trials=15,
         )
        atom.plot_pareto_front()
        ```

        """
        if len(self._metric) == 1:
            raise PermissionError(
                "The plot_pareto_front method is only "
                "available for models with multi-metric runs."
            )

        models_c = self._get_plot_models(models, max_one=True, ensembles=False)
        model = self._check_hyperparams(models_c)[0]

        if len(metric_c := self._get_metric(metric)) < 2:
            raise ValueError(
                "Invalid value for the metric parameter. A minimum "
                f"of two metrics are required, got {metric_c}."
            )

        fig = self._get_figure()
        for i in range((length := len(metric_c) - 1) ** 2):
            x, y = i // length, i % length

            if y <= x:
                # Calculate the distance between subplots
                offset = divide(0.0125, length - 1)

                # Calculate the size of the subplot
                size = (1 - ((offset * 2) * (length - 1))) / length

                # Determine the position for the axes
                x_pos = y * (size + 2 * offset)
                y_pos = (length - x - 1) * (size + 2 * offset)

                xaxis, yaxis = BasePlot._fig.get_axes(
                    x=(x_pos, rnd(x_pos + size)),
                    y=(y_pos, rnd(y_pos + size)),
                )

                fig.add_trace(
                    go.Scatter(
                        x=model.trials[metric_c[y]],
                        y=model.trials[metric_c[x + 1]],
                        mode="markers",
                        marker=dict(
                            size=self.marker_size,
                            color=model.trials.index,
                            colorscale="Teal",
                            line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                        ),
                        customdata=model.trials.index,
                        hovertemplate="(%{x}, %{y})<extra>Trial %{customdata}</extra>",
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                if x < length - 1:
                    fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
                if y > 0:
                    fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

                self._plot(
                    ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                    xlabel=metric_c[y] if x == length - 1 else None,
                    ylabel=metric_c[x + 1] if y == 0 else None,
                )

        BasePlot._fig.used_models.append(model)
        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (500 + 100 * length, 500 + 100 * length),
            plotname="plot_pareto_front",
            filename=filename,
            display=display,
        )

    @crash
    def plot_slice(
        self,
        models: ModelSelector | None = None,
        params: ParamsSelector | None = None,
        metric: MetricSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the parameter relationship in a study.

        The color of the markers indicates the trial. This plot is only
        available for models that ran [hyperparameter tuning][].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_slice()`.

        params: str, segment, sequence or None, default=None
            Hyperparameters to plot. Use a sequence or add `+` between
            options to select more than one. If None, all the model's
            hyperparameters are selected.

        metric: int or str, default=None
            Metric to plot (only for multi-metric runs). If str, add `+`
            between options to select more than one. If None, the metric
            used to run the pipeline is selected.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of hyperparameters shown.

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
        atom.plots:HyperparameterTuningPlot.plot_edf
        atom.plots:HyperparameterTuningPlot.plot_hyperparameters
        atom.plots:HyperparameterTuningPlot.plot_parallel_coordinate

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(
            models="RF",
            metric=["f1", "recall"],
            n_trials=15,
        )
        atom.plot_slice(params=(0, 1, 2))
        ```

        """
        models_c = self._get_plot_models(models, max_one=True, ensembles=False)
        model = self._check_hyperparams(models_c)[0]
        params_c = self._get_hyperparams(params, model)
        metric_c = self._get_metric(metric)

        fig = self._get_figure()
        for i in range(len(params_c) * len(metric_c)):
            x, y = i // len(params_c), i % len(params_c)

            # Calculate the distance between subplots
            x_offset = divide(0.0125, (len(params_c) - 1))
            y_offset = divide(0.0125, (len(metric_c) - 1))

            # Calculate the size of the subplot
            x_size = (1 - ((x_offset * 2) * (len(params_c) - 1))) / len(params_c)
            y_size = (1 - ((y_offset * 2) * (len(metric_c) - 1))) / len(metric_c)

            # Determine the position for the axes
            x_pos = y * (x_size + 2 * x_offset)
            y_pos = (len(metric_c) - x - 1) * (y_size + 2 * y_offset)

            xaxis, yaxis = BasePlot._fig.get_axes(
                x=(x_pos, rnd(x_pos + x_size)),
                y=(y_pos, rnd(y_pos + y_size)),
            )

            fig.add_trace(
                go.Scatter(
                    x=model.trials[params_c[y]],
                    y=model.trials[metric_c[x]],
                    mode="markers",
                    marker=dict(
                        size=self.marker_size,
                        color=model.trials.index,
                        colorscale="Teal",
                        line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                    ),
                    customdata=model.trials.index,
                    hovertemplate="(%{x}, %{y})<extra>Trial %{customdata}</extra>",
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

            if _is_log_scale(model.study.trials, params_c[y]):
                fig.update_layout({f"xaxis{xaxis[1:]}_type": "log"})

            if x < len(metric_c) - 1:
                fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
            if y > 0:
                fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

            self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel=params_c[y] if x == len(metric_c) - 1 else None,
                ylabel=metric_c[x] if y == 0 else None,
            )

        BasePlot._fig.used_models.append(model)
        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (800 + 100 * len(params_c), 500 + 100 * len(metric_c)),
            plotname="plot_slice",
            filename=filename,
            display=display,
        )

    @crash
    def plot_terminator_improvement(
        self,
        models: ModelsSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the potentials for future objective improvement.

        This function visualizes the objective improvement potentials.
        It helps to determine whether you should continue the
        optimization or not. The evaluated error is also plotted. Note
        that this function may take some time to compute the improvement
        potentials. This plot is only available for models that ran
        [hyperparameter tuning][].

        !!! warning
            * The plot_terminator_improvement method is only available
              for models that ran hyperparameter tuning using
              cross-validation, e.g., using `ht_params={'cv': 5}`.
            * This method does not support
              [multi-objective optimizations][multi-metric runs].
            * The calculation of the improvement can be slow. Set the
              [`memory`][atomclassifier-memory] parameter to cache the
              results and speed up repeated calls.

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models that used hyperparameter
            tuning are selected.

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
            Figure's size in pixels, format as (x, y)

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
        atom.plots:HyperparameterTuningPlot.plot_pareto_front
        atom.plots:HyperparameterTuningPlot.plot_timeline
        atom.plots:HyperparameterTuningPlot.plot_trials

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run("RF", n_trials=10, ht_params={"cv": 5})
        atom.plot_terminator_improvement()
        ```

        """
        check_dependency("botorch")

        models_c = self._get_plot_models(models, ensembles=False)
        models_c = self._check_hyperparams(models_c)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()
        for m in models_c:
            if m._ht["cv"] > 1 and len(self._metric) == 1:
                info = self.memory.cache(_get_improvement_info)(m.study, get_error=True)
            else:
                raise PermissionError(
                    "The plot_terminator_improvement method is only available for "
                    "models that ran hyperparameter tuning using cross-validation "
                    "(e.g., using ht_params={'cv': 5}) on a single-metric optimization."
                )

            fig.add_trace(
                self._draw_line(
                    x=m.trials.index,
                    y=info.improvements,
                    error_y=dict(type="data", array=info.errors),
                    mode="markers+lines",
                    parent=m.name,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Trial",
            ylabel="Terminator improvement",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_terminator_improvement",
            filename=filename,
            display=display,
        )

    @crash
    def plot_timeline(
        self,
        models: ModelsSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the timeline of a study.

        This plot is only available for models that ran
        [hyperparameter tuning][].

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models that used hyperparameter
            tuning are selected.

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
            Figure's size in pixels, format as (x, y)

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
        atom.plots:HyperparameterTuningPlot.plot_edf
        atom.plots:HyperparameterTuningPlot.plot_slice
        atom.plots:HyperparameterTuningPlot.plot_terminator_improvement

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from optuna.pruners import PatientPruner
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(
            models="LGB",
            n_trials=15,
            ht_params={"pruner": PatientPruner(None, patience=2)},
        )
        atom.plot_timeline()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)
        models_c = self._check_hyperparams(models_c)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        _cm = {
            "COMPLETE": BasePlot._fig._palette[0],  # Main color
            "FAIL": "rgb(255, 0, 0)",  # Red
            "PRUNED": "rgb(255, 165, 0)",  # Orange
            "RUNNING": "rgb(124, 252, 0)",  # Green
            "WAITING": "rgb(220, 220, 220)",  # Gray
        }

        for m in models_c:
            info = []
            for trial in m.study.get_trials(deepcopy=False):
                date_complete = trial.datetime_complete or datetime.now()
                date_start = trial.datetime_start or date_complete

                # Create nice representation of scores and params for hover
                s = [f'{m}: {trial.values[i]}' for i, m in enumerate(self._metric.keys())]
                p = [f" --> {k}: {v}" for k, v in trial.params.items()]

                info.append(
                    Bunch(
                        number=trial.number,
                        start=date_start,
                        duration=1000 * (date_complete - date_start).total_seconds(),
                        state=trial.state,
                        hovertext=(
                            f"Trial: {trial.number}<br>"
                            f"{'<br>'.join(s)}"
                            f"Parameters:<br>{'<br>'.join(p)}"
                        )
                    )
                )

            for state in sorted(TrialState, key=lambda x: x.name):
                if bars := list(filter(lambda x: x.state == state, info)):
                    fig.add_trace(
                        go.Bar(
                            name=state.name,
                            x=[b.duration for b in bars],
                            y=[b.number for b in bars],
                            base=[b.start.isoformat() for b in bars],
                            text=[b.hovertext for b in bars],
                            textposition="none",
                            hovertemplate=f"%{{text}}<extra>{m.name}</extra>",
                            orientation="h",
                            marker=dict(
                                color=f"rgba({_cm[state.name][4:-1]}, 0.2)",
                                line=dict(width=2, color=_cm[state.name]),
                            ),
                            showlegend=BasePlot._fig.showlegend(_cm[state.name], legend),
                            xaxis=xaxis,
                            yaxis=yaxis,
                        )
                    )

        fig.update_layout({f"xaxis{yaxis[1:]}_type": "date", "barmode": "group"})

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Datetime",
            ylabel="Trial",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_timeline",
            filename=filename,
            display=display,
        )

    @crash
    def plot_trials(
        self,
        models: ModelsSelector = None,
        metric: Int | str | Sequence[Int | str] | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 800),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the hyperparameter tuning trials.

        Creates a figure with two plots: the first plot shows the score
        of every trial and the second shows the distance between the
        last consecutive steps. The best trial is indicated with a star.
        This is the same plot as produced by `ht_params={"plot": True}`.
        This plot is only available for models that ran
        [hyperparameter tuning][].

        Parameters
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Models to plot. If None, all models that used hyperparameter
            tuning are selected.

        metric: int, str, sequence or None, default=None
            Metric to plot (only for multi-metric runs). Add `+` between
            options to select more than one. If None, all metrics are
            selected.

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

        figsize: tuple, default=(900, 800)
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
        atom.plots:PredictionPlot.plot_evals
        atom.plots:HyperparameterTuningPlot.plot_hyperparameters
        atom.plots:PredictionPlot.plot_results

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, flip_y=0.2, random_state=1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run(["ET", "RF"], n_trials=15)
        atom.plot_trials()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)
        models_c = self._check_hyperparams(models_c)
        metric_c = self._get_metric(metric)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(y=(0.31, 1.0))
        xaxis2, yaxis2 = BasePlot._fig.get_axes(y=(0.0, 0.29))
        for m in models_c:
            for met in metric_c:
                # Place a star symbol at the best trial
                symbols = ["circle"] * len(m.trials)
                symbols[m.best_trial.number] = "star"
                sizes = [self.marker_size] * len(m.trials)
                sizes[m.best_trial.number] = self.marker_size * 1.5

                fig.add_trace(
                    self._draw_line(
                        x=m.trials.index,
                        y=m.trials[met],
                        mode="lines+markers",
                        marker_symbol=symbols,
                        marker_size=sizes,
                        hovertemplate=None,
                        parent=m.name,
                        child=self._metric[met].name,
                        legend=legend,
                        xaxis=xaxis2,
                        yaxis=yaxis,
                    )
                )

                fig.add_trace(
                    self._draw_line(
                        x=m.trials.index,
                        y=m.trials[met].diff(),
                        mode="lines+markers",
                        marker_symbol="circle",
                        parent=m.name,
                        child=self._metric[met].name,
                        legend=legend,
                        xaxis=xaxis2,
                        yaxis=yaxis2,
                    )
                )

        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}_anchor": f"x{xaxis2[1:]}",
                f"xaxis{xaxis[1:]}_showticklabels": False,
                "hovermode": "x unified",
            },
        )

        self._plot(
            ax=(f"xaxis{xaxis2[1:]}", f"yaxis{yaxis2[1:]}"),
            xlabel="Trial",
            ylabel="d",
        )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            ylabel="Score",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_trials",
            filename=filename,
            display=display,
        )

# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the FeatureSelectionPlot class.

"""

from __future__ import annotations

from abc import ABC
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from beartype import beartype
from beartype.typing import Any
from sklearn.base import is_classifier

from atom.plots.base import BasePlot
from atom.utils.types import Bool, IntLargerZero, Legend
from atom.utils.utils import crash


@beartype
class FeatureSelectionPlot(BasePlot, ABC):
    """Feature selection plots.

    These plots are accessible from atom or from the FeatureSelector
    class when the appropriate feature selection strategy is used.

    """

    @crash
    def plot_components(
        self,
        show: IntLargerZero | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the explained variance ratio per component.

        Kept components are colored and discarded components are
        transparent. This plot is available only when feature selection
        was applied with strategy="pca".

        Parameters
        ----------
        show: int or None, default=None
            Number of components to show. None to show all.

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
            adapts the size to the number of components shown.

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
        atom.plots:FeatureSelectionPlot.plot_pca
        atom.plots:FeatureSelectionPlot.plot_rfecv

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.feature_selection("pca", n_features=5)
        atom.plot_components(show=10)
        ```

        """
        if not hasattr(self, "pca_"):
            raise ValueError(
                "The plot_pca method is only available for instances "
                "that ran feature selection using the 'pca' strategy, "
                "e.g., atom.feature_selection(strategy='pca')."
            )

        # Get the variance ratio per component
        variance = np.array(self.pca_.explained_variance_ratio_)

        show_c = self._get_show(show, len(variance))

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        # Create a color scheme: first normal and then fully transparent
        color = BasePlot._fig.get_elem("components")
        opacity = [0.2] * self.pca_._comps + [0] * (len(variance) - self.pca_._comps)

        fig.add_trace(
            go.Bar(
                x=variance,
                y=[f"pca{str(i)}" for i in range(len(variance))],
                orientation="h",
                marker=dict(
                    color=[f"rgba({color[4:-1]}, {o})" for o in opacity],
                    line=dict(width=2, color=color),
                ),
                hovertemplate="%{x}<extra></extra>",
                name=f"Variance retained: {variance[:self.pca_._comps].sum():.3f}",
                legendgroup="components",
                showlegend=BasePlot._fig.showlegend("components", legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        fig.update_layout({f"yaxis{yaxis[1:]}": dict(categoryorder="total ascending")})

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Explained variance ratio",
            ylim=(len(variance) - show_c - 0.5, len(variance) - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_components",
            filename=filename,
            display=display,
        )

    @crash
    def plot_pca(
        self,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the explained variance ratio vs number of components.

        If the underlying estimator is [PCA][] (for dense datasets),
        all possible components are plotted. If the underlying estimator
        is [TruncatedSVD][] (for sparse datasets), it only shows the
        selected components. The star marks the number of components
        selected by the user. This plot is available only when feature
        selection was applied with strategy="pca".

        Parameters
        ----------
        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

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
        atom.plots:FeatureSelectionPlot.plot_components
        atom.plots:FeatureSelectionPlot.plot_rfecv

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.feature_selection("pca", n_features=5)
        atom.plot_pca()
        ```

        """
        if not hasattr(self, "pca_"):
            raise ValueError(
                "The plot_components method is only available for instances "
                "that ran feature selection using the 'pca' strategy, "
                "e.g., atom.feature_selection(strategy='pca')."
            )

        # Create star symbol at selected number of components
        symbols = ["circle"] * self.pca_.n_features_in_
        symbols[self.pca_._comps - 1] = "star"
        sizes = [self.marker_size] * self.pca_.n_features_in_
        sizes[self.pca_._comps - 1] = self.marker_size * 1.5

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()
        fig.add_trace(
            go.Scatter(
                x=tuple(range(1, self.pca_.n_features_in_ + 1)),
                y=np.cumsum(self.pca_.explained_variance_ratio_),
                mode="lines+markers",
                line=dict(width=self.line_width, color=BasePlot._fig.get_elem("pca")),
                marker=dict(
                    symbol=symbols,
                    size=sizes,
                    line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                    opacity=1,
                ),
                hovertemplate="%{y}<extra></extra>",
                showlegend=False,
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        fig.update_layout(
            {
                "hovermode": "x",
                f"xaxis{xaxis[1:]}_showspikes": True,
                f"yaxis{yaxis[1:]}_showspikes": True,
            }
        )

        margin = self.pca_.n_features_in_ / 30
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="First N principal components",
            ylabel="Cumulative variance ratio",
            xlim=(1 - margin, self.pca_.n_features_in_ - 1 + margin),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_pca",
            filename=filename,
            display=display,
        )

    @crash
    def plot_rfecv(
        self,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the rfecv results.

        Plot the scores obtained by the estimator fitted on every
        subset of the dataset. Only available when feature selection
        was applied with strategy="rfecv".

        Parameters
        ----------
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
        atom.plots:FeatureSelectionPlot.plot_components
        atom.plots:FeatureSelectionPlot.plot_pca

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.feature_selection("rfecv", solver="Tree")
        atom.plot_rfecv()
        ```

        """
        if not hasattr(self, "rfecv_"):
            raise ValueError(
                "The plot_rfecv method is only available for instances "
                "that ran feature selection using the 'rfecv' strategy, "
                "e.g., atom.feature_selection(strategy='rfecv')."
            )

        try:  # Define the y-label for the plot
            ylabel = self.rfecv_.get_params()["scoring"].name
        except AttributeError:
            ylabel = "accuracy" if is_classifier(self.rfecv_.estimator_) else "r2"

        x = range(self.rfecv_.min_features_to_select, self.rfecv_.n_features_in_ + 1)

        # Create star symbol at selected number of features
        sizes = [6] * len(x)
        sizes[self.rfecv_.n_features_ - self.rfecv_.min_features_to_select] = 12
        symbols = ["circle"] * len(x)
        symbols[self.rfecv_.n_features_ - self.rfecv_.min_features_to_select] = "star"

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        mean = self.rfecv_.cv_results_["mean_test_score"]
        std = self.rfecv_.cv_results_["std_test_score"]

        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=mean,
                mode="lines+markers",
                line=dict(width=self.line_width, color=BasePlot._fig.get_elem("rfecv")),
                marker=dict(
                    symbol=symbols,
                    size=sizes,
                    line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                    opacity=1,
                ),
                name=ylabel,
                legendgroup="rfecv",
                showlegend=BasePlot._fig.showlegend("rfecv", legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        # Add error bands
        fig.add_traces(
            [
                go.Scatter(
                    x=tuple(x),
                    y=mean + std,
                    mode="lines",
                    line=dict(width=1, color=BasePlot._fig.get_elem("rfecv")),
                    hovertemplate="%{y}<extra>upper bound</extra>",
                    legendgroup="rfecv",
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                ),
                go.Scatter(
                    x=tuple(x),
                    y=mean - std,
                    mode="lines",
                    line=dict(width=1, color=BasePlot._fig.get_elem("rfecv")),
                    fill="tonexty",
                    fillcolor=f"rgba{BasePlot._fig.get_elem('rfecv')[3:-1]}, 0.2)",
                    hovertemplate="%{y}<extra>lower bound</extra>",
                    legendgroup="rfecv",
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                ),
            ]
        )

        fig.update_layout({"hovermode": "x unified"})

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            xlabel="Number of features",
            ylabel=ylabel,
            xlim=(min(x) - len(x) / 30, max(x) + len(x) / 30),
            ylim=(min(mean) - 3 * max(std), max(mean) + 3 * max(std)),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_rfecv",
            filename=filename,
            display=display,
        )

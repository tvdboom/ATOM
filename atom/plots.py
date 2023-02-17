# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the plotting classes.

"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
from importlib.util import find_spec
from itertools import chain, cycle
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from joblib import Parallel, delayed
from mlflow.tracking import MlflowClient
from nltk.collocations import (
    BigramCollocationFinder, QuadgramCollocationFinder,
    TrigramCollocationFinder,
)
from optuna.importance import FanovaImportanceEvaluator
from optuna.visualization._parallel_coordinate import (
    _get_dims_from_info, _get_parallel_coordinate_info,
)
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
from typeguard import typechecked

from atom.utils import (
    FLOAT, INT, INT_TYPES, PALETTE, SCALAR, SEQUENCE, Model, bk, check_canvas,
    check_dependency, check_predict_proba, composed, crash, divide,
    get_best_score, get_corpus, get_custom_scorer, has_attr, has_task,
    is_binary, is_multioutput, it, lst, plot_from_model, rnd, to_rgb,
)


@dataclass
class Aesthetics:
    """Keeps track of plot aesthetics."""

    palette: SEQUENCE  # Sequence of colors
    title_fontsize: INT  # Fontsize for titles
    label_fontsize: INT  # Fontsize for labels, legend and hoverinfo
    tick_fontsize: INT  # Fontsize for ticks
    line_width: INT  # Width of the line plots
    marker_size: INT  # Size of the markers


class BaseFigure:
    """Base plotly figure.

    The instance stores the position of the current axes in grid,
    as well as the models used for the plot (to track in mlflow).

    Parameters
    ----------
    rows: int, default=1
        Number of subplot rows in the canvas.

    cols: int, default=1
        Number of subplot columns in the canvas.

    horizontal_spacing: float, default=0.05
        Space between subplot rows in normalized plot coordinates.
        The spacing is relative to the figure's size.

    vertical_spacing: float, default=0.07
        Space between subplot cols in normalized plot coordinates.
        The spacing is relative to the figure's size.

    palette: str or sequence, default="Prism"
        Name or color sequence for the palette.

    is_canvas: bool, default=False
        Whether the figure shows multiple plots.

    backend: str, default="plotly"
        Figure's backend. Choose between plotly or matplotlib.

    create_figure: bool, default=True
        Whether to create a new figure.

    """

    def __init__(
        self,
        rows: INT = 1,
        cols: INT = 1,
        horizontal_spacing: FLOAT = 0.05,
        vertical_spacing: FLOAT = 0.07,
        palette: str | SEQUENCE = "Prism",
        is_canvas: bool = False,
        backend: str = "plotly",
        create_figure: bool = True,
    ):
        self.rows = rows
        self.cols = cols
        self.horizontal_spacing = horizontal_spacing
        self.vertical_spacing = vertical_spacing
        if isinstance(palette, str):
            self.palette = cycle(getattr(px.colors.qualitative, palette))
        else:
            # Convert color names or hex to rgb
            self.palette = cycle(map(to_rgb, palette))
        self.is_canvas = is_canvas
        self.backend = backend
        self.create_figure = create_figure

        self.idx = 0  # N-th plot in the canvas
        self.axes = 0  # N-th axis in the canvas
        if self.create_figure:
            if self.backend == "plotly":
                self.figure = go.Figure()
            else:
                self.figure, _ = plt.subplots()

        self.groups = []
        self.style = dict(colors={}, markers={}, dashes={}, shapes={})
        self.markers = cycle(["circle", "x", "diamond", "pentagon", "star", "hexagon"])
        self.dashes = cycle([None, "dashdot", "dash", "dot", "longdash", "longdashdot"])
        self.shapes = cycle(["", "/", "x", "\\", "-", "|", "+", "."])

        self.pos = {}  # Subplot position to use for title
        self.custom_layout = {}  # Layout params specified by user
        self.used_models = []  # Models plotted in this figure

        # Perform parameter checks
        if not 0 < horizontal_spacing < 1:
            raise ValueError(
                "Invalid value for the horizontal_spacing parameter. The "
                f"value must lie between 0 and 1, got {horizontal_spacing}."
            )

        if not 0 < vertical_spacing < 1:
            raise ValueError(
                "Invalid value for the vertical_spacing parameter. The "
                f"value must lie between 0 and 1, got {vertical_spacing}."
            )

    @property
    def grid(self) -> tuple[INT, INT]:
        """Position of the current axes on the grid.

        Returns
        -------
        int
            X-position.

        int
            Y-position.

        """
        return (self.idx - 1) // self.cols + 1, self.idx % self.cols or self.cols

    @property
    def next_subplot(self) -> go.Figure | plt.Figure | None:
        """Increase the subplot index.

        Returns
        -------
        go.Figure, plt.Figure or None
            Current figure. Returns None if `create_figure=False`.

        """
        # Check if there are too many plots in the canvas
        if self.idx >= self.rows * self.cols:
            raise ValueError(
                "Invalid number of plots in the canvas! Increase "
                "the number of rows and cols to add more plots."
            )
        else:
            self.idx += 1

        if self.create_figure:
            return self.figure

    def get_color(self, elem: SCALAR | str | None = None) -> str:
        """Get the next color.

        This method is used to assign the same color to the same
        elements (columns, models, etc...) in a plot.

        Parameters
        ----------
        elem: int, float or str or None
            Element for which to get the color.

        Returns
        -------
        str
            Color code.

        """
        if elem is None:
            return next(self.palette)
        elif elem in self.style["colors"]:
            return self.style["colors"][elem]
        else:
            return self.style["colors"].setdefault(elem, next(self.palette))

    def get_marker(self, elem: SCALAR | str | None = None) -> str:
        """Get the next marker.

        This method is used to assign the same marker to the same
        elements (e.g. distribution) in a plot.

        Parameters
        ----------
        elem: int, float or str or None
            Element for which to get the marker.

        Returns
        -------
        str
            Marker code.

        """
        if elem is None:
            return next(self.markers)
        elif elem in self.style["markers"]:
            return self.style["markers"][elem]
        else:
            return self.style["markers"].setdefault(elem, next(self.markers))

    def get_dashes(self, elem: SCALAR | str | None = None) -> str:
        """Get the next dash style.

        This method is used to assign the same dash style to the same
        elements (e.g. data set) in a plot.

        Parameters
        ----------
        elem: int, float or str or None
            Element for which to get the dash.

        Returns
        -------
        str
            Dash style.

        """
        if elem is None:
            return next(self.dashes)
        elif elem in self.style["dashes"]:
            return self.style["dashes"][elem]
        else:
            return self.style["dashes"].setdefault(elem, next(self.dashes))

    def get_shapes(self, elem: SCALAR | str | None = None) -> str:
        """Get the next shape pattern.

        This method is used to assign the same shape pattern to the
        same elements in a plot.

        Parameters
        ----------
        elem: int, float or str or None
            Element for which to get the shape.

        Returns
        -------
        str
            Pattern shape.

        """
        if elem is None:
            return next(self.shapes)
        elif elem in self.style["shapes"]:
            return self.style["shapes"][elem]
        else:
            return self.style["shapes"].setdefault(elem, next(self.shapes))

    def showlegend(self, name: str, legend: str | dict | None) -> bool:
        """Get whether the trace should be showed in the legend.

        If there's already a trace with the same name, it's not
        necessary to show it in the plot's legend.

        Parameters
        ----------
        name: str
            Name of the trace.

        legend: str, dict or None
            Legend parameter.

        Returns
        -------
        bool
            Whether the trace should be placed in the legend.

        """
        if name in self.groups:
            return False
        else:
            self.groups.append(name)
            return legend is not None

    def get_axes(
        self,
        x: tuple[INT, INT] = (0, 1),
        y: tuple[INT, INT] = (0, 1),
        coloraxis: dict | None = None,
    ) -> tuple[str, str]:
        """Create and update the plot's axes.

        Parameters
        ----------
        x: tuple of int
            Relative x-size of the plot.

        y: tuple of int
            Relative y-size of the plot.

        coloraxis: dict or None
            Properties of the coloraxis to create. None to ignore.

        Returns
        -------
        str
            Name of the x-axis.

        str
            Name of the y-axis.

        """
        self.axes += 1

        # Calculate the distance between subplots
        x_offset = divide(self.horizontal_spacing, (self.cols - 1))
        y_offset = divide(self.vertical_spacing, (self.rows - 1))

        # Calculate the size of the subplot
        x_size = (1 - ((x_offset * 2) * (self.cols - 1))) / self.cols
        y_size = (1 - ((y_offset * 2) * (self.rows - 1))) / self.rows

        # Calculate the size of the axes
        ax_size = (x[1] - x[0]) * x_size
        ay_size = (y[1] - y[0]) * y_size

        # Determine the position for the axes
        x_pos = (self.grid[1] - 1) * (x_size + 2 * x_offset) + x[0] * x_size
        y_pos = (self.rows - self.grid[0]) * (y_size + 2 * y_offset) + y[0] * y_size

        # Store positions for subplot title
        self.pos[str(self.axes)] = (x_pos + ax_size / 2, rnd(y_pos + ay_size))

        # Update the figure with the new axes
        self.figure.update_layout(
            {
                f"xaxis{self.axes}": dict(
                    domain=(x_pos, rnd(x_pos + ax_size)), anchor=f"y{self.axes}"
                ),
                f"yaxis{self.axes}": dict(
                    domain=(y_pos, rnd(y_pos + ay_size)), anchor=f"x{self.axes}"
                ),
            }
        )

        # Place a colorbar right of the axes
        if coloraxis:
            if title := coloraxis.pop("title", None):
                coloraxis["colorbar_title"] = dict(
                    text=title, side="right", font_size=coloraxis.pop("font_size")
                )

            coloraxis["colorbar_x"] = rnd(x_pos + ax_size) + ax_size / 40
            coloraxis["colorbar_xanchor"] = "left"
            coloraxis["colorbar_y"] = y_pos + ay_size / 2
            coloraxis["colorbar_yanchor"] = "middle"
            coloraxis["colorbar_len"] = ay_size * 0.9
            coloraxis["colorbar_thickness"] = ax_size * 30  # Default width in pixels
            self.figure.update_layout(
                {f"coloraxis{coloraxis.pop('axes', self.axes)}": coloraxis}
            )

        xaxis = f"x{self.axes if self.axes > 1 else ''}"
        yaxis = f"y{self.axes if self.axes > 1 else ''}"
        return xaxis, yaxis


class BasePlot:
    """Base class for all plotting methods.

    This base class defines the properties that can be changed
    to customize the plot's aesthetics.

    """

    _fig = None
    _custom_layout = {}
    _aesthetics = Aesthetics(
        palette=list(PALETTE),
        title_fontsize=24,
        label_fontsize=16,
        tick_fontsize=12,
        line_width=2,
        marker_size=8,
    )

    # Properties =================================================== >>

    @property
    def aesthetics(self) -> dict:
        """All plot aesthetic attributes."""
        return self._aesthetics

    @aesthetics.setter
    @typechecked
    def aesthetics(self, value: dict):
        self.palette = value.get("palette", self.palette)
        self.title_fontsize = value.get("title_fontsize", self.title_fontsize)
        self.label_fontsize = value.get("label_fontsize", self.label_fontsize)
        self.tick_fontsize = value.get("tick_fontsize", self.tick_fontsize)
        self.line_width = value.get("line_width", self.line_width)
        self.marker_size = value.get("marker_size", self.marker_size)

    @property
    def palette(self) -> str | SEQUENCE:
        """Color palette.

        Specify one of plotly's [built-in palettes][palette] or create
        a custom one, e.g. `atom.palette = ["red", "green", "blue"]`.

        """
        return self._aesthetics.palette

    @palette.setter
    @typechecked
    def palette(self, value: str | SEQUENCE):
        if isinstance(value, str) and not hasattr(px.colors.qualitative, value):
            raise ValueError(
                f"Invalid value for the palette parameter, got {value}. Choose "
                f"from one of plotly's built-in qualitative color sequences in "
                f"the px.colors.qualitative module or define your own sequence."
            )

        self._aesthetics.palette = value

    @property
    def title_fontsize(self) -> INT:
        """Fontsize for the plot's title."""
        return self._aesthetics.title_fontsize

    @title_fontsize.setter
    @typechecked
    def title_fontsize(self, value: INT):
        if value <= 0:
            raise ValueError(
                "Invalid value for the title_fontsize parameter. "
                f"Value should be >=0, got {value}."
            )

        self._aesthetics.title_fontsize = value

    @property
    def label_fontsize(self) -> INT:
        """Fontsize for the labels, legend and hover information."""
        return self._aesthetics.label_fontsize

    @label_fontsize.setter
    @typechecked
    def label_fontsize(self, value: INT):
        if value <= 0:
            raise ValueError(
                "Invalid value for the label_fontsize parameter. "
                f"Value should be >=0, got {value}."
            )

        self._aesthetics.label_fontsize = value

    @property
    def tick_fontsize(self) -> INT:
        """Fontsize for the ticks along the plot's axes."""
        return self._aesthetics.tick_fontsize

    @tick_fontsize.setter
    @typechecked
    def tick_fontsize(self, value: INT):
        if value <= 0:
            raise ValueError(
                "Invalid value for the tick_fontsize parameter. "
                f"Value should be >=0, got {value}."
            )

        self._aesthetics.tick_fontsize = value

    @property
    def line_width(self) -> INT:
        """Width of the line plots."""
        return self._aesthetics.line_width

    @line_width.setter
    @typechecked
    def line_width(self, value: INT):
        if value <= 0:
            raise ValueError(
                "Invalid value for the line_width parameter. "
                f"Value should be >=0, got {value}."
            )

        self._aesthetics.line_width = value

    @property
    def marker_size(self) -> INT:
        """Size of the markers."""
        return self._aesthetics.marker_size

    @marker_size.setter
    @typechecked
    def marker_size(self, value: INT):
        if value <= 0:
            raise ValueError(
                "Invalid value for the marker_size parameter. "
                f"Value should be >=0, got {value}."
            )

        self._aesthetics.marker_size = value

    # Methods ====================================================== >>

    @staticmethod
    def _get_show(show: INT | None, model: Model | list[Model]) -> INT:
        """Check and return the number of features to show.

        Parameters
        ----------
        show: int or None
            Number of features to show. If None, select all (max 200).

        model: Model or list
            Models from which to get the features.

        Returns
        -------
        int
            Number of features to show.

        """
        max_fxs = max(m.n_features for m in lst(model))
        if show is None or show > max_fxs:
            # Limit max features shown to avoid maximum figsize error
            show = min(200, max_fxs)
        elif show < 1:
            raise ValueError(
                f"Invalid value for the show parameter. Value should be >0, got {show}."
            )

        return show

    @staticmethod
    def _get_hyperparams(
        params: str | slice | SEQUENCE | None,
        model: Model,
    ) -> list[str]:
        """Check and return a model's hyperparameters.

        Parameters
        ----------
        params: str, slice, sequence or None
            Hyperparameters to get. Use a sequence or add `+` between
            options to select more than one. If None, all the model's
            hyperparameters are selcted.

        model: Model
            Get the params from this model.

        Returns
        -------
        list of str
            Selected hyperparameters.

        """
        if params is None:
            hyperparameters = list(model._ht["distributions"])
        elif isinstance(params, slice):
            hyperparameters = list(model._ht["distributions"])[params]
        else:
            hyperparameters = []
            for param in lst(params):
                if isinstance(param, INT_TYPES):
                    hyperparameters.append(list(model._ht["distributions"])[param])
                elif isinstance(param, str):
                    for p in param.split("+"):
                        if p not in model._ht["distributions"]:
                            raise ValueError(
                                "Invalid value for the params parameter. "
                                f"Hyperparameter {p} was not used during the "
                                f"optimization of model {model.name}."
                            )
                        else:
                            hyperparameters.append(p)

        if not hyperparameters:
            raise ValueError(f"Didn't find any hyperparameters for model {model.name}.")

        return hyperparameters

    def _get_metric(
        self,
        metric: INT | str | SEQUENCE,
        max_one: bool,
    ) -> INT | str | list[INT]:
        """Check and return the provided metric index.

        Parameters
        ----------
        metric: int, str, sequence or None
            Metric to retrieve. If None, all metrics are returned.

        max_one: bool
            Whether one or multiple metrics are allowed.

        Returns
        -------
        int or list
            Position index of the metric. If `max_one=False`, returns
            a list of metric positions.

        """
        if metric is None:
            return list(range(len(self._metric)))
        else:
            inc = []
            for met in lst(metric):
                if isinstance(met, INT_TYPES):
                    if 0 <= met < len(self._metric):
                        inc.append(met)
                    else:
                        raise ValueError(
                            f"Invalid value for the metric parameter. Value {met} is out "
                            f"of range for a pipeline with {len(self._metric)} metrics."
                        )
                elif isinstance(met, str):
                    met = met.lower()
                    for m in met.split("+"):
                        if m in ("time_ht", "time_fit", "time_bootstrap", "time"):
                            inc.append(m)
                        elif (name := get_custom_scorer(m).name) in self.metric:
                            inc.append(self._metric.index(name))
                        else:
                            raise ValueError(
                                "Invalid value for the metric parameter. The "
                                f"{name} metric wasn't used to fit the models."
                            )

        if len(inc) > 1 and max_one:
            raise ValueError(
                "Invalid value for the metric parameter. "
                f"Only one metric is allowed, got {inc}."
            )

        return inc[0] if max_one else inc

    def _get_set(
        self,
        dataset: str | SEQUENCE,
        max_one: bool,
        allow_holdout: bool = True,
    ) -> str | list[str]:
        """Check and return the provided data set.

        Parameters
        ----------
        dataset: str or sequence
            Name(s) of the data set to retrieve.

        max_one: bool
            Whether one or multiple data sets are allowed. If True, return
            the data set instead of a list.

        allow_holdout: bool, default=True
            Whether to allow the retrieval of the holdout set.

        Returns
        -------
        str or list
            Selected data set(s).

        """
        for ds in (dataset := "+".join(lst(dataset)).lower().split("+")):
            if ds == "holdout":
                if allow_holdout:
                    if self.holdout is None:
                        raise ValueError(
                            "Invalid value for the dataset parameter. No holdout "
                            "data set was specified when initializing the instance."
                        )
                else:
                    raise ValueError(
                        "Invalid value for the dataset parameter, got "
                        f"{ds}. Choose from: train or test."
                    )
            elif ds not in ("train", "test"):
                raise ValueError(
                    "Invalid value for the dataset parameter, "
                    f"got {ds}. Choose from: train, test or holdout."
                )

        if max_one and len(dataset) > 1:
            raise ValueError(
                "Invalid value for the dataset parameter, got "
                f"{dataset}. Only one data set is allowed."
            )

        return dataset[0] if max_one else dataset

    def _get_figure(self, **kwargs) -> go.Figure | plt.Figure:
        """Return existing figure if in canvas, else a new figure.

        Every time this method is called from a canvas, the plot
        index is raised by one to keep track in which subplot the
        BaseFigure is at.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for BaseFigure.

        Returns
        -------
        [go.Figure][] or [plt.Figure][]
            Existing figure or newly created.

        """
        if self._fig and self._fig.is_canvas:
            return self._fig.next_subplot
        else:
            self._fig = BaseFigure(palette=self.palette, **kwargs)
            return self._fig.next_subplot

    def _draw_line(
        self,
        parent: str,
        child: str | None = None,
        legend: str | dict = None,
        **kwargs,
    ) -> go.Scatter:
        """Draw a line.

        Unify the style to draw a line, where parent and child
        (e.g. model - data set or column - distribution) keep the
        same style (color or dash). A legendgroup title is only added
        when there is a child element.

        Parameters
        ----------
        parent: str
            Name of the model.

        child: str or None, default=None
            Data set which is plotted.

        legend: str, dict or None
            Legend argument provided by the user.

        **kwargs
            Additional keyword arguments for the trace.

        Returns
        -------
        go.Scatter
            New trace to add to figure.

        """
        legendgrouptitle = dict(text=parent, font_size=self.label_fontsize)
        hover = f"(%{{x}}, %{{y}})<extra>{parent}{' - ' + child if child else ''}</extra>"
        return go.Scatter(
            line=dict(
                width=self.line_width,
                color=self._fig.get_color(parent),
                dash=self._fig.get_dashes(child) if child else None,
            ),
            marker=dict(
                symbol=self._fig.get_marker(child),
                size=self.marker_size,
                color=self._fig.get_color(parent),
                line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
            ),
            hovertemplate=kwargs.pop("hovertemplate", hover),
            name=kwargs.pop("name", child if child else parent),
            legendgroup=kwargs.pop("legendgroup", parent),
            legendgrouptitle=legendgrouptitle if child else None,
            showlegend=self._fig.showlegend(f"{parent}-{child}", legend),
            **kwargs,
        )

    def _draw_straight_line(self, y: SCALAR | str, xaxis: str, yaxis: str):
        """Draw a line across the axis.

        The line can be either horizontal or diagonal. The line should
        be used as reference. It's not added to the legend and doesn't
        show any information on hover.

        Parameters
        ----------
        y: int, float or str, default = "diagonal"
            Coordinates on the y-axis. If a value, draw a horizontal line
            at that value. If "diagonal", draw a diagonal line from x.

        xaxis: str
            Name of the x-axis to draw in.

        yaxis: str
            Name of the y-axis to draw in.

        """
        self._fig.figure.add_shape(
            type="line",
            x0=0,
            x1=1,
            xref=f"{xaxis} domain",
            y0=0 if y == "diagonal" else y,
            y1=1 if y == "diagonal" else y,
            yref=f"{yaxis} domain" if y == "diagonal" else yaxis,
            line=dict(width=1, color="black", dash="dash"),
            opacity=0.6,
            layer="below",
        )

    def _plot(
        self,
        fig: go.Figure | plt.Figure | None = None,
        ax: plt.Axes | tuple[str, str] | None = None,
        **kwargs,
    ) -> go.Figure | plt.Figure | None:
        """Make the plot.

        Customize the axes to the default layout and plot the figure
        if it's not part of a canvas.

        Parameters
        ----------
        fig: go.Figure, plt.Figure or None
            Current figure. If None, use `plt.gcf()`.

        ax: plt.Axes, tuple or None, default=None
            Axis object or names of the axes to update. If None, ignore
            their update.

        **kwargs
            Keyword arguments containing the figure's parameters.

            - title: Name of the title or custom configuration.
            - legend: Whether to show the legend or custom configuration.
            - xlabel: Label for the x-axis.
            - ylabel: Label for the y-axis.
            - xlim: Limits for the x-axis.
            - ylim: Limits for the y-axis.
            - figsize: Size of the figure.
            - filename: Name of the saved file.
            - plotname: Name of the plot.
            - display: Whether to show the plot. If None, return the figure.

        Returns
        -------
        plt.Figure, go.Figure or None
            Created figure. Only returned if `display=None`.

        """
        # Set name with which to save the file
        if kwargs.get("filename"):
            if kwargs["filename"].endswith("auto"):
                name = kwargs["filename"].replace("auto", kwargs["plotname"])
            else:
                name = kwargs["filename"]
        else:
            name = kwargs.get("plotname")

        fig = fig or self._fig.figure
        if self._fig.backend == "plotly":
            if ax:
                fig.update_layout(
                    {
                        f"{ax[0]}_title": dict(
                            text=kwargs.get("xlabel"), font_size=self.label_fontsize
                        ),
                        f"{ax[1]}_title": dict(
                            text=kwargs.get("ylabel"), font_size=self.label_fontsize
                        ),
                        f"{ax[0]}_range": kwargs.get("xlim"),
                        f"{ax[1]}_range": kwargs.get("ylim"),
                        f"{ax[0]}_automargin": True,
                        f"{ax[1]}_automargin": True,
                    }
                )

                if self._fig.is_canvas and (title := kwargs.get("title")):
                    # Add a subtitle to a plot in the canvas
                    default_title = {
                        "x": self._fig.pos[ax[0][5:] or "1"][0],
                        "y": self._fig.pos[ax[0][5:] or "1"][1] + 0.005,
                        "xref": "paper",
                        "yref": "paper",
                        "xanchor": "center",
                        "yanchor": "bottom",
                        "showarrow": False,
                        "font_size": self.title_fontsize - 4,
                    }

                    if isinstance(title, dict):
                        title = {**default_title, **title}
                    else:
                        title = {"text": title, **default_title}

                    fig.update_layout(dict(annotations=fig.layout.annotations + (title,)))

            if not self._fig.is_canvas and kwargs.get("plotname"):
                default_title = dict(
                    x=0.5,
                    y=1,
                    pad=dict(t=15, b=15),
                    xanchor="center",
                    yanchor="top",
                    xref="paper",
                    font_size=self.title_fontsize,
                )
                if isinstance(title := kwargs.get("title"), dict):
                    title = {**default_title, **title}
                else:
                    title = {"text": title, **default_title}

                default_legend = dict(
                    traceorder="grouped",
                    groupclick=kwargs.get("groupclick", "toggleitem"),
                    font_size=self.label_fontsize,
                    bgcolor="rgba(255, 255, 255, 0.5)",
                )
                if isinstance(legend := kwargs.get("legend"), str):
                    position = {}
                    legend = legend.lower()
                    if legend == "upper left":
                        position = dict(x=0.01, y=0.99, xanchor="left", yanchor="top")
                    elif legend == "lower left":
                        position = dict(x=0.01, y=0.01, xanchor="left", yanchor="bottom")
                    elif legend == "upper right":
                        position = dict(x=0.99, y=0.99, xanchor="right", yanchor="top")
                    elif legend == "lower right":
                        position = dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom")
                    elif legend == "upper center":
                        position = dict(x=0.5, y=0.99, xanchor="center", yanchor="top")
                    elif legend == "lower center":
                        position = dict(x=0.5, y=0.01, xanchor="center", yanchor="bottom")
                    elif legend == "center left":
                        position = dict(x=0.01, y=0.5, xanchor="left", yanchor="middle")
                    elif legend == "center right":
                        position = dict(x=0.99, y=0.5, xanchor="right", yanchor="middle")
                    elif legend == "center":
                        position = dict(x=0.5, y=0.5, xanchor="center", yanchor="middle")
                    elif legend != "out":
                        raise ValueError(
                            "Invalid value for the legend parameter. Got unknown "
                            f"position: {legend}. Choose from: upper left, upper "
                            "right, lower left, lower right, upper center, lower "
                            "center, center left, center right, center, out."
                        )
                    legend = {**default_legend, **position}
                elif isinstance(legend, dict):
                    legend = {**default_legend, **legend}

                # Update layout with predefined settings
                space1 = self.title_fontsize if title.get("text") else 10
                space2 = self.title_fontsize * int(bool(fig.layout.annotations))
                fig.update_layout(
                    title=title,
                    legend=legend,
                    showlegend=bool(kwargs.get("legend")),
                    hoverlabel=dict(font_size=self.label_fontsize),
                    font_size=self.tick_fontsize,
                    margin=dict(l=50, b=50, r=0, t=25 + space1 + space2, pad=0),
                    width=kwargs["figsize"][0],
                    height=kwargs["figsize"][1],
                )

                # Update layout with custom settings
                fig.update_layout(**self._custom_layout)

                if kwargs.get("filename"):
                    if "." not in name or name.endswith(".html"):
                        fig.write_html(name if "." in name else name + ".html")
                    else:
                        fig.write_image(name)

                # Log plot to mlflow run of every model visualized
                if getattr(self, "experiment", None) and self.log_plots:
                    for m in set(self._fig.used_models):
                        MlflowClient().log_figure(
                            run_id=m._run.info.run_id,
                            figure=fig,
                            artifact_file=name if "." in name else f"{name}.html",
                        )

                if kwargs.get("display") is True:
                    fig.show()
                elif kwargs.get("display") is None:
                    return fig

        else:
            if kwargs.get("title"):
                ax.set_title(kwargs.get("title"), fontsize=self.title_fontsize, pad=20)
            if kwargs.get("xlabel"):
                ax.set_xlabel(kwargs["xlabel"], fontsize=self.label_fontsize, labelpad=12)
            if kwargs.get("ylabel"):
                ax.set_ylabel(kwargs["ylabel"], fontsize=self.label_fontsize, labelpad=12)
            if ax is not None:
                ax.tick_params(axis="both", labelsize=self.tick_fontsize)

            if kwargs.get("figsize"):
                # Convert from pixels to inches
                fig.set_size_inches(
                    kwargs["figsize"][0] // fig.get_dpi(),
                    kwargs["figsize"][1] // fig.get_dpi(),
                )
            if kwargs.get("filename"):
                fig.savefig(name)

            # Log plot to mlflow run of every model visualized
            if self.experiment and self.log_plots:
                for m in set(self._fig.used_models):
                    MlflowClient().log_figure(
                        run_id=m._run.info.run_id,
                        figure=fig,
                        artifact_file=name if "." in name else f"{name}.png",
                    )

            plt.tight_layout()
            plt.show() if kwargs.get("display") else plt.close()
            if kwargs.get("display") is None:
                return fig

    @composed(contextmanager, crash, typechecked)
    def canvas(
        self,
        rows: INT = 1,
        cols: INT = 2,
        *,
        horizontal_spacing: FLOAT = 0.05,
        vertical_spacing: FLOAT = 0.07,
        title: str | dict | None = None,
        legend: str | dict | None = "out",
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool = True,
    ):
        """Create a figure with multiple plots.

        This `@contextmanager` allows you to draw many plots in one
        figure. The default option is to add two plots side by side.
        See the [user guide][canvas] for an example.

        Parameters
        ----------
        rows: int, default=1
            Number of plots in length.

        cols: int, default=2
            Number of plots in width.

        horizontal_spacing: float, default=0.05
            Space between subplot rows in normalized plot coordinates.
            The spacing is relative to the figure's size.

        vertical_spacing: float, default=0.07
            Space between subplot cols in normalized plot coordinates.
            The spacing is relative to the figure's size.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: bool, str or dict, default="out"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of plots in the canvas.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool, default=True
            Whether to render the plot.

        Yields
        ------
        [go.Figure][]
            Plot object.

        """
        self._fig = BaseFigure(
            rows=rows,
            cols=cols,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            palette=self.palette,
            is_canvas=True,
        )

        try:
            yield self._fig.figure
        finally:
            self._fig.is_canvas = False  # Close the canvas
            self._plot(
                groupclick="togglegroup",
                title=title,
                legend=legend,
                figsize=figsize or (550 + 350 * cols, 200 + 400 * rows),
                plotname="canvas",
                filename=filename,
                display=display,
            )

    def reset_aesthetics(self):
        """Reset the plot [aesthetics][] to their default values."""
        self._custom_layout = {}
        self._aesthetics = Aesthetics(
            palette=PALETTE,
            title_fontsize=24,
            label_fontsize=16,
            tick_fontsize=12,
            line_width=2,
            marker_size=8,
        )

    def update_layout(
        self,
        dict1: dict | None = None,
        overwrite: bool = False,
        **kwargs,
    ):
        """Update the properties of the plot's layout.

        This recursively updates the structure of the original layout
        with the values in the input dict / keyword arguments.

        Parameters
        ----------
        dict1: dict or None, default=None
            Dictionary of properties to be updated.

        overwrite: bool, default=False
            If True, overwrite existing properties. If False, apply
            updates to existing properties recursively, preserving
            existing properties that are not specified in the update
            operation.

        **kwargs
            Keyword/value pair of properties to be updated.

        """
        self._custom_layout = dict(dict1=dict1, overwrite=overwrite, **kwargs)


class FeatureSelectorPlot(BasePlot):
    """Feature selection plots.

    These plots are accessible from atom or from the FeatureSelector
    class when the appropriate feature selection strategy is used.

    """

    @available_if(has_attr("pca"))
    @composed(crash, typechecked)
    def plot_components(
        self,
        show: INT | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the explained variance ratio per component.

        Kept components are colored and discarted components are
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

        filename: str or None, default=None
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
        atom.plots:FeatureSelectorPlot.plot_pca
        atom.plots:FeatureSelectorPlot.plot_rfecv

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.feature_selection("pca", n_features=5)
        >>> atom.plot_components(show=10)

        ```

        :: insert:
            url: /img/plots/plot_components.html

        """
        if show is None or show > self.pca.components_.shape[0]:
            # Limit max features shown to avoid maximum figsize error
            show = min(200, self.pca.components_.shape[0])
        elif show < 1:
            raise ValueError(
                "Invalid value for the show parameter. "
                f"Value should be >0, got {show}."
            )

        # Get the variance ratio per component
        variance = np.array(self.pca.explained_variance_ratio_)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        # Create color scheme: first normal and then fully transparent
        color = self._fig.get_color("components")
        opacity = [0.2] * self.pca._comps + [0] * (len(variance) - self.pca._comps)

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
                name=f"Variance retained: {variance[:self.pca._comps].sum():.3f}",
                legendgroup="components",
                showlegend=self._fig.showlegend("components", legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        fig.update_layout({f"yaxis{yaxis[1:]}": dict(categoryorder="total ascending")})

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Explained variance ratio",
            ylim=(len(variance) - show - 0.5, len(variance) - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_components",
            filename=filename,
            display=display,
        )

    @available_if(has_attr("pca"))
    @composed(crash, typechecked)
    def plot_pca(
        self,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
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

        filename: str or None, default=None
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
        atom.plots:FeatureSelectorPlot.plot_components
        atom.plots:FeatureSelectorPlot.plot_rfecv

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.feature_selection("pca", n_features=5)
        >>> atom.plot_pca()

        ```

        :: insert:
            url: /img/plots/plot_pca.html

        """
        # Create star symbol at selected number of components
        symbols = ["circle"] * self.pca.n_features_in_
        symbols[self.pca._comps - 1] = "star"
        sizes = [self.marker_size] * self.pca.n_features_in_
        sizes[self.pca._comps - 1] = self.marker_size * 1.5

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        fig.add_trace(
            go.Scatter(
                x=tuple(range(1, self.pca.n_features_in_ + 1)),
                y=np.cumsum(self.pca.explained_variance_ratio_),
                mode="lines+markers",
                line=dict(width=self.line_width, color=self._fig.get_color("pca")),
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

        margin = self.pca.n_features_in_ / 30
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="First N principal components",
            ylabel="Cumulative variance ratio",
            xlim=(1 - margin, self.pca.n_features_in_ - 1 + margin),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_pca",
            filename=filename,
            display=display,
        )

    @available_if(has_attr("rfecv"))
    @composed(crash, typechecked)
    def plot_rfecv(
        self,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
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

        filename: str or None, default=None
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
        atom.plots:FeatureSelectorPlot.plot_components
        atom.plots:FeatureSelectorPlot.plot_pca

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.feature_selection("rfecv", solver="Tree")
        >>> atom.plot_rfecv()

        ```

        :: insert:
            url: /img/plots/plot_rfecv.html

        """
        try:  # Define the y-label for the plot
            ylabel = self.rfecv.get_params()["scoring"].name
        except AttributeError:
            ylabel = "accuracy" if self.goal.startswith("class") else "r2"

        x = range(self.rfecv.min_features_to_select, self.rfecv.n_features_in_ + 1)

        # Create star symbol at selected number of features
        sizes = [6] * len(x)
        sizes[self.rfecv.n_features_ - self.rfecv.min_features_to_select] = 12
        symbols = ["circle"] * len(x)
        symbols[self.rfecv.n_features_ - self.rfecv.min_features_to_select] = "star"

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        mean = self.rfecv.cv_results_["mean_test_score"]
        std = self.rfecv.cv_results_["std_test_score"]

        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=mean,
                mode="lines+markers",
                line=dict(width=self.line_width, color=self._fig.get_color("rfecv")),
                marker=dict(
                    symbol=symbols,
                    size=sizes,
                    line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                    opacity=1,
                ),
                name=ylabel,
                legendgroup="rfecv",
                showlegend=self._fig.showlegend("rfecv", legend),
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
                    line=dict(width=1, color=self._fig.get_color("rfecv")),
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
                    line=dict(width=1, color=self._fig.get_color("rfecv")),
                    fill="tonexty",
                    fillcolor=f"rgba{self._fig.get_color('rfecv')[3:-1]}, 0.2)",
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


class DataPlot(BasePlot):
    """Data plots.

    Plots used for understanding and interpretation of the dataset.
    They are only accessible from atom since. The other runners should
    be used for model training only, not for data manipulation.

    """

    @composed(crash, typechecked)
    def plot_correlation(
        self,
        columns: slice | SEQUENCE | None = None,
        method: str = "pearson",
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] = (800, 700),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot a correlation matrix.

        Displays a heatmap showing the correlation between columns in
        the dataset. The colors red, blue and white stand for positive,
        negative, and no correlation respectively.

        Parameters
        ----------
        columns: slice, sequence or None, default=None
            Columns to plot. If None, plot all columns in the dataset.
            Selected categorical columns are ignored.

        method: str, default="pearson"
            Method of correlation. Choose from: pearson, kendall or
            spearman.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple, default=(800, 700)
            Figure's size in pixels, format as (x, y).

        filename: str or None, default=None
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
        atom.plots:DataPlot.plot_qq
        atom.plots:DataPlot.plot_relationships

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.plot_correlation()

        ```

        :: insert:
            url: /img/plots/plot_correlation.html

        """
        columns = self.branch._get_columns(columns, only_numerical=True)
        if method.lower() not in ("pearson", "kendall", "spearman"):
            raise ValueError(
                f"Invalid value for the method parameter, got {method}. "
                "Choose from: pearson, kendall or spearman."
            )

        # Compute the correlation matrix
        corr = self.dataset[columns].corr(method=method.lower())

        # Generate a mask for the lower triangle
        # k=1 means keep outermost diagonal line
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes(
            x=(0, 0.87),
            coloraxis=dict(
                colorscale="rdbu_r",
                cmin=-1,
                cmax=1,
                title=f"{method.lower()} correlation",
                font_size=self.label_fontsize,
            ),
        )

        fig.add_trace(
            go.Heatmap(
                z=corr.mask(mask),
                x=columns,
                y=columns,
                coloraxis=f"coloraxis{xaxis[1:]}",
                hovertemplate="x:%{x}<br>y:%{y}<br>z:%{z}<extra></extra>",
                hoverongaps=False,
                showlegend=False,
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        fig.update_layout(
            {
                "template": "plotly_white",
                f"yaxis{yaxis[1:]}_autorange": "reversed",
                f"xaxis{xaxis[1:]}_showgrid": False,
                f"yaxis{yaxis[1:]}_showgrid": False,
            }
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_correlation",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_distribution(
        self,
        columns: INT | str | slice | SEQUENCE = 0,
        distributions: str | SEQUENCE | None = None,
        show: INT | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper right",
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot column distributions.

        - For numerical columns, plot the probability density
          distribution. Additionally, it's possible to plot any of
          `scipy.stats` distributions fitted to the column.
        - For categorical columns, plot the class distribution.
          Only one categorical column can be plotted at the same time.

        !!! tip
            Use atom's [distribution][atomclassifier-distribution]
            method to check which distribution fits the column best.

        Parameters
        ----------
        columns: int, str, slice or sequence, default=0
            Columns to plot. I's only possible to plot one categorical
            column. If more than one categorical columns are selected,
            all categorical columns are ignored.

        distributions: str, sequence or None, default=None
            Names of the `scipy.stats` distributions to fit to the
            columns. If None, a [Gaussian kde distribution][kde] is
            showed. Only for numerical columns.

        show: int or None, default=None
            Number of classes (ordered by number of occurrences) to
            show in the plot. If None, it shows all classes. Only for
            categorical columns.

        title: str, dict or None, default=None
            Title for the plot.

            - If None: No title is shown.
            - If str: Text for the title.
            - If dict: [title configuration][parameters].

        legend: str, dict or None, default="upper right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the plot's type.

        filename: str or None, default=None
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
        atom.plots:DataPlot.plot_correlation
        atom.plots:DataPlot.plot_qq
        atom.plots:DataPlot.plot_relationships

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> # Add a categorical feature
        >>> animals = ["cat", "dog", "bird", "lion", "zebra"]
        >>> probabilities = [0.001, 0.1, 0.2, 0.3, 0.399]
        >>> X["animals"] = np.random.choice(animals, size=len(X), p=probabilities)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.plot_distribution(columns=[0, 1])

        ```

        :: insert:
            url: /img/plots/plot_distribution_1.html

        ```pycon
        >>> atom.plot_distribution(columns=0, distributions=["norm", "invgauss"])

        ```

        :: insert:
            url: /img/plots/plot_distribution_2.html

        ```pycon
        >>> atom.plot_distribution(columns="animals", legend="lower right")

        ```

        :: insert:
            url: /img/plots/plot_distribution_3.html

        """
        columns = self.branch._get_columns(columns)
        cat_columns = list(self.dataset.select_dtypes(exclude="number").columns)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        if len(columns) == 1 and columns[0] in cat_columns:
            series = self.dataset[columns[0]].value_counts(ascending=True)

            if show is None or show > len(series):
                show = len(series)
            elif show < 1:
                raise ValueError(
                    "Invalid value for the show parameter."
                    f"Value should be >0, got {show}."
                )

            color = self._fig.get_color()
            fig.add_trace(
                go.Bar(
                    x=series,
                    y=series.index,
                    orientation="h",
                    marker=dict(
                        color=f"rgba({color[4:-1]}, 0.2)",
                        line=dict(width=2, color=color),
                    ),
                    hovertemplate="%{x}<extra></extra>",
                    name=f"{columns[0]}: {len(series)} classes",
                    showlegend=self._fig.showlegend("dist", legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

            return self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel="Counts",
                ylim=(len(series) - show - 0.5, len(series) - 0.5),
                title=title,
                legend=legend,
                figsize=figsize or (900, 400 + show * 50),
                plotname="plot_distribution",
                filename=filename,
                display=display,
            )

        else:
            for col in [c for c in columns if c not in cat_columns]:
                fig.add_trace(
                    go.Histogram(
                        x=self.dataset[col],
                        histnorm="probability density",
                        marker=dict(
                            color=f"rgba({self._fig.get_color(col)[4:-1]}, 0.2)",
                            line=dict(width=2, color=self._fig.get_color(col)),
                        ),
                        nbinsx=40,
                        name="dist",
                        legendgroup=col,
                        legendgrouptitle=dict(text=col, font_size=self.label_fontsize),
                        showlegend=self._fig.showlegend(f"{col}-dist", legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                x = np.linspace(self.dataset[col].min(), self.dataset[col].max(), 200)

                # Drop missing values for compatibility with scipy.stats
                missing = self.missing + [np.inf, -np.inf]
                values = self.dataset[col].replace(missing, np.NaN).dropna()

                if distributions:
                    # Get a line for each distribution
                    for j, dist in enumerate(lst(distributions)):
                        params = getattr(stats, dist).fit(values)

                        fig.add_trace(
                            self._draw_line(
                                x=x,
                                y=getattr(stats, dist).pdf(x, *params),
                                parent=col,
                                child=dist,
                                legend=legend,
                                xaxis=xaxis,
                                yaxis=yaxis,
                            )
                        )
                else:
                    # If no distributions specified, draw Gaussian kde
                    fig.add_trace(
                        self._draw_line(
                            x=x,
                            y=stats.gaussian_kde(values)(x),
                            parent=col,
                            child="kde",
                            legend=legend,
                            xaxis=xaxis,
                            yaxis=yaxis,
                        )
                    )

            fig.update_layout(dict(barmode="overlay"))

            return self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel="Values",
                ylabel="Probability density",
                title=title,
                legend=legend,
                figsize=figsize or (900, 600),
                plotname="plot_distribution",
                filename=filename,
                display=display,
            )

    @composed(crash, typechecked)
    def plot_ngrams(
        self,
        ngram: INT | str = "bigram",
        index: INT | str | slice | SEQUENCE | None = None,
        show: INT = 10,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot n-gram frequencies.

        The text for the plot is extracted from the column named
        `corpus`. If there is no column with that name, an exception
        is raised. If the documents are not tokenized, the words are
        separated by spaces.

        !!! tip
            Use atom's [tokenize][atomclassifier-tokenize] method to
            separate the words creating n-grams based on their frequency
            in the corpus.

        Parameters
        ----------
        ngram: str or int, default="bigram"
            Number of contiguous words to search for (size of n-gram).
            Choose from: words (1), bigrams (2), trigrams (3),
            quadgrams (4).

        index: int, str, slice, sequence or None, default=None
            Documents in the corpus to include in the search. If None,
            it selects all documents in the dataset.

        show: int, default=10
            Number of n-grams (ordered by number of occurrences) to
            show in the plot.

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
            adapts the size to the number of n-grams shown.

        filename: str or None, default=None
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
        atom.plots:DataPlot.plot_wordcloud

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import fetch_20newsgroups

        >>> X, y = fetch_20newsgroups(
        ...     return_X_y=True,
        ...     categories=[
        ...         'alt.atheism',
        ...         'sci.med',
        ...         'comp.windows.x',
        ...     ],
        ...     shuffle=True,
        ...     random_state=1,
        ... )
        >>> X = np.array(X).reshape(-1, 1)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.textclean()
        >>> atom.textnormalize()
        >>> atom.plot_ngrams()

        ```

        :: insert:
            url: /img/plots/plot_ngrams.html

        """

        def get_text(column: bk.Series) -> bk.Series:
            """Get the complete corpus as sequence of tokens.

            Parameters
            ----------
            column: series
                Column containing the corpus.

            Returns
            -------
            series
                Corpus of tokens.

            """
            if isinstance(column.iat[0], str):
                return column.apply(lambda row: row.split())
            else:
                return column

        corpus = get_corpus(self.X)
        rows = self.dataset.loc[self.branch._get_rows(index, return_test=False)]

        if str(ngram).lower() in ("1", "word", "words"):
            ngram = "words"
            series = pd.Series(
                [word for row in get_text(rows[corpus]) for word in row]
            ).value_counts(ascending=True)
        else:
            if str(ngram).lower() in ("2", "bigram", "bigrams"):
                ngram, finder = "bigrams", BigramCollocationFinder
            elif str(ngram).lower() in ("3", "trigram", "trigrams"):
                ngram, finder = "trigrams", TrigramCollocationFinder
            elif str(ngram).lower() in ("4", "quadgram", "quadgrams"):
                ngram, finder = "quadgrams", QuadgramCollocationFinder
            else:
                raise ValueError(
                    f"Invalid value for the ngram parameter, got {ngram}. "
                    "Choose from: words, bigram, trigram, quadgram."
                )

            ngram_fd = finder.from_documents(get_text(rows[corpus])).ngram_fd
            series = pd.Series(
                data=[x[1] for x in ngram_fd.items()],
                index=[" ".join(x[0]) for x in ngram_fd.items()],
            ).sort_values(ascending=True)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        fig.add_trace(
            go.Bar(
                x=(data := series[-show:]),
                y=data.index,
                orientation="h",
                marker=dict(
                    color=f"rgba({self._fig.get_color(ngram)[4:-1]}, 0.2)",
                    line=dict(width=2, color=self._fig.get_color(ngram)),
                ),
                hovertemplate="%{x}<extra></extra>",
                name=f"Total {ngram}: {len(series)}",
                legendgroup=ngram,
                showlegend=self._fig.showlegend(ngram, legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Counts",
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_ngrams",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_qq(
        self,
        columns: INT | str | slice | SEQUENCE = 0,
        distributions: str | SEQUENCE = "norm",
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot a quantile-quantile plot.

        Columns are distinguished by color and the distributions are
        distinguished by marker type. Missing values are ignored.

        Parameters
        ----------
        columns: int, str, slice or sequence, default=0
            Columns to plot. Selected categorical columns are ignored.

        distributions: str or sequence, default="norm"
            Names of the `scipy.stats` distributions to fit to the
            columns.

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

        filename: str or None, default=None
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
        atom.plots:DataPlot.plot_correlation
        atom.plots:DataPlot.plot_distribution
        atom.plots:DataPlot.plot_relationships

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> atom = ATOMClassifier(X, y)
        >>> atom.plot_qq(columns=[5, 6])

        ```

        :: insert:
            url: /img/plots/plot_qq_1.html

        ```pycon
        >>> atom.plot_qq(columns=0, distributions=["norm", "invgauss", "triang"])

        ```

        :: insert:
            url: /img/plots/plot_qq_2.html

        """
        columns = self.branch._get_columns(columns)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        percentiles = np.linspace(0, 100, 101)
        for col in columns:
            # Drop missing values for compatibility with scipy.stats
            missing = self.missing + [np.inf, -np.inf]
            values = self.dataset[col].replace(missing, np.NaN).dropna()

            for dist in lst(distributions):
                stat = getattr(stats, dist)
                params = stat.fit(values)
                samples = stat.rvs(*params, size=101, random_state=self.random_state)

                fig.add_trace(
                    self._draw_line(
                        x=np.percentile(samples, percentiles),
                        y=np.percentile(values, percentiles),
                        mode="markers",
                        parent=col,
                        child=dist,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._draw_straight_line(y="diagonal", xaxis=xaxis, yaxis=yaxis)

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Theoretical quantiles",
            ylabel="Observed quantiles",
            title=title,
            legend=legend,
            figsize=figsize or (900, 600),
            plotname="plot_qq",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_relationships(
        self,
        columns: slice | SEQUENCE = (0, 1, 2),
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] = (900, 900),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot pairwise relationships in a dataset.

        Creates a grid of axes such that each numerical column appears
        once on the x-axes and once on the y-axes. The bottom triangle
        contains scatter plots (max 250 random samples), the diagonal
        plots contain column distributions, and the upper triangle
        contains contour histograms for all samples in the columns.

        Parameters
        ----------
        columns: slice or sequence, default=(0, 1, 2)
            Columns to plot. Selected categorical columns are ignored.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple, default=(900, 900)
            Figure's size in pixels, format as (x, y).

        filename: str or None, default=None
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
        atom.plots:DataPlot.plot_correlation
        atom.plots:DataPlot.plot_distribution
        atom.plots:DataPlot.plot_qq

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.plot_relationships(columns=[0, 4, 5])

        ```

        :: insert:
            url: /img/plots/plot_relationships.html

        """
        columns = self.branch._get_columns(columns, only_numerical=True)

        # Use max 250 samples to not clutter the plot
        sample = lambda col: self.dataset[col].sample(
            n=min(len(self.dataset), 250), random_state=self.random_state
        )

        fig = self._get_figure()
        color = self._fig.get_color()
        for i in range(len(columns)**2):
            x, y = i // len(columns), i % len(columns)

            # Calculate the distance between subplots
            offset = divide(0.0125, (len(columns) - 1))

            # Calculate the size of the subplot
            size = (1 - ((offset * 2) * (len(columns) - 1))) / len(columns)

            # Determine the position for the axes
            x_pos = y * (size + 2 * offset)
            y_pos = (len(columns) - x - 1) * (size + 2 * offset)

            xaxis, yaxis = self._fig.get_axes(
                x=(x_pos, rnd(x_pos + size)),
                y=(y_pos, rnd(y_pos + size)),
                coloraxis=dict(
                    colorscale=PALETTE.get(color, "Blues"),
                    cmin=0,
                    cmax=len(self.dataset),
                    showscale=False,
                )
            )

            if x == y:
                fig.add_trace(
                    go.Histogram(
                        x=self.dataset[columns[x]],
                        marker=dict(
                            color=f"rgba({color[4:-1]}, 0.2)",
                            line=dict(width=2, color=color),
                        ),
                        name=columns[x],
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )
            elif x > y:
                fig.add_trace(
                    go.Scatter(
                        x=sample(columns[y]),
                        y=sample(columns[x]),
                        mode="markers",
                        marker=dict(color=color),
                        hovertemplate="(%{x}, %{y})<extra></extra>",
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )
            elif y > x:
                fig.add_trace(
                    go.Histogram2dContour(
                        x=self.dataset[columns[y]],
                        y=self.dataset[columns[x]],
                        coloraxis=f"coloraxis{xaxis[1:]}",
                        hovertemplate="x:%{x}<br>y:%{y}<br>z:%{z}<extra></extra>",
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

            if x < len(columns) - 1:
                fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
            if y > 0:
                fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

            self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel=columns[y] if x == len(columns) - 1 else None,
                ylabel=columns[x] if y == 0 else None,
            )

        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (900, 900),
            plotname="plot_relationships",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_wordcloud(
        self,
        index: INT | str | slice | SEQUENCE | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
        **kwargs,
    ) -> go.Figure | None:
        """Plot a wordcloud from the corpus.

        The text for the plot is extracted from the column named
        `corpus`. If there is no column with that name, an exception
        is raised.

        Parameters
        ----------
        index: int, str, slice, sequence or None, default=None
            Documents in the corpus to include in the wordcloud. If
            None, it selects all documents in the dataset.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        **kwargs
            Additional keyword arguments for the [Wordcloud][] object.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_ngrams
        atom.plots:PredictionPlot.plot_pipeline

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import fetch_20newsgroups

        >>> X, y = fetch_20newsgroups(
        ...     return_X_y=True,
        ...     categories=[
        ...         'alt.atheism',
        ...         'sci.med',
        ...         'comp.windows.x',
        ...     ],
        ...     shuffle=True,
        ...     random_state=1,
        ... )
        >>> X = np.array(X).reshape(-1, 1)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.textclean()
        >>> atom.textnormalize()
        >>> atom.plot_wordcloud()

        ```

        :: insert:
            url: /img/plots/plot_wordcloud.html

        """

        def get_text(column):
            """Get the complete corpus as one long string."""
            if isinstance(column.iat[0], str):
                return " ".join(column)
            else:
                return " ".join([" ".join(row) for row in column])

        check_dependency("wordcloud")
        from wordcloud import WordCloud

        corpus = get_corpus(self.X)
        rows = self.dataset.loc[self.branch._get_rows(index, return_test=False)]

        wordcloud = WordCloud(
            width=figsize[0],
            height=figsize[1],
            background_color=kwargs.pop("background_color", "white"),
            random_state=kwargs.pop("random_state", self.random_state),
            **kwargs,
        )

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        fig.add_trace(
            go.Image(
                z=wordcloud.generate(get_text(rows[corpus])),
                hoverinfo="skip",
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        fig.update_layout(
            {
                f"xaxis{xaxis[1:]}_showticklabels": False,
                f"yaxis{xaxis[1:]}_showticklabels": False,
            }
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            figsize=figsize or (900, 600),
            plotname="plot_wordcloud",
            filename=filename,
            display=display,
        )


class HTPlot(BasePlot):
    """Hyperparameter tuning plots.

    Plots that help interpret the model's study and corresponding
    trials. These plots are accessible from the runners or from the
    models. If called from a runner, the `models` parameter has to be
    specified (if None, uses all models). If called from a model, that
    model is used and the `models` parameter becomes unavailable.

    """

    @composed(crash, plot_from_model, typechecked)
    def plot_edf(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        metric: INT | str | SEQUENCE | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper left",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
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
        models: int, str, Model, slice, sequence or None, default=None
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

        filename: str or None, default=None
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
        atom.plots:HTPlot.plot_hyperparameters
        atom.plots:HTPlot.plot_trials

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from optuna.distributions import IntDistribution

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()

        # We run three models with different search spaces
        >>> atom.run(
        ...     models="RF1",
        ...     n_trials=200,
        ...     ht_params={"distributions": {"n_estimators": IntDistribution(10, 20)}},
        ... )
        >>> atom.run(
        ...     models="RF1",
        ...     n_trials=200,
        ...     ht_params={"distributions": {"n_estimators": IntDistribution(50, 80)}},
        ... )
        >>> atom.run(
        ...     models="RF1",
        ...     n_trials=200,
        ...     ht_params={"distributions": {"n_estimators": IntDistribution(100, 200)}},
        ... )

        >>> atom.plot_edf()

        ```

        :: insert:
            url: /img/plots/plot_edf.html

        """
        # Check there is at least one model that ran hyperparameter tuning
        if not (models := list(filter(lambda x: x.trials is not None, models))):
            raise ValueError(
                "The plot_edf method is only available "
                "for models that ran hyperparameter tuning."
            )

        metric = self._get_metric(metric, max_one=False)

        values = []
        for m in models:
            values.append([])
            for met in metric:
                values[-1].append(
                    np.array([lst(row)[met] for row in m.trials["score"]])
                )

        x_min = np.nanmin(np.array(values))
        x_max = np.nanmax(np.array(values))

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m, val in zip(models, values):
            for met in metric:
                fig.add_trace(
                    self._draw_line(
                        x=(x := np.linspace(x_min, x_max, 100)),
                        y=np.sum(val[met][:, np.newaxis] <= x, axis=0) / len(val[met]),
                        parent=m.name,
                        child=self._metric[met].name,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._fig.used_models.extend(models)
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

    @composed(crash, plot_from_model, typechecked)
    def plot_hyperparameter_importance(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        metric: int | str = 0,
        show: INT | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot a model's hyperparameter importance.

        The hyperparameter importance are calculated using the
        [fANOVA][] importance evaluator. The sum of importances for all
        parameters (per model) is 1. This plot is only available for
        models that ran [hyperparameter tuning][].

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
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

        filename: str or None, default=None
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
        atom.plots:HTPlot.plot_hyperparameters
        atom.plots:HTPlot.plot_trials

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run(["LR", "RF"], n_trials=15)
        >>> atom.plot_hyperparameter_importance()

        ```

        :: insert:
            url: /img/plots/plot_hyperparameter_importance.html

        """
        # Check there is at least one model that ran hyperparameter tuning
        if not (models := list(filter(lambda x: x.trials is not None, models))):
            raise ValueError(
                "The plot_hyperparameter_importance method is only "
                "available for models that ran hyperparameter tuning."
            )

        params = len(set([k for m in lst(models) for k in m._ht["distributions"]]))
        if show is None or show > params:
            # Limit max features shown to avoid maximum figsize error
            show = min(200, params)
        elif show < 1:
            raise ValueError(
                f"Invalid value for the show parameter. Value should be >0, got {show}."
            )

        met = self._get_metric(metric, max_one=True)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            importances = FanovaImportanceEvaluator(seed=self.random_state).evaluate(
                study=m.study,
                target=None if len(self._metric) == 1 else lambda x: x.values[met],
            )

            fig.add_trace(
                go.Bar(
                    x=np.array(list(importances.values())) / sum(importances.values()),
                    y=list(importances.keys()),
                    orientation="h",
                    marker=dict(
                        color=f"rgba({self._fig.get_color(m.name)[4:-1]}, 0.2)",
                        line=dict(width=2, color=self._fig.get_color(m.name)),
                    ),
                    hovertemplate="%{x}<extra></extra>",
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=self._fig.showlegend(m.name, legend),
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

        self._fig.used_models.extend(models)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Normalized hyperparameter importance",
            ylim=(params - show - 0.5, params - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_hyperparameter_importance",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_hyperparameters(
        self,
        models: INT | str | Model | None = None,
        params: str | slice | SEQUENCE = (0, 1),
        metric: int | str = 0,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
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
            from a model, e.g. `atom.lr.plot_hyperparameters()`.

        params: str, slice or sequence, default=(0, 1)
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

        filename: str or None, default=None
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
        atom.plots:HTPlot.plot_hyperparameter_importance
        atom.plots:HTPlot.plot_parallel_coordinate
        atom.plots:HTPlot.plot_trials

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("RF")
        >>> atom.plot_hyperparameters(params=(0, 1, 2))

        ```

        :: insert:
            url: /img/plots/plot_hyperparameters.html

        """
        # Check that the model ran hyperparameter tuning
        if models.trials is None:
            raise ValueError(
                "The plot_hyperparameters method is only "
                "available for models that ran hyperparameter tuning."
            )

        if len(params := self._get_hyperparams(params, models)) < 2:
            raise ValueError(
                "Invalid value for the hyperparameters parameter. A minimum "
                f"of two parameters is required, got {len(params)}."
            )

        met = self._get_metric(metric, max_one=True)

        fig = self._get_figure()
        for i in range((length := len(params) - 1) ** 2):
            x, y = i // length, i % length

            if y <= x:
                # Calculate the size of the subplot
                size = 1 / length

                # Determine the position for the axes
                x_pos = y * size
                y_pos = (length - x - 1) * size

                xaxis, yaxis = self._fig.get_axes(
                    x=(x_pos, rnd(x_pos + size)),
                    y=(y_pos, rnd(y_pos + size)),
                    coloraxis=dict(
                        axes="99",
                        colorscale=PALETTE.get(self._fig.get_color(models.name), "Blues"),
                        cmin=np.nanmin(
                            models.trials.apply(lambda x: lst(x["score"])[met], axis=1)
                        ),
                        cmax=np.nanmax(
                            models.trials.apply(lambda x: lst(x["score"])[met], axis=1)
                        ),
                        showscale=False,
                    )
                )

                x_values = lambda row: row["params"].get(params[y], None)
                y_values = lambda row: row["params"].get(params[x + 1], None)

                customdata = zip(
                    models.trials.index.tolist(),
                    models.trials.apply(lambda x: lst(x["score"])[met], axis=1),
                )

                fig.add_trace(
                    go.Scatter(
                        x=models.trials.apply(x_values, axis=1),
                        y=models.trials.apply(y_values, axis=1),
                        mode="markers",
                        marker=dict(
                            size=self.marker_size,
                            color=self._fig.get_color(models.name),
                            line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                        ),
                        customdata=list(customdata),
                        hovertemplate=(
                            f"{params[y]}:%{{x}}<br>"
                            f"{params[x + 1]}:%{{y}}<br>"
                            f"{self._metric[met].name}:%{{customdata[1]:.4f}}"
                            "<extra>Trial %{customdata[0]}</extra>"
                        ),
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                fig.add_trace(
                    go.Contour(
                        x=models.trials.apply(x_values, axis=1),
                        y=models.trials.apply(y_values, axis=1),
                        z=models.trials.apply(lambda i: lst(i["score"])[met], axis=1),
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
                    xlabel=params[y] if x == length - 1 else None,
                    ylabel=params[x + 1] if y == 0 else None,
                )

        self._fig.used_models.append(models)
        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (800 + 100 * length, 500 + 100 * length),
            plotname="plot_hyperparameters",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_parallel_coordinate(
        self,
        models: INT | str | Model | None = None,
        params: str | slice | SEQUENCE | None = None,
        metric: INT | str = 0,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
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
            from a model, e.g. `atom.lr.plot_parallel_coordinate()`.

        params: str, slice, sequence or None, default=None
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

        filename: str or None, default=None
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
        atom.plots:HTPlot.plot_edf
        atom.plots:HTPlot.plot_hyperparameter_importance
        atom.plots:HTPlot.plot_hyperparameters

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("RF", n_trials=15)
        >>> atom.plot_parallel_coordinate(params=slice(1, 5))

        ```

        :: insert:
            url: /img/plots/plot_parallel_coordinate.html

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
            numbers, categorical = [], []
            for elem in values:
                try:
                    numbers.append(it(float(elem)))
                except (TypeError, ValueError):
                    categorical.append(str(elem))

            return list(map(str, sorted(numbers))) + sorted(categorical)

        # Check that the model ran hyperparameter tuning
        if models.trials is None:
            raise ValueError(
                "The plot_parallel_coordinate method is only "
                "available for models that ran hyperparameter tuning."
            )

        params = self._get_hyperparams(params, models)
        met = self._get_metric(metric, max_one=True)

        dims = _get_dims_from_info(
            _get_parallel_coordinate_info(
                study=models.study,
                params=params,
                target=None if len(self._metric) == 1 else lambda x: x.values[met],
                target_name=self._metric[met].name,
            )
        )

        # Clean up dimensions for nicer view
        for d in [dims[0]] + sorted(dims[1:], key=lambda x: params.index(x["label"])):
            if "ticktext" in d:
                # Skip processing for logarithmic params
                if all(isinstance(i, INT) for i in d["values"]):
                    # Order categorical values
                    mapping = [d["ticktext"][i] for i in d["values"]]
                    d["ticktext"] = sort_mixed_types(d["ticktext"])
                    d["values"] = [d["ticktext"].index(v) for v in mapping]
            else:
                # Round numerical values
                d["tickvals"] = list(
                    map(rnd, np.linspace(min(d["values"]), max(d["values"]), 5))
                )

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes(
            coloraxis=dict(
                colorscale=PALETTE.get(self._fig.get_color(models.name), "Blues"),
                cmin=min(dims[0]["values"]),
                cmax=max(dims[0]["values"]),
                title=self._metric[met].name,
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

        self._fig.used_models.append(models)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            figsize=figsize or (700 + len(params) * 50, 600),
            plotname="plot_parallel_coordinate",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_pareto_front(
        self,
        models: INT | str | Model | None = None,
        metric: str | SEQUENCE | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the Pareto front of a study.

        Shows the trial scores plotted against each other. The marker's
        colors indicate the trial number. This plot is only available
        for models that ran [multi-metric runs][] with
        [hyperparameter tuning][].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g. `atom.lr.plot_pareto_front()`.

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

        filename: str or None, default=None
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
        atom.plots:HTPlot.plot_edf
        atom.plots:HTPlot.plot_slice
        atom.plots:HTPlot.plot_trials

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("RF", metric=["f1", "accuracy", "recall"], n_trials=15)
        >>> atom.plot_pareto_front()

        ```

        :: insert:
            url: /img/plots/plot_pareto_front.html

        """
        # Check that the model ran hyperparameter tuning
        if models.trials is None:
            raise ValueError(
                "The plot_pareto_front method is only "
                "available for models that ran hyperparameter tuning."
            )

        if len(metric := self._get_metric(metric, max_one=False)) < 2:
            raise ValueError(
                "Invalid value for the metric parameter. A minimum "
                f"of two metrics are required, got {len(metric)}."
            )

        fig = self._get_figure()
        for i in range((length := len(metric) - 1) ** 2):
            x, y = i // length, i % length

            if y <= x:
                # Calculate the distance between subplots
                offset = divide(0.0125, length - 1)

                # Calculate the size of the subplot
                size = (1 - ((offset * 2) * (length - 1))) / length

                # Determine the position for the axes
                x_pos = y * (size + 2 * offset)
                y_pos = (length - x - 1) * (size + 2 * offset)

                xaxis, yaxis = self._fig.get_axes(
                    x=(x_pos, rnd(x_pos + size)),
                    y=(y_pos, rnd(y_pos + size)),
                )

                fig.add_trace(
                    go.Scatter(
                        x=models.trials.apply(lambda row: row["score"][y], axis=1),
                        y=models.trials.apply(lambda row: row["score"][x + 1], axis=1),
                        mode="markers",
                        marker=dict(
                            size=self.marker_size,
                            color=models.trials.index,
                            colorscale="Teal",
                            line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                        ),
                        customdata=models.trials.index,
                        hovertemplate="(%{x}, %{y})<extra>Trial %{customdata}</extra>",
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                if x < len(metric) - 1:
                    fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
                if y > 0:
                    fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

                self._plot(
                    ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                    xlabel=self._metric[y].name if x == length - 1 else None,
                    ylabel=self._metric[x + 1].name if y == 0 else None,
                )

        self._fig.used_models.append(models)
        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (500 + 100 * length, 500 + 100 * length),
            plotname="plot_pareto_front",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_slice(
        self,
        models: INT | str | Model | None = None,
        params: str | slice | SEQUENCE | None = None,
        metric: INT | str | SEQUENCE | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the parameter relationship in a study.

        The color of the markers indicate the trial. This plot is only
        available for models that ran [hyperparameter tuning][].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g. `atom.lr.plot_slice()`.

        params: str, slice, sequence or None, default=None
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

        filename: str or None, default=None
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
        atom.plots:HTPlot.plot_edf
        atom.plots:HTPlot.plot_hyperparameters
        atom.plots:HTPlot.plot_parallel_coordinate

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("RF", metric=["f1", "recall"], n_trials=15)
        >>> atom.plot_slice(params=(0, 1, 2))

        ```

        :: insert:
            url: /img/plots/plot_slice.html

        """
        # Check that the model ran hyperparameter tuning
        if models.trials is None:
            raise ValueError(
                "The plot_slice method is only "
                "available for models that ran hyperparameter tuning."
            )

        params = self._get_hyperparams(params, models)
        metric = self._get_metric(metric, max_one=False)

        fig = self._get_figure()
        for i in range(len(params) * len(metric)):
            x, y = i // len(params), i % len(params)

            # Calculate the distance between subplots
            x_offset = divide(0.0125, (len(params) - 1))
            y_offset = divide(0.0125, (len(metric) - 1))

            # Calculate the size of the subplot
            x_size = (1 - ((x_offset * 2) * (len(params) - 1))) / len(params)
            y_size = (1 - ((y_offset * 2) * (len(metric) - 1))) / len(metric)

            # Determine the position for the axes
            x_pos = y * (x_size + 2 * x_offset)
            y_pos = (len(metric) - x - 1) * (y_size + 2 * y_offset)

            xaxis, yaxis = self._fig.get_axes(
                x=(x_pos, rnd(x_pos + x_size)),
                y=(y_pos, rnd(y_pos + y_size)),
            )

            fig.add_trace(
                go.Scatter(
                    x=models.trials.apply(
                        lambda r: r["params"].get(params[y], None), axis=1
                    ),
                    y=models.trials.apply(lambda r: lst(r["score"])[x], axis=1),
                    mode="markers",
                    marker=dict(
                        size=self.marker_size,
                        color=models.trials.index,
                        colorscale="Teal",
                        line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                    ),
                    customdata=models.trials.index,
                    hovertemplate="(%{x}, %{y})<extra>Trial %{customdata}</extra>",
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

            if x < len(metric) - 1:
                fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
            if y > 0:
                fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

            self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel=params[y] if x == len(metric) - 1 else None,
                ylabel=self._metric[x].name if y == 0 else None,
            )

        self._fig.used_models.append(models)
        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (800 + 100 * len(params), 500 + 100 * len(metric)),
            plotname="plot_slice",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_trials(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        metric: INT | str | SEQUENCE | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper left",
        figsize: tuple[INT, INT] = (900, 800),
        filename: str | None = None,
        display: bool | None = True,
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
        models: int, str, Model, slice, sequence or None, default=None
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

        filename: str or None, default=None
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
        atom.plots:HTPlot.plot_hyperparameters
        atom.plots:PredictionPlot.plot_results

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["LR", "RF"], n_trials=15)
        >>> atom.plot_trials()

        ```

        :: insert:
            url: /img/plots/plot_trials.html

        """
        # Check there is at least one model that ran hyperparameter tuning
        if not (models := list(filter(lambda x: x.trials is not None, models))):
            raise ValueError(
                "The plot_trials method is only available "
                "for models that ran hyperparameter tuning."
            )

        metric = self._get_metric(metric, max_one=False)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes(y=(0.31, 1.0))
        xaxis2, yaxis2 = self._fig.get_axes(y=(0.0, 0.29))
        for m in models:
            for met in metric:
                y = m.trials["score"].apply(lambda value: lst(value)[met])

                # Create star symbol at best trial
                symbols = ["circle"] * len(y)
                symbols[m.best_trial.number] = "star"
                sizes = [self.marker_size] * len(y)
                sizes[m.best_trial.number] = self.marker_size * 1.5

                fig.add_trace(
                    self._draw_line(
                        x=list(range(len(y))),
                        y=y,
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
                        x=list(range(1, len(y))),
                        y=np.abs(np.diff(y)),
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
                f"xaxis{xaxis[1:]}_showticklabels": False,
                "hovermode": "x unified",
            },
        )

        self._plot(
            ax=(f"xaxis{xaxis2[1:]}", f"yaxis{yaxis2[1:]}"),
            xlabel="Trial",
            ylabel="d",
        )

        self._fig.used_models.extend(models)
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


class PredictionPlot(BasePlot):
    """Prediction plots.

    Plots that use the model's predictions. These plots are accessible
    from the runners or from the models. If called from a runner, the
    `models` parameter has to be specified (if None, uses all models).
    If called from a model, that model is used and the `models` parameter
    becomes unavailable.

    """

    @available_if(has_task(["binary", "multilabel"]))
    @composed(crash, plot_from_model, typechecked)
    def plot_calibration(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str | SEQUENCE = "test",
        n_bins: INT = 10,
        target: INT | str = 0,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper left",
        figsize: tuple[INT, INT] = (900, 900),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the calibration curve for a binary classifier.

        Well calibrated classifiers are probabilistic classifiers for
        which the output of the `predict_proba` method can be directly
        interpreted as a confidence level. For instance a well
        calibrated (binary) classifier should classify the samples such
        that among the samples to which it gave a `predict_proba` value
        close to 0.8, approx. 80% actually belong to the positive class.
        Read more in sklearn's [documentation][calibration].

        This figure shows two plots: the calibration curve, where the
        x-axis represents the average predicted probability in each bin
        and the y-axis is the fraction of positives, i.e. the proportion
        of samples whose class is the positive class (in each bin); and
        a distribution of all predicted probabilities of the classifier.
        This plot is available only for models with a `predict_proba`
        method in a binary or [multilabel][] classification task.

        !!! tip
            Use the [calibrate][adaboost-calibrate] method to calibrate
            the winning model.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str or sequence, default="test"
            Data set on which to calculate the metric. Use a sequence
            or add `+` between options to select more than one. Choose
            from: "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["RF", "LGB"])
        >>> atom.plot_calibration()

        ```

        :: insert:
            url: /img/plots/plot_calibration.html

        """
        check_predict_proba(models, "plot_calibration")
        dataset = self._get_set(dataset, max_one=False)
        target = self.branch._get_target(target, only_columns=True)

        if n_bins < 5:
            raise ValueError(
                "Invalid value for the n_bins parameter."
                f"Value should be >=5, got {n_bins}."
            )

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes(y=(0.31, 1.0))
        xaxis2, yaxis2 = self._fig.get_axes(y=(0.0, 0.29))
        for m in models:
            for ds in dataset:
                y_true, y_pred = m._get_pred(ds, target, attr="predict_proba")

                # Get calibration (frac of positives and predicted values)
                frac_pos, pred = calibration_curve(y_true, y_pred, n_bins=n_bins)

                fig.add_trace(
                    self._draw_line(
                        x=pred,
                        y=frac_pos,
                        parent=m.name,
                        child=ds,
                        mode="lines+markers",
                        marker_symbol="circle",
                        legend=legend,
                        xaxis=xaxis2,
                        yaxis=yaxis,
                    )
                )

                fig.add_trace(
                    go.Histogram(
                        x=y_pred,
                        xbins=dict(start=0, end=1, size=1. / n_bins),
                        marker=dict(
                            color=f"rgba({self._fig.get_color(m.name)[4:-1]}, 0.2)",
                            line=dict(width=2, color=self._fig.get_color(m.name)),
                        ),
                        name=m.name,
                        legendgroup=m.name,
                        showlegend=False,
                        xaxis=xaxis2,
                        yaxis=yaxis2,
                    )
                )

        self._draw_straight_line(y="diagonal", xaxis=xaxis, yaxis=yaxis)

        fig.update_layout({f"xaxis{xaxis2[1:]}_showgrid": True, "barmode": "overlay"})

        self._plot(
            ax=(f"xaxis{xaxis2[1:]}", f"yaxis{yaxis2[1:]}"),
            xlabel="Predicted value",
            ylabel="Count",
            xlim=(0, 1),
        )

        self._fig.used_models.extend(models)
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

    @available_if(has_task("class"))
    @composed(crash, plot_from_model, typechecked)
    def plot_confusion_matrix(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str = "test",
        target: INT | str = 0,
        threshold: FLOAT = 0.5,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper right",
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
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
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str, default="test"
            Data set on which to calculate the confusion matrix. Choose
            from:` "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["LR", "RF"])
        >>> atom.lr.plot_confusion_matrix()  # For one model

        ```

        :: insert:
            url: /img/plots/plot_confusion_matrix_1.html

        ```pycon
        >>> atom.plot_confusion_matrix()  # For multiple models

        ```

        :: insert:
            url: /img/plots/plot_confusion_matrix_2.html

        """
        ds = self._get_set(dataset, max_one=True)
        target = self.branch._get_target(target, only_columns=True)

        if self.task.startswith("multiclass") and len(models) > 1:
            raise NotImplementedError(
                "The plot_confusion_matrix method does not support "
                "the comparison of multiple models for multiclass "
                "or multiclass-multioutput classification tasks."
            )

        labels = np.array(
            (("True negatives", "False positives"), ("False negatives", "True positives"))
        )

        fig = self._get_figure()
        for m in models:
            y_true, y_pred = m._get_pred(ds, target, attr="predict")
            if threshold != 0.5:
                y_pred = (y_pred > threshold).astype("int")

            cm = confusion_matrix(y_true, y_pred)
            if len(models) == 1:  # Create matrix heatmap
                ticks = m.mapping.get(target, np.unique(m.dataset[target]).astype(str))
                xaxis, yaxis = self._fig.get_axes(
                    x=(0, 0.87),
                    coloraxis=dict(
                        colorscale="Blues",
                        cmin=0,
                        cmax=100,
                        title="Percentage of samples",
                        font_size=self.label_fontsize,
                    ),
                )

                fig.add_trace(
                    go.Heatmap(
                        x=ticks,
                        y=ticks,
                        z=100. * cm / cm.sum(axis=1)[:, np.newaxis],
                        coloraxis=f"coloraxis{xaxis[1:]}",
                        text=cm,
                        customdata=labels,
                        texttemplate="%{text}<br>(%{z:.2f}%)",
                        textfont=dict(size=self.label_fontsize),
                        hovertemplate=(
                            "<b>%{customdata}</b><br>" if is_binary(self.task) else ""
                            "x:%{x}<br>y:%{y}<br>z:%{z}<extra></extra>"
                        ),
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
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
                xaxis, yaxis = self._fig.get_axes()

                color = self._fig.get_color(m.name)
                fig.add_trace(
                    go.Bar(
                        x=cm.ravel(),
                        y=labels.ravel(),
                        orientation="h",
                        marker=dict(
                            color=f"rgba({color[4:-1]}, 0.2)",
                            line=dict(width=2, color=color),
                        ),
                        hovertemplate="%{x}<extra></extra>",
                        name=m.name,
                        legendgroup=m.name,
                        showlegend=self._fig.showlegend(m.name, legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                fig.update_layout(bargroupgap=0.05)

        self._fig.used_models.extend(models)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Predicted label" if len(models) == 1 else "Count",
            ylabel="True label" if len(models) == 1 else None,
            title=title,
            legend=legend,
            figsize=figsize or ((800, 800) if len(models) == 1 else (900, 600)),
            plotname="plot_confusion_matrix",
            filename=filename,
            display=display,
        )

    @available_if(has_task(["binary", "multilabel"]))
    @composed(crash, plot_from_model, typechecked)
    def plot_det(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str | SEQUENCE = "test",
        target: INT | str = 0,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ):
        """Plot the Detection Error Tradeoff curve.

        Read more about [DET][] in sklearn's documentation. Only
        available for binary classification tasks.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str or sequence, default="test"
            Data set on which to calculate the metric. Use a sequence
            or add `+` between options to select more than one. Choose
            from: "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> import pandas as pd

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_det()

        ```

        :: insert:
            url: /img/plots/plot_det.html

        """
        dataset = self._get_set(dataset, max_one=False)
        target = self.branch._get_target(target, only_columns=True)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            for ds in dataset:
                # Get fpr-fnr pairs for different thresholds
                fpr, fnr, _ = det_curve(*m._get_pred(ds, target, attr="thresh"))

                fig.add_trace(
                    self._draw_line(
                        x=fpr,
                        y=fnr,
                        mode="lines",
                        parent=m.name,
                        child=ds,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._fig.used_models.extend(models)
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

    @available_if(has_task("reg"))
    @composed(crash, plot_from_model, typechecked)
    def plot_errors(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str = "test",
        target: INT | str = 0,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot a model's prediction errors.

        Plot the actual targets from a set against the predicted values
        generated by the regressor. A linear fit is made on the data.
        The gray, intersected line shows the identity line. This plot
        can be useful to detect noise or heteroscedasticity along a
        range of the target domain. This plot is available only for
        regression tasks.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMRegressor
        >>> from sklearn.datasets import load_diabetes

        >>> X, y = load_diabetes(return_X_y=True, as_frame=True)

        >>> atom = ATOMRegressor(X, y)
        >>> atom.run(["OLS", "LGB"])
        >>> atom.plot_errors()

        ```

        :: insert:
            url: /img/plots/plot_errors.html

        """
        ds = self._get_set(dataset, max_one=True)
        target = self.branch._get_target(target, only_columns=True)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            y_true, y_pred = m._get_pred(ds, target)

            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode="markers",
                    line=dict(width=2, color=self._fig.get_color(m.name)),
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=self._fig.showlegend(m.name, legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

            # Fit the points using linear regression
            from atom.models import OrdinaryLeastSquares
            model = OrdinaryLeastSquares(goal=self.goal, branch=m.branch)._get_est()
            model.fit(y_true.values.reshape(-1, 1), y_pred)

            fig.add_trace(
                go.Scatter(
                    x=(x := np.linspace(y_true.min(), y_true.max(), 100)),
                    y=model.predict(x[:, np.newaxis]),
                    mode="lines",
                    line=dict(width=2, color=self._fig.get_color(m.name)),
                    hovertemplate="(%{x}, %{y})<extra></extra>",
                    legendgroup=m.name,
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

        self._draw_straight_line(y="diagonal", xaxis=xaxis, yaxis=yaxis)

        self._fig.used_models.extend(models)
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

    @composed(crash, plot_from_model(ensembles=False), typechecked)
    def plot_evals(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str | SEQUENCE = "test",
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot evaluation curves.

        The evaluation curves are the main metric scores achieved by the
        models at every iteration of the training process. This plot is
        available only for models that allow [in-training validation][].

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str or sequence, default="test"
            Data set on which to calculate the evaluation curves. Use a
            sequence or add `+` between options to select more than one.
            Choose from: "train" or "test".

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

        filename: str or None, default=None
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
        atom.plots:HTPlot.plot_trials

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["XGB", "LGB"])
        >>> atom.plot_evals()

        ```

        :: insert:
            url: /img/plots/plot_evals.html

        """
        dataset = self._get_set(dataset, max_one=False, allow_holdout=False)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            if not m.evals:
                raise ValueError(
                    "Invalid value for the models parameter. Model "
                    f"{m.name} has no in-training validation."
                )

            for ds in dataset:
                fig.add_trace(
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
                )

        self._fig.used_models.append(models)
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

    @composed(crash, plot_from_model, typechecked)
    def plot_feature_importance(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        show: INT | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot a model's feature importance.

        The sum of importances for all features (per model) is 1.
        This plot is available only for models whose estimator has
        a `scores_`, `feature_importances_` or `coef` attribute.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_feature_importance(show=10)

        ```

        :: insert:
            url: /img/plots/plot_feature_importance.html

        """
        show = self._get_show(show, models)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            if (fi := m.feature_importance) is None:
                raise ValueError(
                    "Invalid value for the models parameter. The estimator "
                    f"{m.estimator.__class__.__name__} has no feature_importances_ "
                    "nor coef_ attribute."
                )

            fig.add_trace(
                go.Bar(
                    x=fi,
                    y=fi.index,
                    orientation="h",
                    marker=dict(
                        color=f"rgba({self._fig.get_color(m.name)[4:-1]}, 0.2)",
                        line=dict(width=2, color=self._fig.get_color(m.name)),
                    ),
                    hovertemplate="%{x}<extra></extra>",
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=self._fig.showlegend(m.name, legend),
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

        # Unique number of features over all branches
        n_fxs = len(set([fx for m in models for fx in m.features]))

        self._fig.used_models.extend(models)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Normalized feature importance",
            ylim=(n_fxs - show - 0.5, n_fxs - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_feature_importance",
            filename=filename,
            display=display,
        )

    @available_if(has_task(["binary", "multilabel"]))
    @composed(crash, plot_from_model, typechecked)
    def plot_gains(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str | SEQUENCE = "test",
        target: INT | str = 0,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the cumulative gains curve.

        This plot is available only for binary and [multilabel][]
        classification tasks.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str or sequence, default="test"
            Data set on which to calculate the metric. Use a sequence
            or add `+` between options to select more than one. Choose
            from: "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> import pandas as pd

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_gains()

        ```

        :: insert:
            url: /img/plots/plot_gains.html

        """
        dataset = self._get_set(dataset, max_one=False)
        target = self.branch._get_target(target, only_columns=True)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            for ds in dataset:
                y_true, y_pred = m._get_pred(ds, target, attr="thresh")

                fig.add_trace(
                    self._draw_line(
                        x=np.arange(start=1, stop=len(y_true) + 1) / len(y_true),
                        y=np.cumsum(y_true.iloc[np.argsort(y_pred)[::-1]]) / y_true.sum(),
                        mode="lines",
                        parent=m.name,
                        child=ds,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._draw_straight_line(y="diagonal", xaxis=xaxis, yaxis=yaxis)

        self._fig.used_models.extend(models)
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

    @composed(crash, plot_from_model(ensembles=False), typechecked)
    def plot_learning_curve(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        metric: INT | str | SEQUENCE | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the learning curve: score vs number of training samples.

        This plot is available only for models fitted using
        [train sizing][]. [Ensembles][] are ignored.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.train_sizing(["LR", "RF"], n_bootstrap=5)
        >>> atom.plot_learning_curve()

        ```

        :: insert:
            url: /img/plots/plot_learning_curve.html

        """
        metric = self._get_metric(metric, max_one=False)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        for met in metric:
            x, y, std = defaultdict(list), defaultdict(list), defaultdict(list)
            for m in models:
                x[m._group].append(m._train_idx)
                y[m._group].append(get_best_score(m, met))
                if m.bootstrap is not None:
                    std[m._group].append(m.bootstrap.iloc[:, met].std())

            for group in x:
                fig.add_trace(
                    self._draw_line(
                        x=x[group],
                        y=y[group],
                        mode="lines+markers",
                        marker_symbol="circle",
                        error_y=dict(type="data", array=std[group], visible=True),
                        parent=group,
                        child=self._metric[met].name,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                # Add error bands
                if m.bootstrap is not None:
                    fig.add_traces(
                        [
                            go.Scatter(
                                x=x[group],
                                y=np.add(y[group], std[group]),
                                mode="lines",
                                line=dict(width=1, color=self._fig.get_color(group)),
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
                                line=dict(width=1, color=self._fig.get_color(group)),
                                fill="tonexty",
                                fillcolor=f"rgba{self._fig.get_color(group)[3:-1]}, 0.2)",
                                hovertemplate="%{y}<extra>lower bound</extra>",
                                legendgroup=group,
                                showlegend=False,
                                xaxis=xaxis,
                                yaxis=yaxis,
                            ),
                        ]
                    )

        self._fig.used_models.extend(models)
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

    @available_if(has_task(["binary", "multilabel"]))
    @composed(crash, plot_from_model, typechecked)
    def plot_lift(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str | SEQUENCE = "test",
        target: INT | str = 0,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the lift curve.

        Only available for binary classification tasks.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str or sequence, default="test"
            Data set on which to calculate the metric. Use a sequence
            or add `+` between options to select more than one. Choose
            from: "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> import pandas as pd

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_lift()

        ```

        :: insert:
            url: /img/plots/plot_lift.html

        """
        dataset = self._get_set(dataset, max_one=False)
        target = self.branch._get_target(target, only_columns=True)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            for ds in dataset:
                y_true, y_pred = m._get_pred(ds, target, attr="thresh")

                gains = np.cumsum(y_true.iloc[np.argsort(y_pred)[::-1]]) / y_true.sum()
                fig.add_trace(
                    self._draw_line(
                        x=(x := np.arange(start=1, stop=len(y_true) + 1) / len(y_true)),
                        y=gains / x,
                        mode="lines",
                        parent=m.name,
                        child=ds,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._draw_straight_line(y=1, xaxis=xaxis, yaxis=yaxis)

        self._fig.used_models.extend(models)
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

    @composed(crash, plot_from_model, typechecked)
    def plot_parshap(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        columns: INT | str | slice | SEQUENCE | None = None,
        target: INT | str | tuple = 1,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper left",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
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
        map.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        columns: int, str, slice, sequence or None, default=None
            Features to plot. If None, it plots all features.

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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run(["GNB", "RF"])
        >>> atom.rf.plot_parshap(legend=None)

        ```

        :: insert:
            url: /img/plots/plot_parshap_1.html

        ```pycon
        >>> atom.plot_parshap(columns=slice(5, 10))

        ```

        :: insert:
            url: /img/plots/plot_parshap_2.html

        """
        target = self.branch._get_target(target)

        fig = self._get_figure()

        # Colorbar is only needed when a model has feature_importance
        if all(m.feature_importance is None for m in models):
            xaxis, yaxis = self._fig.get_axes()
        else:
            xaxis, yaxis = self._fig.get_axes(
                x=(0, 0.87),
                coloraxis=dict(
                    colorscale="Reds",
                    title="Normalized feature importance",
                    font_size=self.label_fontsize,
                )
            )

        for m in models:
            parshap = {}
            fxs = m.branch._get_columns(columns, include_target=False)

            for ds in ("train", "test"):
                # Calculating shap values is computationally expensive,
                # therefore select a random subsample for large data sets
                if len(data := getattr(m, ds)) > 500:
                    data = data.sample(500, random_state=self.random_state)

                # Replace data with the calculated shap values
                explanation = m._shap.get_explanation(data[m.features], target)
                data[m.features] = explanation.values

                parshap[ds] = pd.Series(index=fxs, dtype=float)
                for fx in fxs:
                    # All other features are covariates
                    covariates = [f for f in data.columns[:-1] if f != fx]
                    cols = [fx, data.columns[-1], *covariates]

                    # Compute covariance
                    V = data[cols].cov()

                    # Inverse covariance matrix
                    Vi = np.linalg.pinv(V, hermitian=True)
                    diag = Vi.diagonal()

                    D = np.diag(np.sqrt(1 / diag))

                    # Partial correlation matrix
                    partial_corr = -1 * (D @ Vi @ D)  # @ is matrix multiplication

                    # Semi-partial correlation matrix
                    with np.errstate(divide="ignore"):
                        V_sqrt = np.sqrt(np.diag(V))[..., None]
                        Vi_sqrt = np.sqrt(np.abs(diag - Vi ** 2 / diag[..., None])).T
                        semi_partial_correlation = partial_corr / V_sqrt / Vi_sqrt

                    # X covariates are removed
                    parshap[ds][fx] = semi_partial_correlation[1, 0]

            # Get the feature importance or coefficients
            if m.feature_importance is not None:
                color = m.feature_importance.loc[fxs]
            else:
                color = self._fig.get_color("parshap")

            fig.add_trace(
                go.Scatter(
                    x=parshap["train"],
                    y=parshap["test"],
                    mode="markers+text",
                    marker=dict(
                        color=color,
                        size=self.marker_size,
                        coloraxis=f"coloraxis{xaxis[1:]}",
                        line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                    ),
                    text=m.features,
                    textposition="top center",
                    customdata=(data := None if isinstance(color, str) else list(color)),
                    hovertemplate=(
                        f"%{{text}}<br>(%{{x}}, %{{y}})"
                        f"{'<br>Feature importance: %{customdata:.4f}' if data else ''}"
                        f"<extra>{m.name}</extra>"
                    ),
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=self._fig.showlegend(m.name, legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

        self._draw_straight_line(y="diagonal", xaxis=xaxis, yaxis=yaxis)

        self._fig.used_models.extend(models)
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

    @composed(crash, plot_from_model, typechecked)
    def plot_partial_dependence(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        columns: INT | str | slice | SEQUENCE | None = None,
        kind: str | SEQUENCE = "average",
        pair: int | str | None = None,
        target: INT | str = 1,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
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
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        columns: int, str, slice, sequence or None, default=None
            Features to get the partial dependence from. If None, it
            uses the first 3 features in the dataset.

        kind: str or sequence, default="average"
            Kind of depedence to plot. Use a sequence or add `+` between
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
            If None, the plots show the partial dependece of single
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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_partial_dependence(kind="average+individual", legend="upper left")

        ```

        :: insert:
            url: /img/plots/plot_partial_dependence_1.html

        ```pycon
        >>> atom.rf.plot_partial_dependence(columns=(3, 4), pair=2)

        ```

        :: insert:
            url: /img/plots/plot_partial_dependence_2.html

        """
        if any(self.task.startswith(t) for t in ("multilabel", "multiclass-multioutput")):
            raise PermissionError(
                "The plot_partial_dependence method is not available for multilabel "
                f"nor multiclass-multioutput classification tasks, got {self.task}."
            )
        elif self.task.startswith("multiclass"):
            _, target = self.branch._get_target(target)
        else:
            target = 0

        kind = "+".join(lst(kind)).lower()
        if any(k not in ("average", "individual") for k in kind.split("+")):
            raise ValueError(
                f"Invalid value for the kind parameter, got {kind}. "
                "Choose from: average, individual."
            )

        axes, names = [], []
        fig = self._get_figure()
        for m in models:
            color = self._fig.get_color(m.name)

            # Since every model can have different fxs, select them
            # every time and make sure the models use the same fxs
            cols = m.branch._get_columns(
                columns=(0, 1, 2) if columns is None else columns,
                include_target=False,
            )

            if not names:
                names = cols
            elif names != cols:
                raise ValueError(
                    "Invalid value for the columns parameter. Not all "
                    f"models use the same features, got {names} and {cols}."
                )

            if pair is not None:
                if len(models) > 1:
                    raise ValueError(
                        f"Invalid value for the pair parameter, got {pair}. "
                        "The value must be None when plotting multiple models"
                    )
                else:
                    pair = m.branch._get_columns(pair, include_target=False)
                    cols = [(c, pair[0]) for c in cols]
            else:
                cols = [(c,) for c in cols]

            # Create new axes
            if not axes:
                for i, col in enumerate(cols):
                    # Calculate the distance between subplots
                    offset = divide(0.025, len(cols) - 1)

                    # Calculate the size of the subplot
                    size = (1 - ((offset * 2) * (len(cols) - 1))) / len(cols)

                    # Determine the position for the axes
                    x_pos = i % len(cols) * (size + 2 * offset)

                    xaxis, yaxis = self._fig.get_axes(x=(x_pos, rnd(x_pos + size)))
                    axes.append((xaxis, yaxis))

            # Compute averaged predictions
            predictions = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(partial_dependence)(
                    estimator=m.estimator,
                    X=m.X_test,
                    features=col,
                    kind="both" if "individual" in kind else "average",
                ) for col in cols
            )

            # Compute deciles for ticks (only if line plots)
            if len(cols[0]) == 1:
                deciles = {}
                for fx in chain.from_iterable(cols):
                    if fx not in deciles:  # Skip if the feature is repeated
                        X_col = _safe_indexing(m.X_test, fx, axis=1)
                        deciles[fx] = mquantiles(X_col, prob=np.arange(0.1, 1.0, 0.1))

            for i, (ax, fx, pred) in enumerate(zip(axes, cols, predictions)):
                # Draw line or contour plot
                if len(pred["values"]) == 1:
                    # For both average and individual: draw ticks on the horizontal axis
                    for line in deciles[fx[0]]:
                        fig.add_shape(
                            type="line",
                            x0=line,
                            x1=line,
                            xref=ax[0],
                            y0=0,
                            y1=0.05,
                            yref=f"{axes[0][1]} domain",
                            line=dict(width=1, color=self._fig.get_color(m.name)),
                            opacity=0.6,
                            layer="below",
                        )

                    # Draw the mean of the individual lines
                    if "average" in kind:
                        fig.add_trace(
                            go.Scatter(
                                x=pred["values"][0],
                                y=pred["average"][target].ravel(),
                                mode="lines",
                                line=dict(width=2, color=color),
                                name=m.name,
                                legendgroup=m.name,
                                showlegend=self._fig.showlegend(m.name, legend),
                                xaxis=ax[0],
                                yaxis=axes[0][1],
                            )
                        )

                    # Draw all individual (per sample) lines (ICE)
                    if "individual" in kind:
                        # Select up to 50 random samples to plot
                        idx = np.random.choice(
                            list(range(len(pred["individual"][target]))),
                            size=min(len(pred["individual"][target]), 50),
                            replace=False,
                        )
                        for sample in pred["individual"][target, idx, :]:
                            fig.add_trace(
                                go.Scatter(
                                    x=pred["values"][0],
                                    y=sample,
                                    mode="lines",
                                    line=dict(width=0.5, color=color),
                                    name=m.name,
                                    legendgroup=m.name,
                                    showlegend=self._fig.showlegend(m.name, legend),
                                    xaxis=ax[0],
                                    yaxis=axes[0][1],
                                )
                            )

                else:
                    fig.add_trace(
                        go.Contour(
                            x=pred["values"][0],
                            y=pred["values"][1],
                            z=pred["average"][target],
                            contours=dict(
                                showlabels=True,
                                labelfont=dict(size=self.tick_fontsize, color="white")
                            ),
                            hovertemplate="x:%{x}<br>y:%{y}<br>z:%{z}<extra></extra>",
                            hoverongaps=False,
                            colorscale=PALETTE.get(self._fig.get_color(m.name), "Teal"),
                            showscale=False,
                            showlegend=False,
                            xaxis=ax[0],
                            yaxis=axes[0][1],
                        )
                    )

                self._plot(
                    ax=(f"xaxis{ax[0][1:]}", f"yaxis{ax[1][1:]}"),
                    xlabel=fx[0],
                    ylabel=(fx[1] if len(fx) > 1 else "Score") if i == 0 else None,
                )

        self._fig.used_models.extend(models)
        return self._plot(
            groupclick="togglegroup",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_partial_dependence",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_permutation_importance(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        show: INT | None = None,
        n_repeats: INT = 10,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the feature permutation importance of models.

        Calculating permutations can be time-consuming, especially
        if `n_repeats` is high. For this reason, the permutations
        are stored under the `permutations` attribute. If the plot
        is called again for the same model with the same `n_repeats`,
        it will use the stored values, making the method considerably
        faster.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_permutation_importance(show=10, n_repeats=7)

        ```

        :: insert:
            url: /img/plots/plot_permutation_importance.html

        """
        show = self._get_show(show, models)

        if n_repeats <= 0:
            raise ValueError(
                "Invalid value for the n_repeats parameter."
                f"Value should be >0, got {n_repeats}."
            )

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        for m in models:
            # If permutations are already calculated and n_repeats is
            # same, use known permutations (for efficient re-plotting)
            if (
                not hasattr(m, "permutations")
                or m.permutations.importances.shape[1] != n_repeats
            ):
                # Permutation importances returns Bunch object
                m.permutations = permutation_importance(
                    estimator=m.estimator,
                    X=m.X_test,
                    y=m.y_test,
                    scoring=self._metric[0],
                    n_repeats=n_repeats,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )

            fig.add_trace(
                go.Box(
                    x=m.permutations["importances"].ravel(),
                    y=list(np.array([[fx] * n_repeats for fx in m.features]).ravel()),
                    marker_color=self._fig.get_color(m.name),
                    boxpoints="outliers",
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=self._fig.showlegend(m.name, legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

        fig.update_traces(orientation="h")
        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}": dict(categoryorder="total ascending"),
                "boxmode": "group",
            }
        )

        # Unique number of features over all branches
        n_fxs = len(set([fx for m in models for fx in m.features]))

        self._fig.used_models.extend(models)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Score",
            ylim=(n_fxs - show - 0.5, n_fxs - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_permutation_importance",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model(check_fitted=False), typechecked)
    def plot_pipeline(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        draw_hyperparameter_tuning: bool = True,
        color_branches: bool | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> plt.Figure | None:
        """Plot a diagram of the pipeline.

        !!! warning
            This plot uses the [schemdraw][] package, which is
            incompatible with [plotly][]. The returned plot is
            therefore a [matplotlib figure][pltfigure].

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
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
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the pipeline drawn.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

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
        >>> from atom import ATOMClassifier

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.scale()
        >>> atom.run(["GNB", "RNN", "SGD", "MLP"])
        >>> atom.voting(models=atom.winners[:2])
        >>> atom.plot_pipeline()

        ```

        ![plot_pipeline](../../img/plots/plot_pipeline_1.png)

        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.scale()
        >>> atom.prune()
        >>> atom.run("RF", n_trials=30)

        >>> atom.branch = "undersample"
        >>> atom.balance("nearmiss")
        >>> atom.run("RF_undersample")

        >>> atom.branch = "oversample_from_master"
        >>> atom.balance("smote")
        >>> atom.run("RF_oversample")

        >>> atom.plot_pipeline()

        ```

        ![plot_pipeline](../../img/plots/plot_pipeline_2.png)

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

        fig = self._get_figure(backend="matplotlib")
        check_canvas(self._fig.is_canvas, "plot_pipeline")

        # Define branches to plot (if called from model, it's only one)
        branches = []
        for branch in getattr(self, "_branches", [self.branch]):
            draw_models, draw_ensembles = [], []
            for m in models:
                if m.branch is branch:
                    if m.acronym not in ("Stack", "Vote"):
                        draw_models.append(m)
                    else:
                        draw_ensembles.append(m)

                        # Additionally, add all dependent models (if not already there)
                        draw_models.extend([i for i in m._models if i not in draw_models])

            if not models or draw_models:
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
                color = next(self._fig.palette)

                # Convert back to format accepted by matplotlib
                branch["color"] = unconvert_from_RGB_255(unlabel_rgb(color))
            else:
                branch["color"] = "black"

        # Create schematic drawing
        d = Drawing(unit=1, backend="matplotlib")
        d.config(fontsize=self.tick_fontsize)
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
        if any(m.scaler for m in models):
            x_pos[-1] = x_pos[-2] = x_pos[-3] + length + 7
        if draw_hyperparameter_tuning and any(m.trials is not None for m in models):
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
                        RoundBox(w=7)
                        .label("Scaler", color="k")
                        .color(branch["color"])
                        .drop("E")
                    )
                    offset = 0

                # Draw hyperparameter tuning
                if draw_hyperparameter_tuning and model.trials is not None:
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
                    Data(w=max(len(model._fullname) * 0.5, 7))
                    .label(model._fullname, color="k")
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

        d.draw(ax=plt.gca(), showframe=False, show=False)
        plt.axis("off")

        self._fig.used_models.extend(models)
        return self._plot(
            ax=plt.gca(),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_pipeline",
            filename=filename,
            display=display,
        )

    @available_if(has_task(["binary", "multilabel"]))
    @composed(crash, plot_from_model, typechecked)
    def plot_prc(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str | SEQUENCE = "test",
        target: INT | str = 0,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower left",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the precision-recall curve.

        Read more about [PRC][] in sklearn's documentation. Only
        available for binary classification tasks.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str or sequence, default="test"
            Data set on which to calculate the metric. Use a sequence
            or add `+` between options to select more than one. Choose
            from: "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> import pandas as pd

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_prc()

        ```

        :: insert:
            url: /img/plots/plot_prc.html

        """
        dataset = self._get_set(dataset, max_one=False)
        target = self.branch._get_target(target, only_columns=True)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            for ds in dataset:
                y_true, y_pred = m._get_pred(ds, target, attr="thresh")

                # Get precision-recall pairs for different thresholds
                prec, rec, _ = precision_recall_curve(y_true, y_pred)

                fig.add_trace(
                    self._draw_line(
                        x=rec,
                        y=prec,
                        mode="lines",
                        parent=m.name,
                        child=ds,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._draw_straight_line(sum(m.y_test) / len(m.y_test), xaxis=xaxis, yaxis=yaxis)

        self._fig.used_models.extend(models)
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

    @available_if(has_task("class"))
    @composed(crash, plot_from_model, typechecked)
    def plot_probabilities(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str = "test",
        target: INT | str | tuple = 1,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the probability distribution of the target classes.

        This plot is available only for models with a `predict_proba`
        method in classification tasks.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> import pandas as pd

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_probabilities()

        ```

        :: insert:
            url: /img/plots/plot_probabilities.html

        """
        check_predict_proba(models, "plot_probabilities")
        ds = self._get_set(dataset, max_one=True)
        col, cls = self.branch._get_target(target)
        col = lst(self.target)[col]

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            y_true, y_pred = getattr(m, f"y_{ds}"), getattr(m, f"predict_proba_{ds}")
            for value in np.unique(m.dataset[col]):
                # Get indices per class
                if is_multioutput(self.task):
                    if self.task.startswith("multilabel"):
                        hist = y_pred.loc[y_true[col] == value, col]
                    else:
                        hist = y_pred.loc[cls, col].loc[y_true[col] == value]
                else:
                    hist = y_pred.loc[y_true == value, str(cls)]

                fig.add_trace(
                    go.Scatter(
                        x=(x := np.linspace(0, 1, 100)),
                        y=stats.gaussian_kde(hist)(x),
                        mode="lines",
                        line=dict(
                            width=2,
                            color=self._fig.get_color(m.name),
                            dash=self._fig.get_dashes(ds),
                        ),
                        fill="tonexty",
                        fillcolor=f"rgba{self._fig.get_color(m.name)[3:-1]}, 0.2)",
                        fillpattern=dict(shape=self._fig.get_shapes(value)),
                        name=f"{col}={value}",
                        legendgroup=m.name,
                        legendgrouptitle=dict(text=m.name, font_size=self.label_fontsize),
                        showlegend=self._fig.showlegend(f"{m.name}-{value}", legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._fig.used_models.extend(models)
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

    @available_if(has_task("reg"))
    @composed(crash, plot_from_model, typechecked)
    def plot_residuals(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str = "test",
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper left",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot a model's residuals.

        The plot shows the residuals (difference between the predicted
        and the true value) on the vertical axis and the independent
        variable on the horizontal axis. The gray, intersected line
        shows the identity line. This plot can be useful to analyze the
        variance of the error of the regressor. If the points are
        randomly dispersed around the horizontal axis, a linear
        regression model is appropriate for the data; otherwise, a
        non-linear model is more appropriate. This plot is only
        available for regression tasks.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMRegressor
        >>> from sklearn.datasets import load_diabetes

        >>> X, y = load_diabetes(return_X_y=True, as_frame=True)

        >>> atom = ATOMRegressor(X, y)
        >>> atom.run(["OLS", "LGB"])
        >>> atom.plot_residuals()

        ```

        :: insert:
            url: /img/plots/plot_residuals.html

        """
        ds = self._get_set(dataset, max_one=True)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes(x=(0, 0.69))
        xaxis2, yaxis2 = self._fig.get_axes(x=(0.71, 1.0))
        for m in models:
            fig.add_trace(
                go.Scatter(
                    x=(x := getattr(m, f"predict_{ds}")),
                    y=(res := np.subtract(x, getattr(m, f"y_{ds}"))),
                    mode="markers",
                    line=dict(width=2, color=self._fig.get_color(m.name)),
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=self._fig.showlegend(m.name, legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

            fig.add_trace(
                go.Histogram(
                    y=res,
                    bingroup="residuals",
                    marker=dict(
                        color=f"rgba({self._fig.get_color(m.name)[4:-1]}, 0.2)",
                        line=dict(width=2, color=self._fig.get_color(m.name)),
                    ),
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=False,
                    xaxis=xaxis2,
                    yaxis=yaxis,
                )
            )

        self._draw_straight_line(y=0, xaxis=xaxis, yaxis=yaxis)

        fig.update_layout({f"yaxis{xaxis[1:]}_showgrid": True, "barmode": "overlay"})

        self._plot(
            ax=(f"xaxis{xaxis2[1:]}", f"yaxis{yaxis2[1:]}"),
            xlabel="Distribution",
            title=title,
        )

        self._fig.used_models.extend(models)
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

    @composed(crash, plot_from_model, typechecked)
    def plot_results(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        metric: INT | str | SEQUENCE | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the model results.

        If all models applied bootstrap, the plot is a boxplot. If
        not, the plot is a barplot. Models are ordered based on
        their score from the top down. The score is either the
        `score_bootstrap` or `score_test` attribute of the model,
        selected in that order.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        metric: int, str, sequence or None, default=None
            Metric to plot (only for multi-metric runs). Other available
            options are "time_bo", "time_fit", "time_bootstrap" and
            "time". If str, add `+` between options to select more than
            one. If None, the metric used to run the pipeline is selected.

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

        filename: str or None, default=None
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
        atom.plots:PredictionPlot.plot_probabilities
        atom.plots:PredictionPlot.plot_threshold

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> import pandas as pd

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["GNB", "LR", "RF", "LGB"], metric=["f1", "recall"])
        >>> atom.plot_results()

        ```

        :: insert:
            url: /img/plots/plot_results_1.html

        ```pycon
        >>> atom.run(["GNB", "LR", "RF", "LGB"], metric=["f1", "recall"], n_bootstrap=5)
        >>> atom.plot_results()

        ```

        :: insert:
            url: /img/plots/plot_results_2.html

        ```pycon
        >>> atom.plot_results(metric="time_fit+time")

        ```

        :: insert:
            url: /img/plots/plot_results_3.html

        """

        def get_std(model: Model, metric: int) -> SCALAR:
            """Get the standard deviation of the bootstrap scores.

            Parameters
            ----------
            model: Model
                 Model to get the std from.

            metric: int
                Index of the metric to get it from.

            Returns
            -------
            int or float
                Standard deviation score or 0 if not bootstrapped.

            """
            if model.bootstrap is None:
                return 0
            else:
                return model.bootstrap.iloc[:, metric].std()

        metric = self._get_metric(metric, max_one=False)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        for met in metric:
            if isinstance(met, str):
                color = self._fig.get_color(met)
                fig.add_trace(
                    go.Bar(
                        x=[getattr(m, met) for m in models],
                        y=[m.name for m in models],
                        orientation="h",
                        marker=dict(
                            color=f"rgba({color[4:-1]}, 0.2)",
                            line=dict(width=2, color=color),
                        ),
                        hovertemplate=f"%{{x}}<extra>{met}</extra>",
                        name=met,
                        legendgroup=met,
                        showlegend=self._fig.showlegend(met, legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )
            else:
                name = self._metric[met].name
                color = self._fig.get_color()

                if all(m.score_bootstrap for m in models):
                    x = np.array([m.bootstrap.iloc[:, met] for m in models]).ravel()
                    y = np.array([[m.name] * len(m.bootstrap) for m in models]).ravel()
                    fig.add_trace(
                        go.Box(
                            x=x,
                            y=list(y),
                            marker_color=color,
                            boxpoints="outliers",
                            name=name,
                            legendgroup=name,
                            showlegend=self._fig.showlegend(name, legend),
                            xaxis=xaxis,
                            yaxis=yaxis,
                        )
                    )
                else:
                    fig.add_trace(
                        go.Bar(
                            x=[get_best_score(m, met) for m in models],
                            y=[m.name for m in models],
                            error_x=dict(
                                type="data",
                                array=[get_std(m, met) for m in models],
                            ),
                            orientation="h",
                            marker=dict(
                                color=f"rgba({color[4:-1]}, 0.2)",
                                line=dict(width=2, color=color),
                            ),
                            hovertemplate="%{x}<extra></extra>",
                            name=name,
                            legendgroup=name,
                            showlegend=self._fig.showlegend(name, legend),
                            xaxis=xaxis,
                            yaxis=yaxis,
                        )
                    )

        fig.update_traces(orientation="h")
        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}": dict(categoryorder="total ascending"),
                "bargroupgap": 0.05,
                "boxmode": "group",
            }
        )

        self._fig.used_models.extend(models)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel=self._metric[metric].name if isinstance(metric, INT) else "time (s)",
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + len(models) * 50),
            plotname="plot_results",
            filename=filename,
            display=display,
        )

    @available_if(has_task(["binary", "multilabel"]))
    @composed(crash, plot_from_model, typechecked)
    def plot_roc(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        dataset: str | SEQUENCE = "test",
        target: INT | str = 0,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot the Receiver Operating Characteristics curve.

        Read more about [ROC][] in sklearn's documentation. Only
        available for classification tasks.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        dataset: str or sequence, default="test"
            Data set on which to calculate the metric. Use a sequence
            or add `+` between options to select more than one. Choose
            from: "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> import pandas as pd

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_roc()

        ```

        :: insert:
            url: /img/plots/plot_roc.html

        """
        dataset = self._get_set(dataset, max_one=False)
        target = self.branch._get_target(target, only_columns=True)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            for ds in dataset:
                # Get False (True) Positive Rate as arrays
                fpr, tpr, _ = roc_curve(*m._get_pred(ds, target, attr="thresh"))

                fig.add_trace(
                    self._draw_line(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        parent=m.name,
                        child=ds,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._draw_straight_line(y="diagonal", xaxis=xaxis, yaxis=yaxis)

        self._fig.used_models.extend(models)
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

    @composed(crash, plot_from_model(ensembles=False), typechecked)
    def plot_successive_halving(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        metric: INT | str | SEQUENCE | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot scores per iteration of the successive halving.

        Only use with models fitted using [successive halving][].
        [Ensembles][] are ignored.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.successive_halving(["Tree", "Bag", "RF", "LGB"], n_bootstrap=5)
        >>> atom.plot_successive_halving()

        ```

        :: insert:
            url: /img/plots/plot_successive_halving.html

        """
        metric = self._get_metric(metric, max_one=False)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        for met in metric:
            x, y, std = defaultdict(list), defaultdict(list), defaultdict(list)
            for m in models:
                x[m._group].append(len(m.branch._idx[1]) // m._train_idx)
                y[m._group].append(get_best_score(m, met))
                if m.bootstrap is not None:
                    std[m._group].append(m.bootstrap.iloc[:, met].std())

            for group in x:
                fig.add_trace(
                    self._draw_line(
                        x=x[group],
                        y=y[group],
                        mode="lines+markers",
                        marker_symbol="circle",
                        error_y=dict(type="data", array=std[group], visible=True),
                        parent=group,
                        child=self._metric[met].name,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                # Add error bands
                if m.bootstrap is not None:
                    fig.add_traces(
                        [
                            go.Scatter(
                                x=x[group],
                                y=np.add(y[group], std[group]),
                                mode="lines",
                                line=dict(width=1, color=self._fig.get_color(group)),
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
                                line=dict(width=1, color=self._fig.get_color(group)),
                                fill="tonexty",
                                fillcolor=f"rgba{self._fig.get_color(group)[3:-1]}, 0.2)",
                                hovertemplate="%{y}<extra>lower bound</extra>",
                                legendgroup=group,
                                showlegend=False,
                                xaxis=xaxis,
                                yaxis=yaxis,
                            ),
                        ]
                    )

        fig.update_layout({f"xaxis{yaxis[1:]}": dict(dtick=1, autorange="reversed")})

        self._fig.used_models.extend(models)
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

    @available_if(has_task(["binary", "multilabel"]))
    @composed(crash, plot_from_model, typechecked)
    def plot_threshold(
        self,
        models: INT | str | Model | slice | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        dataset: str = "test",
        target: INT | str = 0,
        steps: INT = 100,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower left",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot metric performances against threshold values.

        This plot is available only for models with a `predict_proba`
        method in a binary or [multilabel][] classification task.

        Parameters
        ----------
        models: int, str, Model, slice, sequence or None, default=None
            Models to plot. If None, all models are selected.

        metric: str, func, scorer, sequence or None, default=None
            Metric to plot. Choose from any of sklearn's scorers, a
            function with signature `metric(y_true, y_pred)`, a scorer
            object or a sequence of these. Use a sequence or add `+`
            between options to select more than one. If None, the
            metric used to run the pipeline is selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

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

        filename: str or None, default=None
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
        >>> from atom import ATOMClassifier
        >>> import pandas as pd

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_threshold()

        ```

        :: insert:
            url: /img/plots/plot_threshold.html

        """
        check_predict_proba(models, "plot_threshold")
        ds = self._get_set(dataset, max_one=True)
        target = self.branch._get_target(target, only_columns=True)

        # Get all metric functions from the input
        if metric is None:
            metrics = [m._score_func for m in self._metric]
        else:
            metrics = []
            for m in lst(metric):
                if isinstance(m, str):
                    metrics.extend(m.split("+"))
                else:
                    metrics.append(m)
            metrics = [get_custom_scorer(m)._score_func for m in metrics]

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        steps = np.linspace(0, 1, steps)
        for m in models:
            y_true, y_pred = m._get_pred(ds, target, attr="predict_proba")
            for met in metrics:
                fig.add_trace(
                    self._draw_line(
                        x=steps,
                        y=[met(y_true, y_pred >= step) for step in steps],
                        parent=m.name,
                        child=met.__name__,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._fig.used_models.extend(models)
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


class ShapPlot(BasePlot):
    """Shap plots.

    ATOM wrapper for plots made by the shap package, using Shapley
    values for model interpretation. These plots are accessible from
    the runners or from the models. Only one model can be plotted at
    the same time since the plots are not made by ATOM.

    """

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_shap_bar(
        self,
        models: INT | str | Model | None = None,
        index: INT | str | slice | SEQUENCE | None = None,
        show: INT | None = None,
        target: INT | str | tuple = 1,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's bar plot.

        Create a bar plot of a set of SHAP values. If a single sample
        is passed, then the SHAP values are plotted. If many samples
        are passed, then the mean absolute value for each feature
        column is plotted. Read more about SHAP plots in the
        [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g. `atom.lr.plot_shap_bar()`.

        index: int, str, slice, sequence or None, default=None
            Rows in the dataset to plot. If None, it selects all rows
            in the test set.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

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

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_parshap
        atom.plots:ShapPlot.plot_shap_beeswarm
        atom.plots:ShapPlot.plot_shap_scatter

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("LR")
        >>> atom.plot_shap_bar(show=10)

        ```

        ![plot_shap_bar](../../img/plots/plot_shap_bar.png)

        """
        rows = models.X.loc[models.branch._get_rows(index)]
        show = self._get_show(show, models)
        target = self.branch._get_target(target)
        explanation = models._shap.get_explanation(rows, target)

        self._get_figure(backend="matplotlib")
        check_canvas(self._fig.is_canvas, "plot_shap_bar")

        shap.plots.bar(explanation, max_display=show, show=False)

        self._fig.used_models.append(models)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_shap_bar",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_shap_beeswarm(
        self,
        models: INT | str | Model | None = None,
        index: slice | SEQUENCE | None = None,
        show: INT | None = None,
        target: INT | str | tuple = 1,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's beeswarm plot.

        The plot is colored by feature values. Read more about SHAP
        plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g. `atom.lr.plot_shap_beeswarm()`.

        index: tuple, slice or None, default=None
            Rows in the dataset to plot. If None, it selects all rows
            in the test set. The beeswarm plot does not support plotting
            a single sample.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

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

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_parshap
        atom.plots:ShapPlot.plot_shap_bar
        atom.plots:ShapPlot.plot_shap_scatter

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("LR")
        >>> atom.plot_shap_beeswarm(show=10)

        ```

        ![plot_shap_beeswarm](../../img/plots/plot_shap_beeswarm.png)

        """
        rows = models.X.loc[models.branch._get_rows(index)]
        show = self._get_show(show, models)
        target = self.branch._get_target(target)
        explanation = models._shap.get_explanation(rows, target)

        self._get_figure(backend="matplotlib")
        check_canvas(self._fig.is_canvas, "plot_shap_beeswarm")

        shap.plots.beeswarm(explanation, max_display=show, show=False)

        self._fig.used_models.append(models)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_shap_decision(
        self,
        models: INT | str | Model | None = None,
        index: INT | str | slice | SEQUENCE | None = None,
        show: INT | None = None,
        target: INT | str | tuple = 1,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's decision plot.

        Visualize model decisions using cumulative SHAP values. Each
        plotted line explains a single model prediction. If a single
        prediction is plotted, feature values are printed in the
        plot (if supplied). If multiple predictions are plotted
        together, feature values will not be printed. Plotting too
        many predictions together will make the plot unintelligible.
        Read more about SHAP plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g. `atom.lr.plot_shap_decision()`.

        index: int, str, slice, sequence or None, default=None
            Rows in the dataset to plot. If None, it selects all rows
            in the test set.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

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

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ShapPlot.plot_shap_bar
        atom.plots:ShapPlot.plot_shap_beeswarm
        atom.plots:ShapPlot.plot_shap_force

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("LR")
        >>> atom.plot_shap_decision(show=10)

        ```

        ![plot_shap_decision](../../img/plots/plot_shap_decision_1.png)

        ```pycon
        >>> atom.plot_shap_decision(index=-1, show=10)

        ```

        ![plot_shap_decision](../../img/plots/plot_shap_decision_2.png)

        """
        rows = models.X.loc[models.branch._get_rows(index)]
        show = self._get_show(show, models)
        target = self.branch._get_target(target)
        explanation = models._shap.get_explanation(rows, target)

        self._get_figure(backend="matplotlib")
        check_canvas(self._fig.is_canvas, "plot_shap_decision")

        shap.decision_plot(
            base_value=explanation.base_values,
            shap_values=explanation.values,
            features=rows,
            feature_display_range=slice(-1, -show - 1, -1),
            auto_size_plot=False,
            show=False,
        )

        self._fig.used_models.append(models)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_shap_decision",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_shap_force(
        self,
        models: INT | str | Model | None = None,
        index: INT | str | slice | SEQUENCE | None = None,
        target: INT | str | tuple = 1,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] = (900, 300),
        filename: str | None = None,
        display: bool | None = True,
        **kwargs,
    ) -> plt.Figure | None:
        """Plot SHAP's force plot.

        Visualize the given SHAP values with an additive force layout.
        Note that by default this plot will render using javascript.
        For a regular figure use `matplotlib=True` (this option is
        only available when only a single sample is plotted). Read more
        about SHAP plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g. `atom.lr.plot_shap_force()`.

        index: int, str, slice, sequence or None, default=None
            Rows in the dataset to plot. If None, it selects all rows
            in the test set.

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

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=(900, 300)
            Figure's size in pixels, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        **kwargs
            Additional keyword arguments for [shap.plots.force][force].

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ShapPlot.plot_shap_beeswarm
        atom.plots:ShapPlot.plot_shap_scatter
        atom.plots:ShapPlot.plot_shap_decision

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("LR")
        >>> atom.plot_shap_force(index=-2, matplotlib=True, figsize=(1800, 300))

        ```

        ![plot_shap_force](../../img/plots/plot_shap_force.png)

        """
        rows = models.X.loc[models.branch._get_rows(index)]
        target = self.branch._get_target(target)
        explanation = models._shap.get_explanation(rows, target)

        self._get_figure(create_figure=False, backend="matplotlib")
        check_canvas(self._fig.is_canvas, "plot_shap_force")

        plot = shap.force_plot(
            base_value=explanation.base_values,
            shap_values=explanation.values,
            features=rows,
            show=False,
            **kwargs,
        )

        if kwargs.get("matplotlib"):
            self._fig.used_models.append(models)
            return self._plot(
                fig=plt.gcf(),
                ax=plt.gca(),
                title=title,
                legend=legend,
                figsize=figsize,
                plotname="plot_shap_force",
                filename=filename,
                display=display,
            )
        else:
            if filename:  # Save to a html file
                if not filename.endswith(".html"):
                    filename += ".html"
                shap.save_html(filename, plot)
            if display and find_spec("IPython"):
                from IPython.display import display

                shap.initjs()
                display(plot)

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_shap_heatmap(
        self,
        models: INT | str | Model | None = None,
        index: slice | SEQUENCE | None = None,
        show: INT | None = None,
        target: INT | str | tuple = 1,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's heatmap plot.

        This plot is designed to show the population substructure of a
        dataset using supervised clustering and a heatmap. Supervised
        clustering involves clustering data points not by their original
        feature values but by their explanations. Read more about SHAP
        plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g. `atom.lr.plot_shap_heatmap()`.

        index: slice, sequence or None, default=None
            Rows in the dataset to plot. If None, it selects all rows
            in the test set. The plot_shap_heatmap method does not
            support plotting a single sample.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

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

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ShapPlot.plot_shap_decision
        atom.plots:ShapPlot.plot_shap_force
        atom.plots:ShapPlot.plot_shap_waterfall

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("LR")
        >>> atom.plot_shap_heatmap(show=10)

        ```

        ![plot_shap_heatmap](../../img/plots/plot_shap_heatmap.png)

        """
        rows = models.X.loc[models.branch._get_rows(index)]
        show = self._get_show(show, models)
        target = self.branch._get_target(target)
        explanation = models._shap.get_explanation(rows, target)

        self._get_figure(backend="matplotlib")
        check_canvas(self._fig.is_canvas, "plot_shap_heatmap")

        shap.plots.heatmap(explanation, max_display=show, show=False)

        self._fig.used_models.append(models)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_shap_heatmap",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_shap_scatter(
        self,
        models: INT | str | Model | None = None,
        index: slice | SEQUENCE | None = None,
        columns: INT | str = 0,
        target: INT | str | tuple = 1,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's scatter plot.

        Plots the value of the feature on the x-axis and the SHAP value
        of the same feature on the y-axis. This shows how the model
        depends on the given feature, and is like a richer extension of
        the classical partial dependence plots. Vertical dispersion of
        the data points represents interaction effects. Read more about
        SHAP plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g. `atom.lr.plot_shap_scatter()`.

        index: slice, sequence or None, default=None
            Rows in the dataset to plot. If None, it selects all rows
            in the test set. The plot_shap_scatter method does not
            support plotting a single sample.

        columns: int or str, default=0
            Column to plot.

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

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ShapPlot.plot_shap_beeswarm
        atom.plots:ShapPlot.plot_shap_decision
        atom.plots:ShapPlot.plot_shap_force

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("LR")
        >>> atom.plot_shap_scatter(columns="symmetry error")

        ```

        ![plot_shap_scatter](../../img/plots/plot_shap_scatter.png)

        """
        rows = models.X.loc[models.branch._get_rows(index)]
        column = models.branch._get_columns(columns, include_target=False)[0]
        target = self.branch._get_target(target)
        explanation = models._shap.get_explanation(rows, target)

        # Get explanation for a specific column
        explanation = explanation[:, models.columns.get_loc(column)]

        self._get_figure(backend="matplotlib")
        check_canvas(self._fig.is_canvas, "plot_shap_scatter")

        shap.plots.scatter(explanation, color=explanation, ax=plt.gca(), show=False)

        self._fig.used_models.append(models)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            ylabel=plt.gca().get_ylabel(),
            title=title,
            legend=legend,
            plotname="plot_shap_scatter",
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model(max_one=True), typechecked)
    def plot_shap_waterfall(
        self,
        models: INT | str | Model | None = None,
        index: INT | str | None = None,
        show: INT | None = None,
        target: INT | str | tuple = 1,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = None,
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's waterfall plot.

        The SHAP value of a feature represents the impact of the
        evidence provided by that feature on the model’s output. The
        waterfall plot is designed to visually display how the SHAP
        values (evidence) of each feature move the model output from
        our prior expectation under the background data distribution,
        to the final model prediction given the evidence of all the
        features. Features are sorted by the magnitude of their SHAP
        values with the smallest magnitude features grouped together
        at the bottom of the plot when the number of features in the
        models exceeds the `show` parameter. Read more about SHAP plots
        in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g. `atom.lr.plot_shap_waterfall()`.

        index: int, str or None, default=None
            Rows in the dataset to plot. If None, it selects all rows
            in the test set. The plot_shap_waterfall method does not
            support plotting multiple samples.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

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

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ShapPlot.plot_shap_bar
        atom.plots:ShapPlot.plot_shap_beeswarm
        atom.plots:ShapPlot.plot_shap_heatmap

        Examples
        --------
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.run("LR")
        >>> atom.plot_shap_waterfall()

        ```

        ![plot_shap_waterfall](../../img/plots/plot_shap_waterfall.png)

        """
        rows = models.X.loc[[models.branch._get_rows(index)[0]]]
        show = self._get_show(show, models)
        target = self.branch._get_target(target)
        explanation = models._shap.get_explanation(rows, target)

        # Waterfall accepts only one row
        explanation.values = explanation.values[0]
        explanation.data = explanation.data[0]

        # For binary classification, it's a scalar already
        if hasattr(explanation.base_values, "__len__"):
            explanation.base_values = explanation.base_values[target]

        self._get_figure(backend="matplotlib")
        check_canvas(self._fig.is_canvas, "plot_shap_waterfall")

        shap.plots.waterfall(explanation, max_display=show, show=False)

        self._fig.used_models.append(models)
        return self._plot(
            ax=plt.gca(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_shap_waterfall",
            filename=filename,
            display=display,
        )

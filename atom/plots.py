# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the plotting classes.

"""

from collections import defaultdict
from contextlib import contextmanager
from functools import reduce
from importlib.util import find_spec
from itertools import chain, cycle
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.transforms import blended_transform_factory
from mlflow.tracking import MlflowClient
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nltk.collocations import (
    BigramCollocationFinder, QuadgramCollocationFinder,
    TrigramCollocationFinder,
)
from schemdraw import Drawing
from schemdraw.flow import Data, RoundBox, Subroutine, Wire
from schemdraw.util import Point
from scipy import stats
from scipy.stats.mstats import mquantiles
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix, det_curve, precision_recall_curve, roc_curve,
)
from sklearn.utils import _safe_indexing
from sklearn.utils.metaestimators import available_if
from typeguard import typechecked
from wordcloud import WordCloud

from atom.utils import (
    FLOAT, INT, PALETTE, SCALAR, SEQUENCE_TYPES, Model, check_is_fitted,
    check_predict_proba, composed, crash, divide, get_best_score, get_corpus,
    get_custom_scorer, get_feature_importance, has_attr, has_task, lst,
    partial_dependence, plot_from_model, rnd, to_rgb,
)


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

    """

    def __init__(
        self,
        rows: INT = 1,
        cols: INT = 1,
        horizontal_spacing: FLOAT = 0.05,
        vertical_spacing: FLOAT = 0.07,
        palette: Union[str, SEQUENCE_TYPES] = "Prism",
        is_canvas: bool = False,
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

        self.idx = 0  # N-th plot in the canvas
        self.axes = 0  # N-th axis in the canvas
        self.figure = go.Figure()

        self.groups = []
        self.style = dict(colors={}, markers={}, dashes={})

        self.pos = {}  # Subplot position to use for title
        self.custom_layout = {}  # Layout params specified by user
        self.used_models = []  # Models plotted in this figure
        self._markers = cycle(["circle", "x", "diamond", "pentagon", "star", "hexagon"])
        self._dashes = cycle([None, "dashdot", "dash", "dot"])

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
    def grid(self) -> Tuple[int, int]:
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
    def next_subplot(self) -> go.Figure:
        """Increase the subplot index.

        Returns
        -------
        go.Figure
            Current pyplot figure.

        """
        # Check if there are too many plots in the canvas
        if self.idx >= self.rows * self.cols:
            raise ValueError(
                "Invalid number of plots in the canvas! Increase "
                "the number of rows and cols to add more plots."
            )
        else:
            self.idx += 1

        return self.figure

    def get_color(self, elem: Optional[str] = None) -> str:
        """Get the next color.

        This method is used to assign the same color to the same
        elements (columns, models, etc...) in a canvas.

        Parameters
        ----------
        elem: str or None
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

    def get_marker(self, elem: Optional[str] = None) -> str:
        """Get the next marker.

        This method is used to assign the same marker to the same
        elements (e.g. distribution) in a canvas.

        Parameters
        ----------
        elem: str or None
            Element for which to get the marker.

        Returns
        -------
        str
            Marker code.

        """
        if elem is None:
            return next(self._markers)
        elif elem in self.style["markers"]:
            return self.style["markers"][elem]
        else:
            return self.style["markers"].setdefault(elem, next(self._markers))

    def get_dashes(self, elem: Optional[str] = None) -> str:
        """Get the next dash style.

        This method is used to assign the same dash style to the same
        elements (e.g. data set) in a canvas.

        Parameters
        ----------
        elem: str or None
            Element for which to get the dash.

        Returns
        -------
        str
            Dash style.

        """
        if elem is None:
            return next(self._dashes)
        elif elem in self.style["dashes"]:
            return self.style["dashes"][elem]
        else:
            return self.style["dashes"].setdefault(elem, next(self._dashes))

    def showlegend(self, name: str, legend: Optional[Union[str, dict]]) -> bool:
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
        x: Tuple[int, int] = (0, 1),
        y: Tuple[int, int] = (0, 1),
        coloraxis: Optional[dict] = None,
    ) -> Tuple[str, str]:
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
                # Add a title as annotation since plotly's
                # title can't be oriented correctly
                self.figure.add_annotation(
                    x=x_pos + x_size + x_offset / 10,
                    xref="paper",
                    xanchor="left",
                    y=y_pos + ay_size / 2,
                    yref="paper",
                    yanchor="middle",
                    text=title,
                    textangle=90,
                    font_size=coloraxis.pop("font_size"),
                    showarrow=False,
                )

            coloraxis["colorbar_x"] = rnd(x_pos + ax_size)
            coloraxis["colorbar_y"] = y_pos + ay_size / 2
            coloraxis["colorbar_len"] = ay_size * 0.9
            coloraxis["colorbar_thickness"] = ax_size * 20  # Default width in pixels
            self.figure.update_layout({f"coloraxis{self.axes}": coloraxis})

        return f"x{self.axes}", f"y{self.axes}"


class BasePlot:
    """Parent class for all plotting methods.

    This base class defines the plot properties that can
    be changed in order to customize the plot's aesthetics.

    """

    def __init__(self):
        self._fig = None
        self._custom_layout = {}
        self._aesthetics = dict(
            palette=PALETTE,  # Sequence of colors
            title_fontsize=24,  # Fontsize for titles
            label_fontsize=16,  # Fontsize for labels, legend and hoverinfo
            tick_fontsize=12,  # Fontsize for ticks
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

    @property
    def palette(self) -> Union[str, SEQUENCE_TYPES]:
        """Color palette.

        Specify one of plotly's [built-in palettes][palette] or create
        a custom one, e.g. `atom.palette = ["red", "green", "blue"]`.

        """
        return self._aesthetics["palette"]

    @palette.setter
    @typechecked
    def palette(self, value: Union[str, SEQUENCE_TYPES]):
        if isinstance(value, str) and not hasattr(px.colors.qualitative, value):
            raise ValueError(
                f"Invalid value for the palette parameter, got {value}. Choose "
                f"from one of plotly's built-in qualitative color sequences in "
                f"the px.colors.qualitative module or define your own sequence."
            )

        self._aesthetics["palette"] = value

    @property
    def title_fontsize(self) -> int:
        """Fontsize for the plot's title."""
        return self._aesthetics["title_fontsize"]

    @title_fontsize.setter
    @typechecked
    def title_fontsize(self, value: INT):
        if value <= 0:
            raise ValueError(
                "Invalid value for the title_fontsize parameter. "
                f"Value should be >=0, got {value}."
            )
        self._aesthetics["title_fontsize"] = value

    @property
    def label_fontsize(self) -> int:
        """Fontsize for the labels, legend and hover information."""
        return self._aesthetics["label_fontsize"]

    @label_fontsize.setter
    @typechecked
    def label_fontsize(self, value: INT):
        if value <= 0:
            raise ValueError(
                "Invalid value for the label_fontsize parameter. "
                f"Value should be >=0, got {value}."
            )
        self._aesthetics["label_fontsize"] = value

    @property
    def tick_fontsize(self) -> int:
        """Fontsize for the ticks along the plot's axes."""
        return self._aesthetics["tick_fontsize"]

    @tick_fontsize.setter
    @typechecked
    def tick_fontsize(self, value: INT):
        if value <= 0:
            raise ValueError(
                "Invalid value for the tick_fontsize parameter. "
                f"Value should be >=0, got {value}."
            )
        self._aesthetics["tick_fontsize"] = value

    def reset_aesthetics(self):
        """Reset the plot [aesthetics][] to their default values."""
        self._custom_layout = {}
        self._aesthetics = dict(
            palette=PALETTE,
            title_fontsize=24,
            label_fontsize=16,
            tick_fontsize=12,
        )

    def update_layout(
        self,
        dict1: Optional[dict] = None,
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

    # Methods ====================================================== >>

    @staticmethod
    def _draw_line(
        xaxis: str,
        yaxis: str,
        x: Tuple[SCALAR, SCALAR] = (0, 1),
        y: Union[SCALAR, str] = "diagonal",
    ) -> go.Scatter:
        """Draw a line across the axis.

        The line can be either horizontal or diagonal. The line should
        be used as reference. It's not added to the legend and doesn't
        show any information on hover.

        Parameters
        ----------
        xaxis: str
            Name of the x-axis to draw in.

        yaxis: str
            Name of the y-axis to draw in.

        x: tuple
            Coordinates on the x-axis.

        y: int, float or str
            Coordinates on the y-axis. If a value, draw a horizontal line
            at that value. If "diagonal", draw a diagonal line from x.

        Returns
        -------
        go.Scatter
            New trace to add to figure.

        """
        return go.Scatter(
            x=x,
            y=x if y == "diagonal" else [y, y],
            mode="lines",
            line=dict(width=2, color="black", dash="dash"),
            opacity=0.6,
            hoverinfo="skip",
            showlegend=False,
            xaxis=xaxis,
            yaxis=yaxis,
        )

    def _get_figure(self) -> go.Figure:
        """Return existing figure if in canvas, else a new figure.

        Every time this method is called from a canvas, the plot
        index is raised by one to keep track in which subplot the
        BaseFigure is at.

        Returns
        -------
        [go.Figure][]
            Existing figure or newly created.

        """
        if self._fig and self._fig.is_canvas:
            return self._fig.next_subplot
        else:
            self._fig = BaseFigure(palette=self.palette)
            return self._fig.next_subplot

    def _get_subclass(
        self,
        models: Union[str, List[str]],
        max_one: bool = False,
        ensembles: bool = True,
    ) -> Union[Model, List[Model]]:
        """Get model subclasses from names.

        Parameters
        ----------
        models: str or sequence
            Models provided by the plot's parameter.

        max_one: bool, default=False
            Whether one or multiple models are allowed. If True, return
            the model instead of a list.

        ensembles: bool, default=True
            If False, drop ensemble models automatically.

        Returns
        -------
        model or list of models
            Model subclasses retrieved from the names.

        """
        models = list(self._models[self._get_models(models, ensembles)].values())

        if max_one and len(models) > 1:
            raise ValueError("This plot only accepts one model!")

        return models[0] if max_one else models

    def _get_metric(self, metric: Union[int, str]) -> int:
        """Check and return the provided metric index.

        Parameters
        ----------
        metric: int or str
            Name or position of the metric to get.

        Returns
        -------
        int
            Position index of the metric.

        """
        if isinstance(metric, str):
            if metric.lower() in ("time_ht", "time_fit", "time_bootstrap", "time"):
                return metric.lower()
            else:
                name = get_custom_scorer(metric).name
                if name in self.metric:
                    return self._metric.index(name)

        elif 0 <= metric < len(self._metric):
            return metric

        raise ValueError(
            "Invalid value for the metric parameter. Value should be the index "
            f"or name of a metric used to run the pipeline, got {metric}."
        )

    def _get_target(self, target: Union[int, str]) -> int:
        """Check and return the provided target's index.

        Parameters
        ----------
        metric: int or str
            Name or position of the target to get.

        Returns
        -------
        int
            Position index of the target.

        """
        if isinstance(target, str):
            try:
                return self.mapping[self.target][target]
            except (TypeError, KeyError):
                raise ValueError(
                    f"Invalid value for the target parameter. Value {target} "
                    "not found in the mapping of the target column."
                )
        elif not 0 <= target < self.y.nunique(dropna=False):
            raise ValueError(
                "Invalid value for the target parameter. There are "
                f"{self.y.nunique(dropna=False)} classes, got {target}."
            )

        return target

    def _get_set(self, dataset: str, allow_holdout: bool = True) -> List[str]:
        """Check and return the provided metric.

        Parameters
        ----------
        dataset: str
            Name of the data set to retrieve.

        allow_holdout: bool, default=True
            Whether to allow the retrieval of the holdout set.

        Returns
        -------
        list of str
            Selected data sets.

        """
        dataset = dataset.lower()
        if dataset == "both":
            return ["test", "train"]
        elif dataset in ("train", "test"):
            return [dataset]
        elif allow_holdout:
            if dataset == "holdout":
                if self.holdout is None:
                    raise ValueError(
                        "Invalid value for the dataset parameter. No holdout "
                        "data set was specified when initializing the instance."
                    )
                return [dataset]
            else:
                raise ValueError(
                    "Invalid value for the dataset parameter, got "
                    f"{dataset}. Choose from: train, test, both or holdout."
                )
        else:
            raise ValueError(
                "Invalid value for the dataset parameter, got "
                f"{dataset}. Choose from: train, test or both."
            )

    @staticmethod
    def _get_show(show: Optional[int], model: Union[Model, List[Model]]) -> int:
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
        max_fxs = max([m.n_features for m in lst(model)])
        if show is None or show > max_fxs:
            # Limit max features shown to avoid maximum figsize error
            return min(200, max_fxs)
        elif show < 1:
            raise ValueError(
                "Invalid value for the show parameter."
                f"Value should be >0, got {show}."
            )

        return show

    def _plot(
        self, axes: Optional[Tuple[str, str]] = None, **kwargs
    ) -> Optional[Union[go.Figure, plt.Figure]]:
        """Make the plot.

        Customize the axes to the default layout and plot the figure
        if it's not part of a canvas.

        Parameters
        ----------
        axes: tuple or None, default=None
            Names of the axes to update. If None, ignore their update.

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
        if axes:
            self._fig.figure.update_layout(
                {
                    f"{axes[0]}_title": kwargs.get("xlabel"),
                    f"{axes[1]}_title": kwargs.get("ylabel"),
                    f"{axes[0]}_title_font_size": self.label_fontsize,
                    f"{axes[1]}_title_font_size": self.label_fontsize,
                    f"{axes[0]}_range": kwargs.get("xlim"),
                    f"{axes[1]}_range": kwargs.get("ylim"),
                    f"{axes[0]}_automargin": True,
                    f"{axes[1]}_automargin": True,
                }
            )

            if self._fig.is_canvas and (title := kwargs.get("title")):
                # Add a subtitle to a plot in the canvas
                default_title = {
                    "x": self._fig.pos[axes[0][5:]][0],
                    "y": self._fig.pos[axes[0][5:]][1] + 0.005,
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

                self._fig.figure.update_layout(
                    dict(annotations=self._fig.figure.layout.annotations + (title,))
                )

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
            if not isinstance(title := kwargs.get("title"), dict):
                title = {"text": title, **default_title}
            else:
                title = {**default_title, **title}

            default_legend = dict(
                traceorder="grouped",
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
                        "Invalid value for the legend parameter. Got unknown position: "
                        f"{legend}. Choose from: upper left, upper right, lower left, "
                        "lower right, upper center, lower center, center left, center "
                        "right, center, out."
                    )
                legend = {**default_legend, **position}
            elif isinstance(legend, dict):
                legend = {**default_legend, **legend}

            # Update layout with predefined settings
            space1 = self.title_fontsize if title.get("text") else 10
            space2 = self.title_fontsize * int(bool(self._fig.figure.layout.annotations))
            self._fig.figure.update_layout(
                title=title,
                legend=legend,
                showlegend=bool(kwargs.get("legend")),
                hoverlabel=dict(font_size=self.label_fontsize),
                font_size=self.tick_fontsize,
                margin=dict(l=0, b=0, r=0, t=25 + space1 + space2, pad=0),
                width=kwargs["figsize"][0],
                height=kwargs["figsize"][1],
            )

            # Update layout with custom settings
            self._fig.figure.update_layout(**self._custom_layout)

            # Set name with which to save the file
            if kwargs.get("filename"):
                if kwargs["filename"].endswith("auto"):
                    name = kwargs["filename"].replace("auto", kwargs["plotname"])
                else:
                    name = kwargs["filename"]
            else:
                name = kwargs.get("plotname")

            if kwargs.get("filename"):
                if "." not in name or name.endswith(".html"):
                    self._fig.figure.write_html(name if "." in name else name + ".html")
                else:
                    self._fig.figure.write_image(name)

            # Log plot to mlflow run of every model visualized
            if getattr(self, "experiment", None) and self.log_plots:
                for m in set(self._fig.used_models):
                    MlflowClient().log_figure(
                        run_id=m._run.info.run_id,
                        figure=self._fig.figure,
                        artifact_file=name if "." in name else name + ".html",
                    )

            if kwargs.get("display") is True:
                self._fig.figure.show()
            elif kwargs.get("display") is None:
                return self._fig.figure

    @composed(contextmanager, crash, typechecked)
    def canvas(
        self,
        rows: INT = 1,
        cols: INT = 2,
        *,
        horizontal_spacing: FLOAT = 0.05,
        vertical_spacing: FLOAT = 0.07,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "out",
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
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
            (.html, .png, .pdf, etc...). If `filename` has no period,
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
            self._fig.figure.layout.legend.groupclick = "togglegroup"
            self._fig.is_canvas = False  # Close the canvas
            self._plot(
                title=title,
                legend=legend,
                figsize=figsize or (550 + 350 * cols, 200 + 400 * rows),
                plotname="canvas",
                filename=filename,
                display=display,
            )


class FeatureSelectorPlot(BasePlot):
    """Feature selection plots.

    These plots are accessible from atom or from the FeatureSelector
    class when the appropriate feature selection strategy is used.

    """

    @available_if(has_attr("pca"))
    @composed(crash, typechecked)
    def plot_components(
        self,
        show: Optional[INT] = None,
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "lower right",
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
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

        figsize: tuple, default=None
            Figure's size in pixels, format as (x, y).  If None, it
            adapts the size to the number of components shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
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
                x=variance[:show],
                y=[f"pca{str(i)}" for i in range(show)],
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
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            xlabel="Explained variance ratio",
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
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (900, 600),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
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
            (.html, .png, .pdf, etc...). If `filename` has no period,
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
        sizes = [6] * self.pca.n_features_in_
        sizes[self.pca._comps - 1] = 12
        symbols = ["circle"] * self.pca.n_features_in_
        symbols[self.pca._comps - 1] = "star"

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        fig.add_trace(
            go.Scatter(
                x=tuple(range(1, self.pca.n_features_in_ + 1)),
                y=np.cumsum(self.pca.explained_variance_ratio_),
                mode="lines+markers",
                line=dict(width=2, color=self._fig.get_color("pca")),
                marker=dict(symbol=symbols, size=sizes, opacity=1, line=dict(width=0)),
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
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            xlabel="First N principal components",
            ylabel="Cumulative variance ratio",
            xlim=(1 - margin, self.pca.n_features_in_ - 1 + margin),
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
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (900, 600),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
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
            (.html, .png, .pdf, etc...). If `filename` has no period,
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
                x=tuple(x),
                y=mean,
                mode="lines+markers",
                line=dict(width=2, color=self._fig.get_color("rfecv")),
                marker=dict(symbol=symbols, size=sizes, opacity=1, line=dict(width=0)),
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
                    name="bounds",
                    legendgroup="bounds",
                    showlegend=self._fig.showlegend("bounds", legend),
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
                    legendgroup="bounds",
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                ),
            ]
        )

        fig.update_layout(
            {
                "hovermode": "x",
                f"xaxis{xaxis[1:]}_showspikes": True,
                f"yaxis{yaxis[1:]}_showspikes": True,
            }
        )

        return self._plot(
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            xlabel="Number of features",
            ylabel=ylabel,
            xlim=(min(x) - len(x) / 30, max(x) + len(x) / 30),
            ylim=(min(mean) - 3 * max(std), max(mean) + 3 * max(std)),
            figsize=figsize,
            plotname="plot_rfecv",
            filename=filename,
            display=display,
        )


class DataPlot(BasePlot):
    """Data plots.

    These plots are only accessible from atom since they are used
    for understanding and interpretation of the dataset. The other
    runners should be used for model training only, not for data
    manipulation.

    """

    @composed(crash, typechecked)
    def plot_correlation(
        self,
        columns: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        method: str = "pearson",
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (800, 700),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a correlation matrix.

        Displays a heatmap showing the correlation between columns in
        the dataset. The colors red, blue and white stand for positive,
        negative, and no correlation respectively.

        Parameters
        ----------
        columns: slice, sequence or None, default=None
            Slice, names or indices of the columns to plot. If None,
            plot all columns in the dataset. Selected categorical
            columns are ignored.

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
            (.html, .png, .pdf, etc...). If `filename` has no period,
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
        columns = self._get_columns(columns, only_numerical=True)
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
            x=(0, 0.9),
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
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
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
        columns: Union[INT, str, slice, SEQUENCE_TYPES] = 0,
        distributions: Optional[Union[str, SEQUENCE_TYPES]] = None,
        show: Optional[INT] = None,
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "upper right",
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
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
            Slice, names or indices of the columns to plot. It is only
            possible to plot one categorical column. If more than just
            one categorical columns are selected, all categorical
            columns are ignored.

        distributions: str, sequence or None, default=None
            Names of the `scipy.stats` distributions to fit to the
            columns. If None, a [Gaussian kde distribution][kde] is
            showed. Only for numerical columns.

        show: int or None, default=None
            Number of classes (ordered by number of occurrences) to
            show in the plot. None to show all. Only for categorical
            columns.

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

        figsize: tuple, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the plot's type.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
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
        columns = self._get_columns(columns)
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
                    x=(data := series[-show:]),
                    y=data.index,
                    orientation="h",
                    marker=dict(
                        color=f"rgba({color[4:-1]}, 0.2)",
                        line=dict(width=2, color=color),
                    ),
                    hovertemplate="%{x}<extra></extra>",
                    name=f"{columns[0]}: {len(series)} classes",
                    legendgroup=columns[0],
                    showlegend=self._fig.showlegend(columns[0], legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

            return self._plot(
                axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                title=title,
                legend=legend,
                xlabel="Counts",
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
                        name=col,
                        legendgroup=col,
                        showlegend=self._fig.showlegend(col, legend),
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
                            go.Scatter(
                                x=x,
                                y=getattr(stats, dist).pdf(x, *params),
                                mode="lines",
                                line=dict(
                                    width=2,
                                    color=self._fig.get_color(col),
                                    dash=self._fig.get_dashes(dist),
                                ),
                                name=dist,
                                legendgroup=col,
                                showlegend=self._fig.showlegend(f"{col}-{dist}", legend),
                                xaxis=xaxis,
                                yaxis=yaxis,
                            )
                        )
                else:
                    # If no distributions specified, draw Gaussian kde
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=stats.gaussian_kde(values)(x),
                            mode="lines",
                            line=dict(
                                width=2,
                                color=self._fig.get_color(col),
                                dash=self._fig.get_dashes(dist),
                            ),
                            name="kde",
                            legendgroup=col,
                            showlegend=self._fig.showlegend(f"{col}-kde", legend),
                            xaxis=xaxis,
                            yaxis=yaxis,
                        )
                    )

            fig.update_layout(dict(barmode="overlay", legend_groupclick="toggleitem"))

            return self._plot(
                axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                title=title,
                legend=legend,
                xlabel="Values",
                ylabel="Probability density",
                figsize=figsize or (900, 600),
                plotname="plot_distribution",
                filename=filename,
                display=display,
            )

    @composed(crash, typechecked)
    def plot_ngrams(
        self,
        ngram: Union[INT, str] = "bigram",
        index: Optional[Union[INT, str, SEQUENCE_TYPES]] = None,
        show: INT = 10,
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "lower right",
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
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
            Number of contiguous words to search for (size of
            n-gram). Choose from: words (1), bigrams (2),
            trigrams (3), quadgrams (4).

        index: int, str, sequence or None, default=None
            Index names or positions of the documents in the corpus to
            include in the search. If None, it selects all documents in
            the dataset.

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

        figsize: tuple, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of n-grams shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
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

        def get_text(column: pd.Series) -> pd.Series:
            """Get the complete corpus as sequence of tokens.

            Parameters
            ----------
            column: pd.Series
                Column containing the corpus.

            Returns
            -------
            pd.Series
                Corpus of tokens.

            """
            if isinstance(column.iat[0], str):
                return column.apply(lambda row: row.split())
            else:
                return column

        corpus = get_corpus(self.X)
        rows = self.dataset.loc[self._get_rows(index, return_test=False)]

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
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            xlabel="Counts",
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_ngrams",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_qq(
        self,
        columns: Union[INT, str, slice, SEQUENCE_TYPES] = 0,
        distributions: Union[str, SEQUENCE_TYPES] = "norm",
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "lower right",
        figsize: Tuple[SCALAR, SCALAR] = (900, 600),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a quantile-quantile plot.

        Columns are distinguished by color and the distributions are
        distinguished by marker type.

        Parameters
        ----------
        columns: int, str, slice or sequence, default=0
            Slice, names or indices of the columns to plot. Selected
            categorical columns are ignored.

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
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

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
        columns = self._get_columns(columns)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()

        minimum, maximum = np.inf, -np.inf
        percentiles = np.linspace(0, 100, 101)
        for col in columns:
            # Drop missing values for compatibility with scipy.stats
            missing = self.missing + [np.inf, -np.inf]
            values = self.dataset[col].replace(missing, np.NaN).dropna()

            for dist in lst(distributions):
                stat = getattr(stats, dist)
                params = stat.fit(values)
                samples = stat.rvs(*params, size=101, random_state=self.random_state)

                if len(columns) > 1:
                    label = col
                    if len(lst(distributions)) > 1:
                        label += f" - {dist}"
                else:
                    label = dist

                fig.add_trace(
                    go.Scatter(
                        x=(qn_a := np.percentile(samples, percentiles)),
                        y=(qn_b := np.percentile(values, percentiles)),
                        mode="markers",
                        marker_symbol=self._fig.get_marker(dist),
                        line=dict(width=2, color=self._fig.get_color(col)),
                        name=label,
                        legendgroup=label,
                        showlegend=self._fig.showlegend(f"{col}-{dist}", legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                # Keep track of the min and max values for the diagonal line
                minimum = min(minimum, min(qn_a), min(qn_b))
                maximum = max(maximum, max(qn_a), max(qn_b))

        fig.add_trace(self._draw_line(xaxis, yaxis, x=(minimum, maximum)))

        return self._plot(
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            xlabel="Theoretical quantiles",
            ylabel="Observed quantiles",
            figsize=figsize or (900, 600),
            plotname="plot_qq",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_relationships(
        self,
        columns: Union[slice, SEQUENCE_TYPES] = [0, 1, 2],
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (900, 900),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot pairwise relationships in a dataset.

        Creates a grid of axes such that each numerical column appears
        once on the x-axes and once on the y-axes. The bottom triangle
        contains scatter plots (max 250 random samples), the diagonal
        plots contain column distributions, and the upper triangle
        contains contour histograms for all samples in the columns.

        Parameters
        ----------
        columns: slice or sequence, default=[0, 1, 2]
            Slice, names or indices of the columns to plot. Selected
            categorical columns are ignored.

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
            (.html, .png, .pdf, etc...). If `filename` has no period,
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
        columns = self._get_columns(columns, only_numerical=True)

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

            if x < len(columns) - 1:
                fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
            if y > 0:
                fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

            self._plot(
                axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                title=title if x == 0 and y == 1 else None,
                xlabel=columns[y] if x == len(columns) - 1 else None,
                ylabel=columns[x] if y == 0 else None,
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
        index: Optional[Union[INT, str, SEQUENCE_TYPES]] = None,
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (900, 600),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot a wordcloud from the corpus.

        The text for the plot is extracted from the column named
        `corpus`. If there is no column with that name, an exception
        is raised.

        Parameters
        ----------
        index: int, str, sequence or None, default=None
            Index names or positions of the documents in the corpus to
            include in the wordcloud. If None, it selects all documents
            in the dataset.

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
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for the [Wordlcoud][] object.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_ngrams

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

        corpus = get_corpus(self.X)
        rows = self.dataset.loc[self._get_rows(index, return_test=False)]

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
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            figsize=figsize or (900, 600),
            plotname="plot_wordcloud",
            filename=filename,
            display=display,
        )


class ModelPlot(BasePlot):
    """Model plots.

    These plots are accessible from the runners or from the models.
    If called from a runner, the `models` parameter has to be specified
    (if None, uses all models). If called from a model, that model is
    used and the `models` parameter becomes unavailable.

    """

    @available_if(has_task("binary"))
    @composed(crash, plot_from_model, typechecked)
    def plot_calibration(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        n_bins: INT = 10,
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "upper left",
        figsize: Tuple[SCALAR, SCALAR] = (900, 900),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
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
        Only available for binary classification tasks.

        !!! tip
            Use the [calibrate][adaboost-calibrate] method to calibrate
            the winning model.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        n_bins: int, default=10
            Number of bins used for calibration. Minimum of 5
            required.

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
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ModelPlot.plot_lift
        atom.plots:ModelPlot.plot_prc
        atom.plots:ModelPlot.plot_roc

        Examples
        --------

        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X = pd.read_csv("./examples/datasets/weatherAUS.csv")

        >>> atom = ATOMClassifier(X, y="RainTomorrow", n_rows=1e4)
        >>> atom.impute()
        >>> atom.encode()
        >>> atom.run(["LR", "RF"])
        >>> atom.plot_calibration()

        ```

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)

        if n_bins < 5:
            raise ValueError(
                "Invalid value for the n_bins parameter."
                f"Value should be >=5, got {n_bins}."
            )

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes(y=(0.31, 1.0))
        xaxis2, yaxis2 = self._fig.get_axes(y=(0.0, 0.29))
        for m in models:
            if hasattr(m.estimator, "decision_function"):
                prob = m.decision_function_test
                prob = (prob - prob.min()) / (prob.max() - prob.min())
            elif hasattr(m.estimator, "predict_proba"):
                prob = m.predict_proba_test.iloc[:, 1]

            # Get calibration (frac of positives and predicted values)
            frac_pos, pred = calibration_curve(self.y_test, prob, n_bins=n_bins)

            fig.add_trace(
                go.Scatter(
                    x=pred,
                    y=frac_pos,
                    mode="lines+markers",
                    line=dict(width=2, color=self._fig.get_color(m.name)),
                    name=m.name,
                    legendgroup=m.name,
                    showlegend=self._fig.showlegend(m.name, legend),
                    xaxis=xaxis2,
                    yaxis=yaxis,
                )
            )

            fig.add_trace(
                go.Histogram(
                    x=prob,
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

        fig.add_trace(self._draw_line(xaxis2, yaxis))

        fig.update_layout(
            {
                f"xaxis{xaxis2[1:]}_showgrid": True,
                f"xaxis{xaxis[1:]}_showticklabels": False,
                "barmode": "overlay",
            }
        )

        self._plot(
            axes=(f"xaxis{xaxis2[1:]}", f"yaxis{yaxis2[1:]}"),
            xlabel="Predicted value",
            ylabel="Count",
            xlim=(0, 1),
        )

        self._fig.used_models.extend(models)
        return self._plot(
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            ylabel="Fraction of positives",
            ylim=(-0.05, 1.05),
            figsize=figsize,
            plotname="plot_calibration",
            filename=filename,
            display=display,
        )

    @available_if(has_task("class"))
    @composed(crash, plot_from_model, typechecked)
    def plot_confusion_matrix(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        normalize: bool = False,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a model's confusion matrix.

        For one model, the plot shows a heatmap. For multiple models,
        it compares TP, FP, FN and TN in a barplot (not implemented
        for multiclass classification tasks). Only available for
        classification tasks.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        dataset: str, default="test"
            Data set on which to calculate the confusion matrix.
            Choose from:` "train", "test" or "holdout".

        normalize: bool, default=False
           Whether to normalize the matrix.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=None
            Figure's size, format as (x, y). If None, it adapts the
            size to the plot's type.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)

        dataset = dataset.lower()
        if dataset not in ("train", "test", "holdout"):
            raise ValueError(
                "Unknown value for the dataset parameter. "
                "Choose from: train, test or holdout."
            )
        if dataset == "holdout" and self.holdout is None:
            raise ValueError(
                "Invalid value for the dataset parameter. No holdout "
                "data set was specified when initializing the instance."
            )
        if self.task.startswith("multi") and len(models) > 1:
            raise NotImplementedError(
                "The plot_confusion_matrix method does not support the comparison"
                " of various models for multiclass classification tasks."
            )

        # Create dataframe to plot with barh if len(models) > 1
        df = pd.DataFrame(
            index=[
                "True negatives",
                "False positives",
                "False negatives",
                "True positives",
            ]
        )

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        for m in models:
            cm = confusion_matrix(
                getattr(m, f"y_{dataset}"), getattr(m, f"predict_{dataset}")
            )
            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            if len(models) == 1:  # Create matrix heatmap
                im = ax.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))

                # Create an axes on the right side of ax. The under of
                # cax are 5% of ax and the padding between cax and ax
                # are fixed at 0.3 inch.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.3)
                cbar = ax.figure.colorbar(im, cax=cax)
                ax.set(
                    xticks=np.arange(cm.shape[1]),
                    yticks=np.arange(cm.shape[0]),
                    xticklabels=m.mapping.get(m.target, m.y.sort_values().unique()),
                    yticklabels=m.mapping.get(m.target, m.y.sort_values().unique()),
                )

                # Loop over data dimensions and create text annotations
                fmt = ".2f" if normalize else "d"
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            x=j,
                            y=i,
                            s=format(cm[i, j], fmt),
                            ha="center",
                            va="center",
                            fontsize=self.tick_fontsize,
                            color="w" if cm[i, j] > cm.max() / 2.0 else "k",
                        )

                cbar.set_label(
                    label="Count",
                    fontsize=self.label_fontsize,
                    labelpad=15,
                    rotation=270,
                )
                cbar.ax.tick_params(labelsize=self.tick_fontsize)
                ax.grid(False)

            else:
                df[m.name] = cm.ravel()

        self._fig.used_models.extend(models)
        if len(models) > 1:
            df.plot.barh(ax=ax, width=0.6)
            figsize = figsize or (10, 6)
            self._plot(
                ax=ax,
                title=title,
                legend=("best", len(models)),
                xlabel="Count",
            )
        else:
            figsize = figsize or (8, 6)
            self._plot(
                ax=ax,
                title=title,
                xlabel="Predicted label",
                ylabel="True label",
            )

        return self._plot(
            fig=fig,
            figsize=figsize,
            plotname="plot_confusion_matrix",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @composed(crash, plot_from_model, typechecked)
    def plot_det(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "upper right",
        figsize: Tuple[SCALAR, SCALAR] = (900, 600),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the Detection Error Tradeoff curve.

        Read more about [DET][] in sklearn's documentation. Only
        available for binary classification tasks.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

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
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ModelPlot.plot_gains
        atom.plots:ModelPlot.plot_roc
        atom.plots:ModelPlot.plot_prc

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
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            for set_ in dataset:
                if hasattr(m.estimator, "predict_proba"):
                    y_pred = getattr(m, f"predict_proba_{set_}").iloc[:, 1]
                else:
                    y_pred = getattr(m, f"decision_function_{set_}")

                # Get fpr-fnr pairs for different thresholds
                fpr, fnr, _ = det_curve(getattr(m, f"y_{set_}"), y_pred)

                label = m.name + (f" - {set_}" if len(dataset) > 1 else "")
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=fnr,
                        mode="lines",
                        line=dict(
                            width=2,
                            color=self._fig.get_color(m.name),
                            dash=self._fig.get_dashes(set_),
                        ),
                        name=label,
                        legendgroup=label,
                        showlegend=self._fig.showlegend(label, legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._fig.used_models.extend(models)
        return self._plot(
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            xlabel="FPR",
            ylabel="FNR",
            figsize=figsize,
            plotname="plot_det",
            filename=filename,
            display=display,
        )

    @available_if(has_task("reg"))
    @composed(crash, plot_from_model, typechecked)
    def plot_errors(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a model's prediction errors.

        Plot the actual targets from a set against the predicted values
        generated by the regressor. A linear fit is made on the data.
        The gray, intersected line shows the identity line. This pot can
        be useful to detect noise or heteroscedasticity along a range of
        the target domain. Only available for regression tasks.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 6)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        for m in models:
            for set_ in dataset:
                r2 = f" (R$^2$={round(m.evaluate('r2', set_)['r2'], 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + r2
                ax.scatter(
                    x=getattr(self, f"y_{set_}"),
                    y=getattr(m, f"predict_{set_}"),
                    alpha=0.8,
                    label=label,
                )

                # Fit the points using linear regression
                from atom.models import OrdinaryLeastSquares

                model = OrdinaryLeastSquares(self, fast_init=True)._get_est()
                model.fit(
                    X=np.array(getattr(self, f"y_{set_}")).reshape(-1, 1),
                    y=getattr(m, f"predict_{set_}"),
                )

                # Draw the fit
                x = np.linspace(*ax.get_xlim(), 100)
                ax.plot(x, model.predict(x[:, np.newaxis]), lw=2, alpha=1)

        self._draw_line(ax=ax, y="diagonal")

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("best", len(models)),
            xlabel="True value",
            ylabel="Predicted value",
            figsize=figsize,
            plotname="plot_errors",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_evals(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot evaluation curves for the train and test set.

        Only available for models that allow [in-training validation][].

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name of the model to plot. If None, all models in the
            pipeline are selected.

        dataset: str, default="test"
            Data set on which to calculate the evaluation curves.
            Choose from: "train", "test" or "both".

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 6)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models, ensembles=False)
        dataset = self._get_set(dataset, allow_holdout=False)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        for m in models:
            if not m.evals:
                raise ValueError(
                    "Invalid value for the models parameter. Model "
                    f"{m.name} has no in-training validation."
                )

            for set_ in dataset:
                ax.plot(
                    range(len(m.evals[f"{self._metric[0].name}_{set_}"])),
                    m.evals[f"{self._metric[0].name}_{set_}"],
                    lw=2,
                    label=m.name + (f" - {set_}" if len(dataset) > 1 else ""),
                )

        if len(set(m.has_validation for m in models)) == 1:
            xlabel = m.has_validation
        else:
            xlabel = "Iterations"

        self._fig.used_models.append(m)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("best", len(dataset)),
            xlabel=xlabel,
            ylabel=self._metric[0].name,
            figsize=figsize,
            plotname="plot_evals",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_feature_importance(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        show: Optional[INT] = None,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a model's feature importance.

        The feature importance values are normalized in order to be
        able to compare them between models. Only available for models
        whose estimator has a `feature_importances_` or `coef` attribute.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        show: int or None, default=None
            Number of features (ordered by importance) to show.
            None to show all.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, default=None
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        show = self._get_show(show, models)

        # Create dataframe with columns as indices to plot with barh
        df = pd.DataFrame()
        for m in models:
            if (fi := get_feature_importance(m.estimator)) is None:
                raise ValueError(
                    f"Invalid value for the models parameter. The {m._fullname}'s "
                    f"estimator {m.estimator.__class__.__name__} has no "
                    f"feature_importances_ nor coef_ attribute."
                )

            # Normalize to be able to compare different models
            for col, fx in zip(m.features, fi):
                df.at[col, m.name] = fx / max(fi)

        # Select best and sort ascending (by sum of total importances)
        df = df.nlargest(show, columns=df.columns[-1])
        df = df.reindex(sorted(df.index, key=lambda i: df.loc[i].sum()))

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        ax = df.plot.barh(
            ax=ax,
            width=0.75 if len(models) > 1 else 0.6,
            legend=len(models) > 1,
        )
        if len(models) == 1:
            for i, v in enumerate(df[df.columns[0]]):
                ax.text(v + 0.01, i - 0.08, f"{v:.2f}", fontsize=self.tick_fontsize)

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("lower right", len(models)) if len(models) > 1 else None,
            xlim=(0, 1.03 if len(models) > 1 else 1.09),
            xlabel="Score",
            figsize=figsize or (10, 4 + show // 2),
            plotname="plot_feature_importance",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @composed(crash, plot_from_model, typechecked)
    def plot_gains(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "lower right",
        figsize: Tuple[SCALAR, SCALAR] = (900, 600),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the cumulative gains curve.

        Only available for binary classification tasks.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

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
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ModelPlot.plot_det
        atom.plots:ModelPlot.plot_lift
        atom.plots:ModelPlot.plot_roc

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
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            for set_ in dataset:
                y_true = getattr(m, f"y_{set_}")
                if hasattr(m.estimator, "predict_proba"):
                    y_pred = getattr(m, f"predict_proba_{set_}").iloc[:, 1]
                else:
                    y_pred = getattr(m, f"decision_function_{set_}")

                gains = np.cumsum(y_true.iloc[np.argsort(y_pred)[::-1]]) / y_true.sum()
                x = np.arange(start=1, stop=len(y_true) + 1) / len(y_true)

                label = m.name + (f" - {set_}" if len(dataset) > 1 else "")
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=gains,
                        mode="lines",
                        line=dict(
                            width=2,
                            color=self._fig.get_color(m.name),
                            dash=self._fig.get_dashes(set_),
                        ),
                        name=label,
                        legendgroup=label,
                        showlegend=self._fig.showlegend(label, legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        fig.add_trace(self._draw_line(xaxis, yaxis))

        self._fig.used_models.extend(models)
        return self._plot(
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            xlabel="Fraction of sample",
            ylabel="Gain",
            xlim=(0, 1),
            ylim=(0, 1.02),
            figsize=figsize,
            plotname="plot_gains",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_learning_curve(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        metric: Union[INT, str] = 0,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the learning curve: score vs number of training samples.

        Only use with models fitted using train sizing. Ensemble
        models are ignored.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        metric: int or str, default=0
            Index or name of the metric. Only for multi-metric runs.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 6)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models, ensembles=False)
        metric = self._get_metric(metric)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)

        # Prepare dataframes for seaborn lineplot (one df per line)
        # Not using sns hue parameter because of legend formatting
        lines = defaultdict(pd.DataFrame)
        for m in models:
            if m.bootstrap is None:
                values = {"x": [m._train_idx], "y": [get_best_score(m, metric)]}
            else:
                values = {
                    "x": [m._train_idx] * len(m.bootstrap),
                    "y": m.bootstrap.iloc[:, metric],
                }

            # Add the scores to the group's dataframe
            lines[m._group] = pd.concat([lines[m._group], pd.DataFrame(values)])

        for m, df in zip(models, lines.values()):
            df = df.reset_index(drop=True)
            sns.lineplot(data=df, x="x", y="y", marker="o", label=m.acronym, ax=ax)

        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("lower right", len(lines)),
            xlabel="Number of training samples",
            ylabel=self._metric[metric].name,
            figsize=figsize,
            plotname="plot_learning_curve",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @composed(crash, plot_from_model, typechecked)
    def plot_lift(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "upper right",
        figsize: Tuple[SCALAR, SCALAR] = (900, 600),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the lift curve.

        Only available for binary classification tasks.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

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
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ModelPlot.plot_det
        atom.plots:ModelPlot.plot_gains
        atom.plots:ModelPlot.plot_prc

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
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            for set_ in dataset:
                y_true = getattr(m, f"y_{set_}")
                if hasattr(m.estimator, "predict_proba"):
                    y_pred = getattr(m, f"predict_proba_{set_}").iloc[:, 1]
                else:
                    y_pred = getattr(m, f"decision_function_{set_}")

                gains = np.cumsum(y_true.iloc[np.argsort(y_pred)[::-1]]) / y_true.sum()
                x = np.arange(start=1, stop=len(y_true) + 1) / len(y_true)

                label = m.name + (f" - {set_}" if len(dataset) > 1 else "")
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=gains / x,
                        mode="lines",
                        line=dict(
                            width=2,
                            color=self._fig.get_color(m.name),
                            dash=self._fig.get_dashes(set_),
                        ),
                        name=label,
                        legendgroup=label,
                        showlegend=self._fig.showlegend(label, legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        fig.add_trace(self._draw_line(xaxis, yaxis, y=1))

        self._fig.used_models.extend(models)
        return self._plot(
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            xlabel="Fraction of sample",
            ylabel="Lift",
            xlim=(0, 1),
            figsize=figsize,
            plotname="plot_lift",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_parshap(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        columns: Optional[Union[INT, str, SEQUENCE_TYPES]] = None,
        target: Union[INT, str] = 1,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the partial correlation of shap values.

        Plots the train and test correlation between the shap value of
        every feature with its target value, after removing the effect
        of all other features (partial correlation). This plot is
        useful to identify the features that are contributing most to
        overfitting. Features that lie below the bisector (diagonal
        line) performed worse on the test set than on the training set.
        If the estimator has a `feature_importances_` or `coef_`
        attribute, its normalized values are shown in a color map.

        Idea from: https://towardsdatascience.com/which-of-your-features
        -are-overfitting-c46d0762e769

        Code snippets from: https://github.com/raphaelvallat/pingouin/
        blob/master/pingouin/correlation.py

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        columns: int, str, sequence or None, default=None
            Names or indices of the features to plot. None to show all.

        target: int or str, default=1
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 6)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        target = self._get_target(target)

        fxs_importance = {}
        markers = cycle(["o", "^", "s", "p", "D", "H", "p", "*"])

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        for m in models:
            fxs = self._get_columns(columns, include_target=False, branch=m.branch)
            marker = next(markers)

            parshap = {}
            for set_ in ("train", "test"):
                X, y = getattr(m, f"X_{set_}"), getattr(m, f"y_{set_}")
                data = pd.concat([X, y], axis=1)

                # Calculating shap values is computationally expensive,
                # therefore select a random subsample for large data sets
                if len(data) > 500:
                    data = data.sample(500, random_state=self.random_state)

                # Replace data with the calculated shap values
                data.iloc[:, :-1] = m._shap.get_shap_values(data.iloc[:, :-1], target)

                parshap[set_] = pd.Series(index=data.columns[:-1], dtype=float)
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
                    parshap[set_][fx] = semi_partial_correlation[1, 0]

            # Get the feature importance or coefficients
            fi = get_feature_importance(m.estimator)
            if fi is not None:
                fi = pd.Series(fi, index=m.features, dtype=float)
                fxs_importance[m.name] = fi.sort_values()[fxs]

            sns.scatterplot(
                x=parshap["train"],
                y=parshap["test"],
                marker=marker,
                s=50,
                hue=fxs_importance.get(m.name, None),
                palette="Reds",
                legend=False,
                ax=ax,
            )

            # Add hidden point for nice legend
            if len(models) > 1:
                ax.scatter(
                    x=parshap["train"][0],
                    y=parshap["test"][0],
                    marker=marker,
                    s=20,
                    zorder=-2,
                    color="k",
                    label=m.name,
                )

            # Calculate offset for feature names (5% of height)
            offset = .05 * (ax.get_ylim()[1] - ax.get_ylim()[0])

            # Add feature names above the markers
            for txt in parshap["train"].index:
                xy = (parshap["train"][txt], parshap["test"][txt] + offset / 2)
                ax.annotate(txt, xy, ha="center", va="bottom")

        # Draw color bar if feature importances are defined
        if fxs_importance:
            cbar = plt.colorbar(
                mappable=plt.cm.ScalarMappable(plt.Normalize(0, 1), cmap="Reds"),
                ax=ax,
            )
            cbar.set_label(
                label="Normalized feature importance",
                labelpad=15,
                fontsize=self.label_fontsize,
                rotation=270,
            )
            cbar.ax.tick_params(labelsize=self.tick_fontsize)

        self._draw_line(ax=ax, y="diagonal")

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("best", len(models)) if len(models) > 1 else None,
            xlabel="Training set",
            ylabel="Test set",
            figsize=figsize,
            plotname="plot_parshap",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_partial_dependence(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        columns: Optional[Union[INT, str, SEQUENCE_TYPES]] = None,
        kind: str = "average",
        target: Union[INT, str] = 1,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the partial dependence of features.

        The partial dependence of a feature (or a set of features)
        corresponds to the response of the model for each possible
        value of the feature. Two-way partial dependence plots are
        plotted as contour plots (only allowed for single model plots).
        The deciles of the feature values are shown with tick marks
        on the x-axes for one-way plots, and on both axes for two-way
        plots.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        columns: int, str, sequence or None, default=None
            Features or feature pairs (name or index) to get the partial
            dependence from. Maximum of 3 allowed. If None, it uses the
            best 3 features if the `feature_importance` attribute is
            defined, else it uses the first 3 features in the dataset.

        kind: str, default="average"
            - "average": Plot the partial dependence averaged across
                         all the samples in the dataset.
            - "individual": Plot the partial dependence per sample
                            (Individual Conditional Expectation).
            - "both": Plot both the average (as a thick line) and the
                      individual (thin lines) partial dependence.

            This parameter is ignored when plotting feature pairs.

        target: int or str, default=1
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 6)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """

        def get_features(features, m):
            """Select feature list from provided columns."""
            # Default is to select the best or the first 3 features
            if not features:
                if (fxs := m.feature_importance) is not None:
                    features = list(fxs.index[:3])
                else:
                    features = list(m.features[:3])

            features = lst(features)
            if len(features) > 3:
                raise ValueError(
                    "Invalid value for the columns parameter. "
                    f"Maximum 3 allowed, got {len(features)}."
                )

            # Convert features into a sequence of int tuples
            cols = []
            for fxs in features:
                if isinstance(fxs, (int, str)):
                    cols.append((self._get_columns(fxs, False, branch=m.branch)[0],))
                elif len(models) == 1:
                    if len(fxs) == 2:
                        add = []
                        for fx in fxs:
                            add.append(self._get_columns(fx, False, branch=m.branch)[0])
                        cols.append(tuple(add))
                    else:
                        raise ValueError(
                            "Invalid value for the columns parameter. Features "
                            f"should be single or in pairs, got {fxs}."
                        )
                else:
                    raise ValueError(
                        "Invalid value for the columns parameter. Feature pairs "
                        f"are invalid when plotting multiple models, got {fxs}."
                    )
            return cols

        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        target = self._get_target(target) if self.task.startswith("multi") else 0
        palette = cycle(sns.color_palette())

        if kind.lower() not in ("average", "individual", "both"):
            raise ValueError(
                f"Invalid value for the kind parameter, got {kind}. "
                "Choose from: average, individual, both."
            )

        axes = []
        fig = self._get_figure()
        n_cols = 3 if not columns else len(lst(columns))
        gs = GridSpecFromSubplotSpec(1, n_cols, self._fig.grid)
        for i in range(n_cols):
            axes.append(fig.add_subplot(gs[0, i]))

        names = []  # Names of the features (to compare between models)
        for m in models:
            color = next(palette)

            # Since every model can have different fxs, select them again
            cols = get_features(columns, m)

            # Make sure the models use the same features
            if len(models) > 1:
                if not names:
                    names = cols
                elif names != cols:
                    raise ValueError(
                        "Invalid value for the columns parameter. Not all "
                        f"models use the same features, got {names} and {cols}."
                    )

            # Compute averaged predictions
            pd_results = Parallel(n_jobs=self.n_jobs)(
                delayed(partial_dependence)(
                    estimator=m.estimator,
                    X=m.X_test,
                    features=[list(m.features).index(c) for c in col],
                ) for col in cols
            )

            # Get global min and max average predictions of PD grouped by plot type
            pdp_lim = {}
            for avg_pred, pred, values in pd_results:
                min_pd, max_pd = avg_pred[target].min(), avg_pred[target].max()
                old_min, old_max = pdp_lim.get(len(values), (min_pd, max_pd))
                pdp_lim[len(values)] = (min(min_pd, old_min), max(max_pd, old_max))

            deciles = {}
            for fx in chain.from_iterable(cols):
                if fx not in deciles:  # Skip if the feature is repeated
                    X_col = _safe_indexing(m.X_test, fx, axis=1)
                    deciles[fx] = mquantiles(X_col, prob=np.arange(0.1, 1.0, 0.1))

            for axi, fx, (avg_pred, pred, values) in zip(axes, cols, pd_results):
                # For both types: draw ticks on the horizontal axis
                trans = blended_transform_factory(axi.transData, axi.transAxes)
                axi.vlines(deciles[fx[0]], 0, 0.05, transform=trans, color="k")
                axi.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
                self._plot(ax=axi, xlabel=fx[0])

                # Draw line or contour plot
                if len(values) == 1:
                    # Draw the mean of the individual lines
                    if kind.lower() in ("average", "both"):
                        axi.plot(
                            values[0],
                            avg_pred[target].ravel(),
                            lw=2,
                            color=color,
                            label=m.name,
                        )

                    # Draw all individual (per sample) lines (ICE)
                    if kind.lower() in ("individual", "both"):
                        # Select up to 100 random samples to plot
                        idx = np.random.choice(
                            list(range(len(pred[target]))),
                            size=min(len(pred[target]), 100),
                            replace=False,
                        )
                        for sample in pred[target, idx, :]:
                            axi.plot(values[0], sample, lw=0.5, alpha=0.5, color=color)

                else:
                    # Create contour levels for two-way plots
                    levels = np.linspace(pdp_lim[2][0], pdp_lim[2][1] + 1e-9, num=8)

                    # Draw contour plot
                    XX, YY = np.meshgrid(values[0], values[1])
                    Z = avg_pred[target].T
                    CS = axi.contour(XX, YY, Z, levels=levels, linewidths=0.5)
                    axi.clabel(CS, fmt="%2.2f", colors="k", fontsize=10, inline=True)
                    axi.contourf(
                        XX,
                        YY,
                        Z,
                        levels=levels,
                        vmax=levels[-1],
                        vmin=levels[0],
                        alpha=0.75,
                    )

                    # Draw the ticks on the vertical axis
                    trans = blended_transform_factory(axi.transAxes, axi.transData)
                    axi.hlines(deciles[fx[1]], 0, 0.05, transform=trans, color="k")

                    self._plot(
                        ax=axi,
                        ylabel=fx[1],
                        xlim=(min(XX.flatten()), max(XX.flatten())),
                        ylim=(min(YY.flatten()), max(YY.flatten())),
                    )

        # Place y-label and legend on first non-contour plot
        for axi in axes:
            if not axi.get_ylabel():
                self._plot(
                    ax=axi,
                    ylabel="Score",
                    legend=("best", len(models)) if len(models) > 1 else None,
                )
                break

        if title:
            # Place title if not in canvas, else above first or middle image
            if len(cols) == 1 or (len(cols) == 2 and self._fig.is_canvas):
                axes[0].set_title(title, fontsize=self.title_fontsize, pad=20)
            elif len(cols) == 3:
                axes[1].set_title(title, fontsize=self.title_fontsize, pad=20)
            elif not self._fig.is_canvas:
                plt.suptitle(title, fontsize=self.title_fontsize)

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            figsize=figsize,
            plotname="plot_partial_dependence",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_permutation_importance(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        show: Optional[INT] = None,
        n_repeats: INT = 10,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the feature permutation importance of models.

        Calculating permutations can be time-consuming, especially
        if `n_repeats` is high. For this reason, the permutations
        are stored under the `permutations` attribute. If the plot
        is called again for the same model with the same `n_repeats`,
        it will use the stored values, making the method considerably
        faster.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        show: int or None, default=None
            Number of features (ordered by importance) to show.
            None to show all.

        n_repeats: int, default=10
            Number of times to permute each feature.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, default=None
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        show = self._get_show(show, models)

        if n_repeats <= 0:
            raise ValueError(
                "Invalid value for the n_repeats parameter."
                f"Value should be >0, got {n_repeats}."
            )

        rows = []
        for m in models:
            # If permutations are already calculated and n_repeats is
            # same, use known permutations (for efficient re-plotting)
            if (
                not hasattr(m, "permutations")
                or m.permutations.importances.shape[1] == n_repeats
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

            # Append permutation scores to the dataframe
            for i, feature in enumerate(m.features):
                for score in m.permutations.importances[i, :]:
                    rows.append({"features": feature, "score": score, "model": m.name})

        # Get the column names sorted by sum of scores
        df = pd.DataFrame(rows)
        get_idx = df.groupby("features", as_index=False)["score"].sum()
        get_idx = get_idx.sort_values("score", ascending=False)
        column_order = get_idx["features"].values[:show]

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        sns.boxplot(
            x="score",
            y="features",
            hue="model",
            data=df,
            ax=ax,
            order=column_order,
            width=0.75 if len(models) > 1 else 0.6,
        )

        ax.yaxis.label.set_visible(False)
        if len(models) > 1:
            # Remove seaborn's legend title
            handles, labels = ax.showlegend_handles_labels()
            ax.legend(handles=handles[1:], labels=labels[1:])
        else:
            # Hide the legend created by seaborn
            ax.legend().set_visible(False)

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("lower right" if len(models) > 1 else False, len(models)),
            xlabel="Score",
            figsize=figsize or (10, 4 + show // 2),
            plotname="plot_permutation_importance",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_pipeline(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        draw_hyperparameter_tuning: bool = True,
        color_branches: Optional[bool] = None,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a diagram of the pipeline.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models for which to draw the pipeline.
            If None, all pipelines are plotted.

        draw_hyperparameter_tuning: bool, default=True
            Whether to draw if the models used Hyperparameter Tuning.

        color_branches: bool or None, default=None
            Whether to draw every branch in a different color. If None,
            branches are colored when there is more than one.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, default=None
            Figure's size, format as (x, y). If None, it adapts the
            size to the pipeline drawn.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

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

        models = self._get_subclass(models)
        palette = cycle(sns.color_palette())

        # Define branches to plot
        branches = []
        for branch in self._branches.min("og").values():
            draw_models, draw_ensembles = [], []
            for m in models:
                if m.name in branch._get_depending_models():
                    if m.acronym not in ("Stack", "Vote"):
                        draw_models.append(m)
                    else:
                        draw_ensembles.append(m)

                        # Additionally, add all dependent models (if not already there)
                        draw_models.extend(
                            [i for i in m._models.values() if i not in draw_models]
                        )

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
        colors = {}
        for branch in branches:
            if color_branches or (color_branches is None and len(branches) > 1):
                colors[branch["name"]] = branch["color"] = next(palette)
            else:
                branch["color"] = "black"

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        sns.set_style("white")  # Only for this plot

        # Create schematic drawing
        d = Drawing(unit=1, backend="matplotlib")
        d.config(fontsize=self.label_fontsize)
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
                y_pos = [positions[id(m)][1] for m in model._models.values()]
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
                for m in model._models.values():
                    d.here = positions[id(m)]
                    add_wire(max_pos + length, y)

        bbox = d.get_bbox()
        figure = d.draw(ax=ax, showframe=False, show=False)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        plt.axis("off")

        # Draw lines for legend (outside plot)
        for k, v in colors.items():
            plt.plot((-9e9, -9e9), (-9e9, -9e9), color=v, lw=2, zorder=-2, label=k)

        self._fig.used_models.extend(models)
        return self._plot(
            fig=figure.fig,
            ax=figure.ax,
            title=title,
            xlim=xlim,
            ylim=(ylim[0] - 2, ylim[1] + 2),
            legend=("upper left", 6) if colors else None,
            figsize=figsize or (bbox.xmax // 4, 2.5 + (bbox.ymax - bbox.ymin + 1) // 4),
            plotname="plot_pipeline",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @composed(crash, plot_from_model, typechecked)
    def plot_prc(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "upper right",
        figsize: Tuple[SCALAR, SCALAR] = (900, 600),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the precision-recall curve.

        Read more about [PRC][] in sklearn's documentation. Only
        available for binary classification tasks.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

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
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ModelPlot.plot_det
        atom.plots:ModelPlot.plot_lift
        atom.plots:ModelPlot.plot_roc

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
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            for set_ in dataset:
                if hasattr(m.estimator, "predict_proba"):
                    y_pred = getattr(m, f"predict_proba_{set_}").iloc[:, 1]
                else:
                    y_pred = getattr(m, f"decision_function_{set_}")

                # Get precision-recall pairs for different thresholds
                prec, rec, _ = precision_recall_curve(getattr(m, f"y_{set_}"), y_pred)

                label = m.name + (f" - {set_}" if len(dataset) > 1 else "")
                fig.add_trace(
                    go.Scatter(
                        x=rec,
                        y=prec,
                        mode="lines",
                        line=dict(
                            width=2,
                            color=self._fig.get_color(m.name),
                            dash=self._fig.get_dashes(set_),
                        ),
                        name=label,
                        legendgroup=label,
                        showlegend=self._fig.showlegend(label, legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        fig.add_trace(self._draw_line(xaxis, yaxis, y=sum(m.y_test) / len(m.y_test)))

        self._fig.used_models.extend(models)
        return self._plot(
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            xlabel="Recall",
            ylabel="Precision",
            figsize=figsize,
            plotname="plot_prc",
            filename=filename,
            display=display,
        )

    @available_if(has_task("class"))
    @composed(crash, plot_from_model, typechecked)
    def plot_probabilities(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        target: Union[INT, str] = 1,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the probability distribution of the target classes.

        Only available for classification tasks.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        target: int or str, default=1
            Probability of being that class in the target column
            (as index or name). Only for multiclass classification.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 6)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][]
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)
        target = self._get_target(target)
        check_predict_proba(models, "plot_probabilities")
        palette = cycle(sns.color_palette())

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        for m in models:
            for set_ in dataset:
                for value in m.y.sort_values().unique():
                    # Get indices per class
                    idx = np.where(getattr(m, f"y_{set_}") == value)[0]

                    label = m.name + (f" - {set_}" if len(dataset) > 1 else "")
                    sns.histplot(
                        data=getattr(m, f"predict_proba_{set_}").iloc[idx, target],
                        kde=True,
                        bins=50,
                        label=label + f" ({self.target}={value})",
                        color=next(palette),
                        ax=ax,
                    )

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("best", len(models)),
            xlabel="Probability",
            ylabel="Counts",
            xlim=(0, 1),
            figsize=figsize,
            plotname="plot_probabilities",
            filename=filename,
            display=display,
        )

    @available_if(has_task("reg"))
    @composed(crash, plot_from_model, typechecked)
    def plot_residuals(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a model's residuals.

        The plot shows the residuals (difference between the predicted
        and the true value) on the vertical axis and the independent
        variable on the horizontal axis. The gray, intersected line
        shows the identity line. This plot can be useful to analyze the
        variance of the error of the regressor. If the points are
        randomly dispersed around the horizontal axis, a linear
        regression model is appropriate for the data; otherwise, a
        non-linear model is more appropriate. Only available for
        regression tasks.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 6)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        gs = GridSpecFromSubplotSpec(1, 4, self._fig.grid, wspace=0.05)
        ax1 = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:4])
        for m in models:
            for set_ in dataset:
                r2 = f" (R$^2$={round(m.evaluate('r2', set_)['r2'], 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + r2
                res = np.subtract(
                    getattr(m, f"predict_{set_}"),
                    getattr(self, f"y_{set_}"),
                )

                ax1.scatter(getattr(m, f"predict_{set_}"), res, alpha=0.7, label=label)
                ax2.hist(res, orientation="horizontal", histtype="step", linewidth=1.2)

        ax2.set_yticklabels([])
        self._draw_line(ax=ax2, y=0)
        self._plot(ax=ax2, xlabel="Distribution")

        if title:
            if not self._fig.is_canvas:
                plt.suptitle(title, fontsize=self.title_fontsize, y=0.98)
            else:
                ax1.set_title(title, fontsize=self.title_fontsize, pad=20)

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax1,
            legend=("lower right", len(models)),
            ylabel="Residuals",
            xlabel="True value",
            figsize=figsize,
            plotname="plot_residuals",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_results(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        metric: Union[INT, str] = 0,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot of the model results after the evaluation.

        If all models applied bootstrap, the plot is a boxplot. If
        not, the plot is a barplot. Models are ordered based on
        their score from the top down. The score is either the
        `score_bootstrap` or `score_test` attribute of the model,
        selected in that order.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        metric: int or str, default=0
            Index or name of the metric (only for multi-metric runs).
            Other available metrics are "time_bo", "time_fit",
            "time_bootstrap" and "time".

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=None
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of models shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """

        def get_metric(m, metric):
            """Get the metric or the timing attribute."""
            if isinstance(metric, str):
                if getattr(m, metric):
                    return getattr(m, metric)
                else:
                    raise ValueError(
                        "Invalid value for the metric parameter. "
                        f"Model {m.name} doesn't have metric {metric}."
                    )
            else:
                return get_best_score(m, metric)

        def std(m):
            """Get the standard deviation of the bootstrap results."""
            if m.bootstrap is not None:
                return m.bootstrap.iloc[:, metric].std()
            else:
                return 0

        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        metric = self._get_metric(metric)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)

        names = []
        models = sorted(models, key=lambda m: get_metric(m, metric))
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]  # First color

        all_bootstrap = all(m.score_bootstrap for m in models)
        for i, m in enumerate(models):
            names.append(m.name)
            if isinstance(metric, str):
                ax.barh(y=i, width=getattr(m, metric), height=0.5, color=color)
            elif all_bootstrap:
                ax.boxplot(
                    x=[m.bootstrap.iloc[:, metric]],
                    vert=False,
                    positions=[i],
                    widths=0.5,
                    boxprops=dict(color=color),
                )
            else:
                ax.barh(
                    y=i,
                    width=get_best_score(m, metric),
                    height=0.5,
                    xerr=std(m),
                    color=color,
                )

        min_lim = 0.9 * (get_metric(models[0], metric) - std(models[0]))
        max_lim = 1.05 * (get_metric(models[-1], metric) + std(models[-1]))
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(names)

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            xlabel=self._metric[metric].name if isinstance(metric, int) else "time (s)",
            xlim=(min_lim, max_lim) if not all_bootstrap else None,
            figsize=figsize or (10, 4 + len(models) // 2),
            plotname="plot_results",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @composed(crash, plot_from_model, typechecked)
    def plot_roc(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        *,
        title: Optional[Union[str, dict]] = None,
        legend: Optional[Union[str, dict]] = "lower right",
        figsize: Tuple[SCALAR, SCALAR] = (900, 600),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the Receiver Operating Characteristics curve.

        Read more about [ROC][] in sklearn's documentation. Only
        available for binary classification tasks.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

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
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ModelPlot.plot_gains
        atom.plots:ModelPlot.plot_lift
        atom.plots:ModelPlot.plot_prc

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
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        xaxis, yaxis = self._fig.get_axes()
        for m in models:
            for set_ in dataset:
                if hasattr(m.estimator, "predict_proba"):
                    y_pred = getattr(m, f"predict_proba_{set_}").iloc[:, 1]
                else:
                    y_pred = getattr(m, f"decision_function_{set_}")

                # Get False (True) Positive Rate as arrays
                fpr, tpr, _ = roc_curve(getattr(m, f"y_{set_}"), y_pred)

                label = m.name + (f" - {set_}" if len(dataset) > 1 else "")
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        line=dict(
                            width=2,
                            color=self._fig.get_color(m.name),
                            dash=self._fig.get_dashes(set_),
                        ),
                        name=label,
                        legendgroup=label,
                        showlegend=self._fig.showlegend(label, legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        fig.add_trace(self._draw_line(xaxis, yaxis))

        self._fig.used_models.extend(models)
        return self._plot(
            axes=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            xlim=(-0.05, 1.05),
            ylim=(-0.05, 1.05),
            xlabel="FPR",
            ylabel="TPR",
            figsize=figsize,
            plotname="plot_roc",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_successive_halving(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        metric: Union[INT, str] = 0,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot scores per iteration of the successive halving.

        Only use if the models were fitted using successive_halving.
        Ensemble models are ignored.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        metric: int or str, default=0
            Index or name of the metric. Only for multi-metric runs.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 6)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models, ensembles=False)
        metric = self._get_metric(metric)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)

        # Prepare dataframes for seaborn lineplot (one df per line)
        # Not using sns hue parameter because of legend formatting
        lines = defaultdict(pd.DataFrame)
        for m in models:
            n_models = len(m.branch._idx[0]) // m._train_idx  # Number of models in iter
            if m.bootstrap is None:
                values = {"x": [n_models], "y": [get_best_score(m, metric)]}
            else:
                values = {
                    "x": [n_models] * len(m.bootstrap),
                    "y": m.bootstrap.iloc[:, metric],
                }

            # Add the scores to the group's dataframe
            lines[m._group] = pd.concat([lines[m._group], pd.DataFrame(values)])

        for m, df in zip(models, lines.values()):
            df = df.reset_index(drop=True)
            kwargs = dict(err_style="band" if df["x"].nunique() > 1 else "bars", ax=ax)
            sns.lineplot(data=df, x="x", y="y", marker="o", label=m.acronym, **kwargs)

        n_models = [len(self.train) // m._train_idx for m in models]
        ax.set_xlim(max(n_models) + 0.1, min(n_models) - 0.1)
        ax.set_xticks(range(1, max(n_models) + 1))

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("lower right", len(lines)),
            xlabel="n_models",
            ylabel=self._metric[metric].name,
            figsize=figsize,
            plotname="plot_successive_halving",
            filename=filename,
            display=display,
        )

    @available_if(has_task("binary"))
    @composed(crash, plot_from_model, typechecked)
    def plot_threshold(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        steps: INT = 100,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot metric performances against threshold values.

        Only available for binary classification tasks.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name or index of the models to plot. If None, all models
            are selected.

        metric: str, func, scorer, sequence or None, default=None
            Metric to plot. Choose from any of sklearn's scorers, a
            function with signature `metric(y_true, y_pred)`, a scorer
            object or a sequence of these. If None, the metric used
            to run the pipeline is plotted.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        steps: int, default=100
            Number of thresholds measured.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 6)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)
        check_predict_proba(models, "plot_threshold")

        # Get all metric functions from the input
        if metric is None:
            metric_list = [m._score_func for m in self._metric.values()]
        else:
            metric_list = [get_custom_scorer(m)._score_func for m in lst(metric)]

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        steps = np.linspace(0, 1, steps)
        for m in models:
            for met in metric_list:
                for set_ in dataset:
                    results = []
                    for step in steps:
                        pred = getattr(m, f"predict_proba_{set_}").iloc[:, 1] >= step
                        results.append(met(getattr(m, f"y_{set_}"), pred))

                    if len(models) == 1:
                        l_set = f"{set_} - " if len(dataset) > 1 else ""
                        label = f"{l_set}{met.__name__}"
                    else:
                        l_set = f" - {set_}" if len(dataset) > 1 else ""
                        label = f"{m.name}{l_set} ({met.__name__})"
                    ax.plot(steps, results, label=label, lw=2)

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("best", len(models)),
            xlabel="Threshold",
            ylabel="Score",
            figsize=figsize,
            plotname="plot_threshold",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_trials(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        metric: Union[INT, str] = 0,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 8),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the hyperparameter tuning trials.

        Only available for models that ran [hyperparameter tuning][].
        Creates a figure with two plots: the first plot shows the score
        of every trial and the second shows the distance between the
        last consecutive steps. This is the same plot as produced by
        `ht_params={"plot": True}`.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name of the models to plot. If None, all models that
            used hyperparameter tuning are selected.

        metric: int or str, default=0
            Index or name of the metric. Only for multi-metric runs.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 8)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        metric = self._get_metric(metric)

        # Check there is at least one model that run hyperparameter tuning
        if all(m.trials is None for m in models):
            raise PermissionError(
                "The plot_trials method is only available for "
                "models that ran hyperparameter tuning!"
            )

        fig = self._get_figure()
        gs = GridSpecFromSubplotSpec(4, 1, self._fig.grid, hspace=0.05)
        ax1 = fig.add_subplot(gs[0:3, 0])
        ax2 = plt.subplot(gs[3:4, 0], sharex=ax1)
        for m in models:
            if m.score_ht:  # Only models that did run hyperparameter tuning
                y = m.trials["score"].apply(lambda value: lst(value)[metric])
                if len(models) == 1:
                    label = f"Score={round(lst(m.score_ht)[metric], 3)}"
                else:
                    label = f"{m.name} (Score={round(lst(m.score_ht)[metric], 3)})"

                # Draw bullets on all markers except the maximum
                markers = [i for i in range(len(m.trials))]
                markers.remove(int(np.argmax(y)))

                ax1.plot(range(len(y)), y, "-o", markevery=markers, label=label)
                ax2.plot(range(1, len(y)), np.abs(np.diff(y)), "-o")
                ax1.scatter(np.argmax(y), max(y), zorder=10, s=100, marker="*")

        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_xticks(range(max(len(m.trials) for m in models)))

        self._plot(
            ax=ax1,
            title=title,
            legend=("best", len(models)),
            ylabel=self._metric[metric].name,
        )

        self._fig.used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax2,
            xlabel="Trial",
            ylabel="d",
            figsize=figsize,
            plotname="plot_trials",
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

    @composed(crash, plot_from_model, typechecked)
    def plot_shap_bar(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        index: Optional[Union[INT, str, SEQUENCE_TYPES]] = None,
        show: Optional[INT] = None,
        target: Union[INT, str] = 1,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot SHAP's bar plot.

        Create a bar plot of a set of SHAP values. If a single sample
        is passed, then the SHAP values are plotted. If many samples
        are passed, then the mean absolute value for each feature
        column is plotted.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name of the model to plot. If None, all models are selected.
            Note that leaving the default option could raise an exception
            if there are multiple models. To avoid this, call the plot
            from a model, `atom.xgb.bar_plot()`.

        index: int, str, sequence or None, default=None
            Index names or positions of the rows in the dataset to
            plot. If None, it selects all rows in the test set.

        show: int or None, default=None
            Number of features (ordered by importance) to show.
            None to show all.

        target: int or str, default=1
            Index or name of the class in the target column to
            look at. Only for multi-class classification tasks.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, default=None
            Figure's size, format as (x, y). If None, it adapts
            the size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's bar plot.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        show = self._get_show(show, m)
        target = self._get_target(target)
        explanation = m._shap.get_explanation(rows, target)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        shap.plots.bar(explanation, max_display=show, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)

        self._fig.used_models.append(m)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            figsize=figsize or (10, 4 + show // 2),
            plotname="bar_plot",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_shap_beeswarm(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        index: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        show: Optional[INT] = None,
        target: Union[INT, str] = 1,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot SHAP's beeswarm plot.

        The plot is colored by feature values.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name of the model to plot. If None, all models are selected.
            Note that leaving the default option could raise an exception
            if there are multiple models. To avoid this, call the plot
            from a model, `atom.xgb.beeswarm_plot()`.

        index: tuple, slice or None, default=None
            Index names or positions of the rows in the dataset to plot.
            If None, it selects all rows in the test set. The beeswarm
            plot does not support plotting a single sample.

        show: int or None, default=None
            Number of features (ordered by importance) to show. None
            to show all.

        target: int or str, default=1
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, default=None
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's beeswarm plot.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        show = self._get_show(show, m)
        target = self._get_target(target)
        explanation = m._shap.get_explanation(rows, target)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        shap.plots.beeswarm(explanation, max_display=show, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)

        self._fig.used_models.append(m)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            figsize=figsize or (10, 4 + show // 2),
            plotname="beeswarm_plot",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_shap_decision(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        index: Optional[Union[INT, str, SEQUENCE_TYPES]] = None,
        show: Optional[INT] = None,
        target: Union[INT, str] = 1,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot SHAP's decision plot.

        Visualize model decisions using cumulative SHAP values. Each
        plotted line explains a single model prediction. If a single
        prediction is plotted, feature values are printed in the
        plot (if supplied). If multiple predictions are plotted
        together, feature values will not be printed. Plotting too
        many predictions together will make the plot unintelligible.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name of the model to plot. If None, all models are selected.
            Note that leaving the default option could raise an exception
            if there are multiple models. To avoid this, call the plot
            from a model, `atom.xgb.decision_plot()`.

        index: int, str, sequence or None, default=None
            Index names or positions of the rows in the dataset to plot.
            If None, it selects all rows in the test set.

        show: int or None, default=None
            Number of features (ordered by importance) to show. None
            to show all.

        target: int or str, default=1
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=None
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's decision plot.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        show = self._get_show(show, m)
        target = self._get_target(target)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        shap.decision_plot(
            base_value=m._shap.get_expected_value(target),
            shap_values=m._shap.get_shap_values(rows, target),
            features=rows,
            feature_display_range=slice(-1, -show - 1, -1),
            auto_size_plot=False,
            show=False,
            **kwargs,
        )

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)

        self._fig.used_models.append(m)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            figsize=figsize or (10, 4 + show // 2),
            plotname="decision_plot",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_shap_force(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        index: Optional[Union[INT, str, SEQUENCE_TYPES]] = None,
        target: Union[INT, str] = 1,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (14, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot SHAP's force plot.

        Visualize the given SHAP values with an additive force layout.
        Note that by default this plot will render using javascript.
        For a regular figure use `matplotlib=True` (this option is
        only available when only a single sample is plotted).

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name of the model to plot. If None, all models are selected.
            Note that leaving the default option could raise an exception
            if there are multiple models. To avoid this, call the plot
            from a model,`atom.xgb.force_plot()`.

        index: int, str, sequence or None, default=None
            Index names or positions of the rows in the dataset to plot.
            If None, it selects all rows in the test set.

        target: int or str, default=1
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(14, 6)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Name of the file. If matplotlib=False, the figure will
            be saved as a html file. If None, the figure is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's force plot.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only if `display=None` and `matplotlib=True`.

        """
        if getattr(self._fig, "is_canvas", None):
            raise PermissionError(
                "The force_plot method can not be called from a canvas "
                "because of incompatibility between the ATOM and shap API."
            )

        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        target = self._get_target(target)

        self._get_figure(create_figure=False)
        sns.set_style("white")  # Only for this plot
        plot = shap.force_plot(
            base_value=m._shap.get_expected_value(target),
            shap_values=m._shap.get_shap_values(rows, target),
            features=rows,
            figsize=figsize,
            show=False,
            **kwargs,
        )

        if kwargs.get("matplotlib"):
            self._fig.used_models.append(m)
            return self._plot(
                fig=plt.gcf(),
                title=title,
                plotname="force_plot",
                filename=filename,
                display=display,
            )
        else:
            sns.set_style(self.style)  # Reset style
            if filename:  # Save to a html file
                if not filename.endswith(".html"):
                    filename += ".html"
                shap.save_html(filename, plot)
            if display and find_spec("IPython"):
                from IPython.display import display

                shap.initjs()
                display(plot)

    @composed(crash, plot_from_model, typechecked)
    def plot_shap_heatmap(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        index: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        show: Optional[INT] = None,
        target: Union[INT, str] = 1,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot SHAP's heatmap plot.

        This plot is designed to show the population substructure of a
        dataset using supervised clustering and a heatmap. Supervised
        clustering involves clustering data points not by their original
        feature values but by their explanations.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name of the model to plot. If None, all models are selected.
            Note that leaving the default option could raise an exception
            if there are multiple models. To avoid this, call the plot
            from a model, `atom.xgb.heatmap_plot()`.

        index: slice, sequence or None, default=None
            Index names or positions of the rows in the dataset to plot.
            If None, it selects all rows in the test set. The heatmap
            plot does not support plotting a single sample.

        show: int or None, default=None
            Number of features (ordered by importance) to show. None
            to show all.

        target: int or str, default=1
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, default=None
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's heatmap plot.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        show = self._get_show(show, m)
        target = self._get_target(target)
        explanation = m._shap.get_explanation(rows, target)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        shap.plots.heatmap(explanation, max_display=show, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)

        self._fig.used_models.append(m)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            figsize=figsize,
            plotname="heatmap_plot",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_shap_scatter(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        index: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        feature: Union[INT, str] = 0,
        target: Union[INT, str] = 1,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot SHAP's scatter plot.

        Plots the value of the feature on the x-axis and the SHAP value
        of the same feature on the y-axis. This shows how the model
        depends on the given feature, and is like a richer extension of
        the classical partial dependence plots. Vertical dispersion of
        the data points represents interaction effects.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name of the model to plot. If None, all models are selected.
            Note that leaving the default option could raise an exception
            if there are multiple models. To avoid this, call the plot
            from a model, e.g. `atom.xgb.scatter_plot()`.

        index: slice, sequence or None, default=None
            Index names or positions of the rows in the dataset to
            plot. If None, it selects all rows in the test set. The
            scatter plot does not support plotting a single sample.

        feature: int or str, default=0
            Index or name of the feature to plot.

        target: int or str, default=1
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple, default=(10, 6)
            Figure's size, format as (x, y).

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's scatter plot.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        target = self._get_target(target)
        explanation = m._shap.get_explanation(rows, target, feature)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        shap.plots.scatter(explanation, color=explanation, ax=ax, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=self.label_fontsize, labelpad=12)

        self._fig.used_models.append(m)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            plotname="scatter_plot",
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_shap_waterfall(
        self,
        models: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        index: Optional[Union[INT, str]] = None,
        show: Optional[INT] = None,
        target: Union[INT, str] = 1,
        *,
        title: Optional[Union[str, dict]] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
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
        models exceeds the `show` parameter.

        Parameters
        ----------
        models: int, str, slice, sequence or None, default=None
            Name of the model to plot. If None, all models are selected.
            Note that leaving the default option could raise an exception
            if there are multiple models. To avoid this, call the plot
            from a model, `atom.xgb.waterfall_plot()`.

        index: int, str or None, default=None
            Index name or position of the row in the dataset to plot.
            If None, it selects the first row in the test set. The
            waterfall plot does not support plotting multiple
            samples.

        show: int or None, default=None
            Number of features (ordered by importance) to show. None
            to show all.

        target: int or str, default=1
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, default=None
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, default=None
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no period,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[[self._get_rows(index, branch=m.branch)[0]]]
        show = self._get_show(show, m)
        target = self._get_target(target)
        explanation = m._shap.get_explanation(rows, target, only_one=True)

        fig = self._get_figure()
        ax = fig.add_subplot(self._fig.grid)
        shap.plots.waterfall(explanation, max_display=show, show=False)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)

        self._fig.used_models.append(m)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            figsize=figsize or (10, 4 + show // 2),
            plotname="waterfall_plot",
            filename=filename,
            display=display,
        )

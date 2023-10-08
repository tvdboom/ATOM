# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the base classes for plotting.

"""

from __future__ import annotations

from abc import ABC
from contextlib import contextmanager
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, overload

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from beartype import beartype
from beartype.typing import Any, Literal
from mlflow.tracking import MlflowClient

from atom.basetracker import BaseTracker
from atom.basetransformer import BaseTransformer
from atom.utils.constants import PALETTE
from atom.utils.types import (
    Bool, DataFrame, FloatLargerZero, FloatZeroToOneExc, Index, Int,
    IntLargerZero, IntTypes, Legend, MetricSelector, Model, ModelsSelector,
    PlotBackend, RowSelector, Scalar, Sequence, SequenceTypes, Style,
)
from atom.utils.utils import (
    Aesthetics, check_is_fitted, composed, crash, divide, get_custom_scorer,
    lst, rnd, to_rgb,
)


if TYPE_CHECKING:
    from atom.baserunner import BaseRunner


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

    _marker = ["circle", "x", "diamond", "pentagon", "star", "hexagon"]
    _dash = ["solid", "dashdot", "dash", "dot", "longdash", "longdashdot"]
    _shape = ["", "/", "x", "\\", "-", "|", "+", "."]

    def __init__(
        self,
        rows: IntLargerZero = 1,
        cols: IntLargerZero = 1,
        horizontal_spacing: FloatZeroToOneExc = 0.05,
        vertical_spacing: FloatZeroToOneExc = 0.07,
        palette: str | Sequence[str] = "Prism",
        is_canvas: Bool = False,
        backend: PlotBackend = "plotly",
        create_figure: Bool = True,
    ):
        self.rows = rows
        self.cols = cols
        self.horizontal_spacing = horizontal_spacing
        self.vertical_spacing = vertical_spacing
        if isinstance(palette, str):
            self._palette = getattr(px.colors.qualitative, palette)
            self.palette = cycle(self._palette)
        elif isinstance(palette, SequenceTypes):
            # Convert color names or hex to rgb
            self._palette = list(map(to_rgb, palette))
            self.palette = cycle(self._palette)
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

        self.groups: list[str] = []
        self.style: Style = dict(palette={}, marker={}, dash={}, shape={})
        self.marker = cycle(self._marker)
        self.dash = cycle(self._dash)
        self.shape = cycle(self._shape)

        self.pos: dict[str, tuple[Scalar, Scalar]] = {}  # Subplot position for title
        self.custom_layout: dict[str, Any] = {}  # Layout params specified by user
        self.used_models: list[Model] = []  # Models plotted in this figure

    @property
    def grid(self) -> tuple[Int, Int]:
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
        else:
            return None

    def get_elem(
        self,
        name: str | None = None,
        element: Literal["palette", "marker", "dash", "shape"] = "palette",
    ) -> str:
        """Get the plot element for a specific name.

        This method is used to assign the same element (color, marker,
        etc...) to the same columns and models in a plot.

        Parameters
        ----------
        name: int, float or str or None, default=None
            Name for which to get the plot element. The name is stored in
            the element attributes to assign the same element to all calls
            with the same name. If None, return the first element.

        element: str, default="palette"
            Plot element to get. Choose from: palette, marker, dash, shape.

        Returns
        -------
        str
            Element code.

        """
        if name is None:
            return getattr(self, f"_{element}")[0]  # Get first element (default)
        elif name in self.style[element]:
            return self.style[element][name]
        else:
            return self.style[element].setdefault(name, next(getattr(self, element)))

    def showlegend(self, name: str, legend: Legend | dict | None) -> Bool:
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
        x: tuple[Scalar, Scalar] = (0, 1),
        y: tuple[Scalar, Scalar] = (0, 1),
        coloraxis: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Create and update the plot's axes.

        Parameters
        ----------
        x: tuple
            Relative x-size of the plot.

        y: tuple
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
                    text=title,
                    side="right",
                    font_size=coloraxis.pop("font_size"),
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


class BasePlot(BaseTransformer, BaseTracker, ABC):
    """Abstract base class for all plotting methods.

    This base class defines the properties that can be changed
    to customize the plot's aesthetics.

    """

    _fig = BaseFigure()
    _custom_layout: dict[str, Any] = {}
    _custom_traces: dict[str, Any] = {}
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
    def aesthetics(self) -> Aesthetics:
        """All plot aesthetic attributes."""
        return self._aesthetics

    @aesthetics.setter
    def aesthetics(self, value: dict):
        self.palette = value.get("palette", self.palette)
        self.title_fontsize = value.get("title_fontsize", self.title_fontsize)
        self.label_fontsize = value.get("label_fontsize", self.label_fontsize)
        self.tick_fontsize = value.get("tick_fontsize", self.tick_fontsize)
        self.line_width = value.get("line_width", self.line_width)
        self.marker_size = value.get("marker_size", self.marker_size)

    @property
    def palette(self) -> str | Sequence[str]:
        """Color palette.

        Specify one of plotly's [built-in palettes][palette] or create
        a custom one, e.g., `atom.palette = ["red", "green", "blue"]`.

        """
        return self._aesthetics.palette

    @palette.setter
    def palette(self, value: str | Sequence[str]):
        if isinstance(value, str) and not hasattr(px.colors.qualitative, value):
            raise ValueError(
                f"Invalid value for the palette parameter, got {value}. Choose "
                f"from one of plotly's built-in qualitative color sequences in "
                f"the px.colors.qualitative module or define your own sequence."
            )

        self._aesthetics.palette = value

    @property
    def title_fontsize(self) -> FloatLargerZero:
        """Fontsize for the plot's title."""
        return self._aesthetics.title_fontsize

    @title_fontsize.setter
    def title_fontsize(self, value: FloatLargerZero):
        self._aesthetics.title_fontsize = value

    @property
    def label_fontsize(self) -> FloatLargerZero:
        """Fontsize for the labels, legend and hover information."""
        return self._aesthetics.label_fontsize

    @label_fontsize.setter
    def label_fontsize(self, value: FloatLargerZero):
        self._aesthetics.label_fontsize = value

    @property
    def tick_fontsize(self) -> FloatLargerZero:
        """Fontsize for the ticks along the plot's axes."""
        return self._aesthetics.tick_fontsize

    @tick_fontsize.setter
    def tick_fontsize(self, value: FloatLargerZero):
        self._aesthetics.tick_fontsize = value

    @property
    def line_width(self) -> FloatLargerZero:
        """Width of the line plots."""
        return self._aesthetics.line_width

    @line_width.setter
    def line_width(self, value: FloatLargerZero):
        self._aesthetics.line_width = value

    @property
    def marker_size(self) -> FloatLargerZero:
        """Size of the markers."""
        return self._aesthetics.marker_size

    @marker_size.setter
    def marker_size(self, value: FloatLargerZero):
        self._aesthetics.marker_size = value

    # Methods ====================================================== >>

    @staticmethod
    def _get_plot_index(df: DataFrame) -> Index:
        """Return the dataset's index in a plottable format.

        Plotly does not accept all index formats (e.g., pd.Period),
        thus use this utility method to convert to timestamp those
        indices that can, else return as is.

        Parameters
        ----------
        df: dataframe
            Data set to get the index from.

        Returns
        -------
        index
            Index in an acceptable format.

        """
        if hasattr(df.index, "to_timestamp"):
            return df.index.to_timestamp()
        else:
            return df.index

    @staticmethod
    def _get_show(show: IntLargerZero | None, maximum: IntLargerZero) -> Int:
        """Check and return the number of features to show.

        Parameters
        ----------
        show: int or None
            Number of features to show. If None, select up to 200.

        maximum: int
            Maximum number of features allowed.

        Returns
        -------
        int
            Number of features to show.

        """
        if show is None or show > maximum:
            # Limit max features shown to avoid maximum figsize error
            show_c = min(200, maximum)
        else:
            show_c = show

        return show_c

    @staticmethod
    def _get_set(
        rows: str | Sequence[str] | dict[str, RowSelector]
    ) -> dict[str, RowSelector]:
        """Get the row selection.

        Convert provided rows to a dict where the keys are the names of
        the data sets, and the corresponding values are the selection
        rows fed to branch._get_rows().

        Parameters
        ----------
        rows: str, sequence or dict
            Selection of rows to plot.

        Returns
        -------
        dict
            Name of the row set with their corresponding selection.

        """
        if isinstance(rows, SequenceTypes):
            rows_c = {row: row for row in rows}
        elif isinstance(rows, str):
            rows_c = {rows: rows}
        elif isinstance(rows, dict):
            rows_c = rows

        return rows_c

    def _get_metric(self, metric: MetricSelector, max_one: Bool = False) -> list[str]:
        """Check and return the provided metric index.

        Parameters
        ----------
        metric: int, str, sequence or None
            Metric to retrieve. If None, all metrics are returned.

        max_one: bool, default=False
            Whether one or multiple metrics are allowed. If True, raise
            an exception if more than one metric is selected.

        Returns
        -------
        list of str
            Names of the selected metrics.

        """
        if metric is None:
            return self._metric.keys()
        else:
            inc: list[str] = []
            for met in lst(metric):
                if isinstance(met, IntTypes):
                    if int(met) < len(self._metric):
                        inc.append(self._metric[met].name)
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
                        elif (name := get_custom_scorer(m).name) in self._metric:
                            inc.append(name)
                        else:
                            raise ValueError(
                                "Invalid value for the metric parameter. The "
                                f"{name} metric wasn't used to fit the models."
                            )

        if max_one and len(inc) > 1:
            raise ValueError(
                f"Invalid value for the metric parameter, got {metric}. "
                f"This plotting method only accepts one metric."
            )

        return inc

    def _get_plot_models(
        self,
        models: ModelsSelector,
        max_one: Bool = False,
        ensembles: Bool = True,
        check_fitted: Bool = True,
    ) -> list[Model]:
        """If a plot is called from a model, adapt the `models` parameter.

        Parameters
        ----------
        func: func or None
            Function to decorate. When the decorator is called with no
            optional arguments, the function is passed as the first
            argument, and the decorator just returns the decorated
            function.

        max_one: bool, default=False
            Whether one or multiple models are allowed. If True, raise
            an exception if more than one model is selected.

        ensembles: bool, default=True
            If False, drop ensemble models from selection.

        check_fitted: bool, default=True
            Raise an exception if the runner isn't fitted (has no models).

        Returns
        -------
        list
            Models to plot.

        """
        if isinstance(self, BaseRunner):
            if check_fitted:
                check_is_fitted(self, attributes="_models")

            models_c = self._get_models(models, ensembles=ensembles)
            if max_one and len(models_c) > 1:
                raise ValueError(
                    f"Invalid value for the models parameter, got {models_c}. "
                    f"This plotting method only accepts one model."
                )

            return models_c

        return [self]  # type: ignore

    @overload
    def _get_figure(
        self,
        backend: Literal["plotly"] = ...,
        create_figure: Literal[True] = ...,
    ) -> go.Figure: ...

    @overload
    def _get_figure(
        self,
        backend: Literal["matplotlib"],
        create_figure: Literal[True] = ...,
    ) -> plt.Figure: ...

    @overload
    def _get_figure(
        self,
        backend: PlotBackend = ...,
        create_figure: Literal[False] = ...,
    ) -> None: ...

    def _get_figure(
        self,
        backend: PlotBackend = "plotly",
        create_figure: Bool = True,
    ) -> go.Figure | plt.Figure | None:
        """Return an existing figure if in canvas, else a new figure.

        Every time this method is called from a canvas, the plot
        index is raised by one to keep track of which subplot the
        BaseFigure is at.

        Parameters
        ----------
        backend: str, default="plotly"
            Figure's backend. Choose between plotly or matplotlib.

        create_figure: bool, default=True
            Whether to create a new figure.

        Returns
        -------
        [go.Figure][], [plt.Figure][] or None
            Existing figure or newly created. Returns None if kwarg
            `create_figure=False`.

        """
        if BasePlot._fig and BasePlot._fig.is_canvas:
            return BasePlot._fig.next_subplot
        else:
            BasePlot._fig = BaseFigure(
                palette=self.palette,
                backend=backend,
                create_figure=create_figure,
            )
            return BasePlot._fig.next_subplot

    def _draw_line(
        self,
        parent: str,
        child: str | None = None,
        legend: Legend | dict[str, Any] | None = None,
        **kwargs,
    ) -> go.Scatter:
        """Draw a line.

        Unify the style to draw a line, where parent and child
        (e.g., model - data set or column - distribution) keep the
        same style (color or dash). A legendgroup title is only added
        when there is a child element.

        Parameters
        ----------
        parent: str
            Name of the head attribute.

        child: str or None, default=None
            Name of the secondary attribute.

        legend: str, dict or None
            Legend argument provided by the user.

        **kwargs
            Additional keyword arguments for the trace.

        Returns
        -------
        go.Scatter
            New trace to add to figure.

        """
        return go.Scatter(
            line=dict(
                width=self.line_width,
                color=BasePlot._fig.get_elem(parent),
                dash=BasePlot._fig.get_elem(child, "dash"),
            ),
            marker=dict(
                symbol=BasePlot._fig.get_elem(child, "marker"),
                size=self.marker_size,
                color=BasePlot._fig.get_elem(parent),
                line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
            ),
            hovertemplate=kwargs.pop(
                "hovertemplate",
                f"(%{{x}}, %{{y}})<extra>{parent}{f' - {child}' if child else ''}</extra>"
            ),
            name=kwargs.pop("name", child or parent),
            legendgroup=kwargs.pop("legendgroup", parent),
            legendgrouptitle=kwargs.pop(
                "legendgrouptitle",
                dict(text=parent, font_size=self.label_fontsize) if child else None,
            ),
            showlegend=BasePlot._fig.showlegend(f"{parent}-{child}", legend),
            **kwargs,
        )

    @staticmethod
    def _draw_straight_line(y: Scalar | str, xaxis: str, yaxis: str):
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
        BasePlot._fig.figure.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=0 if y == "diagonal" else y,
            y1=1 if y == "diagonal" else y,
            xref=f"{xaxis} domain",
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
        # Set a Path with which to save the file
        if kwargs.get("filename"):
            if (path := Path(kwargs["filename"])).name == "auto":
                path = path.with_name(kwargs["plotname"])
        else:
            path = Path(kwargs.get("plotname", ""))

        fig = fig or BasePlot._fig.figure
        if BasePlot._fig.backend == "plotly":
            if isinstance(ax, tuple):
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

                if BasePlot._fig.is_canvas and (title := kwargs.get("title")):
                    # Add a subtitle to a plot in the canvas
                    default_title = {
                        "x": BasePlot._fig.pos[ax[0][5:] or "1"][0],
                        "y": BasePlot._fig.pos[ax[0][5:] or "1"][1] + 0.005,
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

            if not BasePlot._fig.is_canvas and kwargs.get("plotname"):
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

                # Update plot with custom settings
                fig.update_traces(**self._custom_traces)
                fig.update_layout(**self._custom_layout)

                if kwargs.get("filename"):
                    if path.suffix in ("", ".html"):
                        fig.write_html(path.with_suffix(".html"))
                    else:
                        fig.write_image(path)

                # Log plot to mlflow run of every model visualized
                if getattr(self, "experiment", None) and self.log_plots:
                    for m in set(BasePlot._fig.used_models):
                        MlflowClient().log_figure(
                            run_id=m.run.info.run_id,
                            figure=fig,
                            artifact_file=str(path.with_suffix(".html")),
                        )

                if kwargs.get("display") is True:
                    fig.show()
                elif kwargs.get("display") is None:
                    return fig

        elif BasePlot._fig.backend == "matplotlib":
            if isinstance(ax, plt.Axes):
                if title := kwargs.get("title"):
                    ax.set_title(title, fontsize=self.title_fontsize, pad=20)
                if xlabel := kwargs.get("xlabel"):
                    ax.set_xlabel(xlabel, fontsize=self.label_fontsize, labelpad=12)
                if ylabel := kwargs.get("ylabel"):
                    ax.set_ylabel(ylabel, fontsize=self.label_fontsize, labelpad=12)
                ax.tick_params(axis="both", labelsize=self.tick_fontsize)

            if size := kwargs.get("figsize"):
                # Convert from pixels to inches
                fig.set_size_inches(size[0] // fig.get_dpi(), size[1] // fig.get_dpi())

            plt.tight_layout()
            if kwargs.get("filename"):
                fig.savefig(path.with_suffix(".png"))

            # Log plot to mlflow run of every model visualized
            if self.experiment and self.log_plots:
                for m in set(BasePlot._fig.used_models):
                    MlflowClient().log_figure(
                        run_id=m.run.info.run_id,
                        figure=fig,
                        artifact_file=str(path.with_suffix(".png")),
                    )

            plt.show() if kwargs.get("display") else plt.close()
            if kwargs.get("display") is None:
                return fig

        return None  # display!=None or not final figures

    @composed(beartype, contextmanager, crash)
    def canvas(
        self,
        rows: IntLargerZero = 1,
        cols: IntLargerZero = 2,
        *,
        horizontal_spacing: FloatZeroToOneExc = 0.05,
        vertical_spacing: FloatZeroToOneExc = 0.07,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "out",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool = True,
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

        filename: str, Path or None, default=None
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
        BasePlot._fig = BaseFigure(
            rows=rows,
            cols=cols,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            palette=self.palette,
            is_canvas=True,
        )

        try:
            yield BasePlot._fig.figure
        finally:
            BasePlot._fig.is_canvas = False  # Close the canvas
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
        self._custom_traces = {}
        self._aesthetics = Aesthetics(
            palette=PALETTE,
            title_fontsize=24,
            label_fontsize=16,
            tick_fontsize=12,
            line_width=2,
            marker_size=8,
        )

    def update_layout(self, **kwargs):
        """Update the properties of the plot's layout.

        Recursively update the structure of the original layout with
        the values in the arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the figure's [update_layout][] method.

        """
        self._custom_layout = kwargs

    def update_traces(self, **kwargs):
        """Update the properties of the plot's traces.

        Recursively update the structure of the original traces with
        the values in the arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the figure's [update_traces][] method.

        """
        self._custom_traces = kwargs

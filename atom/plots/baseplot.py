"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the BasePlot class.

"""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from beartype import beartype
from mlflow.tracking import MlflowClient

from atom.basetracker import BaseTracker
from atom.basetransformer import BaseTransformer
from atom.plots.basefigure import BaseFigure
from atom.utils.constants import PALETTE
from atom.utils.types import (
    Bool, FloatLargerZero, FloatZeroToOneExc, Int, IntLargerZero, Legend,
    MetricSelector, Model, ModelsSelector, PlotBackend, RowSelector, Scalar,
    Sequence, int_t, sequence_t,
)
from atom.utils.utils import (
    Aesthetics, check_is_fitted, composed, crash, get_custom_scorer, lst,
)


class BasePlot(BaseTransformer, BaseTracker, metaclass=ABCMeta):
    """Abstract base class for all plotting methods.

    This base class defines the properties that can be changed
    to customize the plot's aesthetics.

    """

    _fig = BaseFigure()
    _custom_layout: ClassVar[dict[str, Any]] = {}
    _custom_traces: ClassVar[dict[str, Any]] = {}
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
    def title_fontsize(self) -> Scalar:
        """Fontsize for the plot's title."""
        return self._aesthetics.title_fontsize

    @title_fontsize.setter
    def title_fontsize(self, value: FloatLargerZero):
        self._aesthetics.title_fontsize = value

    @property
    def label_fontsize(self) -> Scalar:
        """Fontsize for the labels, legend and hover information."""
        return self._aesthetics.label_fontsize

    @label_fontsize.setter
    def label_fontsize(self, value: FloatLargerZero):
        self._aesthetics.label_fontsize = value

    @property
    def tick_fontsize(self) -> Scalar:
        """Fontsize for the ticks along the plot's axes."""
        return self._aesthetics.tick_fontsize

    @tick_fontsize.setter
    def tick_fontsize(self, value: FloatLargerZero):
        self._aesthetics.tick_fontsize = value

    @property
    def line_width(self) -> Scalar:
        """Width of the line plots."""
        return self._aesthetics.line_width

    @line_width.setter
    def line_width(self, value: FloatLargerZero):
        self._aesthetics.line_width = value

    @property
    def marker_size(self) -> Scalar:
        """Size of the markers."""
        return self._aesthetics.marker_size

    @marker_size.setter
    def marker_size(self, value: FloatLargerZero):
        self._aesthetics.marker_size = value

    # Methods ====================================================== >>

    @staticmethod
    def _get_plot_index(df: pd.DataFrame) -> pd.Index:
        """Return the dataset's index in a plottable format.

        Plotly does not accept all index formats (e.g., pd.Period),
        thus use this utility method to convert to timestamp those
        indices that can. Else, return as is.

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
    def _get_show(show: IntLargerZero | None, maximum: IntLargerZero = 200) -> Int:
        """Get the number of elements to show.

        Always limit the maximum elements shown to 200 to avoid
        a maximum figsize error.

        Parameters
        ----------
        show: int or None
            Number of elements to show. If None, select up to 200.

        maximum: int, default=200
            Maximum number of features allowed.

        Returns
        -------
        int
            Number of features to show.

        """
        if show is None or show > maximum:
            show_c = min(200, maximum)
        else:
            show_c = show

        return show_c

    @staticmethod
    def _get_set(
        rows: str | Sequence[str] | dict[str, RowSelector],
    ) -> Iterator[tuple[str, RowSelector]]:
        """Get the row selection.

        Convert provided rows to a dict where the keys are the names of
        the data sets, and the corresponding values are the selection
        rows fed to branch._get_rows().

        Parameters
        ----------
        rows: str, sequence or dict
            Selection of rows to plot.

        Yields
        ------
        str
            Name of the row set.

        RowSelector
            Selection of rows.

        """
        if isinstance(rows, sequence_t):
            rows_c = {row: row for row in rows}
        elif isinstance(rows, str):
            rows_c = {rows: rows}
        elif isinstance(rows, dict):
            rows_c = rows

        yield from rows_c.items()

    def _get_metric(self, metric: MetricSelector, *, max_one: Bool = False) -> list[str]:
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
                if isinstance(met, int_t):
                    if int(met) < len(self._metric):
                        inc.append(self._metric[met].name)
                    else:
                        raise ValueError(
                            f"Invalid value for the metric parameter. Value {met} is out "
                            f"of range for a pipeline with {len(self._metric)} metrics."
                        )
                elif isinstance(met, str):
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
        *,
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
        if hasattr(self, "_get_models"):
            if check_fitted:
                check_is_fitted(self)

            models_c = self._get_models(models, ensembles=ensembles)
            if max_one and len(models_c) > 1:
                raise ValueError(
                    f"Invalid value for the models parameter, got {models_c}. "
                    f"This plotting method only accepts one model."
                )

            return models_c
        else:
            return [self]  # type: ignore[list-item]

    @overload
    def _get_figure(
        self,
        backend: Literal["plotly"] = ...,
        *,
        create_figure: Literal[True] = ...,
    ) -> go.Figure: ...

    @overload
    def _get_figure(
        self,
        backend: Literal["matplotlib"],
        *,
        create_figure: Literal[True] = ...,
    ) -> plt.Figure: ...

    @overload
    def _get_figure(
        self,
        backend: PlotBackend,
        *,
        create_figure: Literal[False],
    ) -> None: ...

    def _get_figure(
        self,
        backend: PlotBackend = "plotly",
        *,
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
    ):
        """Draw a line on the current figure.

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

        legend: str, dict or None, default=None
            Legend argument provided by the user.

        **kwargs
            Additional keyword arguments for the trace.

        """
        BasePlot._fig.figure.add_scatter(
            name=kwargs.pop("name", child or parent),
            mode=kwargs.pop("mode", "lines"),
            line=kwargs.pop(
                "line", {
                    "width": self.line_width,
                    "color": BasePlot._fig.get_elem(parent),
                    "dash": BasePlot._fig.get_elem(child, "dash"),
                }
            ),
            marker=kwargs.pop(
                "marker", {
                    "symbol": BasePlot._fig.get_elem(child, "marker"),
                    "size": self.marker_size,
                    "color": BasePlot._fig.get_elem(parent),
                    "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                }
            ),
            hovertemplate=kwargs.pop(
                "hovertemplate",
                f"(%{{x}}, %{{y}})<extra>{parent}{f' - {child}' if child else ''}</extra>",
            ),
            legendgroup=kwargs.pop("legendgroup", parent),
            legendgrouptitle=kwargs.pop(
                "legendgrouptitle",
                {"text": parent, "font_size": self.label_fontsize} if child else None,
            ),
            showlegend=kwargs.pop(
                "showlegend",
                BasePlot._fig.showlegend(f"{parent}-{child}" if child else parent, legend)
            ),
            **kwargs,
        )

    @staticmethod
    def _draw_diagonal_line(values: tuple, xaxis: str, yaxis: str):
        """Draw a diagonal line across the axis.

        The line should be used as a reference. It's not added to the
        legend and doesn't show any information on hover.

        Parameters
        ----------
        values: tuple of sequence
            Values of the data points required to determine the ranges in
            format (x, y).

        xaxis: str
            Name of the x-axis to draw in.

        yaxis: str
            Name of the y-axis to draw in.

        """
        # Get the ranges with a 5% margin
        y_min, y_max = min(values[1]), max(values[1])
        if np.issubdtype(type(y_min), np.number):
            y_min *= 0.95
            y_max *= 1.05

        BasePlot._fig.figure.add_shape(
            type="line",
            x0=y_min,
            x1=y_max,
            y0=y_min,
            y1=y_max,
            xref=xaxis,
            yref=yaxis,
            line={"width": 1, "color": "black"},
            opacity=0.5,
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
        if isinstance(fig, go.Figure):
            if isinstance(ax, tuple):
                # Hide the axis' label and ticks from non-border subplots
                if not BasePlot._fig.sharex or self._fig.grid[0] == self._fig.rows:
                    fig.update_layout(
                        {
                            f"{ax[0]}_title": {
                                "text": kwargs.get("xlabel"),
                                "font_size": self.label_fontsize,
                            }
                        }
                    )
                else:
                    fig.update_layout({f"{ax[0]}_showticklabels": False})

                if not BasePlot._fig.sharey or self._fig.grid[1] == 1:
                    fig.update_layout(
                        {
                            f"{ax[1]}_title": {
                                "text": kwargs.get("ylabel"),
                                "font_size": self.label_fontsize,
                            }
                        }
                    )
                else:
                    fig.update_layout({f"{ax[1]}_showticklabels": False})

                fig.update_layout(
                    {
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
                        title = default_title | title
                    else:
                        title = {"text": title, **default_title}

                    fig.update_layout({"annotations": (*fig.layout.annotations, title)})

            if not BasePlot._fig.is_canvas and kwargs.get("plotname"):
                default_title = {
                    "x": 0.5,
                    "y": 1,
                    "pad": {"t": 15, "b": 15},
                    "xanchor": "center",
                    "yanchor": "top",
                    "xref": "paper",
                    "font_size": self.title_fontsize,
                }
                if isinstance(title := kwargs.get("title"), dict):
                    title = default_title | title
                else:
                    title = {"text": title, **default_title}

                default_legend = {
                    "traceorder": "grouped",
                    "groupclick": kwargs.get("groupclick", "toggleitem"),
                    "font_size": self.label_fontsize,
                    "bgcolor": "rgba(255, 255, 255, 0.5)",
                }
                if isinstance(legend := kwargs.get("legend"), str):
                    position = {}
                    if legend == "upper left":
                        position = {"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"}
                    elif legend == "lower left":
                        position = {"x": 0.01, "y": 0.01, "xanchor": "left", "yanchor": "bottom"}
                    elif legend == "upper right":
                        position = {"x": 0.99, "y": 0.99, "xanchor": "right", "yanchor": "top"}
                    elif legend == "lower right":
                        position = {"x": 0.99, "y": 0.01, "xanchor": "right", "yanchor": "bottom"}
                    elif legend == "upper center":
                        position = {"x": 0.5, "y": 0.99, "xanchor": "center", "yanchor": "top"}
                    elif legend == "lower center":
                        position = {"x": 0.5, "y": 0.01, "xanchor": "center", "yanchor": "bottom"}
                    elif legend == "center left":
                        position = {"x": 0.01, "y": 0.5, "xanchor": "left", "yanchor": "middle"}
                    elif legend == "center right":
                        position = {"x": 0.99, "y": 0.5, "xanchor": "right", "yanchor": "middle"}
                    elif legend == "center":
                        position = {"x": 0.5, "y": 0.5, "xanchor": "center", "yanchor": "middle"}

                    legend = default_legend | position

                elif isinstance(legend, dict):
                    legend = default_legend | legend

                # Update layout with predefined settings
                space1 = self.title_fontsize if title.get("text") else 10
                space2 = self.title_fontsize * int(bool(fig.layout.annotations))
                fig.update_layout(
                    title=title,
                    legend=legend,
                    showlegend=bool(kwargs.get("legend")),
                    hoverlabel={"font_size": self.label_fontsize},
                    font_size=self.tick_fontsize,
                    margin={"l": 50, "b": 50, "r": 0, "t": 25 + space1 + space2, "pad": 0},
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

        elif isinstance(fig, plt.Figure):
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
        sharex: Bool = False,
        sharey: Bool = False,
        hspace: FloatZeroToOneExc = 0.05,
        vspace: FloatZeroToOneExc = 0.07,
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

        sharex: bool, default=False
            If True, hide the label and ticks from non-border subplots
            on the x-axis.

        sharey: bool, default=False
            If True, hide the label and ticks from non-border subplots
            on the y-axis.

        hspace: float, default=0.05
            Space between subplot rows in normalized plot coordinates.
            The spacing is relative to the figure's size.

        vspace: float, default=0.07
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
            - If str: Position to display the legend.
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
            sharex=sharex,
            sharey=sharey,
            hspace=hspace,
            vspace=vspace,
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

    @classmethod
    def reset_aesthetics(cls):
        """Reset the plot [aesthetics][] to their default values."""
        cls._custom_layout = {}
        cls._custom_traces = {}
        cls._aesthetics = Aesthetics(
            palette=list(PALETTE),
            title_fontsize=24,
            label_fontsize=16,
            tick_fontsize=12,
            line_width=2,
            marker_size=8,
        )

    @classmethod
    def update_layout(cls, **kwargs):
        """Update the properties of the plot's layout.

        Recursively update the structure of the original layout with
        the values in the arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the figure's [update_layout][] method.

        """
        cls._custom_layout = kwargs

    @classmethod
    def update_traces(cls, **kwargs):
        """Update the properties of the plot's traces.

        Recursively update the structure of the original traces with
        the values in the arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the figure's [update_traces][] method.

        """
        cls._custom_traces = kwargs

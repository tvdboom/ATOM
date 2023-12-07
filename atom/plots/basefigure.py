"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the BaseFigure class.

"""

from __future__ import annotations

from itertools import cycle
from typing import Any, Literal

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from atom.utils.types import (
    Bool, FloatZeroToOneExc, Int, IntLargerZero, Legend, Model, PlotBackend,
    Scalar, Sequence, Style, sequence_t,
)
from atom.utils.utils import divide, rnd, to_rgb


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

    _marker = ("circle", "x", "diamond", "pentagon", "star", "hexagon")
    _dash = ("solid", "dashdot", "dash", "dot", "longdash", "longdashdot")
    _shape = ("", "/", "x", "\\", "-", "|", "+", ".")

    def __init__(
        self,
        rows: IntLargerZero = 1,
        cols: IntLargerZero = 1,
        *,
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
        elif isinstance(palette, sequence_t):
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
        self.style: Style = {"palette": {}, "marker": {}, "dash": {}, "shape": {}}
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

    def showlegend(self, name: str, legend: Legend | dict | None) -> bool:
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
                f"xaxis{self.axes}": {
                    "domain": (x_pos, rnd(x_pos + ax_size)),
                    "anchor": f"y{self.axes}",
                },
                f"yaxis{self.axes}": {
                    "domain": (y_pos, rnd(y_pos + ay_size)),
                    "anchor": f"x{self.axes}",
                },
            }
        )

        # Place a colorbar right of the axes
        if coloraxis:
            if title := coloraxis.pop("title", None):
                coloraxis["colorbar_title"] = {
                    "text": title,
                    "side": "right",
                    "font_size": coloraxis.pop("font_size"),
                }

            coloraxis["colorbar_x"] = rnd(x_pos + ax_size) + ax_size / 40
            coloraxis["colorbar_xanchor"] = "left"
            coloraxis["colorbar_y"] = y_pos + ay_size / 2
            coloraxis["colorbar_yanchor"] = "middle"
            coloraxis["colorbar_len"] = ay_size * 0.9
            coloraxis["colorbar_thickness"] = ax_size * 30  # Default width in pixels
            self.figure.update_layout({f"coloraxis{coloraxis.pop('axes', self.axes)}": coloraxis})

        xaxis = f"x{self.axes if self.axes > 1 else ''}"
        yaxis = f"y{self.axes if self.axes > 1 else ''}"
        return xaxis, yaxis

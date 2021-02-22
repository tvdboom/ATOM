# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the plotting classes.

"""

# Standard packages
import numpy as np
import pandas as pd
from itertools import cycle
from itertools import chain
from inspect import signature
from collections import defaultdict
from typeguard import typechecked
from joblib import Parallel, delayed
from contextlib import contextmanager
from scipy.stats.mstats import mquantiles
from typing import Optional, Union, Tuple

# Plotting packages
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import ConnectionStyle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Sklearn
from sklearn.utils import _safe_indexing
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import SCORERS, roc_curve, precision_recall_curve

# Own modules
from atom.basetransformer import BaseTransformer
from .utils import (
    SEQUENCE_TYPES, SCALAR, METRIC_ACRONYMS, lst, check_is_fitted,
    check_method, check_goal, check_binary_task, check_predict_proba,
    get_best_score, partial_dependence, composed, crash, plot_from_model,
)


class BaseFigure:
    """Class that stores the position of the current axes in grid.

    Parameters
    ----------
    nrows: int
        Number of subplot rows in the canvas.

    ncols: int
        Number of subplot columns in the canvas.

    """

    def __init__(self, nrows=1, ncols=1, is_canvas=False):
        self.nrows = nrows
        self.ncols = ncols
        self.is_canvas = is_canvas
        self._idx = -1

        # Create new figure and corresponding grid
        figure = plt.figure(constrained_layout=True if is_canvas else False)
        self.gridspec = GridSpec(nrows=self.nrows, ncols=self.ncols, figure=figure)

    @property
    def figure(self):
        """Get the current figure and increase the subplot index."""
        self._idx += 1

        # Check if there are too many plots in the contextmanager
        if self._idx >= self.nrows * self.ncols:
            raise RuntimeError(
                "Invalid number of plots in the canvas! Increase "
                "the number of rows and cols to add more plots."
            )

        return plt.gcf()

    @property
    def grid(self):
        """Return the position of the current axes in the grid."""
        return self.gridspec[self._idx]


class BasePlotter:
    """Parent class for all plotting methods.

    This base class defines the plot properties that can be changed in
    order to customize the plot's aesthetics. Making the variable
    mutable ensures that it is changed for all classes at the same time.

    """

    _fig = None
    _aesthetics = dict(
        style="darkgrid",  # Seaborn plotting style
        palette="GnBu_r_d",  # Matplotlib color palette
        title_fontsize=20,  # Fontsize for titles
        label_fontsize=16,  # Fontsize for labels and legends
        tick_fontsize=12,  # Fontsize for ticks
    )
    sns.set_style(_aesthetics["style"])
    sns.set_palette(_aesthetics["palette"])

    # Properties =================================================== >>

    @property
    def aesthetics(self):
        return self._aesthetics

    @aesthetics.setter
    @typechecked
    def aesthetics(self, aesthetics: dict):
        self.style = aesthetics.get("style", self.style)
        self.palette = aesthetics.get("palette", self.palette)
        self.title_fontsize = aesthetics.get("title_fontsize", self.title_fontsize)
        self.label_fontsize = aesthetics.get("label_fontsize", self.label_fontsize)
        self.tick_fontsize = aesthetics.get("tick_fontsize", self.tick_fontsize)

    @property
    def style(self):
        return self._aesthetics["style"]

    @style.setter
    @typechecked
    def style(self, style: str):
        styles = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
        if style not in styles:
            raise ValueError(
                "Invalid value for the style parameter, got "
                f"{style}. Choose from {', '.join(styles)}."
            )
        sns.set_style(style)
        self._aesthetics["style"] = style

    @property
    def palette(self):
        return self._aesthetics["palette"]

    @palette.setter
    @typechecked
    def palette(self, palette: str):
        sns.set_palette(palette)
        self._aesthetics["palette"] = palette

    @property
    def title_fontsize(self):
        return self._aesthetics["title_fontsize"]

    @title_fontsize.setter
    @typechecked
    def title_fontsize(self, title_fontsize: int):
        if title_fontsize <= 0:
            raise ValueError(
                "Invalid value for the title_fontsize parameter. "
                f"Value should be >=0, got {title_fontsize}."
            )
        self._aesthetics["title_fontsize"] = title_fontsize

    @property
    def label_fontsize(self):
        return self._aesthetics["label_fontsize"]

    @label_fontsize.setter
    @typechecked
    def label_fontsize(self, label_fontsize: int):
        if label_fontsize <= 0:
            raise ValueError(
                "Invalid value for the label_fontsize parameter. "
                f"Value should be >=0, got {label_fontsize}."
            )
        self._aesthetics["label_fontsize"] = label_fontsize

    @property
    def tick_fontsize(self):
        return self._aesthetics["tick_fontsize"]

    @tick_fontsize.setter
    @typechecked
    def tick_fontsize(self, tick_fontsize: int):
        if tick_fontsize <= 0:
            raise ValueError(
                "Invalid value for the tick_fontsize parameter. "
                f"Value should be >=0, got {tick_fontsize}."
            )
        self._aesthetics["tick_fontsize"] = tick_fontsize

    def reset_aesthetics(self):
        """Reset the plot aesthetics to their default values."""
        self.aesthetics = dict(
            style="darkgrid",
            palette="GnBu_r_d",
            title_fontsize=20,
            label_fontsize=16,
            tick_fontsize=12,
        )

    # Methods ====================================================== >>

    @staticmethod
    def _get_figure():
        """Return existing figure if in canvas, else a new figure."""
        if BasePlotter._fig and BasePlotter._fig.is_canvas:
            return BasePlotter._fig.figure
        else:
            BasePlotter._fig = BaseFigure()
            return BasePlotter._fig.figure

    def _get_subclass(self, models, max_one=False):
        """Check and return the provided parameter models.

        Parameters
        ----------
        models: str or sequence
            Models provided by the plot's parameter.

        max_one: bool, optional (default=False)
            Whether one or multiple models are allowed. If True, return
            the model instead of a list.

        """
        models = self._get_models(models)
        model_subclasses = [m for m in self._models if m.name in models]

        if max_one and len(model_subclasses) > 1:
            raise ValueError("This plot method allows only one model at a time!")

        return model_subclasses[0] if max_one else model_subclasses

    def _get_metric(self, metric):
        """Check and return the index of the provided metric."""
        if isinstance(metric, str):
            if metric.lower() in METRIC_ACRONYMS:
                metric = METRIC_ACRONYMS[metric]
            return self._metric.index(metric)

        elif 0 <= metric < len(self._metric):
            return metric

        raise ValueError(
            "Invalid value for the metric parameter. Value should be the index"
            f" or name of a metric used to run the pipeline, got {metric}."
        )

    @staticmethod
    def _get_set(dataset):
        """Check and return the provided parameter metric."""
        if dataset.lower() == "both":
            return ["train", "test"]
        elif dataset.lower() in ("train", "test"):
            return [dataset.lower()]
        else:
            raise ValueError(
                "Invalid value for the dataset parameter. "
                "Choose between 'train', 'test' or 'both'."
            )

    def _get_show(self, show):
        """Check and return the provided parameter show."""
        max_fxs = max([branch.n_features for branch in self._branches.values()])
        if show is None or show > max_fxs:
            return max_fxs
        elif show < 1:
            raise ValueError(
                "Invalid value for the show parameter."
                f"Value should be >0, got {show}."
            )

        return show

    @staticmethod
    def _get_index(index, model):
        """Check and return the provided parameter index."""
        if index is None:
            rows = model.X_test
        elif isinstance(index, int):
            if index < 0:
                rows = model.X.iloc[[len(model.X) + index]]
            else:
                rows = model.X.iloc[[index]]
        elif isinstance(index, slice):
            rows = model.X.iloc[index]
        else:
            rows = model.X.iloc[slice(*index)]

        if rows.empty:
            raise ValueError(
                "Invalid value for the index parameter. Couldn't find "
                f"the specified rows in the dataset, got: {index}."
            )

        return rows

    def _get_columns(self, columns):
        """Check and return the provided column names (if numerical)."""
        num_cols = self.dataset.select_dtypes(include=["number"]).columns
        if columns is None:
            return num_cols
        elif isinstance(columns, int):
            return self.columns[columns]
        elif isinstance(columns, str):
            return columns
        elif isinstance(columns, slice):
            columns = self.columns[columns]

        cols = []
        for col in lst(columns):
            if isinstance(col, int):
                cols.append(self.columns[col])
            else:
                if col not in self.columns:
                    raise ValueError(
                        "Invalid value for the columns parameter. "
                        f"Column {col} not found in the dataset."
                    )
                cols.append(col)

        return [col for col in cols if col in num_cols]

    def _get_target(self, target):
        """Check and return the provided target's index."""
        if isinstance(target, str):
            if target not in self.mapping:
                raise ValueError(
                    f"Invalid value for the target parameter. Value {target} "
                    "not found in the mapping of the target column."
                )
            return self.mapping[target]

        elif not 0 <= target < len(self.y.unique()):
            raise ValueError(
                "Invalid value for the target parameter. There are "
                f"{len(self.y.unique())} classes, got {target}."
            )

        return target

    @staticmethod
    def _get_shap(model, index, target):
        """Get the SHAP information of a model.

        Parameters
        ----------
        model: class
         Model subclass.

        index: pd.DataFrame
            Rows on which to calculate the shap values.

        target: int
            Index of the class to look at in the target column.

        Returns
        -------
        shap_values: np.ndarray
            SHAP values for the target class.

        expected_value: float or list
            Difference between the model output for that sample and
            the expected value of the model output.

        """
        # Create the explainer if not invoked before
        if not model.explainer:
            model.explainer = shap.Explainer(model.estimator, model.X_train)

        # Get the shap values on the specified rows
        shap_values = model.explainer(index)

        # Select the target values from the array
        if shap_values.values.ndim > 2:
            shap_values = shap_values[:, :, target]
        if shap_values.shape[0] == 1:  # Rows is a df with one row only
            shap_values = shap_values[0]

        return shap_values

    def _plot(self, ax=None, **kwargs):
        """Make the plot.

        Customize the axes to the default layout and plot the figure
        if it's not part of a canvas.

        Parameters
        ----------
        ax: matplotlib.axes.Axes or None, optional (default=None)
            Current plotting axes. If None, ignore the axes.

        **kwargs
            Keyword arguments containing the plot's parameters.
            Axes parameters:
                - title: Axes' title.
                - legend: Location of the legend and number of items.
                - xlabel: Label for the x-axis.
                - ylabel: Label for the y-axis.
                - xlim: Limits for the x-axis.
                - ylim: Limits for the y-axis.
            Figure parameters:
                - figsize: Size of the figure.
                - tight_layout: Whether to apply it (default=True).
                - filename: Name of the saved file.
                - display: Whether to render the plot.

        """
        if kwargs.get("title"):
            ax.set_title(kwargs.get("title"), fontsize=self.title_fontsize, pad=20)
        if kwargs.get("legend"):
            ax.legend(
                loc=kwargs["legend"][0],
                ncol=kwargs["legend"][1] // 3 if kwargs["legend"][1] // 3 > 0 else 1,
                fontsize=self.label_fontsize
            )
        if kwargs.get("xlabel"):
            ax.set_xlabel(kwargs["xlabel"], fontsize=self.label_fontsize, labelpad=12)
        if kwargs.get("ylabel"):
            ax.set_ylabel(kwargs["ylabel"], fontsize=self.label_fontsize, labelpad=12)
        if kwargs.get("xlim"):
            ax.set_xlim(kwargs["xlim"])
        if kwargs.get("ylim"):
            ax.set_ylim(kwargs["ylim"])
        if ax is not None:
            ax.tick_params(axis='both', labelsize=self.tick_fontsize)

        if not getattr(BasePlotter._fig, "is_canvas", None):
            if kwargs.get("figsize"):
                plt.gcf().set_size_inches(*kwargs["figsize"])
            if kwargs.get("tight_layout", True):
                plt.tight_layout()
            if kwargs.get("filename"):
                plt.savefig(kwargs["filename"])
            if "filename" in kwargs:
                plt.show() if kwargs.get("display") else plt.close()

    @composed(contextmanager, crash, typechecked)
    def canvas(
        self,
        nrows: int = 1,
        ncols: int = 2,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Create a figure with multiple plots.

        This `@contextmanager` allows you to draw many plots in one
        figure. The default option is to add two plots side by side.

        Parameters
        ----------
        nrows: int, optional (default=1)
            Number of plots in length.

        ncols: int, optional (default=2)
            Number of plots in width.

        title: str or None, optional (default=None)
            Plot's title. If None, no title is displayed.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to
            the number of plots in the canvas.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        BasePlotter._fig = BaseFigure(nrows=nrows, ncols=ncols, is_canvas=True)
        try:
            yield self
        finally:
            if title:
                plt.suptitle(title, fontsize=self.title_fontsize + 4)
            BasePlotter._fig.is_canvas = False  # Close the canvas
            self._plot(
                figsize=figsize if figsize else (6 + 4 * ncols, 2 + 4 * nrows),
                tight_layout=False,
                filename=filename,
                display=display
            )


class FSPlotter(BasePlotter):
    """Plots for the FeatureSelector class."""

    @composed(crash, typechecked)
    def plot_pca(
        self,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the explained variance ratio vs number of components.

        Parameters
        ----------
        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        if not hasattr(self, "pca"):
            raise PermissionError(
                "The plot_pca method is only available if PCA was applied on the data!"
            )

        n = self.pca.n_components_  # Number of chosen components
        var = np.array(self.pca.explained_variance_ratio_[:n])
        var_all = np.array(self.pca.explained_variance_ratio_)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        ax.scatter(
            x=self.pca.n_components_ - 1,
            y=var.sum(),
            marker="*",
            s=130,
            c="blue",
            edgecolors="b",
            zorder=3,
            label=f"Total variance retained: {round(var.sum(), 3)}",
        )
        ax.plot(range(self.pca.n_features_), np.cumsum(var_all), marker="o")
        ax.axhline(var.sum(), ls="--", color="k")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only int ticks

        self._plot(
            ax=ax,
            title=title,
            legend=("lower right", 1),
            xlabel="First N principal components",
            ylabel="Cumulative variance ratio",
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_components(
        self,
        show: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the explained variance ratio per component.

        Parameters
        ----------
        show: int or None, optional (default=None)
            Number of components to show. None to show all.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size
            to the number of components shown.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        if not hasattr(self, "pca"):
            raise PermissionError(
                "The plot_components method is only available "
                "if PCA was applied on the data!"
            )

        # Set parameters
        if show is None:
            show = self.pca.n_components_
        elif show > self.pca.n_features_:
            show = self.pca.n_features_
        elif show < 1:
            raise ValueError(
                "Invalid value for the show parameter. "
                f"Value should be >0, got {show}."
            )

        var = np.array(self.pca.explained_variance_ratio_)[:show]
        indices = [f"Component {str(i)}" for i in range(len(var))]
        scr = pd.Series(var, index=indices).sort_values()

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        scr.plot.barh(label=f"Total variance retained: {var.sum():.3f}", width=0.6)
        ax.set_xlim(0, max(scr) + 0.1 * max(scr))  # Make extra space for numbers
        for i, v in enumerate(scr):
            ax.text(v + 0.005, i - 0.08, f"{v:.3f}", fontsize=self.tick_fontsize)

        self._plot(
            ax=ax,
            title=title,
            legend=("lower right", 1),
            xlabel="Explained variance ratio",
            figsize=figsize if figsize else (10, 4 + show // 2),
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_rfecv(
        self,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the RFECV results.

        Plot the scores obtained by the estimator fitted on every
        subset of the dataset. Only if RFECV was applied on the
        dataset through the feature_engineering method.

        Parameters
        ----------
        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        if not hasattr(self.branch, "rfecv") or not self.rfecv:
            raise PermissionError(
                "The plot_rfecv method is only available "
                "if RFECV was applied on the data!"
            )

        try:  # Define the y-label for the plot
            ylabel = self.rfecv.get_params()["scoring"].name
        except AttributeError:
            ylabel = "score"
            if self.rfecv.get_params()["scoring"]:
                ylabel = str(self.rfecv.get_params()["scoring"])

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        n_features = self.rfecv.get_params()["min_features_to_select"]
        xline = range(n_features, n_features + len(self.rfecv.grid_scores_))
        ax.plot(xline, self.rfecv.grid_scores_)

        # Set limits before drawing the intersected lines
        xlim = (n_features - 0.5, n_features + len(self.rfecv.grid_scores_) - 0.5)
        ylim = ax.get_ylim()

        # Draw intersected lines
        x = xline[np.argmax(self.rfecv.grid_scores_)]
        y = max(self.rfecv.grid_scores_)
        ax.vlines(x, -1e4, y, ls="--", color="k", alpha=0.7)
        ax.hlines(
            y=y,
            xmin=-1,
            xmax=x,
            ls="--",
            color="k",
            alpha=0.7,
            label=f"Features: {x}   {ylabel}: {round(y, 3)}",
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only int ticks

        self._plot(
            ax=ax,
            title=title,
            legend=("lower right", 1),
            xlabel="Number of features",
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            filename=filename,
            display=display,
        )


class BaseModelPlotter(BasePlotter):
    """Plots for the BaseModel class."""

    @composed(crash, plot_from_model, typechecked)
    def plot_results(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        metric: Union[int, str] = 0,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot of the model results after the evaluation.

        If all models applied bagging, the plot will be a boxplot.
        If not, the plot will be a barplot. Models are ordered based
        on their score from the top down. The score is either the
        `mean_bagging` or `metric_test` attribute of the model,
        selected in that order.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        metric: int or str, optional (default=0)
            Index or name of the metric. Only for multi-metric runs.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to
            the number of models.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        def get_bagging(m):
            """Get the bagging's results for a specific metric."""
            if getattr(m, "metric_bagging", None):
                if len(self._metric) == 1:
                    return m.metric_bagging
                else:
                    return m.metric_bagging[metric]

        def std(m):
            """Get the standard deviation of the bagging's results."""
            if getattr(m, "std_bagging", None):
                return lst(m.std_bagging)[metric]
            else:
                return 0

        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        metric = self._get_metric(metric)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        names = []
        models = sorted(models, key=lambda m: get_best_score(m, metric))
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]  # First color

        all_bagging = all(get_bagging(m) for m in models)
        for i, m in enumerate(models):
            names.append(m.name)
            if all_bagging:
                ax.boxplot(
                    x=get_bagging(m),
                    vert=False,
                    positions=[i],
                    widths=0.5,
                    boxprops=dict(color=color)
                )
            else:
                ax.barh(
                    y=i,
                    width=get_best_score(m, metric),
                    height=0.5,
                    xerr=std(m),
                    color=color
                )

        min_lim = 0.95 * (get_best_score(models[0], metric) - std(models[0]))
        max_lim = 1.01 * (get_best_score(models[-1], metric) + std(models[-1]))
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(names)
        self._plot(
            ax=ax,
            title=title,
            xlabel=self._metric[metric].name,
            xlim=(min_lim, max_lim) if not all_bagging else None,
            figsize=figsize if figsize else (10, 4 + len(models) // 2),
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_bo(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        metric: Union[int, str] = 0,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 8),
        filename: Optional[str] = None,
        display: bool = True,
    ):

        """Plot the bayesian optimization scoring.

        Only for models that ran the hyperparameter optimization. This
        is the same plot as produced by `bo_params={"plot": True}`
        while running the BO. Creates a canvas with two plots: the
        first plot shows the score of every trial and the second shows
        the distance between the last consecutive steps.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline that used bayesian optimization are selected.

        metric: int or str, optional (default=0)
            Index or name of the metric. Only for multi-metric runs.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 8))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        metric = self._get_metric(metric)

        # Check there is at least one model that run the BO
        if all([m.bo.empty for m in models]):
            raise PermissionError(
                "The plot_bo method is only available for models that "
                "ran the bayesian optimization hyperparameter tuning!"
            )

        fig = self._get_figure()
        gs = GridSpecFromSubplotSpec(4, 1, BasePlotter._fig.grid, hspace=0.05)
        ax1 = fig.add_subplot(gs[0:3, 0])
        ax2 = plt.subplot(gs[3:4, 0], sharex=ax1)
        for m in models:
            y = m.bo["score"].apply(lambda value: lst(value)[metric])
            if len(models) == 1:
                label = f"Score={round(lst(m.metric_bo)[metric], 3)}"
            else:
                label = f"{m.name} (Score={round(lst(m.metric_bo)[metric], 3)})"

            # Draw bullets on all markers except the maximum
            markers = [i for i in range(len(m.bo))]
            markers.remove(int(np.argmax(y)))
            ax1.plot(range(1, len(y) + 1), y, "-o", markevery=markers, label=label)
            ax2.plot(range(2, len(y) + 1), np.abs(np.diff(y)), "-o")
            ax1.scatter(np.argmax(y) + 1, max(y), zorder=10, s=100, marker="*")

        plt.setp(ax1.get_xticklabels(), visible=False)
        self._plot(
            ax=ax1,
            title=title,
            legend=("lower right", len(models)),
            ylabel=self._metric[metric].name,
        )
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._plot(
            ax=ax2,
            xlabel="Call",
            ylabel="d",
            figsize=figsize,
            filename=filename,
            display=display
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_evals(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "both",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot evaluation curves for the train and test set.

         Only for models that allow in-training evaluation (XGB, LGB,
        CastB). The metric is provided by the estimator's package and
        is different for every model and every task. For this reason,
        the method only allows plotting one model at a time.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the model to plot. If None, all models in the
            pipeline are selected. Note that leaving the default
            option could raise an exception if there are multiple
            models in the pipeline. To avoid this, call the plot
            from a model, e.g. `atom.lgb.plot_evals()`.

        dataset: str, optional (default="both")
            Data set on which to calculate the evaluation curves.
            Options are "train", "test" or "both".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_method(self, "plot_evals")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        dataset = self._get_set(dataset)

        # Check that the model had in-training evaluation
        if not hasattr(m, "evals"):
            raise AttributeError(
                "The plot_evals method is only available for models "
                f"that allow in-training evaluation, got {m.name}."
            )

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        for set_ in dataset:
            ax.plot(range(len(m.evals[set_])), m.evals[set_], lw=2, label=set_)

        self._plot(
            ax=ax,
            title=title,
            legend=("best", len(dataset)),
            xlabel=m.get_dimensions()[0].name,  # First param is always the iter
            ylabel=m.evals["metric"],
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_roc(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the Receiver Operating Characteristics curve.

        The legend shows the Area Under the ROC Curve (AUC) score.
        Only for binary classification tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train", "test" or "both".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10,6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_roc")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)
        check_predict_proba(models, "plot_roc")

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        for m in models:
            for set_ in dataset:
                # Get False (True) Positive Rate as arrays
                fpr, tpr, _ = roc_curve(
                    getattr(m, f"y_{set_}"), getattr(m, f"predict_proba_{set_}")[:, 1]
                )

                roc = f" (AUC={round(m.scoring('roc_auc', dataset=set_), 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + roc
                ax.plot(fpr, tpr, lw=2, label=label)

        ax.plot([0, 1], [0, 1], lw=2, color="black", alpha=0.7, linestyle="--")

        self._plot(
            ax=ax,
            title=title,
            legend=("lower right", len(models)),
            xlabel="FPR",
            ylabel="TPR",
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_prc(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the precision-recall curve.

        The legend shows the average precision (AP) score. Only for
        binary classification tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train", "test" or "both".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_prc")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)
        check_predict_proba(models, "plot_prc")

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        for m in models:
            for set_ in dataset:
                # Get precision-recall pairs for different thresholds
                precision, recall, _ = precision_recall_curve(
                    getattr(m, f"y_{set_}"), getattr(m, f"predict_proba_{set_}")[:, 1]
                )

                ap = f" (AP={round(m.scoring('ap', dataset=set_), 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + ap
                plt.plot(recall, precision, lw=2, label=label)

        self._plot(
            ax=ax,
            title=title,
            legend=("lower left", len(models)),
            xlabel="Recall",
            ylabel="Precision",
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_permutation_importance(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        show: Optional[int] = None,
        n_repeats: int = 10,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the feature permutation importance of models.

        If a permutation is repeated for the same model with the
        same amount of n_repeats, the calculation is skipped. The
        `feature_importance` attribute is updated with the extracted
        importance ranking.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show in
            the plot. None to show all.

        n_repeats: int, optional (default=10)
            Number of times to permute each feature.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_method(self, "plot_permutation_importance")
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        show = self._get_show(show)

        if n_repeats <= 0:
            raise ValueError(
                "Invalid value for the n_repeats parameter."
                f"Value should be >0, got {n_repeats}."
            )

        # Create dataframe with columns as indices to plot with barh
        df = pd.DataFrame(columns=["features", "score", "model"])

        # Create dictionary to store the permutations per model
        if not hasattr(self, "permutations"):
            self.permutations = {}

        for m in models:
            # If permutations are already calculated and n_repeats is
            # same, use known permutations (for efficient re-plotting)
            if not hasattr(m, "_repts"):
                m._repeats = -np.inf
            if m.name not in self.permutations or m._repeats != n_repeats:
                m._repeats = n_repeats
                # Permutation importances returns Bunch object
                self.permutations[m.name] = permutation_importance(
                    estimator=m.estimator,
                    X=m.X_test,
                    y=m.y_test,
                    scoring=self._metric[0],
                    n_repeats=n_repeats,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )

            # Append data to the dataframe
            for i, feature in enumerate(m.features):
                for score in self.permutations[m.name].importances[i, :]:
                    df = df.append({
                        "features": feature,
                        "score": score,
                        "model": m.name
                    }, ignore_index=True)

        # Get the column names sorted by sum of scores
        get_idx = df.groupby("features", as_index=False)["score"].sum()
        get_idx = get_idx.sort_values("score", ascending=False)
        column_order = get_idx.features.values[:show]

        # Save the best feature order
        self.branch.feature_importance = list(get_idx.features.values)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
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
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[1:], labels=labels[1:])
        else:
            # Hide the legend created by seaborn
            ax.legend().set_visible(False)

        self._plot(
            ax=ax,
            title=title,
            legend=("lower right" if len(models) > 1 else False, len(models)),
            xlabel="Score",
            figsize=figsize if figsize else (10, 4 + show // 2),
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_feature_importance(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        show: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot a tree-based model's feature importance.

        The importances are normalized in order to be able to compare
        them between models. The `feature_importance` attribute is
        updated with the extracted importance ranking.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show in
            the plot. None to show all.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_method(self, "plot_feature_importance")
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        show = self._get_show(show)

        # Create dataframe with columns as indices to plot with barh
        df = pd.DataFrame()

        for m in models:
            # Bagging is a special case where we use the feature_importance per est
            if not hasattr(m.estimator, "feature_importances_") and m.acronym != "Bag":
                raise PermissionError(
                    "The plot_feature_importance method is only available for "
                    f"models with a feature_importances_ attribute, got {m.name}."
                )

            # Bagging has no direct feature importance implementation
            if m.acronym == "Bag":
                feature_importances = np.mean(
                    [fi.feature_importances_ for fi in m.estimator.estimators_], axis=0
                )
            else:
                feature_importances = m.estimator.feature_importances_

            # Normalize for plotting values adjacent to bar
            max_feature_importance = max(feature_importances)
            for col, fx in zip(m.features, feature_importances):
                df.at[col, m.name] = fx / max_feature_importance

        # Save the best feature order
        best_fxs = df.fillna(0).sort_values(by=df.columns[-1], ascending=False)
        self.branch.feature_importance = list(best_fxs.index.values)

        # Select best and sort ascending (by sum of total importances)
        df = df.nlargest(show, columns=df.columns[-1])
        df = df.reindex(sorted(df.index, key=lambda i: df.loc[i].sum()))

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        df.plot.barh(
            ax=ax,
            width=0.75 if len(models) > 1 else 0.6,
            legend=True if len(models) > 1 else False
        )
        if len(models) == 1:
            for i, v in enumerate(df[df.columns[0]]):
                ax.text(v + 0.01, i - 0.08, f"{v:.2f}", fontsize=self.tick_fontsize)

        self._plot(
            ax=ax,
            title=title,
            legend=("lower right" if len(models) > 1 else False, len(models)),
            xlim=(0, 1.03 if len(models) > 1 else 1.09),
            xlabel="Score",
            figsize=figsize if figsize else (10, 4 + show // 2),
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_partial_dependence(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        features: Optional[Union[int, str, SEQUENCE_TYPES]] = None,
        target: Union[int, str] = 1,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the partial dependence of features.

        The partial dependence of a feature (or a set of features)
        corresponds to the average response of the model for each
        possible value of the feature. Two-way partial dependence
        plots are plotted as contour plots (only allowed for single
        model plots). The deciles of the feature values will be shown
        with tick marks on the x-axes for one-way plots, and on both
        axes for two-way plots.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        features: int, str, sequence or None, optional (default=None)
            Features or feature pairs (name or index) to get the partial
            dependence from. Maximum of 3 allowed. If None, it uses the
            best 3 features if the `feature_importance` attribute is
            defined, else it uses the first 3 features in the dataset.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        def convert_feature(feature):
            if isinstance(feature, str):
                try:
                    feature = list(m.features).index(feature)
                except ValueError:
                    raise ValueError(
                        "Invalid value for the features parameter. "
                        f"Feature {feature} not found in the dataset."
                    )
            elif feature > m.X.shape[1] - 1:  # -1 because of index 0
                raise ValueError(
                    "Invalid value for the features parameter. Dataset "
                    f"has {m.X.shape[1]} features, got index {feature}."
                )
            return int(feature)

        def get_features(features, m):
            # Default is to select the best or the first 3 features
            if not features:
                if not m.branch.feature_importance:
                    features = [0, 1, 2]
                else:
                    features = m.branch.feature_importance[:3]

            features = lst(features)
            if len(features) > 3:
                raise ValueError(
                    "Invalid value for the features parameter. "
                    f"Maximum 3 allowed, got {len(features)}."
                )

            # Convert features into a sequence of int tuples
            cols = []
            for fxs in features:
                if isinstance(fxs, (int, str)):
                    cols.append((convert_feature(fxs),))
                elif len(models) == 1:
                    if len(fxs) == 2:
                        cols.append(tuple(convert_feature(fx) for fx in fxs))
                    else:
                        raise ValueError(
                            "Invalid value for the features parameter. Values "
                            f"should be single or in pairs, got {fxs}."
                        )
                else:
                    raise ValueError(
                        "Invalid value for the features parameter. Feature pairs "
                        f"are invalid when plotting multiple models, got {fxs}."
                    )

            return cols

        check_method(self, "plot_partial_dependence")
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        target = self._get_target(target) if self.task.startswith("multi") else 0

        axes = []
        fig = self._get_figure()
        n_cols = 3 if not features else len(lst(features))
        gs = GridSpecFromSubplotSpec(1, n_cols, BasePlotter._fig.grid)
        for i in range(n_cols):
            axes.append(fig.add_subplot(gs[0, i]))

        names = []  # Names of the features (to compare between models)
        for m in models:
            # Since every model can have different fxs, select them again
            cols = get_features(features, m)

            # Make sure the models use the same features
            if len(models) > 1:
                fxs_names = [m.features[col[0]] for col in cols]
                if not names:
                    names = fxs_names
                elif names != fxs_names:
                    raise ValueError(
                        "Invalid value for the features parameter. Not all models "
                        f"use the same features, got {names} and {fxs_names}."
                    )

            # Compute averaged predictions
            pd_results = Parallel(n_jobs=self.n_jobs)(
                delayed(partial_dependence)(m.estimator, m.X_test, col) for col in cols
            )

            # Get global min and max average predictions of PD grouped by plot type
            pdp_lim = {}
            for pred, values in pd_results:
                min_pd, max_pd = pred[target].min(), pred[target].max()
                old_min, old_max = pdp_lim.get(len(values), (min_pd, max_pd))
                pdp_lim[len(values)] = (min(min_pd, old_min), max(max_pd, old_max))

            deciles = {}
            for fx in chain.from_iterable(cols):
                if fx not in deciles:  # Skip if the feature is repeated
                    X_col = _safe_indexing(m.X_test, fx, axis=1)
                    deciles[fx] = mquantiles(X_col, prob=np.arange(0.1, 1.0, 0.1))

            for axi, fx, (pred, values) in zip(axes, cols, pd_results):
                # For both types: draw ticks on the horizontal axis
                trans = blended_transform_factory(axi.transData, axi.transAxes)
                axi.vlines(deciles[fx[0]], 0, 0.05, transform=trans, color="k")
                axi.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
                self._plot(ax=axi, xlabel=m.columns[fx[0]])

                # Draw line or contour plot
                if len(values) == 1:
                    axi.plot(values[0], pred[target].ravel(), lw=2, label=m.name)
                else:
                    # Create contour levels for two-way plots
                    levels = np.linspace(pdp_lim[2][0], pdp_lim[2][1] + 1e-9, num=8)

                    # Draw contour plot
                    XX, YY = np.meshgrid(values[0], values[1])
                    Z = pred[target].T
                    CS = axi.contour(XX, YY, Z, levels=levels, linewidths=0.5)
                    axi.clabel(CS, fmt="%2.2f", colors="k", fontsize=10, inline=True)
                    axi.contourf(
                        XX, YY, Z,
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
                        ylabel=m.columns[fx[1]],
                        xlim=(min(XX.flatten()), max(XX.flatten())),
                        ylim=(min(YY.flatten()), max(YY.flatten())),
                    )

        # Place y-label and legend on first non-contour plot
        for axi in axes:
            if not axi.get_ylabel():
                self._plot(
                    ax=axi,
                    ylabel="Score",
                    legend=("best" if len(models) > 1 else False, len(models))
                )
                break

        if title:
            # Place title if not in canvas, else above first or middle image
            if len(cols) == 1 or (len(cols) == 2 and BasePlotter._fig.is_canvas):
                axes[0].set_title(title, fontsize=self.title_fontsize, pad=20)
            elif len(cols) == 3:
                axes[1].set_title(title, fontsize=self.title_fontsize, pad=20)
            elif not BasePlotter._fig.is_canvas:
                plt.suptitle(title, fontsize=self.title_fontsize)

        self._plot(figsize=figsize, filename=filename, display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_errors(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot a model's prediction errors.

        Plot the actual targets from a set against the predicted values
        generated by the regressor. A linear fit is made on the data.
        The gray, intersected line shows the identity line. This pot can
        be useful to detect noise or heteroscedasticity along a range of
        the target domain. Only for regression tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the errors. Options are
            "train", "test" or "both".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        check_goal(self, "plot_errors", "regression")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        for m in models:
            for set_ in dataset:
                r2 = f" (R$^2$={round(m.scoring('r2', dataset=set_), 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + r2
                ax.scatter(
                    x=getattr(self, f"y_{set_}"),
                    y=getattr(m, f"predict_{set_}"),
                    alpha=0.8,
                    label=label,
                )

                # Fit the points using linear regression
                from .models import OrdinaryLeastSquares

                model = OrdinaryLeastSquares(self).get_estimator()
                model.fit(
                    X=np.array(getattr(self, f"y_{set_}")).reshape(-1, 1),
                    y=getattr(m, f"predict_{set_}"),
                )

                # Draw the fit
                x = np.linspace(*ax.get_xlim(), 100)
                ax.plot(x, model.predict(x[:, np.newaxis]), lw=2, alpha=1)

        # Get limits before drawing the identity line
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        # Draw identity line
        ax.plot(xlim, ylim, ls="--", lw=1, color="k", alpha=0.7)

        self._plot(
            ax=ax,
            title=title,
            legend=("upper left", len(models)),
            xlabel="True value",
            ylabel="Predicted value",
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_residuals(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot a model's residuals.

        The plot shows the residuals (difference between the predicted
        and the true value) on the vertical axis and the independent
        variable on the horizontal axis. The gray, intersected line
        shows the identity line. This plot can be useful to analyze the
        variance of the error of the regressor. If the points are
        randomly dispersed around the horizontal axis, a linear
        regression model is appropriate for the data; otherwise, a
        non-linear model is more appropriate. Only for regression tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train", "test" or "both".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        check_goal(self, "plot_residuals", "regression")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        gs = GridSpecFromSubplotSpec(1, 4, BasePlotter._fig.grid, wspace=0.05)
        ax1 = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:4])

        for m in models:
            for set_ in dataset:
                r2 = f" (R$^2$={round(m.scoring('r2', dataset=set_), 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + r2
                res = np.subtract(
                    getattr(m, f"predict_{set_}"),
                    getattr(self, f"y_{set_}"),
                )

                ax1.scatter(getattr(m, f"predict_{set_}"), res, alpha=0.7, label=label)
                ax2.hist(res, orientation="horizontal", histtype="step", linewidth=1.2)

        # Get limits before drawing the identity line
        xlim, ylim = ax1.get_xlim(), ax1.get_ylim()
        ax1.hlines(0, *xlim, ls="--", lw=1, color="k", alpha=0.8)  # Identity line

        ax2.set_yticklabels([])
        self._plot(ax=ax2, xlabel="Distribution")

        if title:
            if not BasePlotter._fig.is_canvas:
                plt.suptitle(title, fontsize=self.title_fontsize, y=0.98)
            else:
                ax1.set_title(title, fontsize=self.title_fontsize, pad=20)

        self._plot(
            ax=ax1,
            legend=("lower right", len(models)),
            ylabel="Residuals",
            xlabel="True value",
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_confusion_matrix(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        normalize: bool = False,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot a model's confusion matrix.

        Only for classification tasks.
        If 1 model: Plot the confusion matrix in a heatmap.
        If >1 models: Compare TP, FP, FN and TN in a barplot. Not
                      implemented for multiclass classification tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the confusion matrix.
            Options are "train" or "test".

        normalize: bool, optional (default=False)
           Whether to normalize the matrix. Only for the heatmap plot.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to
            the plot's type.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        check_goal(self, "plot_confusion_matrix", "classification")
        models = self._get_subclass(models)

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
        for m in models:
            cm = m.scoring("cm", dataset.lower())

            if len(models) == 1:  # Create matrix heatmap
                if normalize:
                    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

                fig = self._get_figure()
                ax = fig.add_subplot(BasePlotter._fig.grid)
                im = ax.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))

                # Create an axes on the right side of ax. The under of cax will
                # be 5% of ax and the padding between cax and ax will be fixed
                # at 0.3 inch.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.3)
                cbar = ax.figure.colorbar(im, cax=cax)
                ax.set(
                    xticks=np.arange(cm.shape[1]),
                    yticks=np.arange(cm.shape[0]),
                    xticklabels=self.mapping.keys(),
                    yticklabels=self.mapping.keys(),
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

            else:  # Create barplot
                df[m.name] = cm.ravel()

        if len(models) > 1:
            ax = df.plot.barh(width=0.6)
            self._plot(
                ax=ax,
                title=title,
                legend=("best", len(models)),
                xlabel="Count",
                figsize=figsize if figsize else (10, 6),
                filename=filename,
                display=display,
            )
        else:
            self._plot(
                ax=ax,
                title=title,
                xlabel="Predicted label",
                ylabel="True label",
                figsize=figsize if figsize else (8, 6),
                filename=filename,
                display=display
            )

    @composed(crash, plot_from_model, typechecked)
    def plot_threshold(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        steps: int = 100,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot metric performances against threshold values.

        Only for binary classification tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        metric: str, callable, sequence or None, optional (default=None)
            Metric(s) to plot. These can be one of sklearn's SCORERS,
            a metric function or a scorer. If None, the metric used to
            run the pipeline is used.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train", "test" or "both".

        steps: int, optional (default=100)
            Number of thresholds measured.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_threshold")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)
        check_predict_proba(models, "plot_threshold")

        if metric is None:
            metric = self._metric
        elif not isinstance(metric, list):
            metric = [metric]

        # Convert all strings to functions
        metric_list = []
        for met in metric:
            if isinstance(met, str):  # It is one of sklearn predefined metrics
                if met in METRIC_ACRONYMS:
                    met = METRIC_ACRONYMS[met]

                if met not in SCORERS:
                    raise ValueError(
                        "Unknown value for the metric parameter, "
                        f"got {met}. Try one of {list(SCORERS)}."
                    )
                metric_list.append(SCORERS[met]._score_func)
            elif hasattr(met, "_score_func"):  # It is a scorer
                metric_list.append(met._score_func)
            else:  # It is a metric function
                metric_list.append(met)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        steps = np.linspace(0, 1, steps)
        for m in models:
            for met in metric_list:  # Create dict of empty arrays
                for set_ in dataset:
                    results = []
                    for step in steps:
                        predictions = (
                            getattr(m, f"predict_proba_{set_}")[:, 1] >= step
                        ).astype(bool)
                        results.append(met(getattr(m, f"y_{set_}"), predictions))

                    if len(models) == 1:
                        l_set = f"{set_} - " if len(dataset) > 1 else ""
                        label = f"{l_set}{met.__name__}"
                    else:
                        l_set = f" - {set_}" if len(dataset) > 1 else ""
                        label = f"{m.name}{l_set} ({met.__name__})"
                    ax.plot(steps, results, label=label, lw=2)

        self._plot(
            ax=ax,
            title=title,
            legend=("best", len(models)),
            xlabel="Threshold",
            ylabel="Score",
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_probabilities(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        target: Union[int, str] = 1,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the probability distribution of the target classes.

        Only for classification tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train", "test" or "both".

        target: int or str, optional (default=1)
            Probability of being that class in the target column as
            index or name. Only for multiclass classification.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        check_goal(self, "plot_probabilities", "classification")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)
        target = self._get_target(target)
        check_predict_proba(models, "plot_probabilities")
        palette = cycle(sns.color_palette())

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        for m in models:
            for set_ in dataset:
                for key, value in self.mapping.items():
                    # Get indices per class
                    idx = np.where(getattr(m, f"y_{set_}") == value)[0]

                    label = m.name + (f" - {set_}" if len(dataset) > 1 else "")
                    sns.histplot(
                        data=getattr(m, f"predict_proba_{set_}")[idx, target],
                        kde=True,
                        bins=50,
                        label=label + f" ({self.target}={key})",
                        color=next(palette),
                        ax=ax,
                    )

        self._plot(
            ax=ax,
            title=title,
            legend=("best", len(models)),
            xlabel="Probability",
            ylabel="Counts",
            xlim=(0, 1),
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_calibration(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        n_bins: int = 10,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 10),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the calibration curve for a binary classifier.

        Well calibrated classifiers are probabilistic classifiers for
        which the output of the `predict_proba` method can be directly
        interpreted as a confidence level. For instance a well
        calibrated (binary) classifier should classify the samples such
        that among the samples to which it gave a `predict_proba` value
        close to 0.8, approx. 80% actually belong to the positive class.
        This figure shows two plots: the calibration curve, where the
        x-axis represents the average predicted probability in each bin
        and the y-axis is the fraction of positives, i.e. the proportion
        of samples whose class is the positive class (in each bin); and
        a distribution of all predicted probabilities of the classifier.
        Code snippets from https://scikit-learn.org/stable/auto_examples/
        calibration/plot_calibration_curve.html

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        n_bins: int, optional (default=10)
            Number of bins for the calibration calculation and the
            histogram. Minimum of 5 required.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 10))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_calibration")
        models = self._get_subclass(models)

        if n_bins < 5:
            raise ValueError(
                "Invalid value for the n_bins parameter."
                f"Value should be >=5, got {n_bins}."
            )

        fig = self._get_figure()
        gs = GridSpecFromSubplotSpec(4, 1, BasePlotter._fig.grid, hspace=0.05)
        ax1 = fig.add_subplot(gs[:3, 0])
        ax2 = fig.add_subplot(gs[3:4, 0], sharex=ax1)
        ax1.plot([0, 1], [0, 1], color="k", ls="--")
        for m in models:
            if hasattr(m.estimator, "decision_function"):
                prob = m.decision_function_test
                prob = (prob - prob.min()) / (prob.max() - prob.min())
            elif hasattr(m.estimator, "predict_proba"):
                prob = m.predict_proba_test[:, 1]

            # Get calibration (frac of positives and predicted values)
            frac_pos, pred = calibration_curve(self.y_test, prob, n_bins=n_bins)

            ax1.plot(pred, frac_pos, marker="o", lw=2, label=f"{m.name}")
            ax2.hist(prob, n_bins, range=(0, 1), label=m.name, histtype="step", lw=2)

        plt.setp(ax1.get_xticklabels(), visible=False)
        self._plot(
            ax=ax1,
            title=title,
            legend=("lower right" if len(models) > 1 else False, len(models)),
            ylabel="Fraction of positives",
            ylim=(-0.05, 1.05),
        )
        self._plot(
            ax=ax2,
            xlabel="Predicted value",
            ylabel="Count",
            figsize=figsize,
            filename=filename,
            display=display
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_gains(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the cumulative gains curve.

        Only for binary classification tasks. Code snippet from
        https://github.com/reiinakano/scikit-plot/

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the gains curve. Options
            are "train", "test" or "both".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_gains")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)
        check_predict_proba(models, "plot_gains")

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        ax.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.7)
        for m in models:
            for set_ in dataset:
                y_true = getattr(m, f"y_{set_}") == 1  # Make y_true a bool vector

                # Get sorted indices
                sort_idx = np.argsort(getattr(m, f"predict_proba_{set_}")[:, 1])[::-1]

                # Correct indices for the test set (add train set length)
                if set_ == "test":
                    sort_idx = [i + len(m.y_train) for i in sort_idx]

                # Compute cumulative gains
                gains = np.cumsum(y_true.loc[sort_idx]) / float(np.sum(y_true))

                x = np.arange(start=1, stop=len(y_true) + 1) / float(len(y_true))
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "")
                ax.plot(x, gains, lw=2, label=label)

        self._plot(
            ax=ax,
            title=title,
            legend=("lower right", len(models)),
            xlabel="Fraction of sample",
            ylabel="Gain",
            xlim=(0, 1),
            ylim=(0, 1.02),
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_lift(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the lift curve.

        Only for binary classification tasks. Code snippet from
        https://github.com/reiinakano/scikit-plot/

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train", "test" or "both".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_lift")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)
        check_predict_proba(models, "plot_lift")

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        ax.plot([0, 1], [1, 1], "k--", lw=2, alpha=0.7)
        for m in models:
            for set_ in dataset:
                y_true = getattr(m, f"y_{set_}") == 1  # Make y_true a bool vector

                # Get sorted indices and correct for the test set
                sort_idx = np.argsort(getattr(m, f"predict_proba_{set_}")[:, 1])[::-1]

                # Correct indices for the test set (add train set length)
                if set_ == "test":  # Add the training set length to the indices
                    sort_idx = [i + len(m.y_train) for i in sort_idx]

                # Compute cumulative gains
                gains = np.cumsum(y_true.loc[sort_idx]) / float(np.sum(y_true))

                x = np.arange(start=1, stop=len(y_true) + 1) / float(len(y_true))
                lift = f" (Lift={round(m.scoring('lift', dataset=set_), 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + lift
                ax.plot(x, gains / x, lw=2, label=label)

        self._plot(
            ax=ax,
            title=title,
            legend=("upper right", len(models)),
            xlabel="Fraction of sample",
            ylabel="Lift",
            xlim=(0, 1),
            figsize=figsize,
            filename=filename,
            display=display,
        )

    # SHAP plots =================================================== >>

    @composed(crash, plot_from_model, typechecked)
    def bar_plot(
            self,
            models: Optional[Union[str, SEQUENCE_TYPES]] = None,
            index: Optional[Union[int, tuple, slice]] = None,
            show: Optional[int] = None,
            target: Union[int, str] = 1,
            title: Optional[str] = None,
            figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
            filename: Optional[str] = None,
            display: bool = True,
            **kwargs,
    ):
        """Plot SHAP's bar plot.

        Create a bar plot of a set of SHAP values. If a single sample
        is passed, then the SHAP values are plotted. If many samples
        are passed, then the mean absolute value for each feature
        column is plotted.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected. Note that selecting multiple models
            will raise an exception. To avoid this, call the plot from
            a model.

        index: int, tuple, slice or None, optional (default=None)
            Indices of the rows in the dataset to plot. If shape
            (n, m), it selects rows n until m. If None, it selects
            all rows in the test set.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show in
            the plot. None to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        **kwargs
            Additional keyword arguments for SHAP's bar plot.

        """
        check_method(self, "bar_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        index = self._get_index(index, m)
        show = self._get_show(show)
        target = self._get_target(target)
        shap_values = self._get_shap(m, index, target)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.plots.bar(shap_values, max_display=show, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        self._plot(
            ax=ax,
            title=title,
            figsize=figsize if figsize else (10, 4 + show // 2),
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def beeswarm_plot(
            self,
            models: Optional[Union[str, SEQUENCE_TYPES]] = None,
            index: Optional[Union[tuple, slice]] = None,
            show: Optional[int] = None,
            target: Union[int, str] = 1,
            title: Optional[str] = None,
            figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
            filename: Optional[str] = None,
            display: bool = True,
            **kwargs,
    ):
        """Plot SHAP's beeswarm plot.

        The plot is colored by feature values.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected. Note that selecting multiple models
            will raise an exception. To avoid this, call the plot from
            a model.

        index: tuple, slice or None, optional (default=None)
            Indices of the rows in the dataset to plot. If shape
            (n, m), it selects rows n until m. If None, it selects
            all rows in the test set. The beeswarm plot does not
            support plotting a single sample.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show in
            the plot. None to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        **kwargs
            Additional keyword arguments for SHAP's beeswarm plot.

        """
        check_method(self, "beeswarm_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        index = self._get_index(index, m)
        show = self._get_show(show)
        target = self._get_target(target)
        shap_values = self._get_shap(m, index, target)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.plots.beeswarm(shap_values, max_display=show, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        self._plot(
            ax=ax,
            title=title,
            figsize=figsize if figsize else (10, 4 + show // 2),
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def decision_plot(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        index: Optional[Union[int, tuple, slice]] = None,
        show: Optional[int] = None,
        target: Union[int, str] = 1,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: bool = True,
        **kwargs,
    ):
        """Plot SHAP's decision plot.

        Visualize model decisions using cumulative SHAP values. Each
        plotted line explains a single model prediction. If a single
        prediction is plotted, feature values will be printed in the
        plot (if supplied). If multiple predictions are plotted
        together, feature values will not be printed. Plotting too
        many predictions together will make the plot unintelligible.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected. Note that selecting multiple
            models will raise an exception. To avoid this, call
            the plot from a model.

        index: int, tuple, slice or None, optional (default=None)
            Indices of the rows in the dataset to plot. If shape
            (n, m), it selects rows n until m. If None, it selects
            all rows in the test set.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show in
            the plot. None to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size
            to the number of features.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        **kwargs
            Additional keyword arguments for SHAP's decision plot.

        """
        check_method(self, "decision_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        index = self._get_index(index, m)
        show = self._get_show(show)
        target = self._get_target(target)

        # Get shap information from old API
        if not m.explainer:
            m.explainer = shap.Explainer(m.estimator, m.X_train)

        shap_values = m.explainer.shap_values(index)
        expected_value = m.explainer.expected_value

        # Select the values corresponding to the target
        if not np.array(shap_values).shape == (len(index), m.X.shape[1]):
            shap_values = shap_values[target]

        # Select the target expected value or return all
        if isinstance(expected_value, (list, np.ndarray)):
            if len(expected_value) == len(m.y.unique()):
                expected_value = expected_value[target]

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.decision_plot(
            base_value=expected_value,
            shap_values=shap_values,
            features=index,
            feature_display_range=slice(-1, -show - 1, -1),
            auto_size_plot=False,
            show=False,
            **kwargs,
        )

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        self._plot(
            ax=ax,
            title=title,
            figsize=figsize if figsize else (10, 4 + show // 2),
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def force_plot(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        index: Optional[Union[int, tuple, slice]] = None,
        target: Union[str, int] = 1,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (14, 6),
        filename: Optional[str] = None,
        display: bool = True,
        **kwargs,
    ):
        """Plot SHAP's force plot.

        Visualize the given SHAP values with an additive force layout.
        Note that by default this plot will render using javascript.
        For a regular figure use `matplotlib=True` (this option is
        only available when only a single sample is plotted).

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected. Note that selecting multiple models
            will raise an exception. To avoid this, call the plot from
            a model.

        index: int, tuple, slice or None, optional (default=None)
            Indices of the rows in the dataset to plot. If (n, m),
            it selects rows n until m. If None, it selects all rows
            in the test set.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(14, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If matplotlib=False, the figure will
            be saved as an html file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        **kwargs
            Additional keyword arguments for SHAP's force plot.

        """
        if getattr(BasePlotter._fig, "is_canvas", None):
            raise PermissionError(
                "The force_plot method can not be called from a "
                "canvas because of incompatibility of the APIs."
            )

        check_method(self, "force_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        index = self._get_index(index, m)
        target = self._get_target(target)

        # Get shap information from old API
        if not m.explainer:
            m.explainer = shap.Explainer(m.estimator, m.X_train)

        shap_values = m.explainer.shap_values(index)
        expected_value = m.explainer.expected_value

        # Select the values corresponding to the target
        if not np.array(shap_values).shape == (len(index), m.X.shape[1]):
            shap_values = shap_values[target]

        # Select the target expected value or return all
        if isinstance(expected_value, (list, np.ndarray)):
            if len(expected_value) == len(m.y.unique()):
                expected_value = expected_value[target]

        sns.set_style("white")  # Only for this plot
        plot = shap.force_plot(
            base_value=expected_value,
            shap_values=shap_values,
            features=index,
            figsize=figsize,
            show=False,
            **kwargs,
        )

        sns.set_style(self.style)
        if kwargs.get("matplotlib"):
            self._plot(title=title, filename=filename, display=display)
        else:
            if filename:  # Save to an html file
                fn = filename if filename.endswith(".html") else filename + ".html"
                shap.save_html(fn, plot)
            if display:
                try:  # Render if possible (for notebooks)
                    from IPython.display import display

                    shap.initjs()
                    display(plot)
                except ModuleNotFoundError:
                    pass

    @composed(crash, plot_from_model, typechecked)
    def heatmap_plot(
            self,
            models: Optional[Union[str, SEQUENCE_TYPES]] = None,
            index: Optional[Union[tuple, slice]] = None,
            show: Optional[int] = None,
            target: Union[int, str] = 1,
            title: Optional[str] = None,
            figsize: Tuple[SCALAR, SCALAR] = (8, 6),
            filename: Optional[str] = None,
            display: bool = True,
            **kwargs,
    ):
        """Plot SHAP's heatmap plot.

        This plot is designed to show the population substructure of a
        dataset using supervised clustering and a heatmap. Supervised
        clustering involves clustering data points not by their original
        feature values but by their explanations.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected. Note that selecting multiple models
            will raise an exception. To avoid this, call the plot from
            a model.

        index: tuple, slice or None, optional (default=None)
            Indices of the rows in the dataset to plot. If shape
            (n, m), it selects rows n until m. If None, it selects
            all rows in the test set. The heatmap plot does not
            support plotting a single sample.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show in
            the plot. None to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=(8, 6)))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        **kwargs
            Additional keyword arguments for SHAP's heatmap plot.

        """
        check_method(self, "heatmap_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        index = self._get_index(index, m)
        show = self._get_show(show)
        target = self._get_target(target)
        shap_values = self._get_shap(m, index, target)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.plots.heatmap(shap_values, max_display=show, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        self._plot(ax, title=title, figsize=figsize, filename=filename, display=display)

    @composed(crash, plot_from_model, typechecked)
    def scatter_plot(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        index: Optional[Union[tuple, slice]] = None,
        feature: Union[int, str] = 0,
        target: Union[int, str] = 1,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
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
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected. Note that selecting multiple models
            will raise an exception. To avoid this, call the plot from
            a model.

        index: tuple, slice or None, optional (default=None)
            Indices of the rows in the dataset to plot. If shape
            (n, m), it selects rows n until m. If None, it selects
            all rows in the test set. The scatter plot does not
            support plotting a single sample.

        feature: int or str, optional (default=0)
            Index or name of the feature to plot.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        **kwargs
            Additional keyword arguments for SHAP's scatter plot.

        """
        check_method(self, "scatter_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        index = self._get_index(index, m)
        target = self._get_target(target)
        shap_values = self._get_shap(m, index, target)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.plots.scatter(
            shap_values[:, feature],
            color=shap_values,
            ax=ax,
            show=False,
            **kwargs
        )

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=self.label_fontsize, labelpad=12)
        self._plot(ax, title=title, figsize=figsize, filename=filename, display=display)

    @composed(crash, plot_from_model, typechecked)
    def waterfall_plot(
            self,
            models: Optional[Union[str, SEQUENCE_TYPES]] = None,
            index: Optional[int] = None,
            show: Optional[int] = None,
            target: Union[int, str] = 1,
            title: Optional[str] = None,
            figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
            filename: Optional[str] = None,
            display: bool = True,
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
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected. Note that selecting multiple models
            will raise an exception. To avoid this, call the plot from
            a model.

        index: int or None, optional (default=None)
            Index of the row in the dataset to plot. If None,
            it selects the first row in the test set. The
            waterfall plot does not support plotting multiple
            samples.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show in
            the plot. None to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_method(self, "waterfall_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        index = m.X_test.iloc[[0]] if index is None else self._get_index(index, m)
        show = self._get_show(show)
        target = self._get_target(target)
        shap_values = self._get_shap(m, index, target)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.plots.waterfall(shap_values, max_display=show, show=False)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        self._plot(
            ax=ax,
            title=title,
            figsize=figsize if figsize else (10, 4 + show // 2),
            filename=filename,
            display=display,
        )


class SuccessiveHalvingPlotter(BaseModelPlotter):
    """Plots for the SuccessiveHalving classes."""

    @composed(crash, plot_from_model, typechecked)
    def plot_successive_halving(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        metric: Union[int, str] = 0,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot scores per iteration of the successive halving.

        Only available if the models were fitted via successive_halving.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        metric: int or str, optional (default=0)
            Index or name of the metric. Only for multi-metric runs.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        metric = self._get_metric(metric)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        x, y, std = defaultdict(list), defaultdict(list), defaultdict(list)
        for m in models:
            y[m._group].append(get_best_score(m, metric))
            x[m._group].append(m.branch.idx[0] // m._train_idx)
            if m.std_bagging:
                std[m._group].append(lst(m.std_bagging)[metric])

        for k in x:
            if not std:
                ax.plot(x[k], y[k], lw=2, marker="o", label=k)
            else:
                ax.plot(x[k], y[k], lw=2, marker="o")
                ax.errorbar(x[k], y[k], std[k], lw=1, marker="o", label=k)
                plus, minus = np.add(y[k], std[k]), np.subtract(y[k], std[k])
                ax.fill_between(x[k], plus, minus, alpha=0.3)

        n_models = [len(self.train) // m._train_idx for m in models]
        ax.set_xlim(max(n_models) + 0.1, min(n_models) - 0.1)
        ax.set_xticks(range(1, max(n_models) + 1))
        self._plot(
            ax=ax,
            title=title,
            legend=("lower right", len(x)),
            xlabel="n_models",
            ylabel=self._metric[metric].name,
            figsize=figsize,
            filename=filename,
            display=display,
        )


class TrainSizingPlotter(BaseModelPlotter):
    """Plots for the TrainSizing classes."""

    @composed(crash, plot_from_model, typechecked)
    def plot_learning_curve(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        metric: Union[int, str] = 0,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot the learning curve: score vs number of training samples.

        Only available if the models were fitted using train sizing.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        metric: int or str, optional (default=0)
            Index or name of the metric. Only for multi-metric runs.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        metric = self._get_metric(metric)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        x, y, std = defaultdict(list), defaultdict(list), defaultdict(list)
        for m in models:
            y[m._group].append(get_best_score(m, metric))
            x[m._group].append(m._train_idx)
            if m.std_bagging:
                std[m._group].append(lst(m.std_bagging)[metric])

        for k in x:
            if not std:
                ax.plot(x[k], y[k], lw=2, marker="o", label=k)
            else:
                ax.plot(x[k], y[k], lw=2, marker="o")
                ax.errorbar(x[k], y[k], std[k], lw=1, marker="o", label=k)
                plus, minus = np.add(y[k], std[k]), np.subtract(y[k], std[k])
                ax.fill_between(x[k], plus, minus, alpha=0.3)

        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))
        self._plot(
            ax=ax,
            title=title,
            legend=("lower right", len(x)),
            xlabel="Number of training samples",
            ylabel=self._metric[metric].name,
            figsize=figsize,
            filename=filename,
            display=display,
        )


class ATOMPlotter(FSPlotter, SuccessiveHalvingPlotter, TrainSizingPlotter):
    """Plots for the ATOM class."""

    @composed(crash, typechecked)
    def plot_correlation(
        self,
        columns: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        method: str = "pearson",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (8, 7),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot a correlation matrix. Ignores categorical columns.

        Parameters
        ----------
        columns: slice, sequence or None, optional (default=None)
            Slice, names or indices of the columns to plot. If None,
            plot all columns in the dataset.

        method: str, optional (default="pearson")
            Method of correlation. Choose from "pearson", "kendall"
            or "spearman".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(8, 7))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_method(self, "plot_correlation")
        columns = self._get_columns(columns)
        if method.lower() not in ("pearson", "kendall", "spearman"):
            raise ValueError(
                f"Invalid value for the method parameter, got {method}. "
                "Available options are: pearson, kendall or spearman."
            )

        # Compute the correlation matrix
        corr = self.dataset[columns].corr(method=method.lower())

        # Drop first row and last column (diagonal line)
        corr = corr.iloc[1:].drop(columns[-1], axis=1)

        # Generate a mask for the upper triangle
        # k=1 means keep outermost diagonal line
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        sns.set_style("white")  # Only for this plot
        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        sns.heatmap(
            data=corr,
            mask=mask,
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            vmax=0.3,
            center=0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        sns.set_style(self.style)  # Set back to original style
        self._plot(
            ax=ax,
            title=title,
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_scatter_matrix(
        self,
        columns: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 10),
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot a scatter matrix. Ignores categorical columns.

        Parameters
        ----------
        columns: slice, sequence or None, optional (default=None)
            Slice, names or indices of the columns to plot. If None,
            plot all columns in the dataset.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=(10, 10)))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        if getattr(BasePlotter._fig, "is_canvas", None):
            raise PermissionError(
                "The plot_scatter_matrix method can not be called from a "
                "canvas because of incompatibility of the APIs."
            )

        check_method(self, "plot_scatter_matrix")
        columns = self._get_columns(columns)
        grid = sns.pairplot(self.dataset[columns],  diag_kind="kde")

        # Set right fontsize for all axes in grid
        for axi in grid.axes.flatten():
            axi.tick_params(axis='both', labelsize=self.tick_fontsize)
            axi.set_xlabel(axi.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
            axi.set_ylabel(axi.get_ylabel(), fontsize=self.label_fontsize, labelpad=12)

        self._plot(
            title=title,
            figsize=figsize if figsize else (10, 10),
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_distributions(
        self,
        columns: Union[int, str, slice, SEQUENCE_TYPES] = 0,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot column distributions.

        Parameters
        ----------
        columns: int, str, slice or sequence, optional (default=0)
            Slice, names or indices of the columns to plot. It is only
            possible to plot one categorical column (which will show the
            seven most frequent values). If more than one categorical
            columns are selected, the categorical features are ignored.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to
            the length of the pipeline.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_method(self, "plot_distributions")
        columns = self._get_columns(columns)
        palette = cycle(sns.color_palette())

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        if columns in list(self.dataset.select_dtypes(exclude="number").columns):
            data = self.dataset[columns].value_counts(ascending=True)
            data[-min(len(data), 7):].plot.barh(
                ax=ax,
                width=0.6,
                label=data.name + f": {len(data)} classes",
            )

            # Add the counts at the end of the bar
            for i, v in enumerate(data[-min(len(data), 7):]):
                ax.text(v + 0.01 * max(data), i - 0.08, v, fontsize=self.tick_fontsize)

            self._plot(
                ax=ax,
                xlim=(min(data) - 0.1*min(data), max(data) + 0.1 * max(data)),
                title=title,
                xlabel="Counts",
                legend=("lower right", 1),
                figsize=figsize if figsize else (10, 7),
                filename=filename,
                display=display,
            )
        else:
            for c in lst(columns):
                sns.histplot(
                    data=self.dataset,
                    x=c,
                    kde=True,
                    label=c,
                    color=next(palette),
                    ax=ax,
                )

            self._plot(
                ax=ax,
                title=title,
                xlabel="Values",
                ylabel="Counts",
                legend=("best", len(columns)),
                figsize=figsize if figsize else (10, 6),
                filename=filename,
                display=display,
            )

    @composed(crash, typechecked)
    def plot_pipeline(
        self,
        show_params: bool = True,
        branch: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: bool = True,
    ):
        """Plot a diagram of every estimator in a branch.

        Parameters
        ----------
        show_params: bool, optional (default=True)
            Whether to show the parameters used for every estimator.

        branch: str or None, optional (default=None)
            Name of the branch to plot. If None, plot the current
            branch.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to
            the length of the pipeline.

        filename: str or None, optional (default=None)
            Name of the file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        if not branch:
            branch = self._current
        elif branch not in self._branches:
            raise ValueError(
                "Invalid value for the branch parameter. Unknown branch,"
                f" got {branch}. Choose from: {', '.join(self._branches)}."
            )

        # Calculate figure's limits
        params = []
        ylim = 30
        for est in self._branches[branch].pipeline:
            ylim += 15
            if show_params:
                params.append([
                    p for p in signature(est.__init__).parameters
                    if p not in BaseTransformer.attrs + ["self"]
                ])
                ylim += len(params[-1]) * 10

        sns.set_style("white")  # Only for this plot
        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        # Shared parameters for the blocks
        con = ConnectionStyle("angle", angleA=0, angleB=90, rad=0)
        arrow = dict(arrowstyle="<|-", lw=1, color="k", connectionstyle=con)

        # Draw the main class
        ax.text(
            x=20,
            y=ylim - 20,
            s=self.__class__.__name__,
            ha="center",
            size=self.label_fontsize + 2,
        )

        pos_param = ylim - 20
        pos_estimator = pos_param

        for i, est in enumerate(self._branches[branch].pipeline):
            ax.annotate(
                text=est.__class__.__name__,
                xy=(15, pos_estimator),
                xytext=(30, pos_param - 3 - 15),
                ha="left",
                size=self.label_fontsize,
                arrowprops=arrow,
            )

            pos_param -= 15
            pos_estimator = pos_param

            if show_params:
                for j, key in enumerate(params[i]):
                    ax.annotate(
                        text=f"{key}: {getattr(est, key)}",
                        xy=(32, pos_param - 6 if j == 0 else pos_param + 1),
                        xytext=(40, pos_param - 12),
                        ha="left",
                        size=self.label_fontsize - 4,
                        arrowprops=arrow,
                    )

                    pos_param -= 10

        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        plt.axis("off")

        sns.set_style(self.style)  # Set back to original style
        self._plot(
            ax=ax,
            title=title,
            xlim=(0, 100),
            ylim=(0, ylim),
            figsize=figsize if figsize else (8, ylim // 30),
            filename=filename,
            display=display,
        )

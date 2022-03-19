# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the plotting classes.

"""

from collections import defaultdict
from contextlib import contextmanager
from inspect import signature
from itertools import chain, cycle
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Plotting packages
import shap
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionStyle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory
from mlflow.tracking import MlflowClient
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nltk.collocations import (
    BigramCollocationFinder, QuadgramCollocationFinder,
    TrigramCollocationFinder,
)
from scipy import stats
from scipy.stats.mstats import mquantiles
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix, det_curve, precision_recall_curve, roc_curve,
)
# Sklearn
from sklearn.utils import _safe_indexing
from typeguard import typechecked
from wordcloud import WordCloud

from atom.basetransformer import BaseTransformer

from .utils import (
    SCALAR, SEQUENCE_TYPES, check_binary_task, check_dim, check_goal,
    check_is_fitted, check_predict_proba, composed, crash, get_best_score,
    get_corpus, get_custom_scorer, get_feature_importance, get_proba_attr, lst,
    partial_dependence, plot_from_model,
)


class BaseFigure:
    """Base class for the matplotlib figures.

    The instance stores the position of the current axes in grid,
    as well as the models used for the plot (to track in mlflow).

    Parameters
    ----------
    nrows: int, optional (default=1)
        Number of subplot rows in the canvas.

    ncols: int, optional (default=1)
        Number of subplot columns in the canvas.

    create_figure: bool, optional (default=True)
        Whether to create a new figure. Is False when an external
        library creates the figure (e.g. force_plot).

    is_canvas: bool, optional (default=False)
        Whether the figure shows multiple plots.

    """

    def __init__(self, nrows=1, ncols=1, create_figure=True, is_canvas=False):
        self.nrows = nrows
        self.ncols = ncols
        self.create_figure = create_figure
        self.is_canvas = is_canvas
        self._idx = -1  # Index of the current axes
        self._used_models = []  # Models plotted in this figure

        # Create new figure and corresponding grid
        if self.create_figure:
            figure = plt.figure(constrained_layout=is_canvas)
            self.gridspec = GridSpec(nrows=self.nrows, ncols=self.ncols, figure=figure)

    @property
    def figure(self):
        """Get the current figure and increase the subplot index."""
        self._idx += 1

        # Check if there are too many plots in the contextmanager
        if self._idx >= self.nrows * self.ncols:
            raise ValueError(
                "Invalid number of plots in the canvas! Increase "
                "the number of rows and cols to add more plots."
            )

        if self.create_figure:
            return plt.gcf()

    @property
    def grid(self):
        """Return the position of the current axes in the grid."""
        return self.gridspec[self._idx]


class BasePlotter:
    """Parent class for all plotting methods.

    This base class defines the plot properties that can
    be changed in order to customize the plot's aesthetics.

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
    def aesthetics(self, value: dict):
        self.style = value.get("style", self.style)
        self.palette = value.get("palette", self.palette)
        self.title_fontsize = value.get("title_fontsize", self.title_fontsize)
        self.label_fontsize = value.get("label_fontsize", self.label_fontsize)
        self.tick_fontsize = value.get("tick_fontsize", self.tick_fontsize)

    @property
    def style(self):
        return self._aesthetics["style"]

    @style.setter
    @typechecked
    def style(self, value: str):
        styles = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
        if value not in styles:
            raise ValueError(
                "Invalid value for the style parameter, got "
                f"{value}. Choose from: {', '.join(styles)}."
            )
        sns.set_style(value)
        self._aesthetics["style"] = value

    @property
    def palette(self):
        return self._aesthetics["palette"]

    @palette.setter
    @typechecked
    def palette(self, value: str):
        sns.set_palette(value)
        self._aesthetics["palette"] = value

    @property
    def title_fontsize(self):
        return self._aesthetics["title_fontsize"]

    @title_fontsize.setter
    @typechecked
    def title_fontsize(self, value: int):
        if value <= 0:
            raise ValueError(
                "Invalid value for the title_fontsize parameter. "
                f"Value should be >=0, got {value}."
            )
        self._aesthetics["title_fontsize"] = value

    @property
    def label_fontsize(self):
        return self._aesthetics["label_fontsize"]

    @label_fontsize.setter
    @typechecked
    def label_fontsize(self, value: int):
        if value <= 0:
            raise ValueError(
                "Invalid value for the label_fontsize parameter. "
                f"Value should be >=0, got {value}."
            )
        self._aesthetics["label_fontsize"] = value

    @property
    def tick_fontsize(self):
        return self._aesthetics["tick_fontsize"]

    @tick_fontsize.setter
    @typechecked
    def tick_fontsize(self, value: int):
        if value <= 0:
            raise ValueError(
                "Invalid value for the tick_fontsize parameter. "
                f"Value should be >=0, got {value}."
            )
        self._aesthetics["tick_fontsize"] = value

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
    def _draw_line(ax, x):
        """Draw a line across the axis."""
        ax.plot(
            [0, 1],
            [0, 1] if x == "diagonal" else [x, x],
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            zorder=-2,
            transform=ax.transAxes,
        )

    @staticmethod
    def _get_figure(**kwargs):
        """Return existing figure if in canvas, else a new figure."""
        if BasePlotter._fig and BasePlotter._fig.is_canvas:
            return BasePlotter._fig.figure
        else:
            BasePlotter._fig = BaseFigure(**kwargs)
            return BasePlotter._fig.figure

    def _get_subclass(self, models, max_one=False, ensembles=True):
        """Check and return the provided parameter models.

        Parameters
        ----------
        models: str or sequence
            Models provided by the plot's parameter.

        max_one: bool, optional (default=False)
            Whether one or multiple models are allowed. If True, return
            the model instead of a list.

        ensembles: bool, optional (default=True)
            If False, drop ensemble models automatically.

        """
        models = self._get_models(models)
        ensembles = () if ensembles else ("Vote", "Stack")

        model_subclasses = []
        for m in self._models.values():
            if m.name in models and m.acronym not in ensembles:
                model_subclasses.append(m)

        if max_one and len(model_subclasses) > 1:
            raise ValueError("This plot method allows only one model at a time!")

        return model_subclasses[0] if max_one else model_subclasses

    def _get_metric(self, metric):
        """Check and return the index of the provided metric."""
        if isinstance(metric, str):
            name = get_custom_scorer(metric).name
            if name in self.metric:
                return self._metric.index(name)

        elif 0 <= metric < len(self._metric):
            return metric

        raise ValueError(
            "Invalid value for the metric parameter. Value should be the index "
            f"or name of a metric used to run the pipeline, got {metric}."
        )

    def _get_set(self, dataset, allow_holdout=True):
        """Check and return the provided parameter metric."""
        dataset = dataset.lower()
        if dataset == "both":
            return ["train", "test"]
        elif dataset in ("train", "test"):
            return [dataset]
        elif allow_holdout:
            if dataset == "holdout":
                if self.holdout is None:
                    raise ValueError(
                        "Invalid value for the dataset parameter. No holdout "
                        "data set was specified when initializing the trainer."
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

    def _get_target(self, target):
        """Check and return the provided target's index."""
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

    @staticmethod
    def _get_show(show, model):
        """Check and return the provided parameter show."""
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

    def _plot(self, fig=None, ax=None, **kwargs):
        """Make the plot.

        Customize the axes to the default layout and plot the figure
        if it's not part of a canvas.

        Parameters
        ----------
        matplotlib.figure.Figure, optional (default=None)
            Current plotting figure. If None, ignore the figure.

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
                - plotname: Name of the plot.
                - display: Whether to render the plot. If None, return the figure.

        """
        if kwargs.get("title"):
            ax.set_title(kwargs.get("title"), fontsize=self.title_fontsize, pad=20)
        if kwargs.get("legend"):
            ax.legend(
                loc=kwargs["legend"][0],
                ncol=max(1, kwargs["legend"][1] // 3),
                fontsize=self.label_fontsize,
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
            ax.tick_params(axis="both", labelsize=self.tick_fontsize)

        if fig and not getattr(BasePlotter._fig, "is_canvas", None):
            # Set name with which to save the file
            if kwargs.get("filename"):
                if kwargs["filename"].endswith("auto"):
                    name = kwargs["filename"].replace("auto", kwargs["plotname"])
                else:
                    name = kwargs["filename"]
            else:
                name = kwargs.get("plotname")

            if kwargs.get("figsize"):
                fig.set_size_inches(*kwargs["figsize"])
            if kwargs.get("tight_layout", True):
                fig.tight_layout()
            if kwargs.get("filename"):
                fig.savefig(name)

            # Log plot to mlflow run of every model visualized
            if self.experiment and self.log_plots:
                for m in set(BasePlotter._fig._used_models):
                    MlflowClient().log_figure(
                        run_id=m._run.info.run_id,
                        figure=fig,
                        artifact_file=name if name.endswith(".png") else f"{name}.png",
                    )
            plt.show() if kwargs.get("display") else plt.close()
            if kwargs.get("display") is None:
                return fig

    @composed(contextmanager, crash, typechecked)
    def canvas(
        self,
        nrows: int = 1,
        ncols: int = 2,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
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
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of plots in the canvas.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. The None option doesn't do
            anything. Use the `as` syntax to get the figure object.

        """
        BasePlotter._fig = BaseFigure(nrows=nrows, ncols=ncols, is_canvas=True)
        try:
            yield plt.gcf()
        finally:
            if title:
                plt.suptitle(title, fontsize=self.title_fontsize + 4)
            BasePlotter._fig.is_canvas = False  # Close the canvas
            self._plot(
                fig=plt.gcf(),
                figsize=figsize or (6 + 4 * ncols, 2 + 4 * nrows),
                tight_layout=False,
                plotname="canvas",
                filename=filename,
                display=display,
            )


class FSPlotter(BasePlotter):
    """Plots for the FeatureSelector class."""

    @composed(crash, typechecked)
    def plot_pca(
        self,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the explained variance ratio vs number of components.

        If the underlying estimator is pca (for dense datasets), all
        possible components are plotted. If the underlying estimator
        is TruncatedSVD (for sparse datasets), it only shows the
        selected components. The blue star marks the number of
        components selected by the user.

        Parameters
        ----------
        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        if not hasattr(self, "pca"):
            raise PermissionError(
                "The plot_pca method is only available if pca was applied on the data!"
            )

        var = np.array(self.pca.explained_variance_ratio_[:self.pca._comps])
        var_all = np.array(self.pca.explained_variance_ratio_)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        ax.scatter(
            x=self.pca._comps,
            y=var.sum(),
            marker="*",
            s=130,
            c="blue",
            edgecolors="b",
            zorder=3,
            label=f"Variance ratio retained: {round(var.sum(), 3)}",
        )
        ax.plot(range(1, len(var_all) + 1), np.cumsum(var_all), marker="o")
        ax.axhline(var.sum(), ls="--", color="k")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only int ticks

        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("lower right", 1),
            xlabel="First N principal components",
            ylabel="Cumulative variance ratio",
            figsize=figsize,
            plotname="plot_pca",
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
        display: Optional[bool] = True,
    ):
        """Plot the explained variance ratio per component.

        Parameters
        ----------
        show: int or None, optional (default=None)
            Number of components to show. None to show all.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of components shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        if not hasattr(self, "pca"):
            raise PermissionError(
                "The plot_components method is only available "
                "if pca was applied on the data!"
            )

        if show is None or show > self.pca.components_.shape[0]:
            # Limit max features shown to avoid maximum figsize error
            show = min(200, self.pca.components_.shape[0])
        elif show < 1:
            raise ValueError(
                "Invalid value for the show parameter. "
                f"Value should be >0, got {show}."
            )

        var = np.array(self.pca.explained_variance_ratio_)[:show]
        scr = pd.Series(
            data=var,
            index=[f"component_{str(i)}" for i in range(len(var))],
            dtype=float,
        ).sort_values()

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        scr.plot.barh(label=f"Total variance retained: {var.sum():.3f}", width=0.6)
        ax.set_xlim(0, max(scr) + 0.1 * max(scr))  # Make extra space for numbers
        for i, v in enumerate(scr):
            ax.text(v + 0.005, i - 0.08, f"{v:.3f}", fontsize=self.tick_fontsize)

        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("lower right", 1),
            xlabel="Explained variance ratio",
            figsize=figsize or (10, 4 + show // 2),
            plotname="plot_components",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_rfecv(
        self,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the rfecv results.

        Plot the scores obtained by the estimator fitted on every
        subset of the dataset.

        Parameters
        ----------
        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        if not hasattr(self.branch, "rfecv") or not self.rfecv:
            raise PermissionError(
                "The plot_rfecv method is only available "
                "if rfecv was applied on the data!"
            )

        try:  # Define the y-label for the plot
            ylabel = self.rfecv.get_params()["scoring"].name
        except AttributeError:
            ylabel = "score"

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        n_features = self.rfecv.get_params()["min_features_to_select"]
        mean = self.rfecv.cv_results_["mean_test_score"]

        # Prepare dataframe for seaborn lineplot
        df = pd.DataFrame()
        for key, value in self.rfecv.cv_results_.items():
            if key not in ("mean_test_score", "std_test_score"):
                df = pd.concat([df, pd.DataFrame(value, columns=["y"])])

        df["x"] = np.add(df.index, n_features)
        df = df.reset_index(drop=True)

        xline = range(n_features, n_features + len(mean))
        sns.lineplot(data=df, x="x", y="y", marker="o", ax=ax)

        # Set limits before drawing the intersected lines
        xlim = (n_features - 0.5, n_features + len(mean) - 0.5)
        ylim = ax.get_ylim()

        # Draw intersected lines
        x, y = xline[np.argmax(mean)], max(mean)
        ax.vlines(x, ax.get_ylim()[0], y, ls="--", color="k", alpha=0.7)
        label = f"Features: {x}   {ylabel}: {round(y, 3)}"
        ax.hlines(y, xmin=-1, xmax=x, color="k", ls="--", alpha=0.7, label=label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only int ticks

        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("lower right", 1),
            xlabel="Number of features",
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            plotname="plot_rfecv",
            filename=filename,
            display=display,
        )


class BaseModelPlotter(BasePlotter):
    """Plots for the BaseModel class."""

    @composed(crash, plot_from_model, typechecked)
    def plot_successive_halving(
            self,
            models: Optional[Union[str, SEQUENCE_TYPES]] = None,
            metric: Union[int, str] = 0,
            title: Optional[str] = None,
            figsize: Tuple[SCALAR, SCALAR] = (10, 6),
            filename: Optional[str] = None,
            display: Optional[bool] = True,
    ):
        """Plot scores per iteration of the successive halving.

        Only use if the models were fitted using successive_halving.
        Ensemble models are ignored.

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
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models, ensembles=False)
        metric = self._get_metric(metric)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        # Prepare dataframes for seaborn lineplot (one df per line)
        # Not using sns hue parameter because of legend formatting
        lines = defaultdict(pd.DataFrame)
        for m in models:
            n_models = len(m.branch._idx[0]) // m._train_idx  # Number of models in iter
            if m.metric_bootstrap is None:
                values = {"x": [n_models], "y": [get_best_score(m, metric)]}
            else:
                if len(self._metric) == 1:
                    bootstrap = m.metric_bootstrap
                else:
                    bootstrap = m.metric_bootstrap[metric]
                values = {"x": [n_models] * len(bootstrap), "y": bootstrap}

            # Add the scores to the group's dataframe
            lines[m._group] = pd.concat([lines[m._group], pd.DataFrame(values)])

        for m, df in zip(models, lines.values()):
            df = df.reset_index(drop=True)
            kwargs = dict(err_style="band" if df["x"].nunique() > 1 else "bars", ax=ax)
            sns.lineplot(data=df, x="x", y="y", marker="o", label=m.acronym, **kwargs)

        n_models = [len(self.train) // m._train_idx for m in models]
        ax.set_xlim(max(n_models) + 0.1, min(n_models) - 0.1)
        ax.set_xticks(range(1, max(n_models) + 1))

        BasePlotter._fig._used_models.extend(models)
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

    @composed(crash, plot_from_model, typechecked)
    def plot_learning_curve(
            self,
            models: Optional[Union[str, SEQUENCE_TYPES]] = None,
            metric: Union[int, str] = 0,
            title: Optional[str] = None,
            figsize: Tuple[SCALAR, SCALAR] = (10, 6),
            filename: Optional[str] = None,
            display: Optional[bool] = True,
    ):
        """Plot the learning curve: score vs number of training samples.

        Only use with models fitted using train sizing. Ensemble
        models are ignored.

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
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models, ensembles=False)
        metric = self._get_metric(metric)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        # Prepare dataframes for seaborn lineplot (one df per line)
        # Not using sns hue parameter because of legend formatting
        lines = defaultdict(pd.DataFrame)
        for m in models:
            if m.metric_bootstrap is None:
                values = {"x": [m._train_idx], "y": [get_best_score(m, metric)]}
            else:
                if len(self._metric) == 1:
                    bootstrap = m.metric_bootstrap
                else:
                    bootstrap = m.metric_bootstrap[metric]
                values = {"x": [m._train_idx] * len(bootstrap), "y": bootstrap}

            # Add the scores to the group's dataframe
            lines[m._group] = pd.concat([lines[m._group], pd.DataFrame(values)])

        for m, df in zip(models, lines.values()):
            df = df.reset_index(drop=True)
            sns.lineplot(data=df, x="x", y="y", marker="o", label=m.acronym, ax=ax)

        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))

        BasePlotter._fig._used_models.extend(models)
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

    @composed(crash, plot_from_model, typechecked)
    def plot_results(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        metric: Union[int, str] = 0,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot of the model results after the evaluation.

        If all models applied bootstrap, the plot is a boxplot. If
        not, the plot is a barplot. Models are ordered based on
        their score from the top down. The score is either the
        `mean_bootstrap` or `metric_test` attribute of the model,
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
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of models shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """

        def get_bootstrap(m):
            """Get the bootstrap results for a specific metric."""
            # Use getattr since ensembles don't have the attribute
            if isinstance(getattr(m, "metric_bootstrap", None), np.ndarray):
                if len(self._metric) == 1:
                    return m.metric_bootstrap
                else:
                    return m.metric_bootstrap[metric]

        def std(m):
            """Get the standard deviation of the bootstrap results."""
            if getattr(m, "std_bootstrap", None):
                return lst(m.std_bootstrap)[metric]
            else:
                return 0

        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        metric = self._get_metric(metric)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        names = []
        models = sorted(models, key=lambda m: get_best_score(m, metric))
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]  # First color

        all_bootstrap = all(isinstance(get_bootstrap(m), np.ndarray) for m in models)
        for i, m in enumerate(models):
            names.append(m.name)
            if all_bootstrap:
                ax.boxplot(
                    x=get_bootstrap(m),
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

        min_lim = 0.95 * (get_best_score(models[0], metric) - std(models[0]))
        max_lim = 1.01 * (get_best_score(models[-1], metric) + std(models[-1]))
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(names)

        BasePlotter._fig._used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            xlabel=self._metric[metric].name,
            xlim=(min_lim, max_lim) if not all_bootstrap else None,
            figsize=figsize or (10, 4 + len(models) // 2),
            plotname="plot_results",
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
        display: Optional[bool] = True,
    ):
        """Plot the bayesian optimization scores.

        Only for models that ran hyperparameter tuning. This is the
        same plot as produced by `bo_params={"plot": True}` while
        running the BO. Creates a canvas with two plots: the first
        plot shows the score of every trial and the second shows
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
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        metric = self._get_metric(metric)

        # Check there is at least one model that run the BO
        if all(m.bo.empty for m in models):
            raise PermissionError(
                "The plot_bo method is only available for models that "
                "ran the bayesian optimization hyperparameter tuning!"
            )

        fig = self._get_figure()
        gs = GridSpecFromSubplotSpec(4, 1, BasePlotter._fig.grid, hspace=0.05)
        ax1 = fig.add_subplot(gs[0:3, 0])
        ax2 = plt.subplot(gs[3:4, 0], sharex=ax1)
        for m in models:
            if m.metric_bo:  # Only models that did run the BO
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
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        self._plot(
            ax=ax1,
            title=title,
            legend=("lower right", len(models)),
            ylabel=self._metric[metric].name,
        )

        BasePlotter._fig._used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax2,
            xlabel="Call",
            ylabel="d",
            figsize=figsize,
            plotname="plot_bo",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_evals(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "both",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot evaluation curves for the train and test set.

         Only for models that allow in-training evaluation (XGB, LGB,
        CastB). The metric is provided by the estimator's package and
        is different for every model and every task. For this reason,
        the method only allows plotting one model.

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
            Choose from: "train", "test" or "both".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "plot_evals")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        dataset = self._get_set(dataset, allow_holdout=False)

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

        BasePlotter._fig._used_models.append(m)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("best", len(dataset)),
            xlabel=m.get_dimensions()[0].name,  # First param is always the iter
            ylabel=m.evals["metric"],
            figsize=figsize,
            plotname="plot_evals",
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
        display: Optional[bool] = True,
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
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_roc")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        for m in models:
            attr = get_proba_attr(m)
            for set_ in dataset:
                if attr == "predict_proba":
                    y_pred = getattr(m, f"predict_proba_{set_}")[:, 1]
                else:
                    y_pred = getattr(m, f"decision_function_{set_}")

                # Get False (True) Positive Rate as arrays
                fpr, tpr, _ = roc_curve(getattr(m, f"y_{set_}"), y_pred)

                roc = f" (AUC={round(m.evaluate('auc', set_)['roc_auc'], 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + roc
                ax.plot(fpr, tpr, lw=2, label=label)

        self._draw_line(ax=ax, x="diagonal")

        BasePlotter._fig._used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("lower right", len(models)),
            xlabel="FPR",
            ylabel="TPR",
            figsize=figsize,
            plotname="plot_roc",
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
        display: Optional[bool] = True,
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
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_prc")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        for m in models:
            attr = get_proba_attr(m)
            for set_ in dataset:
                if attr == "predict_proba":
                    y_pred = getattr(m, f"predict_proba_{set_}")[:, 1]
                else:
                    y_pred = getattr(m, f"decision_function_{set_}")

                # Get precision-recall pairs for different thresholds
                prec, rec, _ = precision_recall_curve(getattr(m, f"y_{set_}"), y_pred)

                ap = f" (AP={round(m.evaluate('ap', set_)['average_precision'], 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + ap
                plt.plot(rec, prec, lw=2, label=label)

        self._draw_line(ax=ax, x=m.y_test.sort_values().iloc[-1] / len(m.y_test))

        BasePlotter._fig._used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("best", len(models)),
            xlabel="Recall",
            ylabel="Precision",
            figsize=figsize,
            plotname="plot_prc",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_det(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the detection error tradeoff curve.

        Only for binary classification tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_det")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        for m in models:
            attr = get_proba_attr(m)
            for set_ in dataset:
                if attr == "predict_proba":
                    y_pred = getattr(m, f"predict_proba_{set_}")[:, 1]
                else:
                    y_pred = getattr(m, f"decision_function_{set_}")

                # Get fpr-fnr pairs for different thresholds
                fpr, fnr, _ = det_curve(getattr(m, f"y_{set_}"), y_pred)

                plt.plot(fpr, fnr, lw=2, label=m.name)

        BasePlotter._fig._used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("best", len(models)),
            xlabel="FPR",
            ylabel="FNR",
            figsize=figsize,
            plotname="plot_det",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_gains(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
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
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_gains")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        for m in models:
            attr = get_proba_attr(m)
            for set_ in dataset:
                if attr == "predict_proba":
                    y_pred = getattr(m, f"predict_proba_{set_}")[:, 1]
                else:
                    y_pred = getattr(m, f"decision_function_{set_}")

                # Make y_true a bool vector
                y_true = getattr(m, f"y_{set_}") == 1

                # Get sorted indices
                sort_idx = np.argsort(y_pred)[::-1]

                # Correct indices for the test set (add train set length)
                if set_ == "test":
                    sort_idx = [i + len(m.y_train) for i in sort_idx]

                # Compute cumulative gains
                gains = np.cumsum(y_true.loc[sort_idx]) / float(np.sum(y_true))

                x = np.arange(start=1, stop=len(y_true) + 1) / float(len(y_true))
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "")
                ax.plot(x, gains, lw=2, label=label)

        self._draw_line(ax=ax, x="diagonal")

        BasePlotter._fig._used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("best", len(models)),
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
    def plot_lift(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
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
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_lift")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        for m in models:
            attr = get_proba_attr(m)
            for set_ in dataset:
                if attr == "predict_proba":
                    y_pred = getattr(m, f"predict_proba_{set_}")[:, 1]
                else:
                    y_pred = getattr(m, f"decision_function_{set_}")

                # Make y_true a bool vector
                y_true = getattr(m, f"y_{set_}") == 1

                # Get sorted indices
                sort_idx = np.argsort(y_pred)[::-1]

                # Correct indices for the test set (add train set length)
                if set_ == "test":  # Add the training set length to the indices
                    sort_idx = [i + len(m.y_train) for i in sort_idx]

                # Compute cumulative gains
                gains = np.cumsum(y_true.loc[sort_idx]) / float(np.sum(y_true))

                x = np.arange(start=1, stop=len(y_true) + 1) / float(len(y_true))
                lift = f" (Lift={round(m.evaluate('lift', set_)['lift'], 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + lift
                ax.plot(x, gains / x, lw=2, label=label)

        self._draw_line(ax=ax, x=1)

        BasePlotter._fig._used_models.extend(models)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            legend=("best", len(models)),
            xlabel="Fraction of sample",
            ylabel="Lift",
            xlim=(0, 1),
            figsize=figsize,
            plotname="plot_lift",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_errors(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
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
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        check_goal(self, "plot_errors", "regression")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
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
                from .models import OrdinaryLeastSquares

                model = OrdinaryLeastSquares(self, fast_init=True).get_estimator()
                model.fit(
                    X=np.array(getattr(self, f"y_{set_}")).reshape(-1, 1),
                    y=getattr(m, f"predict_{set_}"),
                )

                # Draw the fit
                x = np.linspace(*ax.get_xlim(), 100)
                ax.plot(x, model.predict(x[:, np.newaxis]), lw=2, alpha=1)

        self._draw_line(ax=ax, x="diagonal")

        BasePlotter._fig._used_models.extend(models)
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
    def plot_residuals(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        title: Optional[str] = None,
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
        non-linear model is more appropriate. Only for regression tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

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
                r2 = f" (R$^2$={round(m.evaluate('r2', set_)['r2'], 3)})"
                label = m.name + (f" - {set_}" if len(dataset) > 1 else "") + r2
                res = np.subtract(
                    getattr(m, f"predict_{set_}"),
                    getattr(self, f"y_{set_}"),
                )

                ax1.scatter(getattr(m, f"predict_{set_}"), res, alpha=0.7, label=label)
                ax2.hist(res, orientation="horizontal", histtype="step", linewidth=1.2)

        ax2.set_yticklabels([])
        self._draw_line(ax=ax2, x=0)
        self._plot(ax=ax2, xlabel="Distribution")

        if title:
            if not BasePlotter._fig.is_canvas:
                plt.suptitle(title, fontsize=self.title_fontsize, y=0.98)
            else:
                ax1.set_title(title, fontsize=self.title_fontsize, pad=20)

        BasePlotter._fig._used_models.extend(models)
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
    def plot_feature_importance(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        show: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a model's feature importance.

        The feature importance values are normalized in order to be
        able to compare them between models. Only for models whose
        estimator has a `feature_importances_` or `coef` attribute.
        The trainer's `feature_importance` attribute is updated with
        the extracted importance ranking.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show.
            None to show all.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "plot_feature_importance")
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        show = self._get_show(show, models)

        # Create dataframe with columns as indices to plot with barh
        df = pd.DataFrame()
        for m in models:
            fi = get_feature_importance(m.estimator)
            if fi is None:
                raise ValueError(
                    f"Invalid value for the models parameter. The {m.fullname}'s "
                    f"estimator {m.estimator.__class__.__name__} has no "
                    f"feature_importances_ nor coef_ attribute."
                )

            # Normalize to be able to compare different models
            for col, fx in zip(m.features, fi):
                df.at[col, m.name] = fx / max(fi)

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
            legend=True if len(models) > 1 else False,
        )
        if len(models) == 1:
            for i, v in enumerate(df[df.columns[0]]):
                ax.text(v + 0.01, i - 0.08, f"{v:.2f}", fontsize=self.tick_fontsize)

        BasePlotter._fig._used_models.extend(models)
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

    @composed(crash, plot_from_model, typechecked)
    def plot_permutation_importance(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        show: Optional[int] = None,
        n_repeats: int = 10,
        title: Optional[str] = None,
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
        faster. The trainer's `feature_importance` attribute is updated
        with the extracted importance ranking.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show.
            None to show all.

        n_repeats: int, optional (default=10)
            Number of times to permute each feature.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "plot_permutation_importance")
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

        # Save the best feature order
        self.branch.feature_importance = list(get_idx.columns.values)

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

        BasePlotter._fig._used_models.extend(models)
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
    def plot_partial_dependence(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        columns: Optional[Union[int, str, SEQUENCE_TYPES]] = None,
        kind: str = "average",
        target: Union[int, str] = 1,
        title: Optional[str] = None,
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
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        columns: int, str, sequence or None, optional (default=None)
            Features or feature pairs (name or index) to get the partial
            dependence from. Maximum of 3 allowed. If None, it uses the
            best 3 features if the `feature_importance` attribute is
            defined, else it uses the first 3 features in the dataset.

        kind: str, optional (default="average")
            - "average": Plot the partial dependence averaged across
                         all the samples in the dataset.
            - "individual": Plot the partial dependence per sample
                            (Individual Conditional Expectation).
            - "both": Plot both the average (as a thick line) and the
                      individual (thin lines) partial dependence.

            This parameter is ignored when plotting feature pairs.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """

        def get_features(features, m):
            """Select feature list from provided columns."""
            # Default is to select the best or the first 3 features
            if not features:
                if not m.branch.feature_importance:
                    features = m.features[:3]
                else:
                    features = m.branch.feature_importance[:3]

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

        check_dim(self, "plot_partial_dependence")
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
        gs = GridSpecFromSubplotSpec(1, n_cols, BasePlotter._fig.grid)
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
                    features=[m.features.index(c) for c in col],
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
            if len(cols) == 1 or (len(cols) == 2 and BasePlotter._fig.is_canvas):
                axes[0].set_title(title, fontsize=self.title_fontsize, pad=20)
            elif len(cols) == 3:
                axes[1].set_title(title, fontsize=self.title_fontsize, pad=20)
            elif not BasePlotter._fig.is_canvas:
                plt.suptitle(title, fontsize=self.title_fontsize)

        BasePlotter._fig._used_models.extend(models)
        return self._plot(
            fig=fig,
            figsize=figsize,
            plotname="plot_partial_dependence",
            filename=filename,
            display=display,
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_parshap(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        columns: Optional[Union[int, str, SEQUENCE_TYPES]] = None,
        target: Union[int, str] = 1,
        title: Optional[str] = None,
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
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        columns: int, str, sequence or None, optional (default=None)
            Names or indices of the features to plot. None to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "plot_parshap")
        check_is_fitted(self, attributes="_models")
        models = self._get_subclass(models)
        target = self._get_target(target)

        fxs_importance = {}
        markers = cycle(["o", "^", "s", "p", "D", "H", "p", "*"])

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
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

        self._draw_line(ax=ax, x="diagonal")

        BasePlotter._fig._used_models.extend(models)
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
    def plot_confusion_matrix(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        normalize: bool = False,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a model's confusion matrix.

        For one model, the plot shows a heatmap. For multiple models,
        it compares TP, FP, FN and TN in a barplot (not implemented
        for multiclass classification tasks). Only for classification
        tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the confusion matrix.
            Choose from:` "train", "test" or "holdout".

        normalize: bool, optional (default=False)
           Whether to normalize the matrix.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the plot's type.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        check_goal(self, "plot_confusion_matrix", "classification")
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
                "data set was specified when initializing the trainer."
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
        ax = fig.add_subplot(BasePlotter._fig.grid)
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

        BasePlotter._fig._used_models.extend(models)
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
        display: Optional[bool] = True,
    ):
        """Plot metric performances against threshold values.

        Only for binary classification tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        metric: str, func, scorer, sequence or None, optional (default=None)
            Metric to plot. Choose from any of sklearn's SCORERS, a
            function with signature `metric(y_true, y_pred)`, a scorer
            object or a sequence of these. If None, the metric used
            to run the pipeline is plotted.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        steps: int, optional (default=100)
            Number of thresholds measured.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_is_fitted(self, attributes="_models")
        check_binary_task(self, "plot_threshold")
        models = self._get_subclass(models)
        dataset = self._get_set(dataset)
        check_predict_proba(models, "plot_threshold")

        # Get all metric functions from the input
        if metric is None:
            metric_list = [m._score_func for m in self._metric.values()]
        else:
            metric_list = [get_custom_scorer(m)._score_func for m in lst(metric)]

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        steps = np.linspace(0, 1, steps)
        for m in models:
            for met in metric_list:
                for set_ in dataset:
                    results = []
                    for step in steps:
                        pred = getattr(m, f"predict_proba_{set_}")[:, 1] >= step
                        results.append(met(getattr(m, f"y_{set_}"), pred))

                    if len(models) == 1:
                        l_set = f"{set_} - " if len(dataset) > 1 else ""
                        label = f"{l_set}{met.__name__}"
                    else:
                        l_set = f" - {set_}" if len(dataset) > 1 else ""
                        label = f"{m.name}{l_set} ({met.__name__})"
                    ax.plot(steps, results, label=label, lw=2)

        BasePlotter._fig._used_models.extend(models)
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
    def plot_probabilities(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        target: Union[int, str] = 1,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot the probability distribution of the target classes.

        Only for classification tasks.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Choose from:
            "train", "test", "both" (train and test) or "holdout".

        target: int or str, optional (default=1)
            Probability of being that class in the target column
            (as index or name). Only for multiclass classification.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

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
                for value in m.y.sort_values().unique():
                    # Get indices per class
                    idx = np.where(getattr(m, f"y_{set_}") == value)[0]

                    label = m.name + (f" - {set_}" if len(dataset) > 1 else "")
                    sns.histplot(
                        data=getattr(m, f"predict_proba_{set_}")[idx, target],
                        kde=True,
                        bins=50,
                        label=label + f" ({self.target}={value})",
                        color=next(palette),
                        ax=ax,
                    )

        BasePlotter._fig._used_models.extend(models)
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

    @composed(crash, plot_from_model, typechecked)
    def plot_calibration(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        n_bins: int = 10,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 10),
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
            Number of bins used for calibration. Minimum of 5
            required.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 10))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

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
        self._draw_line(ax=ax1, x="diagonal")

        BasePlotter._fig._used_models.extend(models)
        self._plot(
            ax=ax1,
            title=title,
            legend=("lower right" if len(models) > 1 else False, len(models)),
            ylabel="Fraction of positives",
            ylim=(-0.05, 1.05),
        )
        return self._plot(
            fig=fig,
            ax=ax2,
            xlabel="Predicted value",
            ylabel="Count",
            figsize=figsize,
            plotname="plot_calibration",
            filename=filename,
            display=display,
        )

    # SHAP plots =================================================== >>

    @composed(crash, plot_from_model, typechecked)
    def bar_plot(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        index: Optional[Union[int, str, SEQUENCE_TYPES]] = None,
        show: Optional[int] = None,
        target: Union[int, str] = 1,
        title: Optional[str] = None,
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
        models: str, sequence or None, optional (default=None)
            Name of the model to plot. If None, all models in the
            pipeline are selected. Note that leaving the default
            option could raise an exception if there are multiple
            models in the pipeline. To avoid this, call the plot
            from a model, e.g. `atom.xgb.bar_plot()`.

        index: int, str, sequence or None, optional (default=None)
            Index names or positions of the rows in the dataset to
            plot. If None, it selects all rows in the test set.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show.
            None to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to
            look at. Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts
            the size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's bar plot.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "bar_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        show = self._get_show(show, m)
        target = self._get_target(target)
        explanation = m._shap.get_explanation(rows, target)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.plots.bar(explanation, max_display=show, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)

        BasePlotter._fig._used_models.append(m)
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
    def beeswarm_plot(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        index: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        show: Optional[int] = None,
        target: Union[int, str] = 1,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot SHAP's beeswarm plot.

        The plot is colored by feature values.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Name of the model to plot. If None, all models in the
            pipeline are selected. Note that leaving the default
            option could raise an exception if there are multiple
            models in the pipeline. To avoid this, call the plot
            from a model, e.g. `atom.xgb.beeswarm_plot()`.

        index: tuple, slice or None, optional (default=None)
            Index names or positions of the rows in the dataset to plot.
            If None, it selects all rows in the test set. The beeswarm
            plot does not support plotting a single sample.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show. None
            to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's beeswarm plot.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "beeswarm_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        show = self._get_show(show, m)
        target = self._get_target(target)
        explanation = m._shap.get_explanation(rows, target)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.plots.beeswarm(explanation, max_display=show, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)

        BasePlotter._fig._used_models.append(m)
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
    def decision_plot(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        index: Optional[Union[int, str, SEQUENCE_TYPES]] = None,
        show: Optional[int] = None,
        target: Union[int, str] = 1,
        title: Optional[str] = None,
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
        models: str, sequence or None, optional (default=None)
            Name of the model to plot. If None, all models in the
            pipeline are selected. Note that leaving the default
            option could raise an exception if there are multiple
            models in the pipeline. To avoid this, call the plot
            from a model, e.g. `atom.xgb.decision_plot()`.

        index: int, str, sequence or None, optional (default=None)
            Index names or positions of the rows in the dataset to plot.
            If None, it selects all rows in the test set.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show. None
            to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's decision plot.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "decision_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        show = self._get_show(show, m)
        target = self._get_target(target)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
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

        BasePlotter._fig._used_models.append(m)
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
    def force_plot(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        index: Optional[Union[int, str, SEQUENCE_TYPES]] = None,
        target: Union[str, int] = 1,
        title: Optional[str] = None,
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
        models: str, sequence or None, optional (default=None)
            Name of the model to plot. If None, all models in the
            pipeline are selected. Note that leaving the default
            option could raise an exception if there are multiple
            models in the pipeline. To avoid this, call the plot
            from a model, e.g. `atom.xgb.force_plot()`.

        index: int, str, sequence or None, optional (default=None)
            Index names or positions of the rows in the dataset to plot.
            If None, it selects all rows in the test set.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(14, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. If matplotlib=False, the figure will
            be saved as a html file. If None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's force plot.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only if `display=None` and `matplotlib=True`.

        """
        if getattr(BasePlotter._fig, "is_canvas", None):
            raise PermissionError(
                "The force_plot method can not be called from a "
                "canvas because of incompatibility of the APIs."
            )

        check_dim(self, "force_plot")
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

        sns.set_style(self.style)  # Reset style
        if kwargs.get("matplotlib"):
            BasePlotter._fig._used_models.append(m)
            return self._plot(
                fig=plt.gcf(),
                title=title,
                plotname="force_plot",
                filename=filename,
                display=display,
            )
        else:
            if filename:  # Save to a html file
                if not filename.endswith(".html"):
                    filename += ".html"
                shap.save_html(filename, plot)
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
        index: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        show: Optional[int] = None,
        target: Union[int, str] = 1,
        title: Optional[str] = None,
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
        models: str, sequence or None, optional (default=None)
            Name of the model to plot. If None, all models in the
            pipeline are selected. Note that leaving the default
            option could raise an exception if there are multiple
            models in the pipeline. To avoid this, call the plot
            from a model, e.g. `atom.xgb.heatmap_plot()`.

        index: slice, sequence or None, optional (default=None)
            Index names or positions of the rows in the dataset to plot.
            If None, it selects all rows in the test set. The heatmap
            plot does not support plotting a single sample.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show. None
            to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's heatmap plot.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "heatmap_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        show = self._get_show(show, m)
        target = self._get_target(target)
        explanation = m._shap.get_explanation(rows, target)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.plots.heatmap(explanation, max_display=show, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)

        BasePlotter._fig._used_models.append(m)
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
    def scatter_plot(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        index: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        feature: Union[int, str] = 0,
        target: Union[int, str] = 1,
        title: Optional[str] = None,
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
        models: str, sequence or None, optional (default=None)
            Name of the model to plot. If None, all models in the
            pipeline are selected. Note that leaving the default
            option could raise an exception if there are multiple
            models in the pipeline. To avoid this, call the plot
            from a model, e.g. `atom.xgb.scatter_plot()`.

        index: slice, sequence or None, optional (default=None)
            Index names or positions of the rows in the dataset to
            plot. If None, it selects all rows in the test set. The
            scatter plot does not support plotting a single sample.

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
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for SHAP's scatter plot.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "scatter_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[self._get_rows(index, branch=m.branch)]
        target = self._get_target(target)
        explanation = m._shap.get_explanation(rows, target, feature)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.plots.scatter(explanation, color=explanation, ax=ax, show=False, **kwargs)

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=self.label_fontsize, labelpad=12)

        BasePlotter._fig._used_models.append(m)
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
    def waterfall_plot(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        index: Optional[Union[int, str]] = None,
        show: Optional[int] = None,
        target: Union[int, str] = 1,
        title: Optional[str] = None,
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
        models: str, sequence or None, optional (default=None)
            Name of the model to plot. If None, all models in the
            pipeline are selected. Note that leaving the default
            option could raise an exception if there are multiple
            models in the pipeline. To avoid this, call the plot
            from a model, e.g. `atom.xgb.waterfall_plot()`.

        index: int, str or None, optional (default=None)
            Index name or position of the row in the dataset to plot.
            If None, it selects the first row in the test set. The
            waterfall plot does not support plotting multiple
            samples.

        show: int or None, optional (default=None)
            Number of features (ordered by importance) to show. None
            to show all.

        target: int or str, optional (default=1)
            Index or name of the class in the target column to look at.
            Only for multi-class classification tasks.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of features shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "waterfall_plot")
        check_is_fitted(self, attributes="_models")
        m = self._get_subclass(models, max_one=True)
        rows = m.X.loc[[self._get_rows(index, branch=m.branch)[0]]]
        show = self._get_show(show, m)
        target = self._get_target(target)
        explanation = m._shap.get_explanation(rows, target, only_one=True)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)
        shap.plots.waterfall(
            explanation,
            max_display=show,
            show=True if shap.__version__ == "0.40.0" else False,
        )

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)

        BasePlotter._fig._used_models.append(m)
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            figsize=figsize or (10, 4 + show // 2),
            plotname="waterfall_plot",
            filename=filename,
            display=display,
        )


class ATOMPlotter(FSPlotter, BaseModelPlotter):
    """Plots for the ATOM class."""

    @composed(crash, typechecked)
    def plot_correlation(
        self,
        columns: Optional[Union[slice, SEQUENCE_TYPES]] = None,
        method: str = "pearson",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (8, 7),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a correlation matrix.

        Parameters
        ----------
        columns: slice, sequence or None, optional (default=None)
            Slice, names or indices of the columns to plot. If None,
            plot all columns in the dataset. Selected categorical
            columns are ignored.

        method: str, optional (default="pearson")
            Method of correlation. Choose from: pearson, kendall or
            spearman.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(8, 7))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "plot_correlation")
        columns = self._get_columns(columns, only_numerical=True)
        if method.lower() not in ("pearson", "kendall", "spearman"):
            raise ValueError(
                f"Invalid value for the method parameter, got {method}. "
                "Choose from: pearson, kendall or spearman."
            )

        # Compute the correlation matrix
        corr = self.dataset[columns].corr(method=method.lower())

        # Drop first row and last column (diagonal line)
        corr = corr.iloc[1:].drop(columns[-1], axis=1)

        # Generate a mask for the upper triangle
        # k=1 means keep outermost diagonal line
        mask = np.zeros_like(corr, dtype=bool)
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
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            figsize=figsize,
            plotname="plot_correlation",
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
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot a matrix of scatter plots.

        A subset of max 250 random samples are selected from every
        column to not clutter the plot.

        Parameters
        ----------
        columns: slice, sequence or None, optional (default=None)
            Slice, names or indices of the columns to plot. If None,
            plot all columns in the dataset. Selected categorical
            columns are ignored.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=(10, 10)))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for seaborn's pairplot.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        if getattr(BasePlotter._fig, "is_canvas", None):
            raise PermissionError(
                "The plot_scatter_matrix method can not be called from "
                "a canvas because of incompatibility of the APIs."
            )

        check_dim(self, "plot_scatter_matrix")
        columns = self._get_columns(columns, only_numerical=True)

        # Use max 250 samples to not clutter the plot
        samples = self.dataset[columns].sample(
            n=min(len(self.dataset), 250), random_state=self.random_state
        )

        diag_kind = kwargs.get("diag_kind", "kde")
        grid = sns.pairplot(samples, diag_kind=diag_kind, **kwargs)

        # Set right fontsize for all axes in grid
        for axi in grid.axes.flatten():
            axi.tick_params(axis="both", labelsize=self.tick_fontsize)
            axi.set_xlabel(axi.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
            axi.set_ylabel(axi.get_ylabel(), fontsize=self.label_fontsize, labelpad=12)

        return self._plot(
            fig=plt.gcf(),
            title=title,
            figsize=figsize or (10, 10),
            plotname="plot_scatter_matrix",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_distribution(
        self,
        columns: Union[int, str, slice, SEQUENCE_TYPES] = 0,
        distributions: Optional[Union[str, SEQUENCE_TYPES]] = None,
        show: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot column distributions.

        Additionally, it is possible to plot any of `scipy.stats`
        probability distributions fitted to the column. Missing
        values are ignored.

        Parameters
        ----------
        columns: int, str, slice or sequence, optional (default=0)
            Slice, names or indices of the columns to plot. It is only
            possible to plot one categorical column. If more than just
            one categorical columns are selected, all categorical
            columns are ignored.

        distributions: str, sequence or None, optional (default=None)
            Names of the `scipy.stats` distributions to fit to the
            columns. If None, no distribution is fitted. Only for
            numerical columns.

        show: int or None, optional (default=None)
            Number of classes (ordered by number of occurrences) to
            show in the plot. None to show all. Only for categorical
            columns.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the plot's type.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for seaborn's histplot.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "plot_distribution")
        columns = self._get_columns(columns)
        palette_1 = cycle(sns.color_palette())
        palette_2 = sns.color_palette("Blues_r", 3)

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        cat_columns = list(self.dataset.select_dtypes(exclude="number").columns)
        if len(columns) == 1 and columns[0] in cat_columns:
            series = self.dataset[columns].value_counts(ascending=True)

            if show is None or show > len(series):
                show = len(series)
            elif show < 1:
                raise ValueError(
                    "Invalid value for the show parameter."
                    f"Value should be >0, got {show}."
                )

            data = series[-show:]  # Subset of series to plot
            data.plot.barh(
                ax=ax,
                width=0.6,
                label=f"{columns[0]}: {len(series)} classes",
            )

            # Add the counts at the end of the bar
            for i, v in enumerate(data):
                ax.text(v + 0.01 * max(data), i - 0.08, v, fontsize=self.tick_fontsize)

            return self._plot(
                fig=fig,
                ax=ax,
                xlim=(min(data) - 0.1 * min(data), max(data) + 0.1 * max(data)),
                title=title,
                xlabel="Counts",
                legend=("lower right", 1),
                figsize=figsize or (10, 4 + show // 2),
                plotname="plot_distribution",
                filename=filename,
                display=display,
            )
        else:
            kde = kwargs.pop("kde", False if distributions else True)
            bins = kwargs.pop("bins", 40)
            for i, col in enumerate(columns):
                sns.histplot(
                    data=self.dataset,
                    x=col,
                    kde=kde,
                    label=col,
                    bins=bins,
                    color=next(palette_1),
                    ax=ax,
                    **kwargs,
                )

                if distributions:
                    x = np.linspace(*ax.get_xlim(), 100)

                    # Drop the missing values form the column
                    missing = self.missing + [np.inf, -np.inf]
                    values = self.dataset[col].replace(missing, np.NaN).dropna()

                    # Get the hist values
                    h = np.histogram(values, bins=bins)

                    # Get a line for each distribution
                    for j, dist in enumerate(lst(distributions)):
                        params = getattr(stats, dist).fit(values)

                        # Calculate pdf and scale to match observed data
                        pdf = getattr(stats, dist).pdf(x, *params)
                        scale = np.trapz(h[0], h[1][:-1]) / np.trapz(pdf, x)

                        label = dist if i == 0 else None  # Label for the first iter
                        plt.plot(x, pdf * scale, lw=2, c=palette_2[j], label=label)

            return self._plot(
                fig=fig,
                ax=ax,
                title=title,
                xlabel="Values",
                ylabel="Counts",
                legend=("best", len(columns) + len(lst(distributions))),
                figsize=figsize or (10, 6),
                plotname="plot_distribution",
                filename=filename,
                display=display,
            )

    @composed(crash, typechecked)
    def plot_qq(
        self,
        columns: Union[int, str, slice, SEQUENCE_TYPES] = 0,
        distributions: Union[str, SEQUENCE_TYPES] = "norm",
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a quantile-quantile plot.

        Parameters
        ----------
        columns: int, str, slice or sequence, optional (default=0)
            Slice, names or indices of the columns to plot. Selected
            categorical columns are ignored.

        distributions: str, sequence or None, optional (default="norm")
            Names of the `scipy.stats` distributions to fit to the
            columns.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        check_dim(self, "plot_qq")
        columns = self._get_columns(columns)
        palette = cycle(sns.color_palette())

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        percentiles = np.linspace(0, 100, 101)
        for col in columns:
            color = next(palette)
            m = cycle(["+", "1", "x", "*", "d", "p", "h"])
            qn_b = np.percentile(self.dataset[col], percentiles)
            for dist in lst(distributions):
                stat = getattr(stats, dist)
                params = stat.fit(self.dataset[col])

                # Get the theoretical percentiles
                samples = stat.rvs(*params, size=101, random_state=self.random_state)
                qn_a = np.percentile(samples, percentiles)

                label = col + (" - " + dist if len(lst(distributions)) > 1 else "")
                plt.scatter(qn_a, qn_b, color=color, marker=next(m), s=50, label=label)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        plt.plot((-9e9, 9e9), (-9e9, 9e9), "k--", lw=2, alpha=0.7, zorder=-2)

        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            xlim=xlim,
            ylim=ylim,
            xlabel="Theoretical quantiles",
            ylabel="Observed quantiles",
            legend=("best", len(columns) + len(lst(distributions))),
            figsize=figsize or (10, 6),
            plotname="plot_qq",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_wordcloud(
        self,
        index: Optional[Union[int, str, SEQUENCE_TYPES]] = None,
        title: Optional[str] = None,
        figsize: Tuple[SCALAR, SCALAR] = (10, 6),
        filename: Optional[str] = None,
        display: Optional[bool] = True,
        **kwargs,
    ):
        """Plot a wordcloud from the corpus.

        The text for the plot is extracted from the column
        named `corpus`. If there is no column with that name,
        an exception is raised.

        Parameters
        ----------
        index: int, str, sequence or None, optional (default=None)
            Index names or positions of the documents in the corpus to
            include in the wordcloud. If None, it selects all documents
            in the dataset.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        **kwargs
            Additional keyword arguments for the WordCloud class.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """

        def get_text(column):
            """Get the complete corpus as one long string."""
            if isinstance(column.iloc[0], str):
                return " ".join(column)
            else:
                return " ".join([" ".join(row) for row in column])

        check_dim(self, "plot_wordcloud")
        corpus = get_corpus(self.X)
        rows = self.dataset.loc[self._get_rows(index, return_test=False)]

        fig = self._get_figure()
        ax = fig.add_subplot(BasePlotter._fig.grid)

        background_color = kwargs.pop("background_color", "white")
        random_state = kwargs.pop("random_state", self.random_state)
        wordcloud = WordCloud(
            width=figsize[0] * 100 if figsize else 1000,
            height=figsize[1] * 100 if figsize else 600,
            background_color=background_color,
            random_state=random_state,
            **kwargs,
        )

        plt.imshow(wordcloud.generate(get_text(rows[corpus])))
        plt.axis("off")

        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            figsize=figsize or (10, 6),
            plotname="plot_wordcloud",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_ngrams(
        self,
        ngram: Union[int, str] = "words",
        index: Optional[Union[int, str, SEQUENCE_TYPES]] = None,
        show: int = 10,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot n-gram frequencies.

        The text for the plot is extracted from the column
        named `corpus`. If there is no column with that name,
        an exception is raised. If the documents are not
        tokenized, the words are separated by spaces.

        Parameters
        ----------
        ngram: str or int, optional (default="bigram")
            Number of contiguous words to search for (size of
            n-gram). Choose from: words (1), bigrams (2),
            trigrams (3), quadgrams (4).

        index: int, str, sequence or None, optional (default=None)
            Index names or positions of the documents in the corpus to
            include in the search. If None, it selects all documents in
            the dataset.

        show: int, optional (default=10)
            Number of n-grams (ordered by number of occurrences) to
            show in the plot.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the number of n-grams shown.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """

        def get_text(column):
            """Get the complete corpus as sequence of tokens."""
            if isinstance(column.iloc[0], str):
                return column.apply(lambda row: row.split())
            else:
                return column

        check_dim(self, "plot_ngrams")
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
        ax = fig.add_subplot(BasePlotter._fig.grid)

        data = series[-show:]  # Subset of series to plot
        data[-show:].plot.barh(ax=ax, width=0.6, label=f"Total {ngram}: {len(series)}")

        # Add the counts at the end of the bar
        for i, v in enumerate(data[-show:]):
            ax.text(v + 0.01 * max(data), i - 0.08, v, fontsize=self.tick_fontsize)

        return self._plot(
            fig=fig,
            ax=ax,
            xlim=(min(data) - 0.1 * min(data), max(data) + 0.1 * max(data)),
            title=title,
            xlabel="Counts",
            legend=("lower right", 1),
            figsize=figsize or (10, 4 + show // 2),
            plotname="plot_ngrams",
            filename=filename,
            display=display,
        )

    @composed(crash, typechecked)
    def plot_pipeline(
        self,
        model: Optional[str] = None,
        show_params: bool = True,
        title: Optional[str] = None,
        figsize: Optional[Tuple[SCALAR, SCALAR]] = None,
        filename: Optional[str] = None,
        display: Optional[bool] = True,
    ):
        """Plot a diagram of a model's pipeline.

        Parameters
        ----------
        model: str or None, optional (default=None)
            Model from which to plot the pipeline. If no model is
            specified, the current pipeline is plotted.

        show_params: bool, optional (default=True)
            Whether to show the parameters used for every estimator.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, it adapts the
            size to the length of the pipeline.

        filename: str or None, optional (default=None)
            Name of the file. Use "auto" for automatic naming. If
            None, the figure is not saved.

        display: bool or None, optional (default=True)
            Whether to render the plot. If None, it returns the
            matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            Plot object. Only returned if `display=None`.

        """
        # Define pipeline to plot
        if not model:
            pipeline = self.branch.pipeline.tolist()
        else:
            model = self._models[self._get_model_name(model)[0]]
            pipeline = model.branch.pipeline.tolist() + [model.estimator]

        # Calculate figure's limits
        params = []
        ylim = 30
        for est in pipeline:
            ylim += 15
            if show_params:
                params.append(
                    [
                        param for param in signature(est.__init__).parameters
                        if param not in ["self"] + BaseTransformer.attrs
                    ]
                )
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

        for i, est in enumerate(pipeline):
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
                    # The param isn't always an attr of the class
                    if hasattr(est, key):
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
        return self._plot(
            fig=fig,
            ax=ax,
            title=title,
            xlim=(0, 100),
            ylim=(0, ylim),
            figsize=figsize or (8, ylim // 30),
            plotname="plot_pipeline",
            filename=filename,
            display=display,
        )

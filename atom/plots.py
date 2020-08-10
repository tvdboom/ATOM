# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the plotting classes.

"""

# Standard packages
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union, Sequence, Tuple

# Plotting packages
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# Sklearn
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import SCORERS, roc_curve, precision_recall_curve

# Own modules
from .utils import (
    CAL, TREE_MODELS, METRIC_ACRONYMS, lst, check_is_fitted, get_best_score,
    get_model_name, composed, crash, plot_from_model
    )


# Classes =================================================================== >>

class BasePlotter(object):
    """Parent class for all plotting methods.

    This base class defines the plot properties that can be changed in order
    to customize the plot's aesthetics. Making the variable mutable ensures
    that it is changed for all classes at the same time.

    """

    _aesthetics = dict(style='darkgrid',  # Seaborn plotting style
                       palette='GnBu_d',  # Seaborn color palette
                       title_fontsize=20,  # Fontsize for titles
                       label_fontsize=16,  # Fontsize for labels and legends
                       tick_fontsize=12)  # Fontsize for ticks
    sns.set_style(_aesthetics['style'])
    sns.set_palette(_aesthetics['palette'])

    # Properties ============================================================ >>

    @property
    def aesthetics(self):
        return self._aesthetics

    @aesthetics.setter
    @typechecked
    def aesthetics(self, aesthetics: dict):
        self.style = aesthetics.get('style', self.style)
        self.palette = aesthetics.get('palette', self.palette)
        self.title_fontsize = aesthetics.get('title_fontsize', self.title_fontsize)
        self.label_fontsize = aesthetics.get('label_fontsize', self.label_fontsize)
        self.tick_fontsize = aesthetics.get('tick_fontsize', self.tick_fontsize)

    @property
    def style(self):
        return self._aesthetics['style']

    @style.setter
    @typechecked
    def style(self, style: str):
        styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
        if style not in styles:
            raise ValueError("Invalid value for the style parameter, got " +
                             f"{style}. Choose from {', '.join(styles)}.")
        sns.set_style(style)
        self._aesthetics['style'] = style

    @property
    def palette(self):
        return self._aesthetics['palette']

    @palette.setter
    @typechecked
    def palette(self, palette: str):
        sns.set_palette(palette)
        self._aesthetics['palette'] = palette

    @property
    def title_fontsize(self):
        return self._aesthetics['title_fontsize']

    @title_fontsize.setter
    @typechecked
    def title_fontsize(self, title_fontsize: int):
        if title_fontsize <= 0:
            raise ValueError("Invalid value for the title_fontsize " +
                             "parameter. Value should be >=0, got " +
                             f"{title_fontsize}.")
        self._aesthetics['title_fontsize'] = title_fontsize

    @property
    def label_fontsize(self):
        return self._aesthetics['label_fontsize']

    @label_fontsize.setter
    @typechecked
    def label_fontsize(self, label_fontsize: int):
        if label_fontsize <= 0:
            raise ValueError("Invalid value for the label_fontsize " +
                             "parameter. Value should be >=0, got " +
                             f"{label_fontsize}.")
        self._aesthetics['label_fontsize'] = label_fontsize

    @property
    def tick_fontsize(self):
        return self._aesthetics['tick_fontsize']

    @tick_fontsize.setter
    @typechecked
    def tick_fontsize(self, tick_fontsize: int):
        if tick_fontsize <= 0:
            raise ValueError("Invalid value for the tick_fontsize " +
                             "parameter. Value should be >=0, got " +
                             f"{tick_fontsize}.")
        self._aesthetics['tick_fontsize'] = tick_fontsize

    # Methods =============================================================== >>

    def _check_models(self, models):
        """Check the provided input models.

        Make sure all names are correct and that every model is in the trainer's
        pipeline. If models is None, returns all models in pipeline.

        Parameters
        ----------
        models: str or sequence
            Models provided by the plot's parameter.

        """
        if models is None:
            return self.models_
        elif isinstance(models, str):
            models = [get_model_name(models)]
        else:
            models = [get_model_name(m) for m in models]

        # Check that all models are in the pipeline
        for model in models:
            if model not in self.models:
                raise ValueError(f"Model {model} not found in the pipeline!")

        return [m for m in self.models_ if m.name in models]

    def _check_metric(self, metric):
        """Check the provided input metric_.

        Parameters
        ----------
        metric: int or str
            Metric provided by the metric_ parameter.

        """
        def _raise():
            raise ValueError(
                "Invalid value for the metric_ parameter. Value should be the " +
                f"index or name of a metric_ used to run the pipeline, got {metric}.")

        # If it's a str, return the corresponding idx
        if isinstance(metric, str):
            if metric.lower() in METRIC_ACRONYMS:
                metric = METRIC_ACRONYMS[metric]

            for i, m in enumerate(self.metric_):
                if metric.lower() == m.name:
                    return i
            _raise()  # If no match was found, raise an exception

        # If index not in available metrics, raise an exception
        elif metric < 0 or metric > len(self.metric_) - 1:
            _raise()

        return metric

    def _plot(self, **kwargs):
        """Make the plot.

        Parameters
        ----------
        **kwargs
            Keyword arguments containing the plot's parameters. Can contain:
                - fig: matplotlib.figure.Figure class for the plot.
                - title: Plot's title.
                - legend: Loc to place the legend.
                - xlabel: Label for the plot's x-axis.
                - ylabel: Label for the plot's y-axis.
                - xlim: Limits for the plot's x-axis.
                - ylim: Limits for the plot's y-axis.
                - filename: Name of the file (to save).
                - display: Whether to render the plot.

        """
        if kwargs.get('title'):
            plt.title(kwargs.get('title'), fontsize=self.title_fontsize, pad=12)
        if kwargs.get('legend'):
            plt.legend(loc=kwargs['legend'], fontsize=self.label_fontsize)
        if kwargs.get('xlabel'):
            plt.xlabel(kwargs['xlabel'], fontsize=self.label_fontsize, labelpad=12)
        if kwargs.get('ylabel'):
            plt.ylabel(kwargs['ylabel'], fontsize=self.label_fontsize, labelpad=12)
        if kwargs.get('xlim'):
            plt.xlim(kwargs['xlim'])
        if kwargs.get('ylim'):
            plt.xlim(kwargs['ylim'])
        plt.xticks(fontsize=self.tick_fontsize)
        plt.yticks(fontsize=self.tick_fontsize)
        plt.tight_layout()
        if kwargs.get('filename'):
            plt.savefig(kwargs['filename'])
        plt.show() if kwargs.get('display') else plt.close()


class FeatureSelectorPlotter(BasePlotter):
    """Plots for the FeatureSelector class."""

    @composed(crash, typechecked)
    def plot_pca(self,
                 title: Optional[str] = None,
                 figsize: Optional[Tuple[int, int]] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the explained variance ratio vs the number of components.

        Parameters
        ----------
        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        if not self.pca:
            raise PermissionError(
                "This plot is only available if you applied PCA on the data!")

        n = self.pca.n_components_  # Number of chosen components
        var = np.array(self.pca.explained_variance_ratio_[:n])
        var_all = np.array(self.pca.explained_variance_ratio_)

        fig, ax = plt.subplots(figsize=figsize)
        plt.scatter(self.pca.n_components_ - 1,
                    var.sum(),
                    marker='*',
                    s=130,
                    c='blue',
                    edgecolors='b',
                    zorder=3,
                    label=f"Total variance retained: {round(var.sum(), 3)}")
        plt.plot(range(0, self.pca.n_features_), np.cumsum(var_all), marker='o')
        plt.axhline(var.sum(), ls='--', color='k')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only int ticks

        self._plot(title="PCA explained variances" if title is None else title,
                   legend='lower right',
                   xlabel='First N principal components',
                   ylabel='Cumulative variance ratio',
                   filename=filename,
                   display=display)

    @composed(crash, typechecked)
    def plot_components(self,
                        show: Optional[int] = None,
                        title: Optional[str] = None,
                        figsize: Optional[Tuple[int, int]] = None,
                        filename: Optional[str] = None,
                        display: bool = True):
        """Plot the explained variance ratio per component.

        Parameters
        ----------
        show: int or None, optional (default=None)
            Number of components to show. If None, the number of components
            in the data are plotted.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        if not self.pca:
            raise PermissionError(
                "This plot is only available if you applied PCA on the data!")

        # Set parameters
        if show is None:
            show = self.pca.n_components_
        elif show > self.pca.n_features_:
            show = self.pca.n_features_
        elif show < 1:
            raise ValueError("Invalid value for the show parameter. " +
                             f"Value should be >0, got {show}.")
        if figsize is None:  # Default figsize depends on features shown
            figsize = (10, int(4 + show / 2))

        var = np.array(self.pca.explained_variance_ratio_)[:show]
        indices = ['Component ' + str(i) for i in range(len(var))]
        scr = pd.Series(var, index=indices).sort_values()

        fig, ax = plt.subplots(figsize=figsize)
        scr.plot.barh(label=f"Total variance retained: {var.sum():.3f}", width=0.6)
        plt.xlim(0, max(scr) + 0.1 * max(scr))  # Make extra space for numbers
        for i, v in enumerate(scr):
            ax.text(v + 0.005, i - 0.08, f'{v:.3f}', fontsize=self.tick_fontsize)

        title = "Explained variance per component" if title is None else title
        self._plot(title=title,
                   legend='lower right',
                   xlabel='Explained variance ratio',
                   ylabel='Components',
                   filename=filename,
                   display=display)

    @composed(crash, typechecked)
    def plot_rfecv(self,
                   title: Optional[str] = None,
                   figsize: Optional[Tuple[int, int]] = (10, 6),
                   filename: Optional[str] = None,
                   display: bool = True):
        """Plot the RFECV results.

        Plot the scores obtained by the estimator fitted on every subset of
        the data. Only if RFECV was applied on the dataset through the
        feature_engineering method.

        Parameters
        ----------
        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        if not self.rfecv:
            raise PermissionError(
                "This plot is only available if you applied RFECV on the data!")

        try:  # Define the y-label for the plot
            ylabel = self.rfecv.get_params()['scoring'].name
        except AttributeError:
            ylabel = 'score'
            if self.rfecv.get_params()['scoring']:
                ylabel = str(self.rfecv.get_params()['scoring'])

        fig, ax = plt.subplots(figsize=figsize)
        n_features = self.rfecv.get_params()['min_features_to_select']
        xline = range(n_features, n_features + len(self.rfecv.grid_scores_))
        ax.axvline(xline[int(np.argmax(self.rfecv.grid_scores_))],
                   ls='--',
                   color='k',
                   label=f'Best score: {round(max(self.rfecv.grid_scores_), 3)}')
        plt.plot(xline, self.rfecv.grid_scores_)
        plt.xlim(n_features - 0.5, n_features + len(self.rfecv.grid_scores_) - 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only int ticks

        self._plot(title="RFE cross-validation scores" if title is None else title,
                   legend='lower right',
                   xlabel='Number of features',
                   ylabel=ylabel,
                   filename=filename,
                   display=display)


class BaseModelPlotter(BasePlotter):
    """Plots for the BaseModel class."""

    @composed(crash, plot_from_model, typechecked)
    def plot_bagging(self,
                     models: Union[None, str, Sequence[str]] = None,
                     metric: Union[int, str] = 0,
                     title: Optional[str] = None,
                     figsize: Optional[Tuple[int, int]] = None,
                     filename: Optional[str] = None,
                     display: bool = True):
        """Boxplot of the bagging's results.

        Only available if the models were fitted using bagging.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline that used bagging are selected.

        metric: int or str, optional (default=0)
            Index or name of the metric_ to plot. Only for multi-metric_ runs.

        title: str or None, optional (default=None)
            Plot's title. If None, adapts size to the number of models.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)
        metric = self._check_metric(metric)

        # Check there is at least one model with bagging
        if all([not m.score_bagging for m in models]):
            raise PermissionError("You need to run the pipeline using bagging>0 " +
                                  "to use the plot_bagging method!")

        results, names = [], []
        for m in models:
            if m.score_bagging:
                if len(self.metric_) > 1:  # Is list of lists
                    results.append(lst(m.score_bagging)[metric])
                else:  # Is single list
                    results.append(m.score_bagging)
                names.append(m.name)

        if figsize is None:  # Default figsize depends on number of models
            figsize = (int(8 + len(names)/2), 6)

        fig, ax = plt.subplots(figsize=figsize)
        plt.boxplot(results)
        ax.set_xticklabels(names)

        self._plot(title="Bagging results" if title is None else title,
                   xlabel='Model',
                   ylabel=self.metric_[metric].name,
                   filename=filename,
                   display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_bo(self,
                models: Union[None, str, Sequence[str]] = None,
                metric: Union[int, str] = 0,
                title: Optional[str] = None,
                figsize: Tuple[int, int] = (10, 6),
                filename: Optional[str] = None,
                display: bool = True):

        """Plot the bayesian optimization scoring.

        Only for models that ran the hyperparameter optimization. This is the same
        plot as the one produced by `bo_params{'plot_bo': True}` while running the
        BO. Creates a canvas with two plots: the first plot shows the score of every
        trial and the second shows the distance between the last consecutive steps.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the pipeline that
            used bayesian optimization are selected.

        metric: int or str, optional (default=0)
            Index or name of the metric_ to plot. Only for multi-metric_ runs.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)
        metric = self._check_metric(metric)

        # Check there is at least one model that run the BO
        if all([m.bo.empty for m in models]):
            raise PermissionError(
                "The plot_bo method is only available for models that " +
                "ran the bayesian optimization hyperparameter tuning!")

        plt.subplots(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        for m in models:
            y = m.bo['score'].apply(lambda value: lst(value)[metric])
            if len(models) == 1:
                label = f"Score={round(lst(m.score_bo)[metric], 3)}"
            else:
                label = f"{m.name} (Score={round(lst(m.score_bo)[metric], 3)})"

            # Draw bullets onm all markers except the maximum
            markers = [i for i in range(len(m.bo))]
            markers.remove(int(np.argmax(y)))
            ax1.plot(range(1, len(y)+1), y, '-o', markevery=markers, label=label)
            ax2.plot(range(2, len(y)+1), np.abs(np.diff(y)), '-o')
            ax1.scatter(np.argmax(y)+1, max(y), zorder=10, s=100, marker='*')

        title = "Bayesian optimization scoring" if title is None else title
        ax1.set_title(title, fontsize=self.title_fontsize, pad=12)
        ax1.legend(loc='lower right', fontsize=self.label_fontsize)
        ax2.set_title("Distance between last consecutive iterations",
                      fontsize=self.title_fontsize)
        ax2.set_xlabel('Iteration', fontsize=self.label_fontsize, labelpad=12)
        ax1.set_ylabel(self.metric_[metric].name, fontsize=self.label_fontsize, labelpad=12)
        ax2.set_ylabel('d', fontsize=self.label_fontsize, labelpad=12)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        self._plot(filename=filename, display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_evals(self,
                   models: Union[None, str, Sequence[str]] = None,
                   title: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 6),
                   filename: Optional[str] = None,
                   display: bool = True):
        """Plot evaluation curves for the train and test set.

         Only for models that allow in-training evaluation. Only allows plotting
         of one model at a time.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the pipeline are
            selected. Note that this will raise an exception if there are multiple
            models in the pipeline.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        m = self._check_models(models)
        if len(m) > 1:
            raise ValueError(
                "The plot_evaluation method can only plot one model at a time.")
        else:
            m = m[0]  # Get first (and only) element of list

        # Check that all models have evals
        if not m.evals:
            raise AttributeError(
                "The plot_evaluation method is only available " +
                f"for models that allow in-training evaluation, got {m.name}.")

        plt.subplots(figsize=figsize)
        plt.plot(range(len(m.evals['train'])), m.evals['train'], lw=2, label='train')
        plt.plot(range(len(m.evals['test'])), m.evals['test'], lw=2, label='test')

        self._plot(title="Evaluation curves" if title is None else title,
                   legend='best',
                   xlabel=m.get_domain()[0].name,  # First param is always the iter
                   ylabel=m.evals['metric_'],
                   filename=filename,
                   display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_roc(self,
                 models: Union[None, str, Sequence[str]] = None,
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the Receiver Operating Characteristics curve.

        The legend shows the Area Under the ROC Curve (AUC) score. Only for
        binary classification tasks.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10,6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)

        if not self.task.startswith('bin'):
            raise PermissionError("The plot_ROC method is only available " +
                                  "for binary classification tasks!")

        plt.subplots(figsize=figsize)
        for m in models:
            # Get False (True) Positive Rate as arrays
            fpr, tpr, _ = roc_curve(m.y_test, m.predict_proba_test[:, 1])

            # Draw line
            if len(models) == 1:
                label = f"AUC={m.scoring('roc_auc'):.3f}"
            else:
                label = f"{m.name} (AUC={m.scoring('roc_auc'):.3f})"
            plt.plot(fpr, tpr, lw=2, label=label)

        plt.plot([0, 1], [0, 1], lw=2, color='black', linestyle='--')

        self._plot(title="ROC curve" if title is None else title,
                   legend='lower right',
                   xlabel='FPR',
                   ylabel='TPR',
                   filename=filename,
                   display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_prc(self,
                 models: Union[None, str, Sequence[str]] = None,
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the precision-recall curve.

        The legend shows the average precision (AP) score. Only for binary
        classification tasks.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)

        if not self.task.startswith('binary'):
            raise PermissionError("The plot_PRC method is only available for " +
                                  "binary classification tasks!")

        plt.subplots(figsize=figsize)
        for m in models:
            # Get precision-recall pairs for different probability thresholds
            precision, recall, _ = \
                precision_recall_curve(m.y_test, m.predict_proba_test[:, 1])

            # Draw line
            if len(models) == 1:
                label = f"AP={m.scoring('average_precision'):.3f}"
            else:
                label = f"{m.name} (AP={m.scoring('average_precision'):.3f})"
            plt.plot(recall, precision, lw=2, label=label)

        self._plot(title="Precision-recall curve" if title is None else title,
                   legend='lower left',
                   xlabel='Recall',
                   ylabel='Precision',
                   filename=filename,
                   display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_permutation_importance(self,
                                    models: Union[None, str, Sequence[str]] = None,
                                    show: Optional[int] = None,
                                    n_repeats: int = 10,
                                    title: Optional[str] = None,
                                    figsize: Optional[Tuple[int, int]] = None,
                                    filename: Optional[str] = None,
                                    display: bool = True):
        """Plot the feature permutation importance of models.

        If a permutation is repeated for the same model with the same
        amount of n_repeats, the calculation is skipped.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        show: int, optional (default=None)
            Number of best features to show in the plot. None for all.

        n_repeats: int, optional (default=10)
            Number of times to permute each feature.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional(default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)
        if show is None:
            show = self.X.shape[1]
        elif show <= 0:
            raise ValueError("Invalid value for the show parameter." +
                             f"Value should be >0, got {show}.")
        if n_repeats <= 0:
            raise ValueError("Invalid value for the n_repeats parameter." +
                             f"Value should be >0, got {n_repeats}.")

        # Default figsize depends on features shown
        if figsize is None:
            figsize = (10, int(4 + show/2))

        # Create dataframe with columns as indices to plot with barh
        df = pd.DataFrame(columns=['features', 'score', 'model'])

        # Create dictionary to store the permutations per model
        if not hasattr(self, 'permutations'):
            self.permutations = {}

        for m in models:
            # If permutations are already calculated and n_repeats is same,
            # use known permutations (for efficient re-plotting)
            if not hasattr(m, '_repts'):
                m._rpts = -np.inf
            if m.name not in self.permutations.keys() or m._rpts != n_repeats:
                m._rpts = n_repeats
                # Permutation importances returns Bunch object from sklearn
                self.permutations[m.name] = \
                    permutation_importance(m.model,
                                           m.X_test,
                                           m.y_test,
                                           scoring=self.metric_[0],
                                           n_repeats=n_repeats,
                                           n_jobs=self.n_jobs,
                                           random_state=self.random_state)

            # Append data to the dataframe
            for i, feature in enumerate(self.X.columns):
                for score in self.permutations[m.name].importances[i, :]:
                    df = df.append({'features': feature,
                                    'score': score,
                                    'model': m.name},
                                   ignore_index=True)

        # Get the column names sorted by mean of score
        get_idx = df.groupby('features', as_index=False)['score'].mean()
        get_idx.sort_values('score', ascending=False, inplace=True)
        column_order = get_idx.features.values[:show]

        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x='score',
                    y='features',
                    hue='model',
                    data=df,
                    order=column_order,
                    width=0.75 if len(models) > 1 else 0.6)

        # Remove seaborn's legend title
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])

        self._plot(title=title,
                   legend='lower right',
                   xlabel='Score',
                   ylabel='Features',
                   filename=filename,
                   display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_feature_importance(self,
                                models: Union[None, str, Sequence[str]] = None,
                                show: Optional[int] = None,
                                title: Optional[str] = None,
                                figsize: Optional[Tuple[int, int]] = None,
                                filename: Optional[str] = None,
                                display: bool = True):
        """Plot a tree-based model's feature importance.

        The importances are normalized in order to be able to compare them
        between models.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        show: int, optional (default=None)
            Number of best features to show in the plot. None for all.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)
        if show is None:
            show = self.X.shape[1]
        elif show <= 0:
            raise ValueError("Invalid value for the show parameter." +
                             f"Value should be >0, got {show}.")

        # Create dataframe with columns as indices to plot with barh
        df = pd.DataFrame(index=self.X.columns)

        for m in models:
            if m.name not in TREE_MODELS:
                raise PermissionError(
                    "The plot_feature_importance method is only available for " +
                    f"tree-based models, got {m.longname}!")

            # Bagging has no direct feature importance implementation
            if m.name == 'Bag':
                feature_importances = np.mean([
                    fi.feature_importances_ for fi in m.model.estimators_
                ], axis=0)
            else:
                feature_importances = m.model.feature_importances_

            # Normalize for plotting values adjacent to bar
            df[m.name] = feature_importances/max(feature_importances)

        # Select best and sort ascending
        df = df.nlargest(show, columns=df.columns[-1])
        df.sort_values(by=df.columns[-1], ascending=True, inplace=True)

        if figsize is None:  # Default figsize depends on features shown
            figsize = (10, int(4 + show/2))

        # Plot figure
        ax = df.plot.barh(figsize=figsize, width=0.75 if len(models) > 1 else 0.6)
        if len(models) == 1:
            for i, v in enumerate(df[df.columns[0]]):
                ax.text(v + .01, i - .08, f'{v:.2f}', fontsize=self.tick_fontsize)

        self._plot(title="Normalized feature importance" if title is None else title,
                   legend='lower right',
                   xlim=(0, 1.07),
                   xlabel='Score',
                   ylabel='Features',
                   filename=filename,
                   display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_confusion_matrix(self,
                              models: Union[None, str, Sequence[str]] = None,
                              normalize: bool = False,
                              title: Optional[str] = None,
                              figsize: Optional[Tuple[int, int]] = None,
                              filename: Optional[str] = None,
                              display: bool = True):
        """Plot the confusion matrix.

        For 1 model: plot the confusion matrix in a heatmap.
        For >1 models: compare TP, FP, FN and TN in a barplot.
                       Not implemented for multiclass classification.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        normalize: bool, optional (default=False)
           Whether to normalize the matrix.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to plot type.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)

        if self.task.startswith('reg'):
            raise PermissionError("The plot_confusion_matrix_method is only " +
                                  "available for classification tasks!")

        if self.task.startswith('multi') and len(models) > 1:
            raise NotImplementedError(
                "The plot_confusion_matrix method does not support the " +
                "comparison of various models for multiclass classification tasks.")

        # Create dataframe to plot with barh if len(models) > 1
        df = pd.DataFrame(index=['True negatives', 'False positives',
                                 'False negatives', 'True positives'])
        # Define title
        if title is None and normalize:
            title = "Normalized confusion matrix"
        elif title is None:
            title = "Confusion matrix"

        for m in models:
            cm = m.scoring('confusion_matrix')
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            if len(models) == 1:  # Create matrix heatmap
                if hasattr(self, 'mapping'):
                    ticks = [v for v in self.mapping.keys()]
                else:
                    ticks = list(m.y_test.unique())

                fig, ax = plt.subplots(figsize=(8, 8) if not figsize else figsize)
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

                # Create an axes on the right side of ax. The under of cax will
                # be 5% of ax and the padding between cax and ax will be fixed
                # at 0.3 inch.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.3)
                cbar = ax.figure.colorbar(im, cax=cax)
                ax.set(xticks=np.arange(cm.shape[1]),
                       yticks=np.arange(cm.shape[0]),
                       xticklabels=ticks,
                       yticklabels=ticks)

                # Loop over data dimensions and create text annotations
                fmt = '.2f' if normalize else 'd'
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], fmt),
                                ha='center', va='center',
                                fontsize=self.tick_fontsize,
                                color='w' if cm[i, j] > cm.max() / 2. else 'k')

                ax.set_title(title, fontsize=self.title_fontsize, pad=12)
                ax.set_xlabel('Predicted label',
                              fontsize=self.label_fontsize,
                              labelpad=12)
                ax.set_ylabel('True label',
                              fontsize=self.label_fontsize,
                              labelpad=12)
                cbar.set_label('Count',
                               fontsize=self.label_fontsize,
                               labelpad=15,
                               rotation=270)
                cbar.ax.tick_params(labelsize=self.tick_fontsize)
                ax.grid(False)

            else:  # Create barplot
                df[m.name] = cm.ravel()

        if len(models) > 1:
            df.plot.barh(figsize=(10, 6) if not figsize else figsize, width=0.6)
            self._plot(title=title,
                       legend='best',
                       xlabel='Count',
                       filename=filename,
                       display=display)
        else:
            self._plot(filename=filename, display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_threshold(self,
                       models: Union[None, str, Sequence[str]] = None,
                       metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                       steps: int = 100,
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       filename: Optional[str] = None,
                       display: bool = True):
        """Plot performance metric_(s) against multiple threshold values.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        metric: string, callable, list, tuple or None, optional (default=None)
            Metric(s) to plot. These can be one of the pre-defined sklearn scorers
            as string, a metric_ function or a sklearn scorer object. If None, the
            metric_ used to run the pipeline is used.

        steps: int, optional (default=100)
            Number of thresholds measured.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 10))
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)

        if not self.task.startswith('bin'):
            raise PermissionError("The plot_threshold method is only " +
                                  "available for binary classification tasks!")

        # Check that all models have predict_proba
        for m in models:
            if not hasattr(m.model, 'predict_proba'):
                raise AttributeError(
                    "The plot_probabilities method is only available " +
                    f"for models with a predict_proba method, got {m}.")

        if metric is None:
            metric = self.metric_
        elif not isinstance(metric, list):
            metric = [metric]

        # Convert all strings to functions
        metric_list = []
        for met in metric:
            if isinstance(met, str):  # It is one of sklearn predefined metrics
                if met in METRIC_ACRONYMS:
                    met = METRIC_ACRONYMS[met]

                if met not in SCORERS:
                    raise ValueError("Unknown value for the metric_ parameter, " +
                                     f"got {met}. Try one of {list(SCORERS)}.")
                metric_list.append(SCORERS[met]._score_func)
            elif hasattr(met, '_score_func'):  # It is a scorer
                metric_list.append(met._score_func)
            else:  # It is a metric_ function
                metric_list.append(met)

        plt.subplots(figsize=figsize)
        steps = np.linspace(0, 1, steps)
        for m in models:
            for met in metric_list:  # Create dict of empty arrays
                results = []
                for step in steps:
                    predictions = (m.predict_proba_test[:, 1] >= step).astype(bool)
                    results.append(met(m.y_test, predictions))

                # Draw the line for each metric_
                if len(models) == 1:
                    label = met.__name__
                else:
                    label = f"{m.name} ({met.__name__})"
                plt.plot(steps, results, label=label, lw=2)

        if title is None:
            temp = '' if len(metric) == 1 else 's'
            title = f"Performance metric_{temp} against threshold value"
        self._plot(title=title,
                   legend='best',
                   xlabel='Threshold',
                   ylabel='Score',
                   filename=filename,
                   display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_probabilities(self,
                           models: Union[None, str, Sequence[str]] = None,
                           target: Union[int, str] = 1,
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           filename: Optional[str] = None,
                           display: bool = True):
        """Plot the probability of being the target category for every category.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        target: int or string, optional (default=1)
            Probability of being that class (as index or value_name).

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 10))
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        # Set parameters
        check_is_fitted(self, 'results')
        models = self._check_models(models)

        if self.task.startswith('reg'):
            raise PermissionError("The plot_probabilities method is only " +
                                  "available for classification tasks!")

        # Check that all models have predict_proba
        for m in models:
            if not hasattr(m.model, 'predict_proba'):
                raise AttributeError(
                    "The plot_probabilities method is only available " +
                    f"for models with a predict_proba method, got {m}.")

        # Make inverse target mapping
        if hasattr(self, 'mapping'):
            inv_map = {str(int(v)): k for k, v in self.mapping.items()}
        else:
            self.mapping = inv_map = {str(i): i for i in self.y_test.unique()}
        if isinstance(target, str):  # User provides a string
            target_int = self.mapping[target]
            target_str = target
        else:  # User provides an integer
            target_int = target
            target_str = inv_map[str(target)]

        plt.subplots(figsize=figsize)
        for m in models:
            for key, value in self.mapping.items():
                idx = np.where(m.y_test == value)[0]  # Get indices per class
                if len(models) == 1:
                    label = f"Category: {key}"
                else:
                    label = f"{m.name} (Category: {key})"
                sns.distplot(m.predict_proba_test[idx, target_int],
                             hist=False,
                             kde=True,
                             norm_hist=True,
                             kde_kws={'shade': True},
                             label=label)

        if title is None:
            title = f"Predicted probabilities for {m.y_test.name}={target_str}"
        self._plot(title=title,
                   legend='best',
                   xlabel='Probability',
                   ylabel='Count',
                   xlim=(0, 1),
                   filename=filename,
                   display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_calibration(self,
                         models: Union[None, str, Sequence[str]] = None,
                         n_bins: int = 10,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 10),
                         filename: Optional[str] = None,
                         display: bool = True):
        """Plot the calibration curve for a binary classifier.

        Well calibrated classifiers are probabilistic classifiers for which the
        output of the predict_proba method can be directly interpreted as a
        confidence level. For instance a well calibrated (binary) classifier
        should classify the samples such that among the samples to which it gave
        a predict_proba value close to 0.8, approx. 80% actually belong to the
        positive class. This figure shows two plots: the calibration curve, where
        the x-axis represents the average predicted probability in each bin and the
        y-axis is the fraction of positives, i.e. the proportion of samples whose
        class is the positive class (in each bin); and a distribution of all
        predicted probabilities of the classifier.
        Code snippets from https://scikit-learn.org/stable/auto_examples/
        calibration/plot_calibration_curve.html

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        n_bins: int, optional (default=10)
            Number of bins for the calibration calculation and the histogram.
            Minimum of 5 required.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 10))
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)

        if not self.task.startswith('bin'):
            raise PermissionError("The plot_probabilities method is only " +
                                  "available for binary classification tasks!")

        # Set parameters
        if n_bins < 5:
            raise ValueError("Invalid value for the n_bins parameter." +
                             f"Value should be >=5, got {n_bins}.")

        plt.figure(figsize=figsize)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax1.plot([0, 1], [0, 1], color='k', ls='--')
        for m in models:
            if hasattr(m.model, 'decision_function'):
                prob = m.decision_function_test
                prob = (prob - prob.min()) / (prob.max() - prob.min())
            elif hasattr(m.model, 'predict_proba'):
                prob = m.predict_proba_test[:, 1]

            # Calculate the calibration (fraction of positives and predicted values)
            frac_pos, pred = calibration_curve(self.y_test, prob, n_bins=n_bins)

            # Draw plots
            ax1.plot(pred, frac_pos, marker='o', lw=2, label=f"{m.name}")
            ax2.hist(prob, range=(0, 1), bins=n_bins,
                     label=m.name, histtype="step", lw=2)

        title = 'Calibration curve' if title is None else title
        ax1.set_title(title, fontsize=self.title_fontsize, pad=12)
        ax1.set_ylabel("Fraction of positives",
                       fontsize=self.label_fontsize,
                       labelpad=12)
        ax1.set_ylim([-0.05, 1.05])

        ax2.set_xlabel("Predicted value",
                       fontsize=self.label_fontsize,
                       labelpad=12)
        ax2.set_ylabel("Count", fontsize=self.label_fontsize, labelpad=12)

        # Only draw the legends for more than one model
        if len(models) > 1:
            ax1.legend(loc="lower right", fontsize=self.label_fontsize)
            ax2.legend(loc='best', fontsize=self.label_fontsize, ncol=3)

        self._plot(filename=filename, display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_gains(self,
                   models: Union[None, str, Sequence[str]] = None,
                   title: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 6),
                   filename: Optional[str] = None,
                   display: bool = True):
        """Plot the cumulative gains curve.

        Only for binary classification. Code snippet from https://github.com/
        reiinakano/scikit-plot/

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)

        if not self.task.startswith('bin'):
            raise PermissionError("The plot_gain method is only " +
                                  "available for binary classification tasks!")

        # Check that all models have predict_proba
        for m in models:
            if not hasattr(m.model, 'predict_proba'):
                raise AttributeError(
                    "The plot_probabilities method is only available " +
                    f"for models with a predict_proba method, got {m}.")

        plt.subplots(figsize=figsize)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        for m in models:
            # Compute Cumulative Gain Curves
            y_true = (m.y_test == 1)  # Make y_true a boolean vector

            # Get sorted indices and correct for the test set
            sorted_indices = np.argsort(m.predict_proba_test[:, 1])[::-1]
            sorted_indices = [i + len(m.y_train) for i in sorted_indices]
            gains = np.cumsum(y_true.loc[sorted_indices])/float(np.sum(y_true))

            x = np.arange(start=1, stop=len(y_true) + 1)/float(len(y_true))
            plt.plot(x, gains, lw=2, label=f'{m.name}')

        self._plot(title="Cumulative gains curve" if title is None else title,
                   legend='lower right',
                   xlabel="Fraction of sample",
                   ylabel='Gain',
                   xlim=(0, 1),
                   ylim=(0, 1.02),
                   filename=filename,
                   display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_lift(self,
                  models: Union[None, str, Sequence[str]] = None,
                  title: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 6),
                  filename: Optional[str] = None,
                  display: bool = True):
        """Plot the lift curve.

        Only for binary classification. Code snippet from https://github.com/
        reiinakano/scikit-plot/

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)

        if not self.task.startswith('bin'):
            raise PermissionError("The plot_gain method is only " +
                                  "available for binary classification tasks!")

        # Check that all models have predict_proba
        for m in models:
            if not hasattr(m.model, 'predict_proba'):
                raise AttributeError(
                    "The plot_probabilities method is only available " +
                    f"for models with a predict_proba method, got {m}.")

        plt.subplots(figsize=figsize)
        plt.plot([0, 1], [1, 1], 'k--', lw=2)
        for m in models:
            # Compute Cumulative Gain Curves
            y_true = (m.y_test == 1)  # Make y_true a boolean vector

            sorted_indices = np.argsort(m.predict_proba_test[:, 1])[::-1]
            sorted_indices = [i + len(m.y_train) for i in sorted_indices]
            gains = np.cumsum(y_true.loc[sorted_indices])/float(np.sum(y_true))

            if len(models) == 1:
                label = f"Lift={round(m.scoring('lift'), 3)}"
            else:
                label = f"{m.name} (Lift={round(m.scoring('lift'), 3)})"
            x = np.arange(start=1, stop=len(y_true) + 1)/float(len(y_true))
            plt.plot(x, gains/x, lw=2, label=label)

        self._plot(title="Lift curve" if title is None else title,
                   legend='upper right',
                   xlabel="Fraction of sample",
                   ylabel='Lift',
                   xlim=(0, 1),
                   filename=filename,
                   display=display)


class SuccessiveHalvingPlotter(BaseModelPlotter):
    """Plots for the SuccessiveHalving classes."""

    @composed(crash, plot_from_model, typechecked)
    def plot_successive_halving(self,
                                models: Union[None, str, Sequence[str]] = None,
                                metric: Union[int, str] = 0,
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 6),
                                filename: Optional[str] = None,
                                display: bool = True):
        """Plot of the models' scores per iteration of the successive halving.

        Only available if the models were fitted via successive_halving.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        metric: int or str, optional (default=0)
            Index or name of the metric_ to plot. Only for multi-metric_ runs.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)
        metric = self._check_metric(metric)
        if hasattr(self, 'trainer'):
            trainer = self.trainer.__class__.__name__
        else:
            trainer = self.__class__.__name__
        if not trainer.startswith('SuccessiveHalving'):
            raise PermissionError(
                "You need to run the pipeline using successive " +
                "halving to use the plot_successive_halving method!")

        fig, ax = plt.subplots(figsize=figsize)
        for m in models:
            # Make df with rows for only that model
            df = self._results.xs(m.name, level='model')
            y = df.apply(lambda row: get_best_score(row, metric), axis=1).values
            std = df.apply(lambda row: lst(row.std_bagging)[metric], axis=1)
            plt.plot(range(len(y)), y, lw=2, marker='o', label=m.name)
            if not any(std.isna()):  # Plot fill if std is not all None
                plt.fill_between(range(len(y)), y + std, y - std, alpha=0.3)

        plt.xlim(-0.1, max(self._results.index.get_level_values('run')) + 0.1)
        ax.set_xticks(range(1 + max(self._results.index.get_level_values('run'))))

        self._plot(title="Successive halving results" if title is None else title,
                   legend='lower right',
                   xlabel='Iteration',
                   ylabel=self.metric_[metric].name,
                   filename=filename,
                   display=display)


class TrainSizingPlotter(BaseModelPlotter):
    """Plots for the TrainSizing classes."""

    @composed(crash, plot_from_model, typechecked)
    def plot_learning_curve(self,
                            models: Union[None, str, Sequence[str]] = None,
                            metric: Union[int, str] = 0,
                            title: Optional[str] = None,
                            figsize: Tuple[int, int] = (10, 6),
                            filename: Optional[str] = None,
                            display: bool = True):
        """Plot the model's learning curve: score vs number of training samples.

        Only available if the models were fitted via train_sizing.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        metric: int or str, optional (default=0)
            Index or name of the metric_ to plot. Only for multi-metric_ runs.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)
        metric = self._check_metric(metric)
        if hasattr(self, 'trainer'):
            trainer = self.trainer.__class__.__name__
        else:
            trainer = self.__class__.__name__
        if not trainer.startswith('TrainSizing'):
            raise PermissionError("You need to run the pipeline using train " +
                                  "sizing to use the plot_learning_curve method!")

        plt.subplots(figsize=figsize)
        for m in models:
            # Make df with rows for only that model
            df = self._results.xs(m.name, level='model')
            y = df.apply(lambda row: get_best_score(row, metric), axis=1).values
            std = df.apply(lambda row: lst(row.std_bagging)[metric], axis=1)
            plt.plot(self._sizes, y, lw=2, marker='o', label=m.name)
            if not any(std.isna()):  # Plot fill if std is not all None
                plt.fill_between(self._sizes, y + std, y - std, alpha=0.3)

        if max(self._sizes) > 1e4:
            plt.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))

        self._plot(title="Learning curve" if title is None else title,
                   legend='lower right',
                   xlabel='Number of training samples',
                   ylabel=self.metric_[metric].name,
                   filename=filename,
                   display=display)


class ATOMPlotter(FeatureSelectorPlotter,
                  SuccessiveHalvingPlotter,
                  TrainSizingPlotter):
    """Plots for the ATOM classes."""

    @composed(crash, typechecked)
    def plot_correlation(self,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 10),
                         filename: Optional[str] = None,
                         display: bool = True):
        """Plot the data's correlation matrix. Ignores non-numeric columns.

        Parameters
        ----------
        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 10))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        # Compute the correlation matrix
        corr = self.dataset.corr()

        # Drop first row and last column (diagonal line)
        corr = corr.iloc[1:].drop(self.target, axis=1)

        # Generate a mask for the upper triangle
        # k=1 means keep outermost diagonal line
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        sns.set_style('white')  # Only for this plot
        plt.subplots(figsize=figsize)

        # Draw the heatmap with the mask and correct aspect ratio
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={'shrink': .5})

        sns.set_style(self.style)  # Set back to original style
        self._plot(title="Feature correlation matrix" if title is None else title,
                   filename=filename,
                   display=display)

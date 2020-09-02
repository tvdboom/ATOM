# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the plotting classes.

"""

# Standard packages
import inspect
import numpy as np
import pandas as pd
from itertools import chain
from typeguard import typechecked
from joblib import Parallel, delayed
from scipy.stats.mstats import mquantiles
from typing import Optional, Union, Sequence, Tuple

# Plotting packages
import shap
from matplotlib import transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import ConnectionStyle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# Sklearn
from sklearn.utils import _safe_indexing
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import SCORERS, roc_curve, precision_recall_curve

# Own modules
from atom.basetransformer import BaseTransformer
from .utils import (
    CAL, METRIC_ACRONYMS, lst, check_is_fitted, get_best_score,
    get_model_name, partial_dependence, composed, crash, plot_from_model
)


# Classes =================================================================== >>

class BasePlotter(object):
    """Parent class for all plotting methods.

    This base class defines the plot properties that can be changed in order
    to customize the plot's aesthetics. Making the variable mutable ensures
    that it is changed for all classes at the same time.

    """

    _aesthetics = dict(
        style='darkgrid',  # Seaborn plotting style
        palette='GnBu_d',  # Seaborn color palette
        title_fontsize=20,  # Fontsize for titles
        label_fontsize=16,  # Fontsize for labels and legends
        tick_fontsize=12   # Fontsize for ticks
    )
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
            raise ValueError("Invalid value for the style parameter, got "
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
            raise ValueError("Invalid value for the title_fontsize parameter. "
                             f"Value should be >=0, got {title_fontsize}.")
        self._aesthetics['title_fontsize'] = title_fontsize

    @property
    def label_fontsize(self):
        return self._aesthetics['label_fontsize']

    @label_fontsize.setter
    @typechecked
    def label_fontsize(self, label_fontsize: int):
        if label_fontsize <= 0:
            raise ValueError("Invalid value for the label_fontsize parameter. "
                             f"Value should be >=0, got {label_fontsize}.")
        self._aesthetics['label_fontsize'] = label_fontsize

    @property
    def tick_fontsize(self):
        return self._aesthetics['tick_fontsize']

    @tick_fontsize.setter
    @typechecked
    def tick_fontsize(self, tick_fontsize: int):
        if tick_fontsize <= 0:
            raise ValueError("Invalid value for the tick_fontsize parameter. "
                             f"Value should be >=0, got {tick_fontsize}.")
        self._aesthetics['tick_fontsize'] = tick_fontsize

    # Methods =============================================================== >>

    def _check_models(self, models, max_one=False):
        """Check the provided input models.

        Make sure all names are correct and that every model is in the trainer's
        pipeline. If models is None, returns all models in the pipeline.

        Parameters
        ----------
        models: str or sequence
            Models provided by the plot's parameter.

        max_one: bool
            Whether one or multiple models are allowed. If True, return the model
            instead of a list.

        """
        if models is None:
            models = self.models
        elif isinstance(models, str):
            models = [get_model_name(models)]
        else:
            models = [get_model_name(m) for m in models]

        # Check that all models are in the pipeline
        for model in models:
            if model not in self.models:
                raise ValueError(f"Model {model} not found in the pipeline!")

        model_subclasses = [m for m in self.models_ if m.name in models]

        if max_one and len(model_subclasses) > 1:
            raise ValueError("This plot method allows only one model at a time!")

        return model_subclasses[0] if max_one else model_subclasses

    def _check_metric(self, metric):
        """Check the provided input metric.

        Parameters
        ----------
        metric: int or str
            Metric provided by the metric parameter.

        """
        def _raise():
            raise ValueError(
                "Invalid value for the metric parameter. Value should be the index"
                f" or name of a metric used to run the pipeline, got {metric}.")

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

    @staticmethod
    def _check_set(dataset):
        """Check the provided input metric.

        Parameters
        ----------
        dataset: str
            Metric provided by the metric parameter.

        """
        if dataset.lower() == 'both':
            return ['train', 'test']
        elif dataset.lower() in ('train', 'test'):
            return [dataset.lower()]
        else:
            raise ValueError("Invalid value for the dataset parameter. "
                             "Choose between 'train', 'test' or 'both'.")

    def _get_explainer(self, m):
        """Get the SHAP explainer for a specific model.

        Parameters
        ----------
        m: class
         Model subclass.

        """
        if m.type == 'tree':
            return shap.TreeExplainer(
                model=m.model, feature_perturbation='tree_path_dependent')
        elif m.type == 'linear':
            return shap.LinearExplainer(m.model, data=self.X_train)
        else:
            if len(self.X_train) <= 100:
                data = self.X_train
            else:
                data = shap.kmeans(self.X_train, 10)
            return shap.KernelExplainer(m.model.predict, data=data)

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
                - tight_layout: Tight layout. Default is True.
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
            plt.ylim(kwargs['ylim'])
        plt.xticks(fontsize=self.tick_fontsize)
        plt.yticks(fontsize=self.tick_fontsize)
        if kwargs.get('tight_layout', True):
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
        if not hasattr(self, 'pca') or not self.pca:
            raise PermissionError("The plot_pca method is only available "
                                  "if you applied PCA on the data!")

        n = self.pca.n_components_  # Number of chosen components
        var = np.array(self.pca.explained_variance_ratio_[:n])
        var_all = np.array(self.pca.explained_variance_ratio_)

        fig, ax = plt.subplots(figsize=figsize)
        plt.scatter(
            x=self.pca.n_components_ - 1,
            y=var.sum(),
            marker='*',
            s=130,
            c='blue',
            edgecolors='b',
            zorder=3,
            label=f"Total variance retained: {round(var.sum(), 3)}"
        )
        plt.plot(range(self.pca.n_features_), np.cumsum(var_all), marker='o')
        plt.axhline(var.sum(), ls='--', color='k')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only int ticks

        self._plot(
            title="PCA explained variances" if not title else title,
            legend='lower right',
            xlabel='First N principal components',
            ylabel='Cumulative variance ratio',
            filename=filename,
            display=display
        )

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
        if not hasattr(self, 'pca') or not self.pca:
            raise PermissionError("The plot_components method is only available "
                                  "if you applied PCA on the data!")

        # Set parameters
        if show is None:
            show = self.pca.n_components_
        elif show > self.pca.n_features_:
            show = self.pca.n_features_
        elif show < 1:
            raise ValueError("Invalid value for the show parameter. "
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

        self._plot(
            title="Explained variance per component" if not title else title,
            legend='lower right',
            xlabel='Explained variance ratio',
            ylabel='Components',
            filename=filename,
            display=display
        )

    @composed(crash, typechecked)
    def plot_rfecv(self,
                   title: Optional[str] = None,
                   figsize: Optional[Tuple[int, int]] = (10, 6),
                   filename: Optional[str] = None,
                   display: bool = True):
        """Plot the RFECV results.

        Plot the scores obtained by the estimator fitted on every subset of
        the dataset. Only if RFECV was applied on the dataset through the
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
        if not hasattr(self, 'rfecv') or not self.rfecv:
            raise PermissionError("The plot_rfecv method is only available "
                                  "if you applied RFECV on the data!")

        try:  # Define the y-label for the plot
            ylabel = self.rfecv.get_params()['scoring'].name
        except AttributeError:
            ylabel = 'score'
            if self.rfecv.get_params()['scoring']:
                ylabel = str(self.rfecv.get_params()['scoring'])

        fig, ax = plt.subplots(figsize=figsize)
        n_features = self.rfecv.get_params()['min_features_to_select']
        xline = range(n_features, n_features + len(self.rfecv.grid_scores_))
        plt.plot(xline, self.rfecv.grid_scores_)

        # Set limits before drawing the intersected lines
        xlim = (n_features-0.5, n_features+len(self.rfecv.grid_scores_)-0.5)
        ylim = ax.get_ylim()

        # Draw intersected lines
        x = xline[np.argmax(self.rfecv.grid_scores_)]
        y = max(self.rfecv.grid_scores_)
        ax.vlines(x, -1e4, y, ls='--', color='k', alpha=0.7)
        ax.hlines(y, -1, x, ls='--', color='k', alpha=0.7,
                  label=f'Features: {x}   {ylabel}: {round(y, 3)}')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only int ticks

        self._plot(
            title="RFE cross-validation scores" if not title else title,
            legend='best',
            xlabel='Number of features',
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            filename=filename,
            display=display
        )


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
            Index or name of the metric to plot. Only for multi-metric runs.

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
        if all([not m.metric_bagging for m in models]):
            raise PermissionError("The plot_bagging method is only available "
                                  "you run the pipeline using bagging>0!")

        results, names = [], []
        for m in models:
            if m.metric_bagging:
                if len(self.metric_) > 1:  # Is list of lists
                    results.append(lst(m.metric_bagging)[metric])
                else:  # Is single list
                    results.append(m.metric_bagging)
                names.append(m.name)

        if figsize is None:  # Default figsize depends on number of models
            figsize = (int(8 + len(names)/2), 6)

        fig, ax = plt.subplots(figsize=figsize)
        plt.boxplot(results)
        ax.set_xticklabels(names)

        self._plot(
            title="Bagging results" if not title else title,
            xlabel='Model',
            ylabel=self.metric_[metric].name,
            filename=filename,
            display=display
        )

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
            Index or name of the metric to plot. Only for multi-metric runs.

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
                "The plot_bo method is only available for models that "
                "ran the bayesian optimization hyperparameter tuning!")

        plt.subplots(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        for m in models:
            y = m.bo['score'].apply(lambda value: lst(value)[metric])
            if len(models) == 1:
                label = f"Score={round(lst(m.metric_bo)[metric], 3)}"
            else:
                label = f"{m.name} (Score={round(lst(m.metric_bo)[metric], 3)})"

            # Draw bullets onm all markers except the maximum
            markers = [i for i in range(len(m.bo))]
            markers.remove(int(np.argmax(y)))
            ax1.plot(range(1, len(y)+1), y, '-o', markevery=markers, label=label)
            ax2.plot(range(2, len(y)+1), np.abs(np.diff(y)), '-o')
            ax1.scatter(np.argmax(y)+1, max(y), zorder=10, s=100, marker='*')

        title = "Bayesian optimization scoring" if not title else title
        ax1.set_title(title, fontsize=self.title_fontsize, pad=12)
        ax1.legend(loc='lower right', fontsize=self.label_fontsize)
        ax2.set_title(
            label="Distance between last consecutive iterations",
            fontsize=self.title_fontsize
        )
        ax2.set_xlabel('Iteration', fontsize=self.label_fontsize, labelpad=12)
        ax1.set_ylabel(
            ylabel=self.metric_[metric].name,
            fontsize=self.label_fontsize,
            labelpad=12
        )
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

         The metric is provided by the model's package and is different for every
         model and every task. Only for models that allow in-training evaluation
         (XGB, LGB, CastB). Only allows plotting one model at a time because of
         the different evaluation metric for each.

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
        m = self._check_models(models, max_one=True)

        # Check that the model had in-training evaluation
        if not hasattr(m, 'evals'):
            raise AttributeError(
                "The plot_evals method is only available for models "
                f"that allow in-training evaluation, got {m.name}.")

        plt.subplots(figsize=figsize)
        plt.plot(range(len(m.evals['train'])), m.evals['train'], lw=2, label='train')
        plt.plot(range(len(m.evals['test'])), m.evals['test'], lw=2, label='test')

        self._plot(
            title="Evaluation curves" if not title else title,
            legend='best',
            xlabel=m.get_domain()[0].name,  # First param is always the iter
            ylabel=m.evals['metric'],
            filename=filename,
            display=display
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_roc(self,
                 models: Union[None, str, Sequence[str]] = None,
                 dataset: str = 'test',
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

        dataset: str, optional (default='test')
            Data set on which to calculate the metric. Options are 'train',
            'test' or 'both'.

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
        dataset = self._check_set(dataset)

        if not self.task.startswith('bin'):
            raise PermissionError("The plot_roc method is only available "
                                  "for binary classification tasks!")

        plt.subplots(figsize=figsize)
        for m in models:
            for set_ in dataset:
                # Get False (True) Positive Rate as arrays
                fpr, tpr, _ = roc_curve(getattr(m, f'y_{set_}'),
                                        getattr(m, f'predict_proba_{set_}')[:, 1])

                if len(models) == 1:
                    l_set = f'{set_} - ' if len(dataset) > 1 else ''
                    label = f"{l_set}AUC={m.scoring('roc_auc', set_):.3f}"
                else:
                    l_set = f' - {set_}' if len(dataset) > 1 else ''
                    label = f"{m.name}{l_set} (AUC={m.scoring('roc_auc', set_):.3f})"
                plt.plot(fpr, tpr, lw=2, label=label)

        plt.plot([0, 1], [0, 1], lw=2, color='black', alpha=0.7, linestyle='--')

        self._plot(
            title="ROC curve" if not title else title,
            legend='lower right',
            xlabel='FPR',
            ylabel='TPR',
            filename=filename,
            display=display
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_prc(self,
                 models: Union[None, str, Sequence[str]] = None,
                 dataset: str = 'test',
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

        dataset: str, optional (default='test')
            Data set on which to calculate the metric. Options are 'train',
            'test' or 'both'.

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
        dataset = self._check_set(dataset)

        if not self.task.startswith('binary'):
            raise PermissionError("The plot_prc method is only available for "
                                  "binary classification tasks!")

        plt.subplots(figsize=figsize)
        for m in models:
            for set_ in dataset:
                # Get precision-recall pairs for different probability thresholds
                precision, recall, _ = \
                    precision_recall_curve(getattr(m, f'y_{set_}'),
                                           getattr(m, f'predict_proba_{set_}')[:, 1])

                if len(models) == 1:
                    l_set = f'{set_} - ' if len(dataset) > 1 else ''
                    label = f"{l_set}AP={m.scoring('ap', set_):.3f}"
                else:
                    l_set = f' - {set_}' if len(dataset) > 1 else ''
                    label = f"{m.name}{l_set} (AP={m.scoring('ap', set_):.3f})"
                plt.plot(recall, precision, lw=2, label=label)

        self._plot(
            title="Precision-recall curve" if not title else title,
            legend='lower left',
            xlabel='Recall',
            ylabel='Precision',
            filename=filename,
            display=display
        )

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

        show: int or None, optional (default=None)
            Number of best features to show in the plot. None to show all.

        n_repeats: int, optional (default=10)
            Number of times to permute each feature.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)
        if show is None or show > self.X.shape[1]:
            show = self.X.shape[1]
        elif show <= 0:
            raise ValueError("Invalid value for the show parameter."
                             f"Value should be >0, got {show}.")
        if n_repeats <= 0:
            raise ValueError("Invalid value for the n_repeats parameter."
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
                m._repeats = -np.inf
            if m.name not in self.permutations or m._repeats != n_repeats:
                m._repeats = n_repeats
                # Permutation importances returns Bunch object from sklearn
                self.permutations[m.name] = permutation_importance(
                    estimator=m.model,
                    X=m.X_test,
                    y=m.y_test,
                    scoring=self.metric_[0],
                    n_repeats=n_repeats,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state
                )

            # Append data to the dataframe
            for i, feature in enumerate(self.X.columns):
                for score in self.permutations[m.name].importances[i, :]:
                    df = df.append({
                        'features': feature,
                        'score': score,
                        'model': m.name
                    }, ignore_index=True)

        # Get the column names sorted by mean of score
        get_idx = df.groupby('features', as_index=False)['score'].mean()
        get_idx.sort_values('score', ascending=False, inplace=True)
        column_order = get_idx.features.values[:show]

        # Save the feature order in the best_features attr
        self.best_features = list(column_order)

        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(
            x='score',
            y='features',
            hue='model',
            data=df,
            order=column_order,
            width=0.75 if len(models) > 1 else 0.6
        )

        # Remove seaborn's legend title
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])

        self._plot(
            title=title,
            legend='lower right',
            xlabel='Score',
            ylabel='Features',
            filename=filename,
            display=display
        )

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

        show: int or None, optional (default=None)
            Number of best features to show in the plot. None to show all.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to `show` param.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)
        if show is None or show > self.X.shape[1]:
            show = self.X.shape[1]
        elif show <= 0:
            raise ValueError("Invalid value for the show parameter."
                             f"Value should be >0, got {show}.")

        # Create dataframe with columns as indices to plot with barh
        df = pd.DataFrame(index=self.X.columns)

        for m in models:
            if m.type != 'tree':
                raise PermissionError(
                    "The plot_feature_importance method is only available for "
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

        # Save the feature order in the best_features attr
        self.best_features = list(df.index.values[::-1])

        if figsize is None:  # Default figsize depends on features shown
            figsize = (10, int(4 + show/2))

        # Plot figure
        ax = df.plot.barh(figsize=figsize, width=0.75 if len(models) > 1 else 0.6)
        if len(models) == 1:
            for i, v in enumerate(df[df.columns[0]]):
                ax.text(v + .01, i - .08, f'{v:.2f}', fontsize=self.tick_fontsize)

        self._plot(
            title="Normalized feature importance" if not title else title,
            legend='lower right',
            xlim=(0, 1.07),
            xlabel='Score',
            ylabel='Features',
            filename=filename,
            display=display
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_partial_dependence(self,
                                models: Union[None, str, Sequence[str]] = None,
                                features: Optional[Union[int, str, Sequence]] = None,
                                target: Union[int, str] = 1,
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 6),
                                filename: Optional[str] = None,
                                display: bool = True):
        """Plot the partial dependence of features.

        The partial dependence of a feature (or a set of features) corresponds to the
        average response of the model for each possible value of the feature.
        Two-way partial dependence plots are plotted as contour plots (only allowed
        for single model plots). The deciles of the feature values will be shown
        with tick marks on the x-axes for one-way plots, and on both axes for two-way
        plots.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        features: int, str, sequence or None, optional (default=None)
            Features or feature pairs (name or index) to get the partial
            dependence from. Maximum of 3 allowed. If None, it uses the top
            3 features if `best_features` is defined (see plot_feature_importance
            or plot_permutation_importance), else it uses the first 3 features in
            the dataset.

        target: int or str, optional (default=1)
            Category to look at in the target class as index or name.
            Only for multi-class classification.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        def convert_feature(feature):
            if isinstance(feature, str):
                try:
                    feature = list(self.columns).index(feature)
                except ValueError:
                    raise ValueError("Invalid value for the features parameter. "
                                     f"Unknown column: {feature}.")
            elif feature > self.X.shape[1] - 1:  # -1 because of index 0
                raise ValueError(
                    "Invalid value for the features parameter. Dataset "
                    f"has {self.X.shape[1]} features, got index {feature}.")
            return int(feature)

        check_is_fitted(self, 'results')
        models = self._check_models(models)

        # Prepare features parameter
        if not features:
            if not hasattr(self, 'best_features'):
                features = [0, 1, 2]
            else:
                # If best_features exists, select the 3 best ones
                features = [idx for idx, column in enumerate(self.columns)
                            if column in self.best_features[:3]]
        elif not isinstance(features, (list, tuple)):
            features = [features]

        if len(features) > 3:
            raise ValueError("Invalid value for the features parameter. "
                             f"Maximum 3 allowed, got {len(features)}.")

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
                        f"should be single or in pairs, got {fxs}.")
            else:
                raise ValueError(
                    "Invalid value for the features parameter. Feature pairs "
                    f"are invalid when plotting multiple models, got {fxs}.")

        # Prepare target parameter
        if self.task.startswith('multi'):
            target = self.mapping[target] if isinstance(target, str) else target
        else:
            target = 0  # For binary and regression, target is always 0

        fig, ax = plt.subplots(ncols=len(cols), figsize=figsize)
        if not isinstance(ax, np.ndarray):
            ax = [ax]  # Make iterator in case there is only 1 feature

        for m in models:
            # Compute averaged predictions
            pd_results = Parallel(n_jobs=self.n_jobs)(
                delayed(partial_dependence)(m.model, self.X_test, col)
                for col in cols)

            # Get global min and max average predictions of PD grouped by plot type
            pdp_lim = {}
            for pred, values in pd_results:
                min_pd, max_pd = pred[target].min(), pred[target].max()
                old_min, old_max = pdp_lim.get(len(values), (min_pd, max_pd))
                pdp_lim[len(values)] = (min(min_pd, old_min), max(max_pd, old_max))

            # Create contour levels for two-way plots
            if 2 in pdp_lim:
                z_lvl = np.linspace(pdp_lim[2][0], pdp_lim[2][1] + 1e-9, num=8)

            deciles = {}
            for fx in chain.from_iterable(cols):
                if fx not in deciles:
                    X_col = _safe_indexing(self.X_test, fx, axis=1)
                    deciles[fx] = mquantiles(X_col, prob=np.arange(0.1, 1.0, 0.1))

            for axi, fx, (pred, values) in zip(ax, cols, pd_results):
                if len(values) == 1:
                    axi.plot(values[0], pred[target].ravel(), lw=2, label=m.name)
                else:
                    # Draw contour plot
                    XX, YY = np.meshgrid(values[0], values[1])
                    Z = pred[target].T
                    CS = axi.contour(XX, YY, Z, levels=z_lvl, linewidths=0.5)
                    axi.contourf(XX, YY, Z, levels=z_lvl,
                                 vmax=z_lvl[-1], vmin=z_lvl[0], alpha=0.75)
                    axi.clabel(CS, fmt='%2.2f', colors='k', fontsize=10, inline=True)

                trans = transforms.blended_transform_factory(
                    axi.transData, axi.transAxes)
                axi.vlines(deciles[fx[0]], 0, 0.05, transform=trans, color='k')

                # Set xlabel if it is not already set
                axi.tick_params(labelsize=self.tick_fontsize)
                axi.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
                axi.set_xlabel(
                    xlabel=self.columns[fx[0]],
                    fontsize=self.label_fontsize,
                    labelpad=10
                )

                if len(values) != 1:
                    trans = transforms.blended_transform_factory(
                        axi.transAxes, axi.transData)
                    axi.hlines(deciles[fx[1]], 0, 0.05, transform=trans, color='k')
                    axi.set_ylabel(
                        ylabel=self.columns[fx[1]],
                        fontsize=self.label_fontsize,
                        labelpad=12
                    )

        # Place y-label on first non-contour plot
        for axi in ax:
            if not axi.get_ylabel():
                axi.set_ylabel('Score', fontsize=self.label_fontsize, labelpad=12)
                break

        title = 'Partial dependence plot' if not title else title
        plt.suptitle(title, fontsize=self.title_fontsize, y=0.97)
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        self._plot(
            legend='best' if len(models) > 1 else False,
            tight_layout=False,
            filename=filename,
            display=display
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_errors(self,
                    models: Union[None, str, Sequence[str]] = None,
                    dataset: str = 'test',
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    filename: Optional[str] = None,
                    display: bool = True):
        """Plot a model's prediction errors.

        Plot the actual targets from the test set against the predicted values
        generated by the regressor. A linear fit is made on the data. The gray,
        intersected line shows the identity line. This pot can be useful to detect
        noise or heteroscedasticity along a range of the target domain. Only for
        regression tasks.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default='test')
            Data set on which to calculate the metric. Options are 'train',
            'test' or 'both'.

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
        dataset = self._check_set(dataset)

        if not self.task.startswith('reg'):
            raise PermissionError(
                "The plot_errors method is only available for regression tasks!")

        fig, ax = plt.subplots(figsize=figsize)
        for m in models:
            for set_ in dataset:
                if len(models) == 1:
                    l_set = f'{set_} - ' if len(dataset) > 1 else ''
                    label = f"{l_set}R$^2$={m.scoring('r2', set_):.3f}"
                else:
                    l_set = f' - {set_}' if len(dataset) > 1 else ''
                    label = f"{m.name}{l_set} (R$^2$={m.scoring('r2', set_):.3f})"

                plt.scatter(
                    x=getattr(self, f'y_{set_}'),
                    y=getattr(m, f'predict_{set_}'),
                    alpha=0.8,
                    label=label
                )

                # Fit the points using linear regression
                from .models import OrdinaryLeastSquares
                model = OrdinaryLeastSquares(self).get_model()
                model.fit(np.array(getattr(self, f'y_{set_}')).reshape(-1, 1),
                          getattr(m, f'predict_{set_}'))

                # Draw the fit
                x = np.linspace(*ax.get_xlim(), 100)
                plt.plot(x, model.predict(x[:, np.newaxis]), lw=2, alpha=1)

        # Get limits before drawing the identity line
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        # Draw identity line
        plt.plot(xlim, ylim, ls='--', lw=1, color='k', alpha=0.7)

        self._plot(
            title="Prediction errors" if not title else title,
            legend='upper left',
            xlabel='True value',
            ylabel='Predicted value',
            xlim=xlim,
            ylim=ylim,
            filename=filename,
            display=display
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_residuals(self,
                       models: Union[None, str, Sequence[str]] = None,
                       dataset: str = 'test',
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       filename: Optional[str] = None,
                       display: bool = True):
        """Residual plot of a model.

        The plot shows the residuals (difference between the predicted and the
        true value) on the vertical axis and the independent variable on the
        horizontal axis. The gray, intersected line shows the identity line. This
        plot can be useful to analyze the variance of the error of the regressor.
        If the points are randomly dispersed around the horizontal axis, a linear
        regression model is appropriate for the data; otherwise, a non-linear model
        is more appropriate. Only for regression tasks.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default='test')
            Data set on which to calculate the metric. Options are 'train',
            'test' or 'both'.

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
        dataset = self._check_set(dataset)

        if not self.task.startswith('reg'):
            raise PermissionError(
                "The plot_residuals method is only available for regression tasks!")

        # Create figure with two axes
        fig, ax = plt.subplots(figsize=figsize)
        divider = make_axes_locatable(ax)
        hax = divider.append_axes("right", size=1.5, pad=0.1)

        for m in models:
            for set_ in dataset:
                if len(models) == 1:
                    l_set = f'{set_} - ' if len(dataset) > 1 else ''
                    label = f"{l_set}R$^2$={m.scoring('r2', set_):.3f}"
                else:
                    l_set = f' - {set_}' if len(dataset) > 1 else ''
                    label = f"{m.name}{l_set} (R$^2$={m.scoring('r2', set_):.3f})"

            res = np.subtract(
                getattr(m, f'predict_{set_}'), getattr(self, f'y_{set_}'))
            ax.scatter(getattr(m, f'predict_{set_}'), res, alpha=0.7, label=label)
            hax.hist(res, orientation="horizontal", histtype='step', linewidth=1.2)

        # Get limits before drawing the identity line
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        # Draw identity line
        ax.hlines(0, *xlim, ls='--', lw=1, color='k', alpha=0.8)

        # Set parameters
        ax.legend(loc='best', fontsize=self.label_fontsize)
        ax.set_ylabel('Residuals', fontsize=self.label_fontsize, labelpad=12)
        ax.set_xlabel('True value', fontsize=self.label_fontsize, labelpad=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=self.tick_fontsize)
        hax.set_xlabel('Distribution', fontsize=self.label_fontsize, labelpad=12)
        hax.set_yticklabels([])

        title = 'Residuals plot' if not title else title
        plt.suptitle(title, fontsize=self.title_fontsize, y=0.97)
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        self._plot(tight_layout=False, filename=filename, display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_confusion_matrix(self,
                              models: Union[None, str, Sequence[str]] = None,
                              dataset: str = 'test',
                              normalize: bool = False,
                              title: Optional[str] = None,
                              figsize: Optional[Tuple[int, int]] = None,
                              filename: Optional[str] = None,
                              display: bool = True):
        """Plot the confusion matrix.

        Only for classification tasks.
        For 1 model: plot the confusion matrix in a heatmap.
        For >1 models: compare TP, FP, FN and TN in a barplot.
                       Not implemented for multiclass classification.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default='test')
            Data set on which to calculate the metric. Options are 'train' or 'test'.

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
            raise PermissionError("The plot_confusion_matrix_method is only "
                                  "available for classification tasks!")

        if self.task.startswith('multi') and len(models) > 1:
            raise NotImplementedError(
                "The plot_confusion_matrix method does not support the comparison"
                " of various models for multiclass classification tasks.")

        # Create dataframe to plot with barh if len(models) > 1
        df = pd.DataFrame(index=['True negatives', 'False positives',
                                 'False negatives', 'True positives'])
        # Define title
        if not title and normalize:
            title = "Normalized confusion matrix"
        elif not title:
            title = "Confusion matrix"

        for m in models:
            cm = m.scoring('confusion_matrix', dataset.lower())
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            if len(models) == 1:  # Create matrix heatmap
                if hasattr(self, 'mapping'):
                    ticks = [v for v in self.mapping.keys()]
                else:
                    ticks = list(m.y_test.unique())

                fig, ax = plt.subplots(figsize=(8, 8) if not figsize else figsize)
                im = ax.imshow(
                    X=cm,
                    interpolation='nearest',
                    cmap=plt.get_cmap('Blues')
                )

                # Create an axes on the right side of ax. The under of cax will
                # be 5% of ax and the padding between cax and ax will be fixed
                # at 0.3 inch.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.3)
                cbar = ax.figure.colorbar(im, cax=cax)
                ax.set(
                    xticks=np.arange(cm.shape[1]),
                    yticks=np.arange(cm.shape[0]),
                    xticklabels=ticks,
                    yticklabels=ticks
                )

                # Loop over data dimensions and create text annotations
                fmt = '.2f' if normalize else 'd'
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            x=j,
                            y=i,
                            s=format(cm[i, j], fmt),
                            ha='center',
                            va='center',
                            fontsize=self.tick_fontsize,
                            color='w' if cm[i, j] > cm.max() / 2. else 'k'
                        )

                ax.set_title(title, fontsize=self.title_fontsize, pad=12)
                ax.set_xlabel(
                    xlabel='Predicted label',
                    fontsize=self.label_fontsize,
                    labelpad=12
                )
                ax.set_ylabel(
                    ylabel='True label',
                    fontsize=self.label_fontsize,
                    labelpad=12
                )
                cbar.set_label(
                    label='Count',
                    fontsize=self.label_fontsize,
                    labelpad=15,
                    rotation=270
                )
                cbar.ax.tick_params(labelsize=self.tick_fontsize)
                ax.grid(False)

            else:  # Create barplot
                df[m.name] = cm.ravel()

        if len(models) > 1:
            df.plot.barh(figsize=(10, 6) if not figsize else figsize, width=0.6)
            self._plot(
                title=title,
                legend='best',
                xlabel='Count',
                filename=filename,
                display=display
            )
        else:
            self._plot(filename=filename, display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_threshold(self,
                       models: Union[None, str, Sequence[str]] = None,
                       metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                       dataset: str = 'test',
                       steps: int = 100,
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       filename: Optional[str] = None,
                       display: bool = True):
        """Plot performance metric(s) against multiple threshold values.

        Only for binary classification tasks.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        metric: string, callable, list, tuple or None, optional (default=None)
            Metric(s) to plot. These can be one of the pre-defined sklearn scorers
            as string, a metric function or a sklearn scorer object. If None, the
            metric used to run the pipeline is used.

        dataset: str, optional (default='test')
            Data set on which to calculate the metric. Options are 'train',
            'test' or 'both'.

        steps: int, optional (default=100)
            Number of thresholds measured.

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
        dataset = self._check_set(dataset)

        if not self.task.startswith('bin'):
            raise PermissionError("The plot_threshold method is only "
                                  "available for binary classification tasks!")

        # Check that all models have predict_proba
        for m in models:
            if not hasattr(m.model, 'predict_proba'):
                raise AttributeError(
                    "The plot_probabilities method is only available "
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
                    raise ValueError("Unknown value for the metric parameter, "
                                     f"got {met}. Try one of {list(SCORERS)}.")
                metric_list.append(SCORERS[met]._score_func)
            elif hasattr(met, '_score_func'):  # It is a scorer
                metric_list.append(met._score_func)
            else:  # It is a metric function
                metric_list.append(met)

        plt.subplots(figsize=figsize)
        steps = np.linspace(0, 1, steps)
        for m in models:
            for met in metric_list:  # Create dict of empty arrays
                for set_ in dataset:
                    results = []
                    for step in steps:
                        predictions = (
                                getattr(m, f'predict_proba_{set_}')[:, 1] >= step
                        ).astype(bool)
                        results.append(met(getattr(m, f'y_{set_}'), predictions))

                    if len(models) == 1:
                        l_set = f'{set_} - ' if len(dataset) > 1 else ''
                        label = f"{l_set}{met.__name__}"
                    else:
                        l_set = f' - {set_}' if len(dataset) > 1 else ''
                        label = f"{m.name}{l_set} ({met.__name__})"
                    plt.plot(steps, results, label=label, lw=2)

        self._plot(
            title="Performance against threshold value" if not title else title,
            legend='best',
            xlabel='Threshold',
            ylabel='Score',
            filename=filename,
            display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_probabilities(self,
                           models: Union[None, str, Sequence[str]] = None,
                           dataset: str = 'test',
                           target: Union[int, str] = 1,
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           filename: Optional[str] = None,
                           display: bool = True):
        """Plot the probability of being the target category for every category.

        Only for binary classification tasks.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        dataset: str, optional (default='test')
            Data set on which to calculate the metric. Options are 'train',
            'test' or 'both'.

        target: int or string, optional (default=1)
            Probability of being that category in the target column as index or name.
            Only for multiclass classification.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        # Set parameters
        check_is_fitted(self, 'results')
        models = self._check_models(models)
        dataset = self._check_set(dataset)

        if self.task.startswith('reg'):
            raise PermissionError("The plot_probabilities method is only "
                                  "available for classification tasks!")

        # Check that all models have predict_proba
        for m in models:
            if not hasattr(m.model, 'predict_proba'):
                raise AttributeError(
                    "The plot_probabilities method is only available "
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
            for set_ in dataset:
                for key, value in self.mapping.items():
                    # Get indices per class
                    idx = np.where(getattr(m, f'y_{set_}') == value)[0]

                    if len(models) == 1:
                        l_set = f'{set_} - ' if len(dataset) > 1 else ''
                        label = f"{l_set}Category:{key}"
                    else:
                        l_set = f' - {set_}' if len(dataset) > 1 else ''
                        label = f"{m.name}{l_set} (Category: {key})"
                    sns.distplot(
                        a=getattr(m, f'predict_proba_{set_}')[idx, target_int],
                        hist=False,
                        kde=True,
                        norm_hist=True,
                        kde_kws={'shade': True},
                        label=label
                    )

        if not title:
            title = f"Predicted probabilities for {m.y.name}={target_str}"
        self._plot(
            title=title,
            legend='best',
            xlabel='Probability',
            ylabel='Count',
            xlim=(0, 1),
            filename=filename,
            display=display
        )

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
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        check_is_fitted(self, 'results')
        models = self._check_models(models)

        if not self.task.startswith('bin'):
            raise PermissionError("The plot_probabilities method is only "
                                  "available for binary classification tasks!")

        # Set parameters
        if n_bins < 5:
            raise ValueError("Invalid value for the n_bins parameter."
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

        title = 'Calibration curve' if not title else title
        ax1.set_title(title, fontsize=self.title_fontsize, pad=12)
        ax1.set_ylabel(
            ylabel="Fraction of positives",
            fontsize=self.label_fontsize,
            labelpad=12
        )
        ax1.set_ylim([-0.05, 1.05])

        ax2.set_xlabel(
            xlabel="Predicted value",
            fontsize=self.label_fontsize,
            labelpad=12
        )
        ax2.set_ylabel("Count", fontsize=self.label_fontsize, labelpad=12)

        # Only draw the legends for more than one model
        if len(models) > 1:
            ax1.legend(loc="lower right", fontsize=self.label_fontsize)
            ax2.legend(loc='best', fontsize=self.label_fontsize, ncol=3)

        self._plot(filename=filename, display=display)

    @composed(crash, plot_from_model, typechecked)
    def plot_gains(self,
                   models: Union[None, str, Sequence[str]] = None,
                   dataset: str = 'test',
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

        dataset: str, optional (default='test')
            Data set on which to calculate the metric. Options are 'train',
            'test' or 'both'.

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
        dataset = self._check_set(dataset)

        if not self.task.startswith('bin'):
            raise PermissionError("The plot_gain method is only available "
                                  "for binary classification tasks!")

        # Check that all models have predict_proba
        for m in models:
            if not hasattr(m.model, 'predict_proba'):
                raise AttributeError(
                    "The plot_probabilities method is only available "
                    f"for models with a predict_proba method, got {m}.")

        plt.subplots(figsize=figsize)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7)
        for m in models:
            for set_ in dataset:
                y_true = (getattr(m, f'y_{set_}') == 1)  # Make y_true a bool vector

                # Get sorted indices and correct for the test set
                sorted_indices = np.argsort(
                    getattr(m, f'predict_proba_{set_}')[:, 1])[::-1]

                if set_ == 'test':
                    sorted_indices = [i + len(m.y_train) for i in sorted_indices]

                # Compute cumulative gains
                gains = np.cumsum(y_true.loc[sorted_indices])/float(np.sum(y_true))

                x = np.arange(start=1, stop=len(y_true) + 1)/float(len(y_true))
                label = ''
                if len(models) > 1:
                    label += m.name
                if len(models) > 1 and len (dataset) > 1:
                    label += ' - '
                if len(dataset) > 1:
                    label += set_
                plt.plot(x, gains, lw=2, label=label)

        self._plot(
            title="Cumulative gains curve" if not title else title,
            legend='lower right',
            xlabel="Fraction of sample",
            ylabel='Gain',
            xlim=(0, 1),
            ylim=(0, 1.02),
            filename=filename,
            display=display
        )

    @composed(crash, plot_from_model, typechecked)
    def plot_lift(self,
                  models: Union[None, str, Sequence[str]] = None,
                  dataset: str = 'test',
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

        dataset: str, optional (default='test')
            Data set on which to calculate the metric. Options are 'train',
            'test' or 'both'.

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
        dataset = self._check_set(dataset)

        if not self.task.startswith('bin'):
            raise PermissionError("The plot_gain method is only available "
                                  "for binary classification tasks!")

        # Check that all models have predict_proba
        for m in models:
            if not hasattr(m.model, 'predict_proba'):
                raise AttributeError(
                    "The plot_probabilities method is only available "
                    f"for models with a predict_proba method, got {m}.")

        plt.subplots(figsize=figsize)
        plt.plot([0, 1], [1, 1], 'k--', lw=2, alpha=0.7)
        for m in models:
            for set_ in dataset:
                y_true = (getattr(m, f'y_{set_}') == 1)  # Make y_true a bool vector

                # Get sorted indices and correct for the test set
                sorted_indices = np.argsort(
                    getattr(m, f'predict_proba_{set_}')[:, 1])[::-1]

                if set_ == 'test':  # Add the training set length to the indices
                    sorted_indices = [i + len(m.y_train) for i in sorted_indices]

                # Compute cumulative gains
                gains = np.cumsum(y_true.loc[sorted_indices])/float(np.sum(y_true))

                if len(models) == 1:
                    l_set = f'{set_} - ' if len(dataset) > 1 else ''
                    label = f"{l_set}Lift={round(m.scoring('lift'), 3)}"
                else:
                    l_set = f' - {set_}' if len(dataset) > 1 else ''
                    label = f"{m.name}{l_set} (Lift={round(m.scoring('lift'), 3)})"
                x = np.arange(start=1, stop=len(y_true) + 1)/float(len(y_true))
                plt.plot(x, gains/x, lw=2, label=label)

        self._plot(
            title="Lift curve" if not title else title,
            legend='upper right',
            xlabel="Fraction of sample",
            ylabel='Lift',
            xlim=(0, 1),
            filename=filename,
            display=display
        )

    # SHAP plots ============================================================ >>

    @composed(crash, plot_from_model, typechecked)
    def force_plot(self,
                   models: Union[None, str, Sequence[str]] = None,
                   index: Optional[Union[int, Sequence]] = None,
                   title: Optional[str] = None,
                   figsize: Tuple[int, int] = (14, 6),
                   filename: Optional[str] = None,
                   display: bool = True,
                   **kwargs):
        """Plot SHAP's force plot.

        Visualize the given SHAP values with an additive force layout. The explainer
        will be chosen automatically based on the model's type. For kernelExplainer,
        the data used to estimate the expected values is the complete training set
        when <100 rows, else we summarize it with a set of 10 weighted K-means, each
        weighted by the number of points they represent.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        index: int, sequence or None, optional (default=None)
            Indices of the rows in the dataset to plot. If tuple (n, m), select
            rows n until m. If None, select all rows in the test set.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(12, 4))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If matplotlib=False, the figure will
            be saved as an html file. If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        **kwargs
            Additional keyword arguments for shap's force_plot.

        """
        check_is_fitted(self, 'results')
        m = self._check_models(models, max_one=True)
        explainer = self._get_explainer(m)

        # Get the indices
        if not index:
            rows = self.X_test
        elif isinstance(index, int):
            if index < 0:
                rows = self.X.iloc[[len(self.X) + index]]
            else:
                rows = self.X.iloc[[index]]
        else:
            rows = self.X.iloc[slice(*index)]

        # The expected value needs to be calculated after the shap values
        shap_values = explainer.shap_values(rows)
        expected_value = explainer.expected_value

        sns.set_style('white')  # Only for this plot
        plot = shap.force_plot(
            base_value=expected_value,
            shap_values=shap_values,
            features=rows,
            figsize=figsize,
            show=False,
            **kwargs
        )

        sns.set_style(self.style)
        if kwargs.get('matplotlib'):
            self._plot(
                title='' if not title else title,
                filename=filename,
                display=display
            )
        else:
            if filename:  # Save to an html file
                fn = filename if filename.endswith('.html') else filename + '.html'
                shap.save_html(fn, plot)
            if display:
                try:  # Render if possible (for notebooks)
                    from IPython.display import display
                    shap.initjs()
                    display(plot)
                except ModuleNotFoundError:
                    pass

    @composed(crash, plot_from_model, typechecked)
    def dependence_plot(self,
                        models: Union[None, str, Sequence[str]] = None,
                        ind: Union[int, str] = 'rank(1)',
                        title: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6),
                        filename: Optional[str] = None,
                        display: bool = True,
                        **kwargs):
        """Plot SHAP's dependence plot.

        Plots the value of the feature on the x-axis and the SHAP value of the same
        feature on the y-axis. This shows how the model depends on the given feature,
        and is like a richer extenstion of the classical partial dependence plots.
        Vertical dispersion of the data points represents interaction effects. Grey
        ticks along the y-axis are data points where the feature's value was NaN.
        The explainer will be chosen automatically based on the model's type.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        ind: int or str, optional (default='rank(1)')
            If this is an int it is the index of the feature to plot. If this is a
            string it is either the name of the feature to plot, or it can have the
            form "rank(int)" to specify the feature with that rank (ordered by mean
            absolute SHAP value over all the samples).

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        **kwargs
            Additional keyword arguments for shap's force_plot.

        """
        check_is_fitted(self, 'results')
        m = self._check_models(models, max_one=True)
        explainer = self._get_explainer(m)

        fig, ax = plt.subplots(figsize=figsize)
        shap.dependence_plot(
            ind=ind,
            shap_values=explainer.shap_values(self.X_test),
            features=self.X_test,
            ax=ax,
            show=False,
            **kwargs
        )

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=self.label_fontsize, labelpad=12)
        self._plot(
            title='Dependence plot' if not title else title,
            filename=filename,
            display=display
        )

    @composed(crash, plot_from_model, typechecked)
    def summary_plot(self,
                     models: Union[None, str, Sequence[str]] = None,
                     show: Optional[int] = None,
                     title: Optional[str] = None,
                     figsize: Tuple[int, int] = (10, 6),
                     filename: Optional[str] = None,
                     display: bool = True,
                     **kwargs):
        """Plot SHAP's summary plot.

        Create a SHAP beeswarm plot, colored by feature values when they are
        provided. The explainer will be chosen automatically based on the model's type.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        show: int or None, optional (default=None)
            Number of features to show in the plot. None to show all.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=(10, 6))
            Figure's size, format as (x, y).

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        **kwargs
            Additional keyword arguments for shap's force_plot.

        """
        check_is_fitted(self, 'results')
        m = self._check_models(models, max_one=True)
        explainer = self._get_explainer(m)

        if show is None or show > self.X.shape[1]:
            show = self.X.shape[1]
        elif show <= 0:
            raise ValueError("Invalid value for the show parameter."
                             f"Value should be >0, got {show}.")

        fig, ax = plt.subplots(figsize=figsize)
        shap.summary_plot(
            shap_values=explainer.shap_values(self.X_test),
            features=self.X_test,
            max_display=show,
            plot_size=figsize,
            show=False,
            **kwargs
        )

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        self._plot(
            title='Summary plot' if not title else title,
            filename=filename,
            display=display
        )

    @composed(crash, plot_from_model, typechecked)
    def decision_plot(self,
                      models: Union[None, str, Sequence[str]] = None,
                      index: Optional[Union[int, Sequence]] = None,
                      title: Optional[str] = None,
                      figsize: Optional[Tuple[int, int]] = None,
                      filename: Optional[str] = None,
                      display: bool = True,
                      **kwargs):
        """Plot SHAP's decision plot.

        Visualize model decisions using cumulative SHAP values. Each plotted line
        explains a single model prediction. If a single prediction is plotted,
        feature values will be printed in the plot (if supplied). If multiple
        predictions are plotted together, feature values will not be printed.
        Plotting too many predictions together will make the plot unintelligible.
        The explainer will be chosen automatically based on the model's type.

        Parameters
        ----------
        models: str, list, tuple or None, optional (default=None)
            Name of the models to plot. If None, all models in the
            pipeline are selected.

        index: int, sequence or None, optional (default=None)
            Indices of the rows in the dataset to plot. If tuple (n, m), select
            rows n until m. If None, select all rows in the test set.

        title: str or None, optional (default=None)
            Plot's title. If None, the title is left empty.

        figsize: tuple, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to the
            number of features.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        **kwargs
            Additional keyword arguments for shap's force_plot.

        """
        check_is_fitted(self, 'results')
        m = self._check_models(models, max_one=True)
        explainer = self._get_explainer(m)

        # Get the indices
        if index is None:
            rows = self.X_test
        elif isinstance(index, int):
            if index < 0:
                rows = self.X.iloc[[len(self.X) + index]]
            else:
                rows = self.X.iloc[[index]]
        else:
            rows = self.X.iloc[slice(*index)]

        if self.task.startswith('multi') and len(rows) > 1:
            raise ValueError("Invalid value for the index parameter. The decision_"
                             "plot method only supports single observations "
                             f"for multiclass classification tasks, got {index}.")

        # The expected value needs to be calculated after the shap values
        shap_values = explainer.shap_values(rows)
        expected_value = explainer.expected_value

        if figsize is None:  # Default figsize depends on features shown
            figsize = (10, int(4 + self.X.shape[1]/4))

        fig, ax = plt.subplots(figsize=figsize)
        if not self.task.startswith('multi'):
            shap.decision_plot(
                base_value=expected_value,
                shap_values=shap_values,
                features=rows,
                auto_size_plot=False,
                show=False,
                **kwargs
            )
        else:
            if not hasattr(self, 'mapping'):
                self.mapping = {str(i): i for i in self.y_test.unique()}

            shap.multioutput_decision_plot(
                base_values=list(expected_value),
                shap_values=shap_values,
                row_index=0,
                features=rows,
                legend_labels=self.mapping,
                auto_size_plot=False,
                legend_location='lower right',
                show=False,
                **kwargs
            )

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_fontsize, labelpad=12)
        self._plot(
            title='Decision plot' if not title else title,
            filename=filename,
            display=display
        )


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
            Index or name of the metric to plot. Only for multi-metric runs.

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

        self._plot(
            title="Successive halving results" if not title else title,
            legend='lower right',
            xlabel='Iteration',
            ylabel=self.metric_[metric].name,
            filename=filename,
            display=display
        )


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
            Index or name of the metric to plot. Only for multi-metric runs.

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

        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))
        self._plot(
            title="Learning curve" if not title else title,
            legend='lower right',
            xlabel='Number of training samples',
            ylabel=self.metric_[metric].name,
            filename=filename,
            display=display
        )


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
        sns.heatmap(
            data=corr,
            mask=mask,
            cmap=cmap,
            vmax=.3,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={'shrink': .5}
        )

        sns.set_style(self.style)  # Set back to original style
        self._plot(
            title="Feature correlation matrix" if not title else title,
            filename=filename,
            display=display
        )

    @composed(crash, typechecked)
    def plot_pipeline(self,
                      show_params: bool = True,
                      title: Optional[str] = None,
                      figsize: Optional[Tuple[int, int]] = None,
                      filename: Optional[str] = None,
                      display: bool = True):
        """Create a diagram showing every estimator in ATOM's pipeline.

        Parameters
        ----------
        show_params: bool, optional (default=True)
            Whether to show the parameters of every estimator in the pipeline.

        title: str or None, optional (default=None)
            Plot's title. If None, the default option is used.

        figsize: tuple or None, optional (default=None)
            Figure's size, format as (x, y). If None, adapts size to
            the length of the pipeline.

        filename: str or None, optional (default=None)
            Name of the file (to save). If None, the figure is not saved.

        display: bool, optional (default=True)
            Whether to render the plot.

        """
        # Calculate figure's limits
        params = []
        ylim = 30
        for estimator in self.pipeline:
            ylim += 15
            if show_params:
                if estimator.__class__.__name__.startswith(('Train', 'Success')):
                    params.append(['models', 'metric', 'n_calls',
                                   'n_random_starts', 'bo_params', 'bagging'])
                else:
                    params.append(
                        [key for key in inspect.getfullargspec(estimator.__init__)[0]
                         if key not in BaseTransformer.attrs + ['self']])
                ylim += len(params[-1]) * 10

        sns.set_style('white')  # Only for this plot
        figsize = (8, int(ylim/30)) if figsize is None else figsize
        fig, ax = plt.subplots(figsize=figsize if figsize is None else figsize)

        # Shared parameters for the blocks
        con = ConnectionStyle("angle", angleA=0, angleB=90, rad=0)
        arrow = dict(arrowstyle='<|-', lw=1, color='k', connectionstyle=con)

        # Draw the main class
        plt.text(
            x=20,
            y=ylim - 20,
            s=self.__class__.__name__,
            ha="center",
            size=self.label_fontsize + 2
        )

        pos_param = ylim - 20
        pos_estimator = pos_param

        for i, estimator in enumerate(self.pipeline):
            plt.annotate(
                s=estimator.__class__.__name__,
                xy=(15, pos_estimator),
                xytext=(30, pos_param - 3 - 15),
                ha="left",
                size=self.label_fontsize,
                arrowprops=arrow
            )

            pos_param -= 15
            pos_estimator = pos_param

            if show_params:
                for j, key in enumerate(params[i]):
                    plt.annotate(
                        s=key + ': ' + str(estimator.get_params()[key]),
                        xy=(32, pos_param - 6 if j == 0 else pos_param + 1),
                        xytext=(40, pos_param - 12),
                        ha='left',
                        size=self.label_fontsize - 4,
                        arrowprops=arrow
                    )

                    pos_param -= 10

        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        sns.set_style(self.style)  # Set back to original style
        self._plot(
            title='' if not title else title,
            xlim=(0, 100),
            ylim=(0, ylim),
            filename=filename,
            display=display
        )

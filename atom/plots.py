# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (self)
Author: tvdboom
Description: Module containing plot functions.

"""


# << ============ Import Packages ============ >>

# Standard packages
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

# Own package modules
from .utils import check_is_fitted


# << ======================== Plots ======================== >>

def plot_correlation(self, title, figsize, filename, display):

    """
    Correlation maxtrix plot of the dataset. Ignores non-numeric columns.

    PARAMETERS
    ----------
    title: string or None
        Plot's title. If None, the default option is used.

    figsize: tuple
        Figure's size, format as (x, y).

    filename: string or None
        Name of the file (to save). If None, the figure is not saved.

    display: bool
        Wether to render the plot.

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
    fig, ax = plt.subplots(figsize=figsize)

    # Draw the heatmap with the mask and correct aspect ratio
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={'shrink': .5})

    title = "Feature correlation matrix" if title is None else title
    plt.title(title, fontsize=self.title_fs, pad=12)
    fig.tight_layout()
    sns.set_style(self.style)  # Set back to original style
    if filename is not None:
        plt.savefig(filename)
    plt.show() if display else plt.close()


def plot_PCA(self, show, title, figsize, filename, display):

    """
    Plot the explained variance ratio of the components. Only if PCA
    was applied on the dataset through the feature_selection method.

    Parameters
    ----------
    show: int or None
        Number of components to show. If None, all are plotted.

    title: string or None
        Plot's title. If None, the default option is used.

    figsize: tuple
        Figure's size, format as (x, y).

    filename: string or None
        Name of the file (to save). If None, the figure is not saved.

    display: bool
        Wether to render the plot.

    """

    if not hasattr(self, 'PCA'):
        raise AttributeError("This plot is only availbale if you apply " +
                             "PCA on the dataset through the " +
                             "feature_selection method!")

    # Set parameters
    var = np.array(self.PCA.explained_variance_ratio_)
    if show is None or show > len(var):
        show = len(var)
    elif show < 1:
        raise ValueError("Invalid value for the show parameter." +
                         f"Value should be >0, got {show}.")
    if figsize is None:  # Default figsize depends on features shown
        figsize = (10, int(4 + show/2))

    scr = pd.Series(var, index=self.X.columns).nlargest(show).sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    scr.plot.barh(label=f"Total variance retained: {round(var.sum(), 3)}")
    for i, v in enumerate(scr):
        ax.text(v + 0.005, i - 0.08, f'{v:.3f}', fontsize=self.tick_fs)

    plt.title("Explained variance ratio", fontsize=self.title_fs, pad=12)
    plt.legend(loc='lower right', fontsize=self.label_fs)
    plt.xlabel('Variance ratio', fontsize=self.label_fs, labelpad=12)
    plt.ylabel('Components', fontsize=self.label_fs, labelpad=12)
    plt.xticks(fontsize=self.tick_fs)
    plt.yticks(fontsize=self.tick_fs)
    plt.xlim(0, max(scr) + 0.1 * max(scr))  # Make extra space for numbers
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show() if display else plt.close()


def plot_bagging(self, models, title, figsize, filename, display):

    """
    Plot a boxplot of the bagging's results.

    Parameters
    ----------
    models: string, list, tuple or None
        Name of the models to plot. If None, all the models in the
        pipeline are selected. Note that if successive halving=True only
        the last model is saved, so avoid plotting models from different
        iterations together.

    title: string or None
        Plot's title. If None, the default option is used.

    figsize: tuple
        Figure's size, format as (x, y).

    filename: string or None
        Name of the file (to save). If None, the figure is not saved.

    display: bool
        Wether to render the plot.

    """

    check_is_fitted(self._is_fitted)
    if not self.bagging:
        raise AttributeError("You need to run the pipeline using bagging" +
                             " before calling the plot_bagging method!")

    if models is None:
        models = [self.winner.name] if self.successive_halving else self.models
    elif isinstance(models, str):
        models = [models]

    results, names = [], []
    for model in models:
        if hasattr(self, model.lower()):
            results.append(getattr(self, model.lower()).bagging_scores)
            names.append(getattr(self, model.lower()).name)
        else:
            raise ValueError(f"Model {model} not found in pipeline!")

    if figsize is None:  # Default figsize depends on number of models
        figsize = (int(8 + len(names)/2), 6)

    fig, ax = plt.subplots(figsize=figsize)
    plt.boxplot(results)

    title = 'Bagging results' if title is None else title
    plt.title(title, fontsize=self.title_fs, pad=12)
    plt.xlabel('Model', fontsize=self.label_fs, labelpad=12)
    plt.ylabel(self.metric.name,
               fontsize=self.label_fs,
               labelpad=12)
    ax.set_xticklabels(names)
    plt.xticks(fontsize=self.tick_fs)
    plt.yticks(fontsize=self.tick_fs)
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show() if display else plt.close()


def plot_successive_halving(self, models, title, figsize, filename, display):

    """
    Plot of the models' scores per iteration of the successive halving.

    Parameters
    ----------
    models: string, list, tuple or None
        Name of the models to plot. If None, all the models in the
        pipeline are selected.

    title: string or None
        Plot's title. If None, the default option is used.

    figsize: tuple
        Figure's size, format as (x, y).

    filename: string or None
        Name of the file (to save). If None, the figure is not saved.

    display: bool
        Wether to render the plot.

    """

    check_is_fitted(self._is_fitted)
    if not self.successive_halving:
        raise AttributeError("You need to run the pipeline using a " +
                             "successive halving approach before " +
                             "calling the plot_successive_halving method!")

    if models is None:
        models = self.results[0].model  # List of models in first iteration
    elif isinstance(models, str):
        models = [models]

    col = 'score_test' if self.bagging is None else 'bagging_mean'
    names = []
    liny = [[] for _ in models]
    for n, model in enumerate(models):
        if hasattr(self, model.lower()):  # If model in pipeline
            names.append(getattr(self, model.lower()).name)
            for m, df in enumerate(self.results):
                if names[-1] in df.model.values:
                    idx = np.where(names[-1] == df.model.values)[0]
                    liny[n].append(df[col].iloc[idx].values[0])
                else:
                    liny[n].append(np.NaN)
        else:
            raise ValueError(f"Model {model} not found in pipeline!")

    fig, ax = plt.subplots(figsize=figsize)
    for y, label in zip(liny, names):
        plt.plot(range(len(self.results)), y, lw=2, marker='o', label=label)
    plt.xlim(-0.1, len(self.results)-0.9)

    title = "Successive halving results" if title is None else title
    plt.title(title, fontsize=self.title_fs, pad=12)
    plt.legend(frameon=False, fontsize=self.label_fs)
    plt.xlabel('Iteration', fontsize=self.label_fs, labelpad=12)
    plt.ylabel(self.metric.name,
               fontsize=self.label_fs,
               labelpad=12)
    ax.set_xticks(range(len(self.results)))
    plt.xticks(fontsize=self.tick_fs)
    plt.yticks(fontsize=self.tick_fs)
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show() if display else plt.close()


def plot_ROC(self, models, title, figsize, filename, display):

    """
    Plot the Receiver Operating Characteristics curve.
    Only for binary classification tasks.

    Parameters
    ----------
    models: string, list, tuple or None
        Name of the models to plot. If None, all the models in the
        pipeline are selected.

    title: string or None
        Plot's title. If None, the default option is used.

    figsize: tuple
        Figure's size, format as (x, y).

    filename: string or None
        Name of the file (to save). If None, the figure is not saved.

    display: bool
        Wether to render the plot.

    """

    if not self.task.startswith('binary'):
        raise AttributeError("The plot_ROC method is only available for " +
                             "binary classification tasks!")

    check_is_fitted(self._is_fitted)
    if models is None:
        models = self.models
    elif isinstance(models, str):
        models = [models]

    fig, ax = plt.subplots(figsize=figsize)
    for model in models:
        if hasattr(self, model):
            m = getattr(self, model)

            # Get False (True) Positive Rate
            fpr, tpr, _ = roc_curve(m.y_test, m.predict_proba_test[:, 1])
            plt.plot(fpr, tpr, lw=2, label=f"{m.name} (AUC={m.roc_auc:.3f})")

        else:
            raise ValueError(f"Model {model} not found in pipeline!")

    plt.plot([0, 1], [0, 1], lw=2, color='black', linestyle='--')

    title = 'ROC curve' if title is None else title
    plt.title(title, fontsize=self.title_fs, pad=12)
    plt.legend(loc='lower right',
               frameon=False,
               fontsize=self.label_fs)
    plt.xlabel('FPR', fontsize=self.label_fs, labelpad=12)
    plt.ylabel('TPR', fontsize=self.label_fs, labelpad=12)
    plt.xticks(fontsize=self.tick_fs)
    plt.yticks(fontsize=self.tick_fs)
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show() if display else plt.close()


def plot_PRC(self, models, title, figsize, filename, display):

    """
    Plot the precision-recall curve. Only for binary classification tasks.

    Parameters
    ----------
    models: string, list, tuple or None
        Name of the models to plot. If None, all the models in the
        pipeline are selected.

    title: string or None
        Plot's title. If None, the default option is used.

    figsize: tuple
        Figure's size, format as (x, y).

    filename: string or None
        Name of the file (to save). If None, the figure is not saved.

    display: bool
        Wether to render the plot.

    """

    if not self.task.startswith('binary'):
        raise AttributeError("The plot_PRC method is only available for " +
                             "binary classification tasks!")

    check_is_fitted(self._is_fitted)
    if models is None:
        models = self.models
    elif isinstance(models, str):
        models = [models]

    fig, ax = plt.subplots(figsize=figsize)
    for model in models:
        if hasattr(self, model):
            m = getattr(self, model)
            ap = m.average_precision

            # Get precision-recall pairs for different probability thresholds
            predict_proba = m.predict_proba_test[:, 1]
            prec, recall, _ = precision_recall_curve(m.y_test, predict_proba)
            plt.plot(recall, prec, lw=2, label=f"{m.name} (AP={ap:.3f})")

        else:
            raise ValueError(f"Model {model} not found in pipeline!")

    title = "Precision-recall curve" if title is None else title
    plt.title(title, fontsize=self.title_fs, pad=12)
    plt.legend(loc='lower left',
               frameon=False,
               fontsize=self.label_fs)
    plt.xlabel('Recall', fontsize=self.label_fs, labelpad=12)
    plt.ylabel('Precision', fontsize=self.label_fs, labelpad=12)
    plt.xticks(fontsize=self.tick_fs)
    plt.yticks(fontsize=self.tick_fs)
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show() if display else plt.close()


# << ====================== Utilities ====================== >>

def save(self, filename):

    """
    Save class to a pickle file.

    Parameters
    ----------
    filename: str or None, optional (default=None)
        Name of the file when saved (as .html). None to not save anything.

    """

    filename = filename if filename.endswith('.pkl') else filename + '.pkl'
    pickle.dump(self, open(filename, 'wb'))

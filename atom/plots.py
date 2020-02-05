# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (self)
Author: tvdboom
Description: Module containing plot functions.

'''


# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

# Own package modules
from .utils import raise_TypeError, raise_ValueError, check_isFit


# << ====================== Functions ====================== >>

def check_params(title, figsize, filename, display):
    ''' Check al all standard input parameters '''

    if not isinstance(title, (type(None), str)):
        raise_TypeError('title', title)
    if not isinstance(figsize, (type(None), tuple)):
        raise_TypeError('figsize', figsize)
    elif figsize is not None and len(figsize) != 2:
        raise_ValueError('figsize', figsize)
    if not isinstance(filename, (type(None), str)):
        raise_TypeError('filename', filename)
    if not isinstance(display, bool) and display not in (0, 1):
        raise_TypeError('display', display)


# << ======================== Plots ======================== >>

# << ======================== self ========================= >>

def plot_correlation(self, title, figsize, filename, display):

    '''
    DESCRIPTION -----------------------------------

    Plot the feature's correlation matrix. Ignores non-numeric columns.

    PARAMETERS -------------------------------------

    title    --> plot's title. None for default title
    figsize  --> figure size: format as (x, y)
    filename --> name of the file to save
    display  --> wether to display the plot

    '''

    check_params(title, figsize, filename, display)  # Test input params

    # Compute the correlation matrix
    corr = self.dataset.corr()
    # Drop first row and last column (diagonal line)
    corr = corr.iloc[1:].drop(self.dataset.columns[-1], axis=1)

    # Generate a mask for the upper triangle
    # k=1 means keep outermost diagonal line
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.set_style('white')  # Only for this plot
    fig, ax = plt.subplots(figsize=figsize)

    # Draw the heatmap with the mask and correct aspect ratio
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    title = 'Feature correlation matrix' if title is None else title
    plt.title(title, fontsize=self.title_fs, pad=12)
    fig.tight_layout()
    sns.set_style(self.style)  # Set back to original style
    if filename is not None:
        plt.savefig(filename)
    plt.show() if display else plt.close()


def plot_PCA(self, show, title, figsize, filename, display):

    '''
    DESCRIPTION -----------------------------------

    Plot the explained variance ratio of the components. Only if PCA
    was applied on the dataset through the feature_selection method.

    PARAMETERS -------------------------------------

    show     --> number of components to show in the plot. None for all
    title    --> plot's title. None for default title
    figsize  --> figure size: format as (x, y)
    filename --> name of the file to save
    display  --> wether to display the plot

    '''

    if not hasattr(self, 'PCA'):
        raise AttributeError('This plot is only availbale if you apply ' +
                             'PCA on the dataset through the ' +
                             'feature_selection method!')

    check_params(title, figsize, filename, display)  # Test input params
    if not isinstance(show, (type(None), int)):
        raise_TypeError('show', show)
    elif show is not None and show < 1:
        raise_ValueError('show', show)

    # Set parameters
    var = np.array(self.PCA.explained_variance_ratio_)
    if show is None or show > len(var):
        show = len(var)
    if figsize is None:  # Default figsize depends on features shown
        figsize = (10, int(4 + show/2))

    scr = pd.Series(var, index=self.X.columns).nlargest(show).sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    scr.plot.barh(label=f'Total variance retained: {round(var.sum(), 3)}')
    for i, v in enumerate(scr):
        ax.text(v + 0.005, i - 0.08, f'{v:.3f}', fontsize=self.tick_fs)

    plt.title('Explained variance ratio', fontsize=self.title_fs, pad=12)
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


# << ======================== Both ======================== >>

def plot_bagging(self, models, title, figsize, filename, display):

    '''
    DESCRIPTION -----------------------------------

    Plot a boxplot of the bagging's results.

    PARAMETERS -------------------------------------

    models    --> models to plot. None for default (all or just the one)
    title     --> plot's title. None for default title
    figsize   --> figure size: format as (x, y)
    filename  --> name of the file to save
    display   --> wether to display the plot

    '''

    if self.bagging is None:
        raise AttributeError('You need to fit the class using bagging ' +
                             'before calling the boxplot method!')

    check_isFit(self._isFit)
    check_params(title, figsize, filename, display)
    if not isinstance(models, (type(None), str, list)):
        raise_TypeError('models', models)
    elif models is None:
        models = [self.winner.name] if self.successive_halving else self.models
    elif isinstance(models, str):
        models = [models]

    results, names = [], []
    for model in models:
        if hasattr(self, model.lower()):
            results.append(getattr(self, model.lower()).bagging_scores)
            names.append(getattr(self, model.lower()).name)
        else:
            raise ValueError(f'Model {model} not found in pipeline!')

    if figsize is None:  # Default figsize depends on number of models
        figsize = (int(8 + len(names)/2), 6)

    fig, ax = plt.subplots(figsize=figsize)
    plt.boxplot(results)

    title = 'Bagging results' if title is None else title
    plt.title(title, fontsize=self.title_fs, pad=12)
    plt.xlabel('Model', fontsize=self.label_fs, labelpad=12)
    plt.ylabel(self.metric.longname,
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

    '''
    DESCRIPTION -----------------------------------

    Plot the successive halving scores.

    PARAMETERS -------------------------------------

    models   --> models to plot. None for default (all or just the one)
    title    --> plot's title. None for default title
    figsize  --> figure size: format as (x, y)
    filename --> name of the file to save
    display  --> wether to display the plot

    '''

    if not self.successive_halving:
        raise AttributeError('You need to fit the class using a ' +
                             'successive halving approach before ' +
                             'calling the plot_successive_halving method!')

    check_isFit(self._isFit)
    check_params(title, figsize, filename, display)
    if not isinstance(models, (type(None), str, list)):
        raise_TypeError('models', models)
    elif models is None:
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
            raise ValueError(f'Model {model} not found in pipeline!')

    fig, ax = plt.subplots(figsize=figsize)
    for y, label in zip(liny, names):
        plt.plot(range(len(self.results)), y, lw=2, marker='o', label=label)
    plt.xlim(-0.1, len(self.results)-0.9)

    title = 'Successive halving results' if title is None else title
    plt.title(title, fontsize=self.title_fs, pad=12)
    plt.legend(frameon=False, fontsize=self.label_fs)
    plt.xlabel('Iteration', fontsize=self.label_fs, labelpad=12)
    plt.ylabel(self.metric.longname,
               fontsize=self.label_fs,
               labelpad=12)
    ax.set_xticks(range(len(self.results)))
    plt.xticks(fontsize=self.tick_fs)
    plt.yticks(fontsize=self.tick_fs)
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show() if display else plt.close()


def plot_ROC(self, models=None, title=None,
             figsize=(10, 6), filename=None, display=True):

    '''
    DESCRIPTION -----------------------------------

    Plot Receiver Operating Characteristics curve.

    PARAMETERS -------------------------------------

    models   --> models to plot. None for default (all or just the one)
    title    --> plot's title. None for default title
    figsize  --> figure size: format as (x, y)
    filename --> name of the file to save
    display  --> wether to display the plot

    '''

    if not self.task.startswith('binary'):
        raise AttributeError('The plot_ROC method only works for binary ' +
                             'classification tasks!')

    check_isFit(self._isFit)
    check_params(title, figsize, filename, display)
    if not isinstance(models, (type(None), str, list)):
        raise_TypeError('models', models)
    elif models is None:
        models = self.models
    elif isinstance(models, str):
        models = [models]

    fig, ax = plt.subplots(figsize=figsize)
    for model in models:
        if hasattr(self, model):
            m = getattr(self, model)

            # Get False (True) Positive Rate
            fpr, tpr, _ = roc_curve(m.y_test, m.predict_proba_test[:, 1])
            plt.plot(fpr, tpr, lw=2, label=f'{m.name} (AUC={m.auc:.3f})')

        else:
            raise ValueError(f'Model {model} not found in pipeline!')

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


def plot_PRC(self, models=None, title=None,
             figsize=(10, 6), filename=None, display=True):

    '''
    DESCRIPTION -----------------------------------

    Plot precision-recall curve.

    PARAMETERS -------------------------------------

    models   --> models to plot. None for default (all or just the one)
    title    --> plot's title. None for default title
    figsize  --> figure size: format as (x, y)
    filename --> name of the file to save
    display  --> wether to display the plot

    '''

    if not self.task.startswith('binary'):
        raise AttributeError('The plot_PRC method only works for binary ' +
                             'classification tasks!')

    check_isFit(self._isFit)
    check_params(title, figsize, filename, display)
    if not isinstance(models, (type(None), str, list)):
        raise_TypeError('models', models)
    elif models is None:
        models = self.models
    elif isinstance(models, str):
        models = [models]

    fig, ax = plt.subplots(figsize=figsize)
    for model in models:
        if hasattr(self, model):
            m = getattr(self, model)

            # Get precision-recall pairs for different probability thresholds
            predict_proba = m.predict_proba_test[:, 1]
            prec, recall, _ = precision_recall_curve(m.y_test, predict_proba)
            plt.plot(recall, prec, lw=2, label=f'{m.name} (AP={m.ap:.3f})')

        else:
            raise ValueError(f'Model {model} not found in pipeline!')

    title = 'Precision-recall curve' if title is None else title
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

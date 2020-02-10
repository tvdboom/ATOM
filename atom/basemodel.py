# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Module containing the parent class for all model subclasses

"""

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
import math
import warnings
from time import time
from collections import deque
from typeguard import typechecked
from typing import Optional, Union

# Sklearn
from sklearn.utils import resample
from sklearn.metrics import SCORERS
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

# Others
from GPyOpt.methods import BayesianOptimization

# Plots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Own package modules
from .utils import composed, crash, params_to_log, time_to_string
from .plots import (
        save, plot_bagging, plot_successive_halving, plot_ROC, plot_PRC
        )


# << ====================== Global variables ====================== >>

# Variable types
scalar = Union[int, float]

# List of tree-based models
tree_models = ['Tree', 'Bag', 'ET', 'RF', 'AdaB', 'GBM', 'XGB', 'LGB', 'CatB']

# List of models that don't use the Bayesian Optimization
no_BO = ['GP', 'GNB', 'OLS']


# << ====================== Classes ====================== >>

class BaseModel(object):

    # Define class variables for plot settings
    style = 'darkgrid'
    palette = 'GnBu_d'
    title_fs = 20
    label_fs = 16
    tick_fs = 12

    def __init__(self, **kwargs):

        """
        Initialize class.

        PARAMETERS
        ----------

        data: dict
            Dictionary of the data used for this model (train, test and all).

        T: class
            ATOM class. To avoid having to pass attributes throw params.

        """

        # Set attributes from ATOM to the model's parent class
        self.__dict__.update(kwargs)
        self.error = "No exceptions encountered!"

    @composed(crash, params_to_log, typechecked)
    def bayesian_optimization(self,
                              max_iter: int = 10,
                              max_time: scalar = np.inf,
                              init_points: int = 5,
                              cv: int = 3,
                              plot_bo: bool = False):

        """
        Run the bayesian optmization algorithm to search for the best
        combination of hyperparameters. The function to optimize is evaluated
        either with a K-fold cross-validation on the training set or using
        a validation set.

        Parameters
        ----------
        max_iter: int, list or tuple, optional (default=10)
            Maximum number of iterations of the BO. If 0, skip the BO and fit
            the model on its default parameters.

        max_time: int, float, list or tuple, optional (default=np.inf)
            Maximum time allowed for the BO per model (in seconds). If 0, skip
            the BO and fit the model on its default parameters.

        init_points: int, list or tuple, optional (default=5)
            Initial number of tests of the BO before fitting the surrogate
            function.

        cv: int, list or tuple, optional (default=3)
            Strategy to fit and score the model selected after every step
            of the BO.
                - if 1, randomly split into a train and validation set
                - if >1, perform a k-fold cross validation on the training set

        plot_bo: bool, optional (default=False)
            Wether to plot the BO's progress as it runs. Creates a canvas with
            two plots: the first plot shows the score of every trial and the
            second shows the distance between the last consecutive steps. Don't
            forget to call %matplotlib at the start of the cell if you are
            using jupyter notebook!

        """

        def animate_plot(x, y1, y2, line1, line2, ax1, ax2):

            """
            Plot the BO's progress as it runs. Creates a canvas with two plots.
            The first plot shows the score of every trial and the second shows
            the distance between the last consecutive steps.

            PARAMETERS
            ----------

            x: list
                Values of both on the X-axis.

            y1, y2: list
                Values of the first and second plot on the y-axis.

            line1, line2: callable
                Line objects (from matplotlib) of the first and second plot.

            ax1, ax2: callable
                Axes objects (from matplotlib) of the first and second plot.

            """

            if line1 == []:  # At the start of the plot...
                # This is the call to matplotlib that allows dynamic plotting
                plt.ion()

                # Initialize plot
                fig = plt.figure(figsize=(10, 6))
                gs = GridSpec(2, 1, height_ratios=[2, 1])

                # First subplot (without xtick labels)
                ax1 = plt.subplot(gs[0])
                # Create a variable for the line so we can later update it
                line1, = ax1.plot(x, y1, '-o', alpha=0.8)
                ax1.set_title(f"Bayesian Optimization for {self.longname}",
                              fontsize=BaseModel.title_fs)
                ax1.set_ylabel(self.T.metric.longname,
                               fontsize=BaseModel.label_fs,
                               labelpad=12)
                ax1.set_xlim(min(self.x)-0.5, max(self.x)+0.5)

                # Second subplot
                ax2 = plt.subplot(gs[1], sharex=ax1)
                line2, = ax2.plot(x, y2, '-o', alpha=0.8)
                ax2.set_title("Metric distance between last consecutive steps",
                              fontsize=BaseModel.title_fs)
                ax2.set_xlabel('Step',
                               fontsize=BaseModel.label_fs,
                               labelpad=12)
                ax2.set_ylabel('d',
                               fontsize=BaseModel.label_fs,
                               labelpad=12)
                ax2.set_xticks(self.x)
                ax2.set_xlim(min(self.x)-0.5, max(self.x)+0.5)
                ax2.set_ylim([-0.05, 0.1])

                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.subplots_adjust(hspace=.0)
                plt.xticks(fontsize=BaseModel.tick_fs)
                plt.yticks(fontsize=BaseModel.tick_fs)
                fig.tight_layout()
                plt.show()

            # Update plot
            line1.set_xdata(x)
            line1.set_ydata(y1)
            line2.set_xdata(x)
            line2.set_ydata(y2)
            ax1.set_xlim(min(self.x)-0.5, max(self.x)+0.5)
            ax2.set_xlim(min(self.x)-0.5, max(self.x)+0.5)
            ax1.set_xticks(self.x)  # Update x-ticks
            ax2.set_xticks(self.x)  # Update x-ticks

            # Adjust y limits if new data goes beyond bounds
            lim = line1.axes.get_ylim()
            if np.nanmin(y1) <= lim[0] or np.nanmax(y1) >= lim[1]:
                ax1.set_ylim([np.nanmin(y1) - np.nanstd(y1),
                              np.nanmax(y1) + np.nanstd(y1)])
            lim = line2.axes.get_ylim()
            if np.nanmax(y2) >= lim[1]:
                ax2.set_ylim([-0.05, np.nanmax(y2) + np.nanstd(y2)])

            # Pause the data so the figure/axis can catch up
            plt.pause(0.01)

            # Return line and axes to update the plot again next iteration
            return line1, line2, ax1, ax2

        def optimize(x):

            """
            Function to be optimized by the BO.

            Parameters
            ----------
            x: dict
                Dictionary of the model's  hyperparameters.

            Returns
            -------
            score: float
                Score achieved by the model.

            """

            t_iter = time()  # Get current time for start of the iteration

            params = self.get_params(x)
            self.BO['params'].append(params)

            # Print iteration and time
            self._iter += 1
            if self._iter > self.init_points:
                _iter = self._iter - self.init_points
                point = 'Iteration'
            else:
                _iter = self._iter
                point = 'Initial point'

            len_ = '-' * (46 - len(point) - len(str(self._iter)))
            self.T._log(f"{point}: {_iter} {len_}", 2)
            self.T._log(f"Parameters --> {params}", 2)

            algorithm = self.get_model(params)

            if self.cv == 1:
                # Split each iteration in different train and validation set
                X_subtrain, X_validation, y_subtrain, y_validation = \
                    train_test_split(self.X_train,
                                     self.y_train,
                                     test_size=self.T.test_size,
                                     shuffle=True)

                # Calculate metric on the validation set
                algorithm.fit(X_subtrain, y_subtrain)
                score = self.T.metric(algorithm, X_validation, y_validation)

            else:  # Use cross validation to get the output of BO

                # Determine number of folds for the cross_val_score
                if self.T.task != 'regression':
                    # Folds are made preserving the % of samples for each class
                    # Use same splits for every model
                    kfold = StratifiedKFold(n_splits=self.cv, random_state=1)
                else:
                    kfold = KFold(n_splits=self.cv, random_state=1)

                # Run cross-validation (get mean of results)
                score = cross_val_score(algorithm,
                                        self.X_train,
                                        self.y_train,
                                        cv=kfold,
                                        scoring=self.T.metric,
                                        n_jobs=self.T.n_jobs).mean()

            # Save output of the BO and plot progress
            self.BO['score'].append(score)
            self.T._log(f"Evaluation --> {self.T.metric.name}: {score:.4f}", 2)

            t = time_to_string(t_iter)
            t_tot = time_to_string(self._init_bo)
            self.BO['time'] = t
            self.BO['total_time'] = t_tot
            self.T._log(f"Time elapsed: {t}   Total time: {t_tot}", 2)

            if self.plot_bo:
                # Start to fill NaNs with encountered metric values
                if np.isnan(self.y1).any():
                    for i, value in enumerate(self.y1):
                        if math.isnan(value):
                            self.y1[i] = score
                            if i > 0:  # The first value must remain empty
                                self.y2[i] = abs(self.y1[i] - self.y1[i-1])
                            break
                else:  # If no NaNs anymore, continue deque
                    self.x.append(max(self.x)+1)
                    self.y1.append(score)
                    self.y2.append(abs(self.y1[-1] - self.y1[-2]))

                self.line1, self.line2, self.ax1, self.ax2 = \
                    animate_plot(self.x, self.y1, self.y2,
                                 self.line1, self.line2, self.ax1, self.ax2)

            return score

        # << ================= Running optimization ================= >>

        # Check parameters
        if max_iter < 0:
            raise ValueError("Invalid value for the max_iter parameter." +
                             f"Value should be >=0, got {max_iter}.")
        if max_time < 0:
            raise ValueError("Invalid value for the max_time parameter." +
                             f"Value should be >=0, got {max_time}.")
        if init_points < 1:
            raise ValueError("Invalid value for the init_points parameter." +
                             f"Value should be >0, got {init_points}.")
        if cv < 1:
            raise ValueError("Invalid value for the cv parameter." +
                             f"Value should be >0, got {cv}.")

        # Update attibutes
        self._has_BO = True if max_iter > 0 and max_time > 0 else False
        self.max_iter = max_iter
        self.max_time = max_time
        self.init_points = init_points
        self.cv = cv
        self.plot_bo = plot_bo

        # Skip BO for GNB and GP (no hyperparameter tuning)
        if self.name not in no_BO and self._has_BO:
            self.T._log(f"\n\nRunning BO for {self.longname}...", 1)

            # Save dictionary of BO steps
            self._iter = 0
            self._init_bo = time()
            self.BO = {}
            self.BO['params'] = []
            self.BO['score'] = []

            if self.plot_bo:  # Create plot variables
                maxlen = 15  # Maximum steps to show at once in the plot
                self.x = deque(list(range(maxlen)), maxlen=maxlen)
                self.y1 = deque([np.NaN for _ in self.x], maxlen=maxlen)
                self.y2 = deque([np.NaN for _ in self.x], maxlen=maxlen)
                self.line1, self.line2 = [], []  # Plot lines
                self.ax1, self.ax2 = 0, 0  # Plot axes

            # Default SKlearn or multiple random initial points
            kwargs = {}
            if self.init_points > 1:
                kwargs['initial_design_numdata'] = self.init_points
            else:
                kwargs['X'] = self.get_init_values()
            BO = BayesianOptimization(f=optimize,
                                      domain=self.get_domain(),
                                      model_update_interval=1,
                                      maximize=True,
                                      initial_design_type='random',
                                      normalize_Y=False,
                                      num_cores=self.T.n_jobs,
                                      **kwargs)

            # eps = Minimum distance in hyperparameters between two
            #       consecutive steps of the BO
            BO.run_optimization(max_iter=self.max_iter,
                                max_time=self.max_time,
                                eps=1e-8,
                                verbosity=False)

            # Optimal score of the BO (neg due to the implementation of GpyOpt)
            self.score_bo = -BO.fx_opt

            if self.plot_bo:
                plt.close()

            # Set to same shape as GPyOpt (2d-array)
            self.best_params = self.get_params(np.round([BO.x_opt], 4))

            # Save best model (not yet fitted)
            self.best_model = self.get_model(self.best_params)

            self.time_bo = time_to_string(self._init_bo)  # Get the BO duration

            # Print results
            self.T._log('', 2)  # Print extra line (note verbosity level 2)
            self.T._log(f"Final results for {self.longname}:{' ':9s}", 1)
            self.T._log("Bayesian Optimization ---------------------------", 1)
            self.T._log(f"Best hyperparameters: {self.best_params}", 1)
            self.T._log(f"Best score on the BO: {self.score_bo:.4f}", 1)
            self.T._log(f"Time elapsed: {self.time_bo}", 1)

        else:
            self.best_model = self.get_model()

    @composed(crash, params_to_log)
    def fit(self):
        """ Fit the best model to the test set """

        t_init = time()

        # In case the bayesian_optimization method wasn't called
        if not hasattr(self, 'best_model'):
            self.best_model = self.get_model()

        # Fit the selected model on the complete training set
        self.best_model_fit = self.best_model.fit(self.X_train, self.y_train)

        # Save predictions
        self.predict_train = self.best_model_fit.predict(self.X_train)
        self.predict_test = self.best_model_fit.predict(self.X_test)

        # Save probability predictions
        if self.T.task != 'regression':
            try:  # Only works if model has predict_proba attribute
                self.predict_proba_train = \
                    self.best_model_fit.predict_proba(self.X_train)
                self.predict_proba_test = \
                    self.best_model_fit.predict_proba(self.X_test)
            except AttributeError:
                pass

        # Save scores on complete training and test set
        self.score_train = self.T.metric(
                            self.best_model_fit, self.X_train, self.y_train)
        self.score_test = self.T.metric(
                            self.best_model_fit, self.X_test, self.y_test)

        # Calculate all pre-defined metrics on the test set
        for key in SCORERS.keys():
            try:
                metric = getattr(self.T.metric, key)
                score = metric(self.best_model_fit, self.X_test, self.y_test)
                setattr(self, key, score)
            except Exception:
                msg = f"This metric is unavailable for {self.T.task} tasks!"
                setattr(self, key, msg)

        # << ================= Print stats ================= >>

        if not self._has_BO:
            self.T._log('\n', 1)  # Print 2 extra lines
            self.T._log(f"Final results for {self.longname}:{' ':9s}", 1)
        if self.name in no_BO and self._has_BO:
            self.T._log('\n', 1)  # Print 2 extra lines
            self.T._log(f"Final results for {self.longname}:{' ':9s}", 1)
        self.T._log("Fitting -----------------------------------------", 1)
        self.T._log(f"Score on the training set: {self.score_train:.4f}", 1)
        self.T._log(f"Score on the test set: {self.score_test:.4f}", 1)

        # Get duration and print to log
        duration = time_to_string(t_init)
        self.fit_time = duration
        self.T._log(f"Time elapsed: {duration}", 1)

    @composed(crash, params_to_log, typechecked)
    def bagging(self, bagging: Optional[int] = 5):

        """
        Take bootstrap samples from the training set and test them on the test
        set to get a distribution of the model's results.

        Parameters
        ----------
        bagging: int or None, optional (default=5)
            Number of data sets (bootstrapped from the training set) to use in
            the bagging algorithm. If None or 0, no bagging is performed.

        """

        t_init = time()

        if bagging is None or bagging == 0:
            return None  # Do not perform bagging
        elif bagging < 0:
            raise ValueError("Invalid value for the cv parameter." +
                             f"Value should be >=0, got {bagging}.")

        self.bagging_scores = []  # List of the scores
        for _ in range(bagging):
            # Create samples with replacement
            sample_x, sample_y = resample(self.X_train, self.y_train)

            # Fit on bootstrapped set and predict on the independent test set
            algorithm = self.best_model.fit(sample_x, sample_y)
            score = self.T.metric(algorithm, self.X_test, self.y_test)

            # Append metric result to list
            self.bagging_scores.append(score)

        # Numpy array for mean and std
        self.bagging_scores = np.array(self.bagging_scores)
        self.T._log("Bagging -----------------------------------------", 1)
        self.T._log("Mean: {:.4f}   Std: {:.4f}".format(
                    self.bagging_scores.mean(), self.bagging_scores.std()), 1)

        # Get duration and print to log
        duration = time_to_string(t_init)
        self.bs_time = duration
        self.T._log(f"Time elapsed: {duration}", 1)

    # << ==================== Plot functions ==================== >>

    @composed(crash, params_to_log)
    def plot_bagging(self, title=None,
                     figsize=(10, 6), filename=None, display=True):
        """ Plot a boxplot of the bagging's results """

        plot_bagging(self.T, self.name, title, figsize, filename, display)

    @composed(crash, params_to_log)
    def plot_successive_halving(self, title=None,
                                figsize=(10, 6), filename=None, display=True):
        """ Plot the successive halving scores """

        plot_successive_halving(self.T, self.name,
                                title, figsize, filename, display)

    @composed(crash, params_to_log)
    def plot_threshold(self, metric=None, steps=100, title=None,
                       figsize=(10, 6), filename=None, display=True):

        """
        DESCRIPTION ------------------------------------

        Plot performance metrics against multiple threshold values.

        ARGUMENTS -------------------------------------

        metric   --> metric(s) to plot
        steps    --> Number of thresholds to try between 0 and 1
        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save
        display  --> wether to display the plot

        """

        if self.T.task != 'binary classification':
            raise AttributeError('This method is only available for ' +
                                 'binary classification tasks.')

        # Set metric parameter
        if metric is None:
            metric = self.T.metric.function
        if not isinstance(metric, list):
            metric = [metric]

        # Convert all strings to functions
        mlist = []
        for m in metric:
            if isinstance(m, str):
                mlist.append(getattr(self.T.metric, m).function)
            else:
                mlist.append(m)

        # Get results ignoring annoying warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            results = {}
            for m in mlist:  # Create dict of empty arrays
                results[m] = []
            space = np.linspace(0, 1, steps)
            for step in space:
                for m in mlist:
                    pred = (self.predict_proba_test[:, 1] >= step).astype(bool)
                    results[m].append(m(self.y_test, pred))

        fig, ax = plt.subplots(figsize=figsize)
        for i, m in enumerate(mlist):
            plt.plot(space, results[m], label=mlist[i].__name__, lw=2)

        if title is None:
            temp = '' if len(metric) == 1 else 's'
            title = f'Performance metric{temp} against threshold value'
        plt.title(title, fontsize=BaseModel.title_fs, pad=12)
        plt.legend(frameon=False, fontsize=BaseModel.label_fs)
        plt.xlabel('Threshold', fontsize=BaseModel.label_fs, labelpad=12)
        plt.ylabel('Score', fontsize=BaseModel.label_fs, labelpad=12)
        plt.xticks(fontsize=BaseModel.tick_fs)
        plt.yticks(fontsize=BaseModel.tick_fs)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show() if display else plt.close()

    @composed(crash, params_to_log)
    def plot_probabilities(self, target=1, title=None,
                           figsize=(10, 6), filename=None, display=True):

        """
        DESCRIPTION -----------------------------------

        Plot a function of the probability of the classes
        of being the target class.

        PARAMETERS -------------------------------------

        target   --> probability of being that class (as idx or string)
        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save
        display  --> wether to display the plot

        """

        if self.T.task == 'regression':
            raise AttributeError('This method is only available for ' +
                                 'classification tasks.')

        # Make target mapping
        inv_map = {str(v): k for k, v in self.T.mapping.items()}
        if isinstance(target, str):  # User provides a string
            target_int = self.T.mapping[target]
            target_str = target
        else:  # User provides an integer
            target_int = target
            target_str = inv_map[str(target)]

        fig, ax = plt.subplots(figsize=figsize)
        for key, value in self.T.mapping.items():
            idx = np.where(self.y_test == value)  # Get indices per class
            sns.distplot(self.predict_proba_test[idx, target_int],
                         hist=False,
                         kde=True,
                         norm_hist=True,
                         kde_kws={"shade": True},
                         label='Class=' + key)

        if title is None:
            title = f'Predicted probabilities for {self.y.name}={target_str}'
        plt.title(title, fontsize=BaseModel.title_fs, pad=12)
        plt.legend(frameon=False, fontsize=BaseModel.label_fs)
        plt.xlabel('Probability', fontsize=BaseModel.label_fs, labelpad=12)
        plt.ylabel('Counts', fontsize=BaseModel.label_fs, labelpad=12)
        plt.xlim(0, 1)
        plt.xticks(fontsize=BaseModel.tick_fs)
        plt.yticks(fontsize=BaseModel.tick_fs)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show() if display else plt.close()

    @composed(crash, params_to_log)
    def plot_permutation_importance(self, show=None, n_repeats=10, title=None,
                                    figsize=None, filename=None, display=True):

        """
        DESCRIPTION -----------------------------------

        Plot a model's feature permutation importance.

        PARAMETERS -------------------------------------

        show      --> number of best features to show in the plot
        n_repeats --> number of times to permute a feature
        title     --> plot's title. None for default title
        figsize   --> figure size: format as (x, y)
        filename  --> name of the file to save
        display   --> wether to display the plot

        """

        # Set parameters
        show = self.X.shape[1] if show is None else int(show)
        if figsize is None:  # Default figsize depends on features shown
            figsize = (10, int(4 + show/2))

        # Calculate the permutation importances
        # Force random state on function (won't work with numpy default)
        self.permutations = \
            permutation_importance(self.best_model_fit,
                                   self.X_test,
                                   self.y_test,
                                   scoring=self.T.metric,
                                   n_repeats=n_repeats,
                                   n_jobs=self.T.n_jobs,
                                   random_state=self.T.random_state)

        # Get indices of permutations sorted by the mean
        idx = self.permutations.importances_mean.argsort()[:show]

        fig, ax = plt.subplots(figsize=figsize)
        plt.boxplot(self.permutations.importances[idx].T,
                    vert=False,
                    labels=self.X.columns[idx])

        title = 'Feature permutation importance' if title is None else title
        plt.title(title, fontsize=BaseModel.title_fs, pad=12)
        plt.xlabel('Score', fontsize=BaseModel.label_fs, labelpad=12)
        plt.ylabel('Features', fontsize=BaseModel.label_fs, labelpad=12)
        plt.xticks(fontsize=BaseModel.tick_fs)
        plt.yticks(fontsize=BaseModel.tick_fs)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show() if display else plt.close()

    @composed(crash, params_to_log)
    def plot_feature_importance(self, show=None, title=None,
                                figsize=None, filename=None, display=True):

        """
        DESCRIPTION -----------------------------------

        Plot a (Tree based) model's normalized feature importance.

        PARAMETERS -------------------------------------

        show     --> number of best features to show in the plot. None for all
        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save
        display  --> wether to display the plot

        """

        if self.name not in tree_models:
            raise AttributeError('Only availabe for tree-based models!')

        # Set parameters
        if show is None or show > self.X.shape[1]:
            show = self.X.shape[1]
        if figsize is None:  # Default figsize depends on features shown
            figsize = (10, int(4 + show/2))

        # Bagging has no direct feature importance implementation
        if self.name == 'Bag':
            feature_importances = np.mean([
                est.feature_importances_ for est in self.best_model.estimators_
            ], axis=0)
        else:
            feature_importances = self.best_model_fit.feature_importances_

        # Normalize for plotting values adjacent to bar
        feature_importances = feature_importances/max(feature_importances)
        scores = pd.Series(feature_importances,
                           index=self.X.columns).nlargest(show).sort_values()

        fig, ax = plt.subplots(figsize=figsize)
        scores.plot.barh()
        for i, v in enumerate(scores):
            ax.text(v + 0.01, i - 0.08, f'{v:.2f}', fontsize=BaseModel.tick_fs)

        title = 'Normalized feature importance' if title is None else title
        plt.title(title, fontsize=BaseModel.title_fs, pad=12)
        plt.xlabel('Score', fontsize=BaseModel.label_fs, labelpad=12)
        plt.ylabel('Features', fontsize=BaseModel.label_fs, labelpad=12)
        plt.xticks(fontsize=BaseModel.tick_fs)
        plt.yticks(fontsize=BaseModel.tick_fs)
        plt.xlim(0, 1.07)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show() if display else plt.close()

    @composed(crash, params_to_log)
    def plot_ROC(self, title=None,
                 figsize=(10, 6), filename=None, display=True):
        """ Plot Receiver Operating Characteristics curve """

        plot_ROC(self.T, self.name, title, figsize, filename, display)

    @composed(crash, params_to_log)
    def plot_PRC(self, title=None,
                 figsize=(10, 6), filename=None, display=True):
        """ Plot precision-recall curve """

        plot_PRC(self.T, self.name, title, figsize, filename, display)

    @composed(crash, params_to_log)
    def plot_confusion_matrix(self, normalize=True, title=None,
                              figsize=(10, 6), filename=None, display=True):

        """
        DESCRIPTION -----------------------------------

        Plot the confusion matrix in a heatmap.

        PARAMETERS -------------------------------------

        normalize --> wether to normalize the matrix
        title     --> plot's title. None for default title
        figsize   --> figure size: format as (x, y)
        filename  --> name of the file to save
        display   --> wether to display the plot

        """

        if self.T.task == 'regression':
            raise AttributeError('This method only works for ' +
                                 'classification tasks.')

        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, self.predict_test)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        ticks = [v for v in self.T.mapping.keys()]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=ticks,
               yticklabels=ticks)

        # Loop over data dimensions and create text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        fontsize=BaseModel.tick_fs,
                        color="white" if cm[i, j] > thresh else "black")

        if title is None and normalize:
            title = 'Normalized confusion matrix'
        elif title is None:
            title = 'Confusion matrix'
        plt.title(title, fontsize=BaseModel.title_fs, pad=12)
        plt.xlabel('Predicted label', fontsize=BaseModel.label_fs, labelpad=12)
        plt.ylabel('True label', fontsize=BaseModel.label_fs, labelpad=12)
        plt.xticks(fontsize=BaseModel.tick_fs)
        plt.yticks(fontsize=BaseModel.tick_fs)
        cbar.ax.tick_params(labelsize=BaseModel.tick_fs)  # Colorbar's ticks
        ax.grid(False)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show() if display else plt.close()

    # << ============ Utility functions ============ >>

    @composed(crash, params_to_log)
    def save(self, filename=None):
        """ Save the model subclass to a pickle file """

        save(self, 'ATOM_' + self.name if filename is None else filename)
        self.T._log(self.longname + ' model subclass saved successfully!', 1)

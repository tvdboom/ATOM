# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Module containing the parent class for all model subclasses

'''

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
import math
import pickle
import warnings
from time import time
from collections import deque

# Sklearn
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (
        make_scorer, confusion_matrix, roc_curve, precision_recall_curve
        )
# Others
from GPyOpt.methods import BayesianOptimization

# Plots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Own package modules
from .utils import composed, crash, params_to_log, timer, prlog, time_to_string


# << ============ Global variables ============ >>

# List of tree-based models
tree_models = ['Tree', 'Bag', 'ET', 'RF', 'AdaB', 'GBM', 'XGB', 'LGB', 'CatB']

# List of models that don't use the Bayesian Optimization
no_BO = ['GP', 'GNB', 'OLS']

# List of models with no (or sometimes no) predict_proba method
no_predict_proba = ['OLS', 'Ridge', 'Lasso', 'EN', 'BR',
                    'lSVM', 'kSVM', 'PA', 'SGD']


# << ============ Classes ============ >>
class BaseModel(object):

    # Define class variables for plot settings
    style = 'darkgrid'
    palette = 'GnBu_d'
    title_fs = 20
    label_fs = 16
    tick_fs = 12

    def __init__(self, **kwargs):

        '''
        DESCRIPTION -----------------------------------

        Initialize class.

        PARAMETERS -------------------------------------

        data --> dictionary of the data (train, test and all)
        T    --> ATOM class. To avoid having to pass attributes throw params

        '''

        # Set attributes from ATOM to the model's parent class
        self.__dict__.update(kwargs)
        self.error = 'No exceptions encountered!'

    def BayesianOpt(self):

        '''
        Run the bayesian optmization algorithm to search for the best
        combination of hyperparameters. The function to optimize is evaluated
        either with a K-fold cross-validation on the training set or using
        a validation set.
        '''

        def animate_plot(x, y1, y2, line1, line2, ax1, ax2):

            '''
            DESCRIPTION -----------------------------------

            Plot the BO's progress as it runs. Creates a canvas with two plots.
            The first plot shows the score of every trial and the second shows
            the distance between the last consecutive steps.

            PARAMETERS -------------------------------------

            x    --> x values of the plots
            y    --> y values of the plots
            line --> line element of the plots
            ax   --> plot's axes

            '''

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
                ax1.set_title(f'Bayesian Optimization for {self.longname}',
                              fontsize=BaseModel.title_fs)
                ax1.set_ylabel(self.T.metric.longname,
                               fontsize=BaseModel.label_fs,
                               labelpad=12)
                ax1.set_xlim(min(self.x)-0.5, max(self.x)+0.5)

                # Second subplot
                ax2 = plt.subplot(gs[1], sharex=ax1)
                line2, = ax2.plot(x, y2, '-o', alpha=0.8)
                ax2.set_title('Metric distance between last consecutive steps',
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
            line1.set_xdata(x)   # Update x-data
            line1.set_ydata(y1)  # Update y-data
            line2.set_xdata(x)   # Update x-data
            line2.set_ydata(y2)  # Update y-data
            ax1.set_xlim(min(self.x)-0.5, max(self.x)+0.5)  # Update x-axis
            ax2.set_xlim(min(self.x)-0.5, max(self.x)+0.5)  # Update x-axis
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
            ''' Function to be optimized by the BO '''

            t_iter = time()  # Get current time for start of the iteration

            params = self.get_params(x)
            self.BO['params'].append(params)

            # Print iteration and time
            self._iter += 1
            if self._iter > self.T.init_points:
                _iter = self._iter - self.T.init_points
                point = 'Iteration'
            else:
                _iter = self._iter
                point = 'Initial point'

            len_ = '-' * (46 - len(point) - len(str(self._iter)))
            prlog(f"{point}: {_iter} {len_}", self, 1)
            prlog(f'Parameters --> {params}', self, 2)

            algorithm = self.get_model(params)

            if self.T.cv == 1:
                # Split each iteration in different train and validation set
                X_subtrain, X_validation, Y_subtrain, Y_validation = \
                    train_test_split(self.X_train,
                                     self.y_train,
                                     test_size=self.T.test_size,
                                     shuffle=True)

                # Models without the predict_proba() method need probs with ccv
                # Not prefit to not have to make an extra cut in the data
                if self.T.metric.needs_proba:
                    if self.name in no_predict_proba:
                        ccv = CalibratedClassifierCV(algorithm, cv=None)
                        ccv.fit(X_subtrain, Y_subtrain)
                        y_pred = ccv.predict_proba(X_validation)
                    else:
                        algorithm.fit(X_subtrain, Y_subtrain)
                        y_pred = algorithm.predict_proba(X_validation)

                else:
                    algorithm.fit(X_subtrain, Y_subtrain)
                    y_pred = algorithm.predict(X_validation)

                # Calculate metric on the validation set
                out = self.T.metric.func(Y_validation, y_pred)

            else:  # Use cross validation to get the output of BO

                # Define the estimator dependent on needs_proba
                if self.name in no_predict_proba and self.T.metric.needs_proba:
                    estimator = CalibratedClassifierCV(algorithm, cv=None)
                else:
                    estimator = algorithm

                # Make scoring function for the cross_validator
                # .function (not .func) since make_scorer handles automatically
                scoring = make_scorer(self.T.metric.function,
                                      greater_is_better=self.T.metric.gib,
                                      needs_proba=self.T.metric.needs_proba)

                # Determine number of folds for the cross_val_score
                if self.T.task != 'regression':
                    # Folds are made preserving the % of samples for each class
                    # Use same splits for every model
                    kfold = StratifiedKFold(n_splits=self.T.cv, random_state=1)
                else:
                    kfold = KFold(n_splits=self.T.cv, random_state=1)

                # Run cross-validation (get mean of results)
                out = cross_val_score(estimator,
                                      self.X_train,
                                      self.y_train,
                                      cv=kfold,
                                      scoring=scoring,
                                      n_jobs=self.T.n_jobs).mean()

                # cross_val_score returns negative loss for minimizing metrics
                out = out if self.T.metric.gib and out != 0 else -out

            # Save output of the BO and plot progress
            self.BO['score'].append(out)
            prlog('Evaluation --> {0}: {1:.{2}f}'.format(
                    self.T.metric.longname, out, self.T.metric.dec), self, 2)

            t = time_to_string(t_iter)
            t_tot = time_to_string(self._init_bo)
            self.BO['time'] = t
            self.BO['total_time'] = t_tot
            prlog(f'Time elapsed: {t}   Total time: {t_tot}', self, 2)

            if self.T.plot_bo:
                # Start to fill NaNs with encountered metric values
                if np.isnan(self.y1).any():
                    for i, value in enumerate(self.y1):
                        if math.isnan(value):
                            self.y1[i] = out
                            if i > 0:  # The first value must remain empty
                                self.y2[i] = abs(self.y1[i] - self.y1[i-1])
                            break
                else:  # If no NaNs anymore, continue deque
                    self.x.append(max(self.x)+1)
                    self.y1.append(out)
                    self.y2.append(abs(self.y1[-1] - self.y1[-2]))

                self.line1, self.line2, self.ax1, self.ax2 = \
                    animate_plot(self.x, self.y1, self.y2,
                                 self.line1, self.line2, self.ax1, self.ax2)

            return out

        # << ============ Running optimization ============ >>

        # Skip BO for GNB and GP (no hyperparameter tuning)
        if self.name not in no_BO and self.T._BO:
            prlog(f'\n\nRunning BO for {self.longname}...', self, 1)

            # Save dictionary of BO steps
            self._iter = 0
            self._init_bo = time()
            self.BO = {}
            self.BO['params'] = []
            self.BO['score'] = []

            # BO plot variables
            maxlen = 15  # Maximum steps to show at once in the plot
            self.x = deque(list(range(maxlen)), maxlen=maxlen)
            self.y1 = deque([np.NaN for _ in self.x], maxlen=maxlen)
            self.y2 = deque([np.NaN for _ in self.x], maxlen=maxlen)
            self.line1, self.line2 = [], []  # Plot lines
            self.ax1, self.ax2 = 0, 0  # Plot axes

            # Minimize or maximize the function depending on the metric
            maximize = True if self.T.metric.gib else False
            # Default SKlearn or multiple random initial points
            kwargs = {}
            if self.T.init_points > 1:
                kwargs['initial_design_numdata'] = self.T.init_points
            else:
                kwargs['X'] = self.get_init_values()
            opt = BayesianOptimization(f=optimize,
                                       domain=self.get_domain(),
                                       model_update_interval=self.T.batch_size,
                                       maximize=maximize,
                                       initial_design_type='random',
                                       normalize_Y=False,
                                       num_cores=self.T.n_jobs,
                                       **kwargs)

            opt.run_optimization(max_iter=self.T.max_iter,
                                 max_time=self.T.max_time,
                                 eps=self.T.eps,
                                 verbosity=False)

            if self.T.plot_bo:
                plt.close()

            # Optimal score of the BO
            self.score_bo = -opt.fx_opt if self.T.metric.gib else opt.fx_opt

            # Set to same shape as GPyOpt (2d-array)
            self.best_params = self.get_params(
                                        np.array(np.round([opt.x_opt], 4)))

            # Save best model (not yet fitted)
            self.best_model = self.get_model(self.best_params)

            self.time_bo = time_to_string(self._init_bo)  # Get the BO duration

        else:
            self.best_model = self.get_model()

    @timer
    def fitting(self):
        ''' Fit the best model to the test set '''

        # Fit the selected model on the complete training set
        self.best_model_fit = self.best_model.fit(self.X_train, self.y_train)

        # Save predictions
        self.predict_train = self.best_model_fit.predict(self.X_train)
        self.predict_test = self.best_model_fit.predict(self.X_test)

        # Models without the predict_proba() method need probs with ccv
        if self.name in no_predict_proba and self.T.task != 'regression':
            ccv = CalibratedClassifierCV(self.best_model, cv=None)
            ccv.fit(self.X_train, self.y_train)
            self.predict_proba_train = ccv.predict_proba(self.X_train)
            self.predict_proba_test = ccv.predict_proba(self.X_test)
        elif self.T.task != 'regression':
            self.predict_proba_train = \
                self.best_model_fit.predict_proba(self.X_train)
            self.predict_proba_test = \
                self.best_model_fit.predict_proba(self.X_test)

        # Get metric scores
        if self.T.metric.needs_proba:
            self.score_train = self.T.metric.func(self.y_train,
                                                  self.predict_proba_train)
            self.score_test = self.T.metric.func(self.y_test,
                                                 self.predict_proba_test)
        else:
            self.score_train = self.T.metric.func(self.y_train,
                                                  self.predict_train)
            self.score_test = self.T.metric.func(self.y_test,
                                                 self.predict_test)

        # Calculate some standard metrics on the test set
        for m in self.T.metric.__dict__.keys():
            # Skip all non-metric attributes
            if m in ['function', 'name', 'longname',
                     'gib', 'needs_proba', 'task', 'dec']:
                continue

            try:
                metric = getattr(self.T.metric, m)
                if metric.needs_proba and self.T.task != 'regression':
                    y_pred = self.predict_proba_test
                else:
                    y_pred = self.predict_test
                setattr(self, m, metric.func(self.y_test, y_pred))
            except Exception:
                msg = f'This metric is unavailable for {self.T.task} tasks!'
                setattr(self, m, msg)

        # << ================= Print stats ================= >>

        if not self.T._BO:
            prlog('\n', self, 1)  # Print extra line
        if self.name in no_BO and self.T._BO:
            prlog('\n', self, 1)  # Print 2 extra lines
        elif self.T._BO:
            prlog('', self, 2)  # Print extra line

        prlog('Final results for {}:{:9s}'.format(self.longname, ' '), self, 1)
        if self.name not in no_BO and self.T._BO:
            prlog('Bayesian Optimization ---------------------------', self, 1)
            prlog(f'Best hyperparameters: {self.best_params}', self, 1)
            prlog('Best score on the BO: {0:.{1}f}'
                  .format(self.score_bo, self.T.metric.dec), self, 1)
            prlog(f'Time elapsed: {self.time_bo}', self, 1)
        prlog('Fitting -----------------------------------------', self, 1)
        prlog('Score on the training set: {0:.{1}f}'
              .format(self.score_train, self.T.metric.dec), self, 1)
        prlog('Score on the test set: {0:.{1}f}'
              .format(self.score_test, self.T.metric.dec), self, 1)

    @timer
    def bagging(self):

        '''
        Take bootstrap samples from the training set and test them on the test
        set to get a distribution of the model's results.
        '''

        self.bagging_scores = []  # List of the scores
        for _ in range(self.T.bagging):
            # Create samples with replacement
            sample_x, sample_y = resample(self.X_train, self.y_train)

            # Fit on bootstrapped set and predict on the independent test set
            if self.T.metric.needs_proba:
                if self.name in no_predict_proba:
                    ccv = CalibratedClassifierCV(self.best_model, cv=None)
                    ccv.fit(sample_x, sample_y)
                    y_pred = ccv.predict_proba(self.X_test)
                else:
                    algorithm = self.best_model.fit(sample_x, sample_y)
                    y_pred = algorithm.predict_proba(self.X_test)

            else:
                algorithm = self.best_model.fit(sample_x, sample_y)
                y_pred = algorithm.predict(self.X_test)

            # Append metric result to list
            self.bagging_scores.append(self.T.metric.func(self.y_test, y_pred))

        # Numpy array for mean and std
        self.bagging_scores = np.array(self.bagging_scores)
        prlog('Bagging -----------------------------------------', self, 1)
        prlog('Mean: {:.4f}   Std: {:.4f}'
              .format(self.bagging_scores.mean(), self.bagging_scores.std()),
              self, 1)

    # << ============ Plot functions ============ >>

    @composed(crash, params_to_log)
    def plot_threshold(self, metric=None, steps=100, title=None,
                       figsize=(10, 6), filename=None, display=True):

        '''
        DESCRIPTION ------------------------------------

        Plot performance metrics against multiple threshold values.

        ARGUMENTS -------------------------------------

        metric   --> metric(s) to plot
        steps    --> Number of thresholds to try between 0 and 1
        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save
        display  --> wether to display the plot

        '''

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

        '''
        DESCRIPTION -----------------------------------

        Plot a function of the probability of the classes
        of being the target class.

        PARAMETERS -------------------------------------

        target   --> probability of being that class (as idx or string)
        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save
        display  --> wether to display the plot

        '''

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

        '''
        DESCRIPTION -----------------------------------

        Plot a model's feature permutation importance.

        PARAMETERS -------------------------------------

        show      --> number of best features to show in the plot
        n_repeats --> number of times to permute a feature
        title     --> plot's title. None for default title
        figsize   --> figure size: format as (x, y)
        filename  --> name of the file to save
        display   --> wether to display the plot

        '''

        # Set parameters
        show = self.X.shape[1] if show is None else int(show)
        if figsize is None:  # Default figsize depends on features shown
            figsize = (10, int(4 + show/2))

        # Calculate the permutation importances
        # Force random state on function (won't work with numpy default)
        scoring = make_scorer(self.T.metric.function,
                              greater_is_better=self.T.metric.gib,
                              needs_proba=self.T.metric.needs_proba)
        self.permutations = \
            permutation_importance(self.best_model_fit,
                                   self.X_test,
                                   self.y_test,
                                   scoring=scoring,
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

        '''
        DESCRIPTION -----------------------------------

        Plot a (Tree based) model's normalized feature importance.

        PARAMETERS -------------------------------------

        show     --> number of best features to show in the plot. None for all
        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save
        display  --> wether to display the plot

        '''

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
    def plot_ROC(self, title=None, figsize=(10, 6),
                 filename=None, display=True):

        '''
        DESCRIPTION -----------------------------------

        Plot Receiver Operating Characteristics curve.

        PARAMETERS -------------------------------------

        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save
        display  --> wether to display the plot

        '''

        if self.T.task != 'binary classification':
            raise AttributeError('This method only works for binary ' +
                                 'classification tasks.')

        # Get False (True) Positive Rate
        fpr, tpr, _ = roc_curve(self.y_test, self.predict_proba_test[:, 1])

        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(fpr, tpr, lw=2, label=f'{self.name} (AUC={self.auc:.3f})')
        plt.plot([0, 1], [0, 1], lw=2, color='black', linestyle='--')

        title = 'ROC curve' if title is None else title
        plt.title(title, fontsize=BaseModel.title_fs, pad=12)
        plt.legend(loc='lower right',
                   frameon=False,
                   fontsize=BaseModel.label_fs)
        plt.xlabel('FPR', fontsize=BaseModel.label_fs, labelpad=12)
        plt.ylabel('TPR', fontsize=BaseModel.label_fs, labelpad=12)
        plt.xticks(fontsize=BaseModel.tick_fs)
        plt.yticks(fontsize=BaseModel.tick_fs)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show() if display else plt.close()

    @composed(crash, params_to_log)
    def plot_PRC(self, title=None, figsize=(10, 6),
                 filename=None, display=True):

        '''
        DESCRIPTION -----------------------------------

        Plot precision-recall curve.

        PARAMETERS -------------------------------------

        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save
        display  --> wether to display the plot

        '''

        if self.T.task != 'binary classification':
            raise AttributeError('This method only works for binary ' +
                                 'classification tasks.')

        # Get precision-recall pairs for different probability thresholds
        prec, recall, _ = precision_recall_curve(self.y_test,
                                                 self.predict_proba_test[:, 1])

        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(recall, prec, lw=2, label=f'{self.name} (AP={self.ap:.3f})')

        title = 'Precision-recall curve' if title is None else title
        plt.title(title, fontsize=BaseModel.title_fs, pad=12)
        plt.legend(loc='lower left',
                   frameon=False,
                   fontsize=BaseModel.label_fs)
        plt.xlabel('Recall', fontsize=BaseModel.label_fs, labelpad=12)
        plt.ylabel('Precision', fontsize=BaseModel.label_fs, labelpad=12)
        plt.xticks(fontsize=BaseModel.tick_fs)
        plt.yticks(fontsize=BaseModel.tick_fs)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show() if display else plt.close()

    @composed(crash, params_to_log)
    def plot_confusion_matrix(self, normalize=True, title=None,
                              figsize=(10, 6), filename=None, display=True):

        '''
        DESCRIPTION -----------------------------------

        Plot the confusion matrix in a heatmap.

        PARAMETERS -------------------------------------

        normalize --> wether to normalize the matrix
        title     --> plot's title. None for default title
        figsize   --> figure size: format as (x, y)
        filename  --> name of the file to save
        display   --> wether to display the plot

        '''

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

    @composed(crash, params_to_log)
    def save(self, filename=None):
        ''' Save model subclass to pickle file '''

        if filename is None:
            filename = 'ATOM_' + self.name
        filename = filename if filename.endswith('.pkl') else filename + '.pkl'
        pickle.dump(self, open(filename, 'wb'))
        prlog(self.longname + ' model subclass saved successfully!', self, 1)

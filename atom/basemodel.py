# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom

'''

# << ============ Import Packages ============ >>

# Standard
import numpy as np
import pandas as pd
import math
import pickle
from time import time
from datetime import datetime
from collections import deque

# Sklearn
import sklearn
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
     KFold, StratifiedKFold, cross_val_score
    )
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score, roc_auc_score,
    r2_score, jaccard_score, roc_curve, confusion_matrix, max_error,
    mean_absolute_error, mean_squared_error, mean_squared_log_error
    )

# Others
from GPyOpt.methods import BayesianOptimization
try:
    import xgboost
    import lightgbm
except ImportError:
    pass

# Plots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set(style='darkgrid', palette="GnBu_d")


# << ============ Functions ============ >>

def timing(f):
    ''' Decorator to time a function '''

    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()

        # args[0]=class instance
        prlog('Elapsed time: {:.1f} seconds'.format(end-start), args[0], 1)
        return result

    return wrapper


def prlog(string, class_, level=0, time=False):

    '''
    DESCRIPTION -----------------------------------

    Print and save output to log file.

    PARAMETERS -------------------------------------

    class_ --> class of the element
    string --> string to output
    level  --> minimum verbosity level to print
    time   --> Wether to add the timestamp to the log

    '''

    if class_.verbose > level:
        print(string)

    if class_.log is not None:
        with open(class_.log, 'a+') as file:
            if time:
                # Datetime object containing current date and time
                now = datetime.now()
                date = now.strftime("%d/%m/%Y %H:%M:%S")
                file.write(date + '\n' + string + '\n')
            else:
                file.write(string + '\n')


def set_init(data, metric, goal, log, verbose, scaled=False):
    ''' Returns BaseModel's (class) parameters as dictionary '''

    if scaled:
        params = {'X': data['X_scaled'],
                  'X_train': data['X_train_scaled'],
                  'X_test': data['X_test_scaled']}
    else:
        params = {'X': data['X'],
                  'X_train': data['X_train'],
                  'X_test': data['X_test']}

    for p in ('Y', 'Y_train', 'Y_test'):
        params[p] = data[p]

    params['metric'] = metric
    params['goal'] = goal
    params['log'] = log
    params['verbose'] = verbose

    return params


# << ============ Classes ============ >>

class BaseModel(object):

    def __init__(self, **kwargs):

        '''
        DESCRIPTION -----------------------------------

        Initialize class.

        PARAMETERS -------------------------------------

        data    --> dictionary of the data (train, test and complete set)
        metric  --> metric to maximize (or minimize) in the BO
        goal    --> classification or regression
        log     --> name of the log file
        verbose --> verbosity level (0, 1, 2)

        '''

        # List of metrics where the goal is to minimize it
        self.metric_min = ['max_error', 'MAE', 'MSE', 'MSLE']

        # List of tree-based models
        self.tree = ['Tree', 'Extra-Trees', 'RF',
                     'AdaBoost', 'GBM', 'XGB', 'LGBM']

        # List of models that don't use the Bayesian Optimization
        self.not_bo = ['GNB', 'GP']

        # Set attributes to child class
        self.__dict__.update(kwargs)

    @timing
    def BayesianOpt(self, test_size, max_iter, max_time, eps,
                    batch_size, init_points, cv, plot_bo, n_jobs):

        '''
        DESCRIPTION -----------------------------------

        Run the bayesian optmization algorithm to search for the best
        combination of hyperparameters. The function to optimize is evaluated
        either with a K-fold cross-validation on the training set or using
        a validation set.

        PARAMETERS -------------------------------------

        max_iter    --> maximum number of iterations
        max_time    --> maximum time for the BO (in seconds)
        eps         --> minimum distance between two consecutive x's
        batch_size  --> size of the batch in which the objective is evaluated
        init_points --> number of initial random tests of the BO
        cv          --> splits for the cross validation
        plot_bo     --> boolean to plot the BO's progress
        n_jobs      --> number of cores to use for parallel processing

        '''

        def animate_plot(x, y1, y2, line1, line2, ax1, ax2):

            '''
            DESCRIPTION -----------------------------------

            Plot the BO's progress.

            PARAMETERS -------------------------------------

            x    --> x values of the plot
            y    --> y values of the plot
            line --> line element of the plot

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
                ax1.set_title('Bayesian Optimization for {}'
                              .format(self.name), fontsize=16)
                ax1.set_ylabel(self.metric, fontsize=16, labelpad=12)
                ax1.set_xlim(min(self.x)-0.5, max(self.x)+0.5)

                # Second subplot
                ax2 = plt.subplot(gs[1], sharex=ax1)
                line2, = ax2.plot(x, y2, '-o', alpha=0.8)
                ax2.set_title('Metric distance between last consecutive steps'
                              .format(self.name), fontsize=16)
                ax2.set_xlabel('Step', fontsize=16, labelpad=12)
                ax2.set_ylabel('d', fontsize=16, labelpad=12)
                ax2.set_xticks(self.x)
                ax2.set_xlim(min(self.x)-0.5, max(self.x)+0.5)
                ax2.set_ylim([-0.05, 0.1])

                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.subplots_adjust(hspace=.0)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
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
            ''' Function to optimize by ht BO '''

            params = self.get_params(x)
            self.BO['params'].append(params)
            prlog(f'Parameters --> {params}', self, 2, True)

            algorithm = self.get_model(params)

            if cv == 1:
                # Split each iteration in different train and validation set
                X_subtrain, X_val, Y_subtrain, Y_val = \
                    train_test_split(self.X_train,
                                     self.Y_train,
                                     test_size=test_size,
                                     shuffle=True)

                algorithm.fit(X_subtrain, Y_subtrain)
                self.predict = algorithm.predict(X_val)

                # Calculate metric on the validation set
                output = getattr(self, self.metric)(true=Y_val)

            else:  # Use cross validation to get the output of BO
                # Set scoring metric for those that are different
                if self.metric == 'MAE':
                    scoring = 'neg_mean_absolute_error'
                elif self.metric == 'MSE':
                    scoring = 'neg_mean_squared_error'
                elif self.metric == 'MSLE':
                    scoring = 'neg_mean_squared_log_error'
                elif self.metric == 'LogLoss':
                    scoring = 'neg_log_loss'
                elif self.metric == 'AUC':
                    scoring = 'roc_auc'
                else:
                    scoring = self.metric.lower()

                if self.goal != 'regression':
                    # Folds are made preserving the % of samples for each class
                    # Use same splits for every model
                    kfold = StratifiedKFold(n_splits=cv, random_state=1)
                else:
                    kfold = KFold(n_splits=cv, random_state=1)

                # Run cross-validation (get mean of results)
                output = cross_val_score(algorithm,
                                         self.X_train,
                                         self.Y_train,
                                         cv=kfold,
                                         scoring=scoring,
                                         n_jobs=n_jobs).mean()

            # cross_val_score returns negative loss for minimizing metrics
            if self.metric in self.metric_min:
                output = -output

            # Save output of the BO and plot progress
            self.BO['score'].append(output)
            prlog(f'Evaluation --> {self.metric}: {output:.4f}', self, 2)

            if plot_bo:
                # Start to fill NaNs with encountered metric values
                if np.isnan(self.y1).any():
                    for i, value in enumerate(self.y1):
                        if math.isnan(value):
                            self.y1[i] = output
                            if i > 0:  # The first value must remain empty
                                self.y2[i] = abs(self.y1[i] - self.y1[i-1])
                            break
                else:  # If no NaNs anymore, continue deque
                    self.x.append(max(self.x)+1)
                    self.y1.append(output)
                    self.y2.append(abs(self.y1[-1] - self.y1[-2]))

                self.line1, self.line2, self.ax1, self.ax2 = \
                    animate_plot(self.x, self.y1, self.y2,
                                 self.line1, self.line2, self.ax1, self.ax2)

            return output

        # << ============ Running optimization ============ >>

        # Skip BO for GNB and GP (no hyperparameter tuning)
        if self.shortname not in self.not_bo:
            prlog(f'\n\nRunning BO for {self.name}...', self, 1)

            # Save dictionary of BO steps
            self.BO = {}
            self.BO['params'] = []
            self.BO['score'] = []

            # BO plot variables
            maxlen = 15  # Steps to show in the BO progress plot
            self.x = deque(list(range(maxlen)), maxlen=maxlen)
            self.y1 = deque([np.NaN for i in self.x], maxlen=maxlen)
            self.y2 = deque([np.NaN for i in self.x], maxlen=maxlen)
            self.line1, self.line2 = [], []  # Plot lines
            self.ax1, self.ax2 = 0, 0  # Plot axes

            # Minimize or maximize the function depending on the metric
            maximize = False if self.metric in self.metric_min else True
            # Default SKlearn or multiple random initial points
            kwargs = {}
            if init_points > 1:
                kwargs['initial_design_numdata'] = init_points
            else:
                kwargs['X'] = self.get_init_values()
            opt = BayesianOptimization(f=optimize,
                                       domain=self.get_domain(),
                                       model_update_interval=batch_size,
                                       maximize=maximize,
                                       initial_design_type='random',
                                       normalize_Y=False,
                                       num_cores=n_jobs,
                                       **kwargs)

            opt.run_optimization(max_iter=max_iter,
                                 max_time=max_time,  # No time restriction
                                 eps=eps,
                                 verbosity=True if self.verbose > 2 else False)

            if plot_bo:
                plt.close()

            # Optimal score of the BO
            bo = opt.fx_opt if self.metric in self.metric_min else -opt.fx_opt

            # Set to same shape as GPyOpt (2d-array)
            self.best_params = self.get_params(
                                        np.array(np.round([opt.x_opt], 4)))

            # Save best model (not yet fitted)
            self.best_model = self.get_model(self.best_params)

        else:
            self.best_model = self.get_model()

        # Fit the selected model on the complete training set
        self.best_model_fit = self.best_model.fit(self.X_train, self.Y_train)

        # Save best predictions and probabilities on the test set
        self.predict = self.best_model_fit.predict(self.X_test)

        # Get metric score on test set
        self.score = getattr(self, self.metric)()

        not_proba = ['LinReg', 'XGB', 'lSVM', 'kSVM']
        if self.shortname in not_proba and self.goal != 'regression':
            # Models without predict_proba() method need probs with ccv
            self.ccv = CalibratedClassifierCV(self.best_model_fit, cv='prefit')
            self.ccv.fit(self.X_test, self.Y_test)
            self.predict_proba = self.ccv.predict_proba(self.X_test)
        elif self.goal != 'regression':
            self.predict_proba = self.best_model_fit.predict_proba(self.X_test)

        # Print stats
        prlog('', self, 2)  # Print extra line
        prlog('Final statistics for {}:{:9s}'.format(self.name, ' '), self, 1)
        if self.shortname not in self.not_bo:
            prlog('Best hyperparameters: {}'.format(self.best_params), self, 1)
            prlog('Best {} on the BO: {:.4f}'.format(self.metric, bo), self, 1)
        prlog('Best {} on test set: {:.4f}'.format(self.metric, self.score),
              self, 1)

    @timing
    def bootstrap(self, n_splits=3):

        '''
        DESCRIPTION -----------------------------------

        Bootstrap the training set and test on the test set to get a
        distribution of the model´s results..

        PARAMETERS -------------------------------------

        n_splits --> number of bootstrap samples to take

        '''

        self.results = []
        for _ in range(n_splits):
            # Create samples with replacement
            sample_x, sample_y = resample(self.X_train, self.Y_train)
            print(len(np.unique(sample_x)), np.unique(sample_x))
            # Fit on bootstrapped set and predict on the independent test set
            algorithm = self.best_model.fit(sample_x, sample_y)
            pred = algorithm.predict(self.X_test)

            # Append metric result to list
            self.results.append(getattr(self, self.metric)(pred=pred))

        self.results = np.array(self.results)  # Numpy array for mean and std
        prlog('--------------------------------------------------', self, 1)
        prlog('Bootstrap {} score --> Mean: {:.4f}   Std: {:.4f}'
              .format(self.metric, self.results.mean(), self.results.std()),
              self, 1)

    # << ============ Evaluation metric functions ============ >>

    def Precision(self, pred=None, true=None):
        avg = 'binary' if self.goal == 'binary classification' else 'weighted'
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return precision_score(true, pred, average=avg)

    def Recall(self, pred=None, true=None):
        avg = 'binary' if self.goal == 'binary classification' else 'weighted'
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return recall_score(true, self.predict, average=avg)

    def F1(self, pred=None, true=None):
        avg = 'binary' if self.goal == 'binary classification' else 'weighted'
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return f1_score(true, self.predict, average=avg)

    def Jaccard(self, pred=None, true=None):
        avg = 'binary' if self.goal == 'binary classification' else 'weighted'
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return jaccard_score(true, self.predict, average=avg)

    def Accuracy(self, pred=None, true=None):
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return accuracy_score(true, self.predict)

    def AUC(self, pred=None, true=None):
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return roc_auc_score(true, self.predict)

    def MAE(self, pred=None, true=None):
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return mean_absolute_error(true, self.predict)

    def MSE(self, pred=None, true=None):
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return mean_squared_error(true, self.predict)

    def MSLE(self, pred=None, true=None):
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return mean_squared_log_error(true, self.predict)

    def R2(self, pred=None, true=None):
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return r2_score(true, self.predict)

    def max_error(self, pred=None, true=None):
        pred = self.predict if pred is None else pred
        true = self.Y_test if true is None else true
        return max_error(true, self.predict)

    # << ============ Plot functions ============ >>

    def plot_probabilities(self, target_class=1,
                           figsize=(10, 6), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot a function of the probability of the classes
        of being the target class.

        PARAMETERS -------------------------------------

        target_class --> probability of being that class (numeric)
        figsize      --> figure size: format as (x, y)
        filename     --> name of the file to save

        '''

        if self.goal == 'regression':
            raise ValueError('This method is only available for ' +
                             'classification tasks.')

        no_prob = ['XGB', 'lSVM', 'kSVM']  # Models without predict_proba
        mod = self.ccv if self.shortname in no_prob else self.best_model_fit

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        classes = list(set(self.Y))
        colors = ['r', 'b', 'g']
        for n in range(len(classes)):
            # Get features per class
            class_ = self.X[np.where(self.Y == classes[n])]
            predicted_proba = mod.predict_proba(class_)[:, target_class]
            sns.distplot(predicted_proba,
                         hist=False,
                         kde=True,
                         norm_hist=True,
                         color=colors[n],
                         kde_kws={"shade": True},
                         label='Class=' + str(classes[n]))

        plt.title('Predicted probabilities for class=' +
                  str(classes[target_class]), fontsize=16)
        plt.legend(frameon=False, fontsize=16)
        plt.xlabel('Probability', fontsize=16, labelpad=12)
        plt.ylabel('Counts', fontsize=16, labelpad=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(0, 1)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_feature_importance(self, show=20,
                                figsize=(10, 15), filename=None):
        ''' Plot a (Tree based) model's feature importance '''

        if self.shortname not in self.tree:
            raise ValueError('This method only works for tree-based ' +
                             f'models. Try one of the following: {self.tree}')

        features = self.X.columns  # Get column names

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        pd.Series(self.best_model_fit.feature_importances_,
                  index=features).nlargest(show).sort_values().plot.barh()

        plt.xlabel('Score', fontsize=16, labelpad=12)
        plt.ylabel('Features', fontsize=16, labelpad=12)
        plt.title('Importance of Features', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_ROC(self, figsize=(10, 6), filename=None):
        ''' Plot Receiver Operating Characteristics curve '''

        if self.goal != 'binary classification':
            raise ValueError('This method only works for binary ' +
                             'classification problems.')

        # Get False (True) Positive Rate
        fpr, tpr, thresholds = roc_curve(self.Y_test, self.predict_proba[:, 1])

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(fpr, tpr,
                 lw=2, color='red', label='AUC={:.3f}'.format(self.AUC()))

        plt.plot([0, 1], [0, 1], lw=2, color='black', linestyle='--')

        plt.xlabel('FPR', fontsize=16, labelpad=12)
        plt.ylabel('TPR', fontsize=16, labelpad=12)
        plt.title('ROC curve', fontsize=16)
        plt.legend(loc='lower right', frameon=False, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_confusion_matrix(self, normalize=True,
                              figsize=(10, 6), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot the confusion matrix in a heatmap.

        PARAMETERS -------------------------------------

        normalize --> boolean to normalize the matrix
        figsize   --> figure size: format as (x, y)
        filename  --> name of the file to save

        '''

        if self.goal != 'binary classification':
            raise ValueError('This method only works for binary ' +
                             'classification problems.')

        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(self.Y_test, self.predict)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['0', '1'],
               yticklabels=['0', '1'])

        # Loop over data dimensions and create text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.title(title, fontsize=16)
        plt.xlabel('Predicted label', fontsize=16, labelpad=12)
        plt.ylabel('True label', fontsize=16, labelpad=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.grid(False)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_tree(self, num_trees=0, max_depth=None,
                  rotate=False, figsize=(14, 10), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Visualize a single decision tree.

        PARAMETERS -------------------------------------

        num_trees --> number of the tree to plot (for ensembles)
        max_depth --> maximum depth to plot (None for complete tree)
        rotate    --> when set to True, orient tree left-right, not top-down
        figsize   --> figure size: format as (x, y)
        filename  --> name of file to save

        '''

        sklearn_trees = ['Tree', 'Extra-Trees', 'RF', 'AdaBoost', 'GBM']
        if self.shortname in sklearn_trees:
            fig, ax = plt.subplots(figsize=figsize)

            # A single decision tree has only one estimator
            if self.shortname != 'Tree':
                estimator = self.best_model_fit.estimators_[num_trees]
            else:
                estimator = self.best_model_fit

            sklearn.tree.plot_tree(estimator,
                                   max_depth=max_depth,
                                   rotate=rotate,
                                   rounded=True,
                                   filled=True,
                                   fontsize=14)

        elif self.shortname == 'XGB':
            xgboost.plot_tree(self.best_model_fit,
                              num_trees=num_trees,
                              rankdir='LR' if rotate else 'UT')

        elif self.shortname == 'LGBM':
            lightgbm.plot_tree(self.best_model_fit,
                               ax=ax,
                               tree_index=num_trees)

        else:
            raise ValueError('This method only works for tree-based models.' +
                             f' Try on of the following: {self.tree}')

        if filename is not None:
            plt.savefig(filename)

    def save(self, filename=None):
        ''' Save model to pickle file '''

        if filename is None:
            filename = 'ATOM_' + self.shortname
        filename = filename if filename.endswith('.pkl') else filename + '.pkl'
        pickle.dump(self.best_model_fit, open(filename, 'wb'))
        prlog('File saved successfully!', self, 1)

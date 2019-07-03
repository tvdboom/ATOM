# -*- coding: utf-8 -*-
"""

Title: AutoML pipeline
Author: Marco van den Boom
Date: 21-Jun-2019

Description
------------------------
Compare multiple ML models on the same data. All models are implemented
using the SKlearn python package. Note that the data needs to be adapted to
the models you want to use in terms of categorical/missing data.
The algorithm first starts selecting the optimal hyperparameters per model
using a Bayesian Optimization (BO) approach implemented with the GPyOpt
library. The data is fitted to the provided metric. The parameters and
domains of the algorithms are pre-set. For this, the data is split in a train
and test set with sizes of 70% and 30% respectively.
Hereafter it performs a K-fold cross validation on the complete data set
provided. This is needed to avoid having a bias towards the hyperparamters
plotted in a boxplot.
The function returns a dictionary of the model as classes, on which you
can call any metric or extra plots.

Usage
------------------------
Load module in with:
from automl import AutoML

Call the pipeline function (it returns a dictionary of the models used):
models = AutoML(X, Y,
                models=['LinReg', 'KNN', 'RF', 'GBM', MLP'],
                metric="MAE",
                ratio=0.25,
                max_iter=5,
                batch_size=1,
                cross_val=True,
                n_splits=5,
                verbose=1)

Call the plotting functions for the specific models:
models['SVM'].plot_proba(target_class=1)
models['GBM'].plot_feature_importance(save_plot='feature_importance.png')


Parameters
------------------------
X      --> array or dataframe of target features
Y      --> array or dataframe of target classes
models --> list of models to use. Possible values are:
               LinReg for linear regression (with elasticnet regularization)
               LogReg for Logistic Regression
               KNN for K_Nearest Neighbors
               Tree for a single Decision Tree
               ET for Extra-Trees
               RF for Random Forest
               AdaBoost for Adaptive Boosting
               GBM for Gradient Boosting Machine
               XGBoost for Extreme Gradient Boosting (if library is available)
               SVM for Support Vector Machine
               MLP for Multilayer Perceptron
metric --> metric on which the BO performs its fit. Possible values are:
               For binary and multiclass classification or regression:
                   max_error
                   r2
                   MAE for Mean Absolute Error
                   MSE for Mean Squared Error
                   MSLE for Mean Squared Log Error
               Only binary classification:
                   Precision
                   Recall
                   Accuracy
                   F1
                   Jaccard
                   AUC for Area Under Curve
                   LogLoss for binary cross-entropy
ratio      --> train/test split ratio
max_iter   --> Maximum number of iterations of the BO algorithm
batch_size --> Size of the batches processed in the BO before fitting
cross_val  --> Boolean wether to perform K-fold cross validation
n_splits   --> Number of splits for the K-fold cross validation
n_jobs     --> Number of cores for parallel processing
percentage --> percentage of the data to use
save_plot  --> Directory to save plot to. If None, plot is not saved
verbose    --> verbosity level of the pipeline. Only works if n_jobs=1.
               Possible values:
                  0 to print only the final stats
                  1 to print stats per algorithm as it gets fit by the BO
                  2 to print every step of the BO


Class methods (plots)
--------------------------
plot_proba(target_class, save_plot):
    Plots the probability of every class in the target variable against the
    class selected by target_class (default=2nd class). Works for multi-class.

plot_feature_importance(save_plot):
    Plots the feature importance scores. Only works with tree based
    algorithms (Tree, ET, RF, AdaBoost, GBM and XGBoost).

plot_ROC(save_plot):
    Plots the ROC curve. Works only for binary classification.

plot_confusion_matrix(normalize, save_plot):
    Plot the confusion matrix for the model. Works only for binary
    classification.

plot_decision_tree(num_trees, rotate, save_plot):
    Plot a single decision tree of a tree-based model. Only works with
    tree-based algorithms.

Class methods (metrics)
--------------------------
Call any of the possible metrics as a method. It will return the metric
(evaluated on the test set) for the best model found by the BO.
e.g. models['KNN'].AUC()        # Get AUC score for the best trained KNN
     models['AdaBoost'].MSE()   # Get MSE score for the best trained AdaBoost

Class attributes
--------------------------
The dictionary returned by the AutoML pipeline can be used to call for the
plot functions described above as well as for other handy features.
e.g. models['MLP'].best_params  # Get parameters of the MLP with highest score
     models['SVM'].best_model   # Get model of the SVM with highest score
     models['Tree'].prediction  # Get the predictions on the test set

"""

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import multiprocessing
from joblib import Parallel, delayed
import warnings

# Sklearn
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, r2_score, jaccard_score
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import max_error, log_loss, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Others
from GPyOpt.methods import BayesianOptimization
try:
    from xgboost import XGBClassifier, XGBRegressor, plot_tree
except ImportError:
    xgb_import = False
else:
    xgb_import = True

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', palette="GnBu_d")


# << ============ Functions ============ >>
def warn(*args, **kwargs):
    pass


# Turn off warnings (scikit forces anoying warnings sometimes)
warnings.warn = warn


def shuffle(a, b):
    ''' Shuffle two arrays in unison '''

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def timing(f):
    ''' Decorator to time a function '''

    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        if args[0].verbose > 0:  # args[0]=class instance
            print('Elapsed time: {:.1f} seconds'.format(end-start))
        return result
    return wrapper


def make_boxplot(algs, save_plot=False):
    ''' Plot a boxplot of the found metric results '''

    results, names = [], []
    for m in algs:
        results.append(algs[m].results)
        names.append(algs[m].shortname)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.boxplot(results)
    ax.set_xticklabels(names)

    plt.xlabel('Model', fontsize=16, labelpad=12)
    plt.ylabel(next(iter(algs.values())).metric,
               fontsize=16,
               labelpad=12)
    plt.title('Model comparison', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_plot is not None:
        plt.savefig(save_plot)
    plt.show()


def AutoML(X, Y, models=None, metric=None, ratio=0.3,
           max_iter=5, batch_size=1, cross_val=True, n_splits=5,
           n_jobs=1, percentage=100, save_plot=None, verbose=1):

    '''
    DESCRIPTION -----------------------------------

    Run a bayesian optmization algorithm for different
    models in the pipeline to get the best hyperparameters
    to perform a kfold cross validation.

    ARGUMENTS -------------------------------------

    X, Y       --> data features and targets (array or dataframe)
    models     --> list of models to use
    metric     --> metric to perform evaluation on
    ratio      --> train/test split ratio
    max_iter   --> maximum number of iterations of the BO
    batch_size --> batch size for the BO algorithm
    cross_val  --> perform kfold cross validation
    n_splits   --> number of splits for the stratified kfold
    n_jobs     --> number of cores for parallel processing
    percentage --> percentage of data to use
    save_plot  --> directory to save plot to
    verbose    --> verbosity level

    RETURNS ----------------------------------------

    Dictionary of models.

    '''

    def run_model(data, model, metric, goal,
                  max_iter, batch_size, cross_val, n_splits, verbose):
        ''' Run every independent model '''

        algs[model] = algorithm(data=data,
                                model=model,
                                metric=metric,
                                goal=goal,
                                verbose=verbose)

        algs[model].Bayesian_Optimization(max_iter=max_iter,
                                          batch_size=batch_size)
        if cross_val:
            algs[model].cross_val_evaluation(n_splits=n_splits)

        return algs

    # << ============ Parameters tests ============ >>

    # Set algorithm goal (regression, binaryclass or multiclass)
    classes = len(set(Y))
    if classes < 2:
        raise ValueError('There are not enough target values.')
    elif classes == 2:
        print('Algorithm set to binary classification.')
        goal = 'binary classification'
    elif 2 < classes < 0.1*len(Y):
        print('Algorithm set to multiclass classification.' +
              f' Number of classes: {classes}')
        goal = 'multiclass classification'
    else:
        print('Algorithm set to regression.')
        goal = 'regression'

    # Check validity models
    model_list = ['LinReg', 'LogReg', 'KNN', 'Tree', 'ET', 'RF',
                  'AdaBoost', 'GBM', 'XGBoost', 'SVM', 'MLP']
    final_models = []  # Final list of models to be used
    if models is None:  # Use all possible models (default)
        final_models = model_list.copy()
    else:
        # If only one model, make list for enumeration
        if isinstance(models, str):
            models = [models]
        for ix, m in enumerate(models):
            # Compare strings case insensitive
            if m.lower() not in map(str.lower, model_list):
                print(f"Unknown model {m}. Removed from pipeline.")
            else:
                for n in model_list:
                    if m.lower() == n.lower():
                        final_models.append(n)
                        break

    # Check if XGBoost is available
    while 'XGBoost' in final_models and not xgb_import:
        print("Unable to import XGBoost. Removing model from pipeline.")
        final_models.remove('XGBoost')

    # Linear regression can't perform classification
    if 'LinReg' in final_models and goal != 'regression':
        print("Linear Regression can't solve classification problems." +
              " Removing model from pipeline.")
        final_models.remove('LinReg')

    # Logistic regression can't perform regression
    if 'LogReg' in final_models and goal == 'regression':
        print("Logistic Regression can't solve regression problems." +
              " Removing model from pipeline.")
        final_models.remove('LogReg')

    # Check if there are still valid models
    if len(final_models) == 0:
        raise ValueError(f"No models found. Try {model_list}")

    print(f'Models in pipeline: {final_models}')

    # Set default metric
    if metric is None and goal == 'binary classification':
        metric = 'F1'
    elif metric is None:
        metric = 'MSE'

    # Check validity metric
    metric_class = ['Precision', 'Recall', 'Accuracy', 'F1', 'AUC',
                    'LogLoss', 'Jaccard']
    metric_reg = ['r2', 'max_error', 'MAE', 'MSE', 'MSLE']
    for m in metric_class + metric_reg:
        if metric.lower() == m.lower():  # Compare strings case insensitive
            metric = m

    if metric not in metric_class + metric_reg:
        raise ValueError('Unknown metric. Try one of {}.'
                         .format(metric_class if goal ==
                                 'binary classification' else metric_reg))
    elif metric not in metric_reg and goal != 'binary classification':
        raise ValueError("Invalid metric for {}. Try one of {}."
                         .format(goal, metric_reg))

    # Check number of cores for multiprocessing
    n_cores = multiprocessing.cpu_count()
    if n_jobs > n_cores:
        print('Warning! n_jobs was reduced to the number of cores available.')
        n_jobs = n_cores
    elif n_jobs < 1:
        print("Warning! Value of n_jobs can't be {}. Automatically set to 1."
              .format(n_jobs))
        n_jobs = 1

    # << ============ Core ============ >>

    data = {}  # Dictionary of data (complete, train, test)
    if percentage < 100:
        X, Y = shuffle(X, Y)  # Shuffle X and Y

        # Return percentage of data
        data['X'] = X[0:int(len(X)*percentage/100)]
        data['Y'] = Y[0:int(len(Y)*percentage/100)]
    else:
        data['X'] = X
        data['Y'] = Y

    print('\nData stats ==================>')
    print(f'Feature shape: {X.shape}\nTarget shape: {Y.shape}')

    # Split train and test for the BO
    data['X_train'], data['X_test'], data['Y_train'], data['Y_test'] = \
        train_test_split(data['X'],
                         data['Y'],
                         test_size=ratio,
                         random_state=1)

    print('Size of training set: {}\nSize of validation set: {}'
          .format(len(data['X_train']), len(data['X_test'])))

    # Loop over models to get score
    algs = {}  # Dictionary of algorithms (to be returned by function)
    if n_jobs > 1:
        print(f'\nParallel processing with {n_jobs} cores.')
        algs = Parallel(n_jobs=n_jobs)(delayed(run_model)
                                       (data,
                                        model,
                                        metric,
                                        goal,
                                        max_iter,
                                        batch_size,
                                        cross_val,
                                        n_splits,
                                        0) for model in tqdm(final_models))

        # Parallel returns list of dictionaries --> convert to one dict
        algs = {k: v for x in algs for k, v in x.items()}

    else:
        for model in final_models:
            algs = run_model(data, model, metric, goal, max_iter, batch_size,
                             cross_val, n_splits, verbose)

    if cross_val:
        max_len = max([len(algs[m].name) for m in final_models])

        # Print final results
        print('\n\nFinal stats ================>>')
        print(f'Target metric: {metric}')
        print('------------------------------------')
        for m in final_models:
            print('{0:{1}s} --> Mean: {2:.3f}   Std: {3:.3f}'
                  .format(algs[m].name,
                          max_len,
                          algs[m].results.mean(),
                          algs[m].results.std()))

        make_boxplot(algs, save_plot=save_plot)

    return algs  # Return dictionary of models


# << ============ Classes ============ >>

class algorithm(object):

    def __init__(self, data, model, metric, goal, verbose):

        '''
        DESCRIPTION -----------------------------------

        Initialize class.

        ARGUMENTS -------------------------------------

        data    --> data features and targets
        model   --> model type
        metric  --> metric to maximize (or minimize) in the BO
        goal    --> classification or regression
        verbose --> verbosity level (0, 1, 2)

        '''

        # List of metrics where the goal is to minimize
        # (Largely same as the regression metrics)
        self.metric_min = ['max_error', 'MAE', 'MSE', 'MSLE']

        # List of tree-based models
        self.tree = ['Tree', 'Extra-Trees', 'RF', 'AdaBoost', 'GBM', 'XGBoost']

        # Define data groups
        self.X = data['X']
        self.Y = data['Y']
        self.X_train = data['X_train']
        self.Y_train = data['Y_train']
        self.X_test = data['X_test']
        self.Y_test = data['Y_test']

        # Set class attributes
        self.metric = metric
        self.goal = goal
        self.verbose = verbose

        # Set model
        if model == 'KNN':
            self.model = KNearest_Neighbors(self.goal)
        elif model == 'Tree':
            self.model = Decision_Tree(self.goal)
        elif model == 'ET':
            self.model = ExtraTrees(self.goal)
        elif model == 'RF':
            self.model = Random_Forest(self.goal)
        elif model == 'AdaBoost':
            self.model = AdaBoost(self.goal)
        elif model == 'GBM':
            self.model = GradientBoost(self.goal)
        elif model == 'XGBoost':
            self.model = XGBoost(self.goal)
        else:
            # Normalize features to mean=0, std=1
            scaler = StandardScaler().fit(self.X_train)
            self.X_train = scaler.transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            self.X = StandardScaler().fit_transform(self.X)

            if model == 'LinReg':
                self.model = Linear_Regression(self.goal)
            elif model == 'LogReg':
                self.model = Logistic_Regression(self.goal)
            elif model == 'SVM':
                self.model = Support_Vector_Machine(self.goal)
            elif model == 'MLP':
                self.model = Multilayer_Perceptron(self.goal)

        print(self.X)
        self.X = np.array(self.X)  # Fix for plot_proba for non normalized algs
        print(self.X)
        self.name, self.shortname = self.model.get_name()

    @timing
    def Bayesian_Optimization(self, max_iter=50, batch_size=1):

        '''
        DESCRIPTION -----------------------------------

        Run the bayesian optmization algorithm.

        ARGUMENTS -------------------------------------

        max_iter   --> maximum number of iterations
        batch_size --> size of BO bacthes

        '''

        def optimize(x):
            ''' Function to optimize '''

            # Call model object
            params = self.model.get_params(x)
            if self.verbose > 1:
                print(f'Parameters --> {params}')
            self.algorithm = self.model.get_model(params)
            self.algorithm.fit(self.X_train, self.Y_train)
            self.prediction = self.algorithm.predict(self.X_test)

            out = getattr(self, self.metric)

            if self.verbose > 1:
                print('Evaluation --> {}: {:.4f}'.format(self.metric, out()))

            return out()

        # << ============ Running optimization ============ >>
        if self.verbose == 1:
            print(f'\nRunning BO for {self.name}...', end='\r')
        elif self.verbose == 2:
            print(f'\nRunning BO for {self.name}...')

        # Minimize or maximize the function depending on the metric
        maximize = False if self.metric in self.metric_min else True

        opt = BayesianOptimization(f=optimize,
                                   domain=self.model.get_domain(),
                                   batch_size=batch_size,
                                   maximize=maximize,
                                   X=self.model.get_init_values(),
                                   normalize_Y=False)

        opt.run_optimization(max_iter=max_iter,
                             verbosity=True if self.verbose > 1 else False)

        # Set to same shape as GPyOpt (2d-array)
        self.best_params = \
            self.model.get_params(np.array(np.round([opt.x_opt], 3)))

        # Save best model (not yet fitted)
        self.best_model = self.model.get_model(self.best_params)

        # Save best predictions
        self.model_fit = self.best_model.fit(self.X_train, self.Y_train)
        self.prediction = self.model_fit.predict(self.X_test)

        if self.metric in self.metric_min:
            opt.fx_opt = -opt.fx_opt

        # Print stats
        if self.verbose > 0:
            if self.verbose > 1:
                print('\nFinal Statistics for {}:{:9s}'.format(self.name, ' '))
            else:
                print('Final Statistics for {}:{:9s}'.format(self.name, ' '))
            print('Optimal parameters BO: {}'.format(self.best_params))
            print('Optimal {} score: {:.3f}'.format(self.metric, -opt.fx_opt))

    @timing
    def cross_val_evaluation(self, n_splits=5):
        ''' Run the kfold cross validation '''

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
            # Folds are made preserving the % of samples for each class.
            kfold = StratifiedKFold(n_splits=n_splits, random_state=1)
        else:
            kfold = KFold(n_splits=n_splits, random_state=1)

        self.results = cross_val_score(self.model.get_model(self.best_params),
                                       self.X,
                                       self.Y,
                                       cv=kfold,
                                       scoring=scoring)

        # cross_val_score returns negative loss for minimizing metrics
        if self.metric in self.metric_min:
            self.results = -self.results

        if self.verbose > 0:
            print('--------------------------------------------------')
            print('Cross_val {} score --> Mean: {:.3f}   Std: {:.3f}'
                  .format(self.metric,
                          self.results.mean(),
                          self.results.std()))

    # << ============ Evaluation metric functions ============ >>
    def Precision(self):
        return precision_score(self.Y_test,
                               self.prediction,
                               average='binary' if self.goal ==
                               'binary classification' else 'macro')

    def Recall(self):
        return recall_score(self.Y_test,
                            self.prediction,
                            average='binary' if self.goal ==
                            'binary classification' else 'macro')

    def Accuracy(self):
        return accuracy_score(self.Y_test,
                              self.prediction,
                              average='binary' if self.goal ==
                              'binary classification' else 'macro')

    def F1(self):
        return f1_score(self.Y_test,
                        self.prediction,
                        average='binary' if self.goal ==
                        'binary classification' else 'macro')

    def Jaccard(self):
        return jaccard_score(self.Y_test,
                             self.prediction,
                             average='binary' if self.goal ==
                             'binary classification' else 'macro')

    def AUC(self):
        return roc_auc_score(self.Y_test, self.prediction)

    def LogLoss(self):
        return log_loss(self.Y_test, self.prediction)

    def MAE(self):
        return mean_absolute_error(self.Y_test, self.prediction)

    def MSE(self):
        return mean_squared_error(self.Y_test, self.prediction)

    def MSLE(self):
        return mean_squared_log_error(self.Y_test, self.prediction)

    def r2(self):
        return r2_score(self.Y_test, self.prediction)

    def max_error(self):
        return max_error(self.Y_test, self.prediction)

    # << ============ Plot functions ============ >>
    def plot_proba(self, target_class=1, save_plot=None):

        '''
        DESCRIPTION -----------------------------------

        Plot a function of the probability of the classes
        of being the target class.

        ARGUMENTS -------------------------------------

        target_class --> probability of being that class (numeric)

        '''

        if self.goal == 'regression':
            raise ValueError('This method only works for ' +
                             'classification problems.')

        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate new probabilities more accurate with cv
        clf = CalibratedClassifierCV(self.best_model, cv=3).fit(self.X, self.Y)

        classes = list(set(self.Y))
        colors = ['r', 'b', 'g']
        for n in range(len(classes)):
            # Get features per class
            cl = self.X[np.where(self.Y == classes[n])]
            pred = clf.predict_proba(cl)[:, target_class]

            sns.distplot(pred,
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
        if save_plot is not None:
            plt.savefig(save_plot)
        plt.show()

    def plot_feature_importance(self, save_plot=None):
        ''' Plot a (Tree based) model's feature importance '''

        if self.shortname not in self.tree:
            raise ValueError('This method only works for tree-based ' +
                             f'models. Try one of the following: {self.tree}')

        features = list(self.X_train)
        fig, ax = plt.subplots(figsize=(10, 15))
        pd.Series(self.model_fit.feature_importances_,
                  features).sort_values().plot.barh()

        plt.xlabel('Features', fontsize=16, labelpad=12)
        plt.ylabel('Score', fontsize=16, labelpad=12)
        plt.title('Importance of Features', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if save_plot is not None:
            plt.savefig(save_plot)
        plt.show()

    def plot_ROC(self, save_plot=None):
        ''' Plot ROC '''

        if self.goal != 'binary classification':
            raise ValueError('This method only works for binary ' +
                             'classification problems.')

        # Calculate new probabilities more accurate with cv
        clf = CalibratedClassifierCV(self.best_model, cv=3).fit(self.X, self.Y)
        pred = clf.predict_proba(self.X)[:, 1]

        # Get False (True) Positive Rate
        fpr, tpr, thresholds = roc_curve(self.Y, pred)

        fig, ax = plt.subplots(figsize=(10, 6))
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
        if save_plot is not None:
            plt.savefig(save_plot)
        plt.show()

    def plot_confusion_matrix(self, normalize=True, save_plot=None):

        '''
        DESCRIPTION -----------------------------------

        Plot the confusion matrix in a heatmap.

        ARGUMENTS -------------------------------------

        normalize --> boolean to normalize the matrix
        save_plot --> boolean to save the plot

        '''

        if self.goal != 'binary classification':
            raise ValueError('This method only works for binary ' +
                             'classification problems.')

        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(self.Y_test, self.prediction)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 6))
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
        if save_plot is not None:
            plt.savefig(save_plot)
        plt.show()

    def plot_decision_tree(self, num_trees=0, rotate=False, save_plot=None):

        '''
        DESCRIPTION -----------------------------------

        Visualize a single decision tree.

        ARGUMENTS -------------------------------------

        num_trees --> number of the tree to plot (for ensembles)
        rotate    --> when set to True, orient tree left-right, not top-down.

        '''

        sklearn_trees = ['Tree', 'Extra-Trees', 'RF', 'AdaBoost', 'GBM']
        if self.shortname in sklearn_trees:
            fig, ax = plt.subplots(figsize=(14, 10))

            # A single decision tree has only one estimator
            if self.shortname != 'Tree':
                estimator = self.model_fit.estimators_[num_trees]
            else:
                estimator = self.model_fit

            sklearn.tree.plot_tree(estimator,
                                   rotate=rotate,
                                   filled=True,
                                   fontsize=14)

        elif self.shortname == 'XGBoost':
            plot_tree(self.model_fit,
                      num_trees=num_trees,
                      rankdir='LR' if rotate else 'UT')
        else:
            raise ValueError('This method only works for tree-based models.' +
                             f' Try on of the following: {self.tree}')

        if save_plot is not None:
            plt.savefig(save_plot)


class Linear_Regression(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'Linear Regression', 'LinReg'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'max_iter': int(x[0, 0]),
                  'alpha': float(np.round(x[0, 1], 2)),
                  'l1_ratio': float(np.round(x[0, 2], 1))}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return ElasticNet(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'max_iter',
                 'type': 'continuous',
                 'domain': (100, 500)},
                {'name': 'alpha',
                 'type': 'continuous',
                 'domain': (0.01, 5)},
                {'name': 'l1_ratio',
                 'type': 'continuous',
                 'domain': (0, 1)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[250, 1.0, 0.5]])
        return values


class Logistic_Regression(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'Logistic Regression', 'LogReg'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        regularization = ['l1', 'l2', 'elasticnet', 'none']
        penalty = regularization[int(x[0, 2])]
        params = {'max_iter': int(x[0, 0]),
                  'C': float(np.round(x[0, 1], 1)),
                  'penalty': penalty}

        if penalty == 'elasticnet':  # Add extra parameter l1_ratio
            params['l1_ratio'] = float(np.round(x[0, 3], 1))

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return LogisticRegression(solver='saga', multi_class='auto', **params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'max_iter',
                 'type': 'continuous',
                 'domain': (100, 500)},
                {'name': 'C',
                 'type': 'continuous',
                 'domain': (0.1, 5)},
                {'name': 'penalty',
                 'type': 'discrete',
                 'domain': (0, 1, 2, 3)},
                {'name': 'l1_ratio',
                 'type': 'discrete',
                 'domain': (0.1, 0.3, 0.5, 0.7, 0.9)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[250, 1.0, 1, 0.5]])
        return values


class KNearest_Neighbors(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'K-Nearest Neighbors', 'KNN'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'n_neighbors': int(x[0, 0])}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return KNeighborsClassifier(**params)
        else:
            return KNeighborsRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_neighbors',
                 'type': 'continuous',
                 'domain': (2, 50)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[5]])
        return values


class Decision_Tree(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'Decision Tree', 'Tree'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        max_features = ['sqrt', 'log2', None]
        params = {'max_depth': int(x[0, 0]),
                  'max_features': max_features[int(x[0, 1])]}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return DecisionTreeClassifier(**params)
        else:
            return DecisionTreeRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'max_depth',
                 'type': 'continuous',
                 'domain': (2, 12)},
                {'name': 'max_features',
                 'type': 'discrete',
                 'domain': (0, 1, 2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[4, 0]])
        return values


class ExtraTrees(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'Extremely Randomized Trees', 'Extra-Trees'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        max_features = ['sqrt', 'log2', None]
        params = {'n_estimators': int(x[0, 0]),
                  'max_depth': int(x[0, 1]),
                  'max_features': max_features[int(x[0, 2])]}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return ExtraTreesClassifier(**params)
        else:
            return ExtraTreesRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'continuous',
                 'domain': (20, 500)},
                {'name': 'max_depth',
                 'type': 'continuous',
                 'domain': (2, 12)},
                {'name': 'max_features',
                 'type': 'discrete',
                 'domain': (0, 1, 2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[50, 4, 0]])
        return values


class Random_Forest(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'Random Forest', 'RF'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        max_features = ['sqrt', 'log2', None]
        params = {'n_estimators': int(x[0, 0]),
                  'max_depth': int(x[0, 1]),
                  'max_features': max_features[int(x[0, 2])]}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'continuous',
                 'domain': (20, 500)},
                {'name': 'max_depth',
                 'type': 'continuous',
                 'domain': (2, 12)},
                {'name': 'max_features',
                 'type': 'discrete',
                 'domain': (0, 1, 2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[50, 4, 0]])
        return values


class AdaBoost(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'Adaptive Boosting', 'AdaBoost'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': float(np.round(x[0, 1], 3))}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return AdaBoostClassifier(**params)
        else:
            return AdaBoostRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'continuous',
                 'domain': (20, 500)},
                {'name': 'learning_rate',
                 'type': 'continuous',
                 'domain': (0.01, 1)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[50, 1]])
        return values


class GradientBoost(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'Gradient Boosting Machine', 'GBM'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': float(np.round(x[0, 1], 3)),
                  'subsample': float(np.round(x[0, 2], 1)),
                  'max_depth': int(x[0, 3])}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return GradientBoostingClassifier(**params)
        else:
            return GradientBoostingRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'continuous',
                 'domain': (20, 500)},
                {'name': 'learning_rate',
                 'type': 'continuous',
                 'domain': (0.01, 1)},
                {'name': 'subsample',
                 'type': 'discrete',
                 'domain': (0.4, 0.6, 0.8, 1.0)},
                {'name': 'max_depth',
                 'type': 'discrete',
                 'domain': (2, 3, 4, 6, 8)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[50, 1, 0.8, 3]])
        return values


class XGBoost(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'Extreme Gradient Boosting', 'XGBoost'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': float(np.round(x[0, 1], 3)),
                  'min_child_weight': int(x[0, 2]),
                  'reg_alpha': int(x[0, 3]),
                  'reg_lambda': int(x[0, 4]),
                  'subsample': float(np.round(x[0, 5], 1)),
                  'max_depth': int(x[0, 6])}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return XGBClassifier(**params, verbosity=0)
        else:
            return XGBRegressor(**params, verbosity=0)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'continuous',
                 'domain': (20, 500)},
                {'name': 'learning_rate',
                 'type': 'continuous',
                 'domain': (0.01, 1)},
                {'name': 'min_child_weight',
                 'type': 'continuous',
                 'domain': (1, 7)},
                {'name': 'reg_alpha',
                 'type': 'continuous',
                 'domain': (0, 50)},
                {'name': 'reg_lambda',
                 'type': 'continuous',
                 'domain': (0, 50)},
                {'name': 'subsample',
                 'type': 'discrete',
                 'domain': (0.4, 0.6, 0.8, 1.0)},
                {'name': 'max_depth',
                 'type': 'discrete',
                 'domain': (2, 3, 4, 6, 8)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[100, 0.1, 1, 0, 1, 1.0, 3]])
        return values


class Support_Vector_Machine(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'Support Vector Machine', 'SVM'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'C': float(np.round(x[0, 0], 1)),
                  'max_iter': int(x[0, 1])}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return LinearSVC(**params)
        else:
            return LinearSVR(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'C',
                 'type': 'continuous',
                 'domain': (0.1, 5)},
                {'name': 'max_iter',
                 'type': 'continuous',
                 'domain': (1e3, 1e4)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[1.0, 5e3]])
        return values


class Multilayer_Perceptron(object):

    def __init__(self, goal):
        self.goal = goal

    def get_name(self):
        return 'Multilayer Perceptron', 'MLP'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        # Set the number of neurons per layer
        n1, n2, n3 = int(x[0, 0]), int(x[0, 1]), int(x[0, 2])
        if n2 == 0:
            layers = (n1,)
        elif n3 == 0:
            layers = (n1, n2)
        else:
            layers = (n1, n2, n3)

        params = {'hidden_layer_sizes': layers,
                  'alpha': float(np.round(x[0, 3], 4)),
                  'learning_rate_init': float(np.round(x[0, 4], 3)),
                  'max_iter': int(x[0, 5]),
                  'batch_size': int(x[0, 6])}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return MLPClassifier(**params)
        else:
            return MLPRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'hidden_layer_1',
                 'type': 'continuous',
                 'domain': (10, 100)},
                {'name': 'hidden_layer_2',
                 'type': 'continuous',
                 'domain': (0, 100)},
                {'name': 'hidden_layer_3',
                 'type': 'continuous',
                 'domain': (0, 100)},
                {'name': 'alpha',
                 'type': 'continuous',
                 'domain': (0.0001, 0.1)},
                {'name': 'learning_rate_init',
                 'type': 'continuous',
                 'domain': (0.001, 0.1)},
                {'name': 'max_iter',
                 'type': 'continuous',
                 'domain': (100, 500)},
                {'name': 'batch_size',
                 'type': 'discrete',
                 'domain': (1, 8, 32, 64)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[20, 0, 0, 0.0001, 0.001, 200, 32]])
        return values

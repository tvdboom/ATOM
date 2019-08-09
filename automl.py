# -*- coding: utf-8 -*-
"""

Title: AutoML pipeline
Author: tvdboom

Description
------------------------
Compare multiple machine learning models on the same data. All models are
implemented using the SKlearn python package (https://scikit-learn.org/stable/)
except for the Extreme Gradient Booster which is implemented with XGBoost
(https://xgboost.readthedocs.io/en/latest/). Note that the data needs to be
adapted to the models you want to use in terms of categorical/missing data.
The pipeline does not do the data pre-processing for you!
The algorithm first starts selecting the optimal hyperparameters per model
using a Bayesian Optimization (BO) approach implemented with the GPyOpt
library (https://sheffieldml.github.io/GPyOpt/). The data is fitted to the
selected metric. The tunable parameters and their respective domains are
pre-set.
Hereafter, the pipleine performs a K-fold cross validation on the complete
data set provided. This is needed to avoid having a bias towards the
hyperparameters selected by the BO and provides a better statistical overview
of the final results.
The class contains the models as subclasses, on which you can call extra
methods and attributes.


Usage
------------------------
Load module in with:
from automl import AutoML

Call the pipeline class:
aml = AutoML(models=['LinReg', 'KNN', 'RF', 'GBM', MLP'],
             metric="MAE",
             ratio=0.25,
             max_iter=5,
             batch_size=1,
             n_splits=5,
             verbose=1)

Run the pipeline:
aml.fit(X, Y)

Make plots and analyse results:
aml.boxplot('boxplot.png')


Class Parameters (default)
------------------------
models --> list of models to use. Possible values are: (all)
               GNB for Gaussian Naïve Bayes
               MNB for Multinomial Naïve Bayes
               BNB for Bernoulli Naïve Bayes
               LinReg for linear regression (with elasticnet regularization)
               LogReg for Logistic Regression
               LDA for Linear Discriminant Analysis
               QDA for Quadratic Discriminant Analysis
               KNN for K_Nearest Neighbors
               Tree for a single Decision Tree
               ET for Extra-Trees
               RF for Random Forest
               AdaBoost for Adaptive Boosting
               GBM for Gradient Boosting Machine
               XGBoost for Extreme Gradient Boosting (if library is available)
               lSVM for Linear Support Vector Machine
               kSVM for Kernel Support Vector Machine
               PA for Passive Aggressive
               SGD for Stochastic Gradient Descent
               MLP for Multilayer Perceptron
metric --> metric on which the BO performs its fit. Possible values are:
               For binary and multiclass classification or regression:
                   max_error
                   r2
                   MAE for Mean Absolute Error
                   MSE for Mean Squared Error (regression or multiclass)
                   MSLE for Mean Squared Log Error
               Only binary classification:
                   Precision
                   Recall
                   Accuracy
                   F1 (binary classification)
                   Jaccard
                   AUC for Area Under Curve
                   LogLoss for binary cross-entropy
ratio      --> train/test split ratio for BO (0.3)
max_iter   --> Maximum number of iterations of the BO (15)
batch_size --> Size of the batches processed in the BO before fitting (1)
cv         --> Boolean wether to perform K-fold cross validation (True)
n_splits   --> Number of splits for the K-fold cross validation (5)
n_jobs     --> Number of CPUs for parallel processing (1)
                  1 to not run the pipeline in parallel
                  -1 will use as many cores as available
                  below -1, (n_cpus + 1 + n_jobs) are used
log        --> name of the log file, None to not save any log (AutoML_log.txt)
verbose    --> verbosity level of the pipeline. Only works if n_jobs=1. (1)
               Possible values:
                  0 to print only the final stats
                  1 to print stats per algorithm as it gets fit by the BO
                  2 to print every step of the BO

Fit parameters
--------------------------
X          --> array or dataframe of target features
Y          --> array or dataframe of target classes
percentage --> percentage of the data to use for the BO


Class methods (plots)
--------------------------
plot_proba(target_class, filename):
    Plots the probability of every class in the target variable against the
    class selected by target_class (default=2nd class). Works for multi-class.

plot_feature_importance(filename):
    Plots the feature importance scores. Only works with tree based
    algorithms (Tree, ET, RF, AdaBoost, GBM and XGBoost).

plot_ROC(filename):
    Plots the ROC curve. Works only for binary classification.

plot_confusion_matrix(normalize, filename):
    Plot the confusion matrix for the model. Works only for binary
    classification.

plot_decision_tree(num_trees, rotate, filename):
    Plot a single decision tree of a tree-based model. Only works with
    tree-based algorithms.


Class methods (metrics)
--------------------------
Call any of the possible metrics as a method. It will return the metric
(evaluated on the test set) for the best model found by the BO.
e.g. aml.KNN.AUC()        # Get AUC score for the best trained KNN
     aml.AdaBoost.MSE()   # Get MSE score for the best trained AdaBoost

Class attributes
--------------------------
The model subclasses of the AutoML class can be used to call for the
plot functions described above as well as for other handy features.
e.g. aml.MLP.best_params  # Get parameters of the MLP with highest score
     aml.SVM.best_model   # Get model of the SVM with highest score
     aml.Tree.prediction  # Get the predictions on the test set
     aml.KNN.BO           # Dictionary for the score and params of steps in BO
     aml.GBM.errors       # If the model didn't work, this shows the error

"""

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
import warnings

# Sklearn
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV

# Metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, r2_score, jaccard_score
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import max_error, log_loss, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# Models
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR
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
try:
    import seaborn as sns
    sns.set(style='darkgrid', palette="GnBu_d")
except ImportError:
    sns_import = False
else:
    sns_import = True


# << ============ Functions ============ >>
def warn(*args, **kwargs):
    pass


# Turn off warnings (scikit forces anoying warnings sometimes)
warnings.warn = warn


def timing(f):
    ''' Decorator to time a function '''

    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()

        # args[0]=class instance
        prlog('Elapsed time: {:.1f} seconds'.format(end-start), args[0], 0)
        return result

    return wrapper


def prlog(string, cl, level, time=False):

    '''
    DESCRIPTION -----------------------------------

    Print and save output to log file.

    ARGUMENTS -------------------------------------

    cl     --> class of the element
    string --> string to output
    level  --> minimum verbosity level to print
    time   --> Wether to add the timestamp to the log

    '''

    if cl.verbose > level:
        print(string)

    if cl.log is not None:
        # Datetime object containing current date and time
        now = datetime.now()
        date = now.strftime("%d/%m/%Y %H:%M:%S")

        # Not using logging because, although this is slower, logging doesn't
        # work when running the code from jupyter notebook or spyder
        f = open(cl.log, 'a+')
        if time:
            f.write(date + '\n' + string + '\n')
        else:
            f.write(string + '\n')
        f.close()


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

class AutoML(object):

    def __init__(self, models=None, metric=None, ratio=0.3, max_iter=5,
                 batch_size=1, cv=True, n_splits=5, n_jobs=1,
                 log='AutoML_log.txt', verbose=1):

        '''
        DESCRIPTION -----------------------------------

        Initialize AutoML class.

        ARGUMENTS -------------------------------------

        models     --> list of models to use
        metric     --> metric to perform evaluation on
        ratio      --> train/test split ratio
        max_iter   --> maximum number of iterations of the BO
        batch_size --> batch size for the BO algorithm
        cv         --> perform kfold cross validation
        n_splits   --> number of splits for the stratified kfold
        n_jobs     --> number of cores for parallel processing
        log        --> keep log file
        verbose    --> verbosity level (0, 1 or 2)

        '''

        # Set attributes to class (set to default if input is invalid)
        self.models = models
        self.metric = metric
        self.ratio = ratio if 0 < ratio < 1 else 0.3
        self.max_iter = int(max_iter) if max_iter > 0 else 5
        self.batch_size = int(batch_size)
        self.cv = cv
        self.n_splits = int(n_splits) if n_splits > 0 else 5
        self.verbose = verbose if verbose in (0, 1, 2) else 1

        # Initialize log file
        self.log = log if log is None or log.endswith('.txt') else log + '.txt'

        # Save model erros (if any)
        self.errors = ''

        # Check number of cores for multiprocessing
        n_cores = multiprocessing.cpu_count()
        self.n_jobs = int(n_jobs)  # Make sure it is an integer
        if self.n_jobs > n_cores:
            prlog('\nWarning! No {} cores available. n_jobs reduced to {}.'
                  .format(self.n_jobs, n_cores), self, -1)
            self.n_jobs = n_cores

        elif self.n_jobs == 0:
            prlog("\nWarning! Value of n_jobs can't be {}. Using 1 core."
                  .format(self.n_jobs), self, -1)
            self.n_jobs = 1

        else:
            if self.n_jobs == -1:
                self.n_jobs = n_cores
            elif n_jobs < -1:
                self.n_jobs = n_cores + 1 + self.n_jobs

            # Final check
            if self.n_jobs < 1 or self.n_jobs > n_cores:
                raise ValueError('Invalid value for n_jobs!')

    def fit(self, X, Y, percentage=100):

        '''
        DESCRIPTION -----------------------------------

        Run the pipeline.

        ARGUMENTS -------------------------------------

        X          --> data features: pandas dataframe or array
        Y          --> data targets: array
        percentage --> percentage of data to use

        '''

        # << ============ Inner Function ============ >>

        def run_model(data, model, metric, goal, max_iter, batch_size,
                      cv, n_splits, log, verbose):
            ''' Run every independent model. Needed for parallelization. '''

            dct = {'GNB': Gaussian_Naive_Bayes,
                   'MNB': Multinomial_Naive_Bayes,
                   'BNB': Bernoulli_Naive_Bayes,
                   'LinReg': Linear_Regression,
                   'LogReg': Logistic_Regression,
                   'LDA': Linear_Discriminant_Analysis,
                   'QDA': Quadratic_Discriminant_Analysis,
                   'KNN': K_Nearest_Neighbors,
                   'Tree': Decision_Tree,
                   'ET': Extra_Trees,
                   'RF': Random_Forest,
                   'AdaBoost': Adaptive_Boosting,
                   'GBM': Gradient_Boosting_Machine,
                   'XGBoost': Extreme_Gradient_Boosting,
                   'lSVM': Linear_Support_Vector_Machine,
                   'kSVM': Kernel_Support_Vector_Machine,
                   'PA': Passive_Aggressive,
                   'SGD': Stochastic_Gradient_Descent,
                   'MLP': Multilayer_Perceptron}

            # Set model class
            setattr(self, model, dct[model](data, metric, goal, log, verbose))

            try:  # If errors occure, just skip the model
                if model != 'GNB':  # GNB has no parameters to tune
                    getattr(self, model).BayesianOpt(max_iter, batch_size)
                if cv:
                    getattr(self, model).cross_val_evaluation(n_splits)

            except Exception as ex:
                prlog('Exception encountered while running '
                      + f'the {model} model. Removing model from pipeline.'
                      + f'\n{type(ex).__name__}: {ex}', self, 0, True)

                # Save the exception to model attribute
                exception = type(ex).__name__ + ': ' + str(ex)
                getattr(self, model).error = exception

                # Append exception to AutoML errors
                self.errors += (model + ' --> ' + exception + '\n')

                # Replace model with value X for later removal
                # Can't remove at once to not disturb list order
                self.final_models = \
                    ['X' if x == model else x for x in self.final_models]

        def not_regression(final_models):
            ''' Remove classification-only models from pipeline '''

            class_models = ['LogReg', 'GNB', 'MNB', 'BNB', 'LDA', 'QDA']
            for model in class_models:
                if model in final_models:
                    prlog(f"{model} can't perform regression tasks."
                          + " Removing model from pipeline.", self, 0)
                    final_models.remove(model)

            return final_models

        # << ============ Make dataframe from input ============ >>

        # Convert array to dataframe
        if not isinstance(X, pd.DataFrame):
            columns = ['Feature ' + str(i) for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=columns)

        # Max 1 column for target array
        if len(Y.shape) != 1:
            raise ValueError('Target input may only contain 1 column!')

        # Convert dataframe to array
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        # << ============ Parameters tests ============ >>

        prlog('\n<=================== AutoML ===================>\n', self, -1)
        if self.n_jobs != 1:
            prlog(f'Parallel processing with {self.n_jobs} cores.', self, -1)

        # Set algorithm goal (regression, binaryclass or multiclass)
        classes = set(Y)
        if len(classes) < 2:
            raise ValueError(f'Only found one target value: {classes}!')
        elif len(classes) == 2:
            prlog('Algorithm set to binary classification.', self, -1)
            goal = 'binary classification'
        elif 2 < len(classes) < 0.1*len(Y):
            prlog('Algorithm set to multiclass classification.' +
                  f' Number of classes: {classes}', self, -1)
            goal = 'multiclass classification'
        else:
            prlog('Algorithm set to regression.', self, -1)
            goal = 'regression'

        # Check validity models
        # BNB not in standard list because it only works with boolean features
        model_list = ['GNB', 'MNB', 'LinReg', 'LogReg', 'LDA', 'QDA', 'KNN',
                      'Tree', 'ET', 'RF', 'AdaBoost', 'GBM', 'XGBoost',
                      'lSVM', 'kSVM', 'PA', 'SGD', 'MLP']

        # Final list of models to be used
        # Class attribute because needed for boxplot
        self.final_models = []
        if self.models is None:  # Use all possible models (default)
            self.final_models = model_list.copy()
        else:
            # If only one model, make list for enumeration
            if isinstance(self.models, str):
                self.models = [self.models]

            # Remove duplicates
            # Use and on the None output of set.add to call the function
            self.models = [not set().add(x.lower()) and x
                           for x in self.models if x.lower() not in set()]

            # Set models to right name
            for m in self.models:
                # Compare strings case insensitive
                if m.lower() not in map(str.lower, model_list):
                    prlog(f"Unknown model {m}. Removed from pipeline.",
                          self, -1)
                else:
                    for n in model_list:
                        if m.lower() == n.lower():
                            self.final_models.append(n)
                            break

        # Check if XGBoost is available
        if 'XGBoost' in self.final_models and not xgb_import:
            prlog("Unable to import XGBoost. Model removed from pipeline.",
                  self, -1)
            self.final_models.remove('XGBoost')

        # Linear regression can't perform classification
        if 'LinReg' in self.final_models and goal != 'regression':
            prlog("Linear Regression can't perform classification tasks."
                  + " Removing model from pipeline.", self, 0)
            self.final_models.remove('LinReg')

        # Remove classification-only models from pipeline
        if goal == 'regression':
            self.final_models = not_regression(self.final_models)

        # Check if there are still valid models
        if len(self.final_models) == 0:
            raise ValueError(f"No models found in pipeline. Try {model_list}")

        prlog(f'Models in pipeline: {self.final_models}', self, -1)

        # Set default metric
        if self.metric is None and goal == 'binary classification':
            self.metric = 'F1'
        elif self.metric is None:
            self.metric = 'MSE'

        # Check validity metric
        metric_class = ['Precision', 'Recall', 'Accuracy', 'F1', 'AUC',
                        'LogLoss', 'Jaccard']
        metric_reg = ['r2', 'max_error', 'MAE', 'MSE', 'MSLE']
        for m in metric_class + metric_reg:
            # Compare strings case insensitive
            if self.metric.lower() == m.lower():
                self.metric = m

        if self.metric not in metric_class + metric_reg:
            raise ValueError('Unknown metric. Try one of {}.'
                             .format(metric_class if goal ==
                                     'binary classification' else metric_reg))
        elif self.metric not in metric_reg and goal != 'binary classification':
            raise ValueError("{} is an invalid metric for {}. Try one of {}."
                             .format(self.metric, goal, metric_reg))

        # << ============ Data preparation ============ >>

        data = {}  # Dictionary of data (complete, train, test and scaled)
        idx = np.random.randint(0, len(X), size=int(len(X)*percentage/100))
        data['X'] = X.ix[idx]
        data['Y'] = Y.ix[idx]

        # Split train and test for the BO on percentage of data
        data['X_train'], data['X_test'], data['Y_train'], data['Y_test'] = \
            train_test_split(data['X'],
                             data['Y'],
                             test_size=self.ratio,
                             shuffle=True,
                             random_state=1)

        prlog('\nData stats =====================>', self, -1)
        prlog('Number of features: {}\nNumber of instances: {}'
              .format(data['X'].shape[1], data['X'].shape[0]), self, -1)
        prlog('Size of the training set: {}\nSize of the validation set: {}'
              .format(len(data['X_train']), len(data['X_test'])), self, -1)
        if goal != 'regression':  # Print count of target values
            unique, counts = np.unique(data['Y'], return_counts=True)
            lenx = max(max([len(str(i)) for i in unique]), 5)
            prlog('\nNumber of instances per target class:', self, -1)
            prlog(f'{"Value":{lenx}} --> Count', self, -1)
            for i in range(len(unique)):
                prlog(f'{unique[i]:<{lenx}} --> {counts[i]}', self, -1)

        # Check if features need to be scaled
        scaling_models = ['LinReg', 'LogReg', 'KNN', 'XGBoost',
                          'lSVM', 'kSVM', 'PA', 'SGD', 'MLP']
        if any(model in self.final_models for model in scaling_models):
            # Normalize features to mean=0, std=1
            data['X_scaled'] = StandardScaler().fit_transform(data['X'])
            scaler = StandardScaler().fit(data['X_train'])
            data['X_train_scaled'] = scaler.transform(data['X_train'])
            data['X_test_scaled'] = scaler.transform(data['X_test'])

        # << ============ Core ============ >>

        prlog('\n\nRunning pipeline =====================>', self, -1)

        # If multiprocessing or verbose=0, use tqdm to evaluate process
        if self.n_jobs > 1 or (self.n_jobs == 1 and self.verbose == 0):
            loop = tqdm(self.final_models)
        else:
            loop = self.final_models

        # If parallel, verbose is automatically 0
        verbose = 0 if self.n_jobs > 1 else self.verbose

        # Call function in parallel (verbose=0 if multiprocessing)
        Parallel(n_jobs=self.n_jobs)(delayed(run_model)
                                            (data, model, self.metric, goal,
                                             self.max_iter, self.batch_size,
                                             self.cv, self.n_splits,
                                             self.log, verbose
                                             ) for model in loop)

        # Remove faulty models (replaced with X)
        while 'X' in self.final_models:
            self.final_models.remove('X')

        if self.cv:
            # Print final results (summary of cross-validation)
            prlog('\n\nFinal stats ================>>', self, -1)
            prlog(f'Target metric: {self.metric}', self, -1)
            prlog('------------------------------------', self, -1)

            lenx = max([len(getattr(self, m).name) for m in self.final_models])
            max_mean = max([getattr(self, m).results.mean()
                            for m in self.final_models])

            for m in self.final_models:
                name = getattr(self, m).name
                mean = getattr(self, m).results.mean()
                std = getattr(self, m).results.std()
                if mean == max_mean:  # Highlight best score
                    prlog(u'{0:{1}s} --> {2:.3f} \u00B1 {3:.3f} !!'
                          .format(name, lenx, mean, std), self, -1)
                else:
                    prlog(u'{0:{1}s} --> {2:.3f} \u00B1 {3:.3f}'
                          .format(name, lenx, mean, std), self, -1)

    def boxplot(self, filename=None):
        ''' Plot a boxplot of the found metric results '''

        results, names = [], []
        try:  # Can't make plot before running fit!
            for m in self.final_models:
                results.append(getattr(self, m).results)
                names.append(getattr(self, m).shortname)
        except AttributeError:
            raise Exception('You need to fit the class before plotting!')

        fig, ax = plt.subplots(figsize=(int(8+len(names)/2), 6))
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.xlabel('Model', fontsize=16, labelpad=12)
        plt.ylabel(getattr(self, self.final_models[0]).metric,
                   fontsize=16,
                   labelpad=12)
        plt.title('Model comparison', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()


class BaseModel(object):

    def __init__(self, **kwargs):

        '''
        DESCRIPTION -----------------------------------

        Initialize class.

        ARGUMENTS -------------------------------------

        data    --> dictionary of the data (train, test and complete set)
        metric  --> metric to maximize (or minimize) in the BO
        goal    --> classification or regression
        log     --> name of the log file
        verbose --> verbosity level (0, 1, 2)

        '''

        # List of metrics where the goal is to minimize
        # (Largely same as the regression metrics)
        self.metric_min = ['max_error', 'MAE', 'MSE', 'MSLE']

        # List of tree-based models
        self.tree = ['Tree', 'Extra-Trees', 'RF', 'AdaBoost', 'GBM', 'XGBoost']

        # Set attributes to child class
        self.__dict__.update(kwargs)

    @timing
    def BayesianOpt(self, max_iter=15, batch_size=1):

        '''
        DESCRIPTION -----------------------------------

        Run the bayesian optmization algorithm.

        ARGUMENTS -------------------------------------

        max_iter   --> maximum number of iterations
        batch_size --> size of BO bacthes

        '''

        def optimize(x):
            ''' Function to optimize '''

            params = self.get_params(x)
            self.BO['params'].append(params)
            prlog(f'Parameters --> {params}', self, 1, True)

            alg = self.get_model(params).fit(self.X_train, self.Y_train)
            self.prediction = alg.predict(self.X_test)

            out = getattr(self, self.metric)

            self.BO['score'].append(out())
            prlog(f'Evaluation --> {self.metric}: {out():.4f}', self, 1)

            return out()

        # << ============ Running optimization ============ >>
        prlog(f'\n\nRunning BO for {self.name}...', self, 0)

        # Save dictionary of BO steps
        self.BO = {}
        self.BO['params'] = []
        self.BO['score'] = []

        # Minimize or maximize the function depending on the metric
        maximize = False if self.metric in self.metric_min else True
        opt = BayesianOptimization(f=optimize,
                                   domain=self.get_domain(),
                                   batch_size=batch_size,
                                   maximize=maximize,
                                   X=self.get_init_values(),
                                   normalize_Y=False)

        opt.run_optimization(max_iter=max_iter,
                             verbosity=True if self.verbose > 1 else False)

        # Set to same shape as GPyOpt (2d-array)
        self.best_params = self.get_params(np.array(np.round([opt.x_opt], 4)))

        # Save best model (not yet fitted)
        self.best_model = self.get_model(self.best_params)

        # Save best predictions
        self.model_fit = self.best_model.fit(self.X_train, self.Y_train)
        self.prediction = self.model_fit.predict(self.X_test)

        # Optimal score of the BO
        score = opt.fx_opt if self.metric in self.metric_min else -opt.fx_opt

        # Print stats
        prlog('', self, 1)  # Print extra line
        prlog('Final Statistics for {}:{:9s}'.format(self.name, ' '), self, 0)
        prlog('Optimal parameters BO: {}'.format(self.best_params), self, 0)
        prlog('Optimal {} score: {:.3f}'.format(self.metric, score), self, 0)

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
            # Folds are made preserving the % of samples for each class
            kfold = StratifiedKFold(n_splits=n_splits, random_state=1)
        else:
            kfold = KFold(n_splits=n_splits, random_state=1)

        # GNB has no best_model yet cause no BO was performed
        if self.shortname == 'GNB':
            self.best_model = self.get_model()
            self.model_fit = self.best_model.fit(self.X_train, self.Y_train)
            self.prediction = self.model_fit.predict(self.X_test)

            # Print stats
            prlog('\n', self, 1)
            prlog(f"Final Statistics for {self.name}:{' ':9s}", self, 0)

        self.results = cross_val_score(self.best_model,
                                       self.X,
                                       self.Y,
                                       cv=kfold,
                                       scoring=scoring)

        # cross_val_score returns negative loss for minimizing metrics
        if self.metric in self.metric_min:
            self.results = -self.results

        prlog('--------------------------------------------------', self, 0)
        prlog('Cross_val {} score --> Mean: {:.3f}   Std: {:.3f}'
              .format(self.metric, self.results.mean(), self.results.std()),
              self, 0)

    # << ============ Evaluation metric functions ============ >>
    def Precision(self):
        average = 'binary' if self.goal == 'binary classification' else 'macro'
        return precision_score(self.Y_test, self.prediction, average=average)

    def Recall(self):
        average = 'binary' if self.goal == 'binary classification' else 'macro'
        return recall_score(self.Y_test, self.prediction, average=average)

    def F1(self):
        average = 'binary' if self.goal == 'binary classification' else 'macro'
        return f1_score(self.Y_test, self.prediction, average=average)

    def Jaccard(self):
        average = 'binary' if self.goal == 'binary classification' else 'macro'
        return jaccard_score(self.Y_test, self.prediction, average=average)

    def Accuracy(self):
        return accuracy_score(self.Y_test, self.prediction)

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
    def plot_proba(self, target_class=1, filename=None):

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

        # From dataframe to array (fix for not scaled models)
        X = np.array(self.X_train)
        Y = np.array(self.Y_train)

        # SVM has no predict_proba() method
        if self.shortname in ('XGBoost', 'lSVM', 'kSVM'):
            # Calculate new probabilities more accurate with cv
            mod = CalibratedClassifierCV(self.best_model, cv=3).fit(X, Y)
        else:
            mod = self.model_fit

        fig, ax = plt.subplots(figsize=(10, 6))
        classes = list(set(Y))
        colors = ['r', 'b', 'g']
        for n in range(len(classes)):
            # Get features per class
            cl = X[np.where(Y == classes[n])]
            pred = mod.predict_proba(cl)[:, target_class]

            if sns_import:
                sns.distplot(pred,
                             hist=False,
                             kde=True,
                             norm_hist=True,
                             color=colors[n],
                             kde_kws={"shade": True},
                             label='Class=' + str(classes[n]))
            else:
                plt.hist(pred,
                         density=True,
                         histtype='stepfilled',
                         color=colors[n],
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

    def plot_feature_importance(self, filename=None):
        ''' Plot a (Tree based) model's feature importance '''

        if self.shortname not in self.tree:
            raise ValueError('This method only works for tree-based ' +
                             f'models. Try one of the following: {self.tree}')

        features = list(self.X)
        fig, ax = plt.subplots(figsize=(10, 15))
        pd.Series(self.model_fit.feature_importances_,
                  features).sort_values().plot.barh()

        plt.xlabel('Features', fontsize=16, labelpad=12)
        plt.ylabel('Score', fontsize=16, labelpad=12)
        plt.title('Importance of Features', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_ROC(self, filename=None):
        ''' Plot Receiver Operating Characteristics curve '''

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
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_confusion_matrix(self, normalize=True, filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot the confusion matrix in a heatmap.

        ARGUMENTS -------------------------------------

        normalize --> boolean to normalize the matrix
        filename --> boolean to save the plot

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
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_decision_tree(self, num_trees=0, rotate=False, filename=None):

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

        if filename is not None:
            plt.savefig(filename)


class Gaussian_Naive_Bayes(BaseModel):

    def __init__(self, *args):
        ''' Class initializer '''

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Gaussian Naïve Bayes', 'GNB'

    def get_params(self, x):
        ''' GNB has no hyperparameters to optimize '''

        return False

    def get_model(self):
        ''' Returns the sklearn model '''

        return GaussianNB()

    def get_domain(self):
        return False

    def get_init_values(self):
        return False


class Multinomial_Naive_Bayes(BaseModel):

    def __init__(self, *args):
        ''' Class initializer '''

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Multinomial Naïve Bayes', 'MNB'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        prior = [True, False]
        params = {'alpha': round(x[0, 0], 2),
                  'fit_prior': prior[int(x[0, 1])]}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return MultinomialNB(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        return [{'name': 'alpha',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)},
                {'name': 'fit_prior',
                 'type': 'discrete',
                 'domain': range(2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[1, 0]])
        return values


class Bernoulli_Naive_Bayes(BaseModel):

    def __init__(self, *args):
        ''' Class initializer '''

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Bernoulli Naïve Bayes', 'BNB'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        prior = [True, False]
        params = {'alpha': round(x[0, 0], 2),
                  'fit_prior': prior[int(x[0, 1])]}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return BernoulliNB(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        return [{'name': 'alpha',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)},
                {'name': 'fit_prior',
                 'type': 'discrete',
                 'domain': range(2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[1, 0]])
        return values


class Linear_Regression(BaseModel):

    def __init__(self, *args):
        ''' Class initializer '''

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Linear Regression', 'LinReg'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'max_iter': int(x[0, 0]),
                  'alpha': round(x[0, 1], 2),
                  'l1_ratio': round(x[0, 2], 1)}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return ElasticNet(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'max_iter',
                 'type': 'discrete',
                 'domain': range(100, 501)},
                {'name': 'alpha',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 5, 500)},
                {'name': 'l1_ratio',
                 'type': 'discrete',
                 'domain': np.linspace(0, 1, 10)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[250, 1.0, 0.5]])
        return values


class Logistic_Regression(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Logistic Regression', 'LogReg'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        regularization = ['l1', 'l2', 'elasticnet', 'none']
        penalty = regularization[int(x[0, 2])]
        params = {'max_iter': int(x[0, 0]),
                  'C': round(x[0, 1], 1),
                  'penalty': penalty}

        if penalty == 'elasticnet':  # Add extra parameter: l1_ratio
            params['l1_ratio'] = float(np.round(x[0, 3], 1))

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return LogisticRegression(solver='saga', multi_class='auto', **params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'max_iter',
                 'type': 'discrete',
                 'domain': range(100, 501)},
                {'name': 'C',
                 'type': 'discrete',
                 'domain': np.linspace(0.1, 5, 50)},
                {'name': 'penalty',
                 'type': 'discrete',
                 'domain': range(4)},
                {'name': 'l1_ratio',
                 'type': 'discrete',
                 'domain': np.linspace(0.1, 0.9, 9)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[250, 1.0, 1, 0.5]])
        return values


class Linear_Discriminant_Analysis(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Linear Discriminant Analysis', 'LDA'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        solver_types = ['svd', 'lsqr', 'eigen']
        solver = solver_types[int(x[0, 0])]
        params = {'solver': solver,
                  'n_components': int(x[0, 2]),
                  'tol': round(x[0, 3], 5)}

        if solver != 'svd':  # Add extra parameter: shrinkage
            params['shrinkage'] = x[0, 1]

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return LinearDiscriminantAnalysis(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'solver',
                 'type': 'discrete',
                 'domain': range(3)},
                {'name': 'shrinkage',
                 'type': 'discrete',
                 'domain': np.linspace(0, 1, 11)},
                {'name': 'n_components',
                 'type': 'discrete',
                 'domain': range(1, 251)},
                {'name': 'tol',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[0, 0, 200, 1e-3]])
        return values


class Quadratic_Discriminant_Analysis(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Quadratic Discriminant Analysis', 'QDA'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'reg_param': round(x[0, 0], 1)}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return QuadraticDiscriminantAnalysis(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'reg_param',
                 'type': 'discrete',
                 'domain': np.linspace(0.1, 1, 10)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[0]])
        return values


class K_Nearest_Neighbors(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'K-Nearest Neighbors', 'KNN'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        weights = ['distance', 'uniform']
        params = {'n_neighbors': int(x[0, 0]),
                  'p': int(x[0, 1]),
                  'weights': weights[int(x[0, 2])]}
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
                 'type': 'discrete',
                 'domain': range(1, 101)},
                {'name': 'p',
                 'type': 'discrete',
                 'domain': range(1, 3)},
                {'name': 'weights',
                 'type': 'discrete',
                 'domain': range(2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[5, 2, 1]])
        return values


class Decision_Tree(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Decision Tree', 'Tree'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.goal != 'regression':
            criterion = ['entropy', 'gini']
        else:
            criterion = ['mse', 'mae', 'friedman_mse']

        params = {'criterion': criterion[int(x[0, 0])],
                  'max_depth': int(x[0, 1]),
                  'min_samples_split': int(x[0, 2]),
                  'min_samples_leaf': int(x[0, 3])}
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
        return [{'name': 'criterion',
                 'type': 'discrete',
                 'domain': range(2 if self.goal != 'regression' else 3)},
                {'name': 'max_depth',
                 'type': 'discrete',
                 'domain': range(1, 11)},
                {'name': 'min_samples_split',
                 'type': 'discrete',
                 'domain': range(2, 21)},
                {'name': 'min_samples_leaf',
                 'type': 'discrete',
                 'domain': range(1, 21)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[0, 3, 2, 1]])
        return values


class Extra_Trees(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Extremely Randomized Trees', 'Extra-Trees'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.goal != 'regression':
            criterion = ['entropy', 'gini']
        else:
            criterion = ['mse', 'mae']
        bootstrap = [True, False]
        params = {'n_estimators': int(x[0, 0]),
                  'max_features': round(x[0, 1], 1),
                  'criterion': criterion[int(x[0, 2])],
                  'bootstrap': bootstrap[int(x[0, 3])],
                  'min_samples_split': int(x[0, 4]),
                  'min_samples_leaf': int(x[0, 5])}
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
                 'type': 'discrete',
                 'domain': range(20, 501)},
                {'name': 'max_features',
                 'type': 'discrete',
                 'domain': np.linspace(0.1, 1, 10)},
                {'name': 'criterion',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'bootstrap',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'min_samples_split',
                 'type': 'discrete',
                 'domain': range(2, 21)},
                {'name': 'min_samples_leaf',
                 'type': 'discrete',
                 'domain': range(1, 21)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[50, 1, 1, 0, 2, 1]])
        return values


class Random_Forest(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Random Forest', 'RF'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.goal != 'regression':
            criterion = ['entropy', 'gini']
        else:
            criterion = ['mse', 'mae', 'friedman_mse']
        bootstrap = [True, False]
        params = {'n_estimators': int(x[0, 0]),
                  'max_features': round(x[0, 1], 1),
                  'criterion': criterion[int(x[0, 2])],
                  'bootstrap': bootstrap[int(x[0, 3])],
                  'min_samples_split': int(x[0, 4]),
                  'min_samples_leaf': int(x[0, 5])}
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
                 'type': 'discrete',
                 'domain': range(2, 101)},
                {'name': 'max_features',
                 'type': 'discrete',
                 'domain': np.linspace(0.1, 1, 10)},
                {'name': 'criterion',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'bootstrap',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'min_samples_split',
                 'type': 'discrete',
                 'domain': range(2, 21)},
                {'name': 'min_samples_leaf',
                 'type': 'discrete',
                 'domain': range(1, 21)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[10, 1, 1, 0, 2, 1]])
        return values


class Adaptive_Boosting(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Adaptive Boosting', 'AdaBoost'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': round(x[0, 1], 2)}
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
                 'type': 'discrete',
                 'domain': range(50, 501)},
                {'name': 'learning_rate',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[50, 1]])
        return values


class Gradient_Boosting_Machine(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Gradient Boosting Machine', 'GBM'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        criterion = ['friedman_mse', 'mae', 'mse']
        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': round(x[0, 1], 2),
                  'subsample': round(x[0, 2], 1),
                  'max_depth': int(x[0, 3]),
                  'criterion': criterion[int(x[0, 4])],
                  'min_samples_split': int(x[0, 5]),
                  'min_samples_leaf': int(x[0, 6])}
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
                 'type': 'discrete',
                 'domain': range(50, 501)},
                {'name': 'learning_rate',
                 'type': 'discrete',
                 'domain': (0.01, 1)},
                {'name': 'subsample',
                 'type': 'discrete',
                 'domain': np.linspace(0.2, 1.0, 9)},
                {'name': 'max_depth',
                 'type': 'discrete',
                 'domain': range(1, 11)},
                {'name': 'criterion',
                 'type': 'discrete',
                 'domain': range(3)},
                {'name': 'min_samples_split',
                 'type': 'discrete',
                 'domain': range(2, 21)},
                {'name': 'min_samples_leaf',
                 'type': 'discrete',
                 'domain': range(1, 21)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[100, 0.1, 1.0, 3, 0, 2, 1]])
        return values


class Extreme_Gradient_Boosting(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Extreme Gradient Boosting', 'XGBoost'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': round(x[0, 1], 2),
                  'min_child_weight': int(x[0, 2]),
                  'reg_alpha': round(x[0, 3], 1),
                  'reg_lambda': round(x[0, 4], 1),
                  'subsample': round(x[0, 5], 1),
                  'max_depth': int(x[0, 6])}
        return params

    def get_model(self, params):
        ''' Returns the model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return XGBClassifier(**params, verbosity=0)
        else:
            return XGBRegressor(**params, verbosity=0)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'discrete',
                 'domain': range(20, 501)},
                {'name': 'learning_rate',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)},
                {'name': 'min_child_weight',
                 'type': 'discrete',
                 'domain': range(1, 21)},
                {'name': 'reg_alpha',
                 'type': 'discrete',
                 'domain': np.linspace(0, 80, 800)},
                {'name': 'reg_lambda',
                 'type': 'discrete',
                 'domain': np.linspace(0, 80, 800)},
                {'name': 'subsample',
                 'type': 'discrete',
                 'domain': np.linspace(0.1, 1.0, 10)},
                {'name': 'max_depth',
                 'type': 'discrete',
                 'domain': range(1, 11)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[100, 0.1, 1, 0, 1, 1.0, 3]])
        return values


class Linear_Support_Vector_Machine(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Linear Support Vector Machine', 'lSVM'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.goal != 'regression':
            losses = ['hinge', 'squared_hinge']
        else:
            losses = ['epsilon_insensitive', 'squared_epsilon_insensitive']
        loss = losses[int(x[0, 1])]
        penalties = ['l1', 'l2']

        # l1 regularization can't be combined with hinge
        penalty = penalties[int(x[0, 2])] if loss == 'squared_hinge' else 'l2'

        # l1 regularization can't be combined with squared_hinge when dual=True
        dual = True if penalty == 'l2' else False

        params = {'C': round(x[0, 0], 2),
                  'loss': loss,
                  'tol': round(x[0, 3], 4),
                  'dual': dual}

        if self.goal != 'regression':
            params['penalty'] = penalty

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
                 'type': 'discrete',
                 'domain': (0.01, 0.1, 1, 10, 100, 1e3, 1e4)},
                {'name': 'loss',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'penalty',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'tol',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[1, 1, 1, 1e-3]])
        return values


class Kernel_Support_Vector_Machine(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Kernel Support Vector Machine', 'kSVM'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        kernels = ['poly', 'rbf', 'sigmoid']
        kernel = kernels[int(x[0, 4])]
        gamma = ['auto', 'scale']
        shrinking = [True, False]

        params = {'C': round(x[0, 0], 2),
                  'degree': int(x[0, 1]),
                  'gamma': gamma[int(x[0, 2])],
                  'kernel': kernel,
                  'shrinking': shrinking[int(x[0, 5])],
                  'tol': round(x[0, 6], 4)}

        if kernel != 'rbf':
            params['coef0'] = round(x[0, 3], 2)

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return SVC(**params)
        else:
            return SVR(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'C',
                 'type': 'discrete',
                 'domain': (0.01, 0.1, 1, 10, 100, 1e3, 1e4)},
                {'name': 'degree',
                 'type': 'discrete',
                 'domain': range(2, 6)},
                {'name': 'gamma',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'coef0',
                 'type': 'discrete',
                 'domain': np.linspace(-1, 1, 200)},
                {'name': 'kernel',
                 'type': 'discrete',
                 'domain': range(3)},
                {'name': 'shrinking',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'tol',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[1, 3, 0, 0, 1, 0, 1e-3]])
        return values


class Passive_Aggressive(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Passive Aggressive', 'PA'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.goal != 'regression':
            loss = ['hinge', 'squared_hinge']
        else:
            loss = ['epsilon_insensitive', 'squared_epsilon_insensitive']
        average = [True, False]

        params = {'loss': loss[int(x[0, 0])],
                  'C': round(x[0, 1], 4),
                  'tol': round(x[0, 2], 4),
                  'average': average[int(x[0, 3])]}

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return PassiveAggressiveClassifier(**params)
        else:
            return PassiveAggressiveRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'loss',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'C',
                 'type': 'discrete',
                 'domain': (1e-4, 1e-3, 0.01, 0.1, 1, 10)},
                {'name': 'tol',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)},
                {'name': 'average',
                 'type': 'discrete',
                 'domain': range(2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[0, 1,  1e-3, 1]])
        return values


class Stochastic_Gradient_Descent(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Stochastic Gradient Descent', 'SGD'
        self.goal = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.goal != 'regression':
            loss = ['hinge', 'log', 'modified_huber', 'squared_hinge',
                    'perceptron', 'squared_loss', 'huber',
                    'epsilon_insensitive', 'squared_epsilon_insensitive']
        else:
            loss = ['squared_loss', 'huber',
                    'epsilon_insensitive', 'squared_epsilon_insensitive']
        penalty = ['none', 'l1', 'l2', 'elasticnet']
        average = [True, False]
        lr = ['constant', 'invscaling', 'optimal', 'adaptive']

        params = {'loss': loss[int(x[0, 0])],
                  'penalty': penalty[int(x[0, 1])],
                  'alpha': round(x[0, 2], 4),
                  'average': average[int(x[0, 3])],
                  'epsilon': round(x[0, 4], 4),
                  'learning_rate': lr[int(x[0, 6])],
                  'power_t': round(x[0, 8], 4),
                  'tol': round(x[0, 9], 4)}

        if penalty == 'elasticnet':
            params['l1_ratio'] = round(x[0, 7], 2)

        if lr != 'optimal':
            params['eta0'] = round(x[0, 5], 5)

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.goal != 'regression':
            return SGDClassifier(**params)
        else:
            return SGDRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'loss',
                 'type': 'discrete',
                 'domain': range(9 if self.goal != 'regression' else 4)},
                {'name': 'penalty',
                 'type': 'discrete',
                 'domain': range(4)},
                {'name': 'alpha',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)},
                {'name': 'average',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'epsilon',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)},
                {'name': 'eta0',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)},
                {'name': 'learning_rate',
                 'type': 'discrete',
                 'domain': range(4)},
                {'name': 'l1_ratio',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)},
                {'name': 'power_t',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)},
                {'name': 'tol',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[0, 2, 1e-3, 1, 0.1, 0.01, 2, 0.15, 0.5, 1e-3]])
        return values


class Multilayer_Perceptron(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Multilayer Perceptron', 'MLP'
        self.goal = args[2]

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
                  'alpha': round(x[0, 3], 4),
                  'learning_rate_init': round(x[0, 4], 3),
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
                 'type': 'discrete',
                 'domain': range(10, 101)},
                {'name': 'hidden_layer_2',
                 'type': 'discrete',
                 'domain': range(101)},
                {'name': 'hidden_layer_3',
                 'type': 'discrete',
                 'domain': range(101)},
                {'name': 'alpha',
                 'type': 'discrete',
                 'domain': (1e-4, 1e-3, 0.01, 0.1)},
                {'name': 'learning_rate_init',
                 'type': 'discrete',
                 'domain': np.linspace(0.001, 0.1, 1e2)},
                {'name': 'max_iter',
                 'type': 'discrete',
                 'domain': range(100, 501)},
                {'name': 'batch_size',
                 'type': 'discrete',
                 'domain': (1, 8, 32, 64)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        values = np.array([[20, 0, 0, 1e-4, 1e-3, 200, 32]])
        return values

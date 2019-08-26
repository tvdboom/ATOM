# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom

'''

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from collections import deque
import math
from datetime import datetime
import warnings
import multiprocessing

# Sklearn
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import (
    SelectPercentile, SelectKBest, f_classif, f_regression,
    VarianceThreshold, SelectFromModel
    )
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, cross_val_score
    )


# Metrics
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score, roc_auc_score,
    r2_score, jaccard_score, roc_curve, confusion_matrix, max_error, log_loss,
    mean_absolute_error, mean_squared_error, mean_squared_log_error
    )

# Models
from sklearn.gaussian_process import (
    GaussianProcessClassifier, GaussianProcessRegressor
    )
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    )
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
    )
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import (
    PassiveAggressiveClassifier, PassiveAggressiveRegressor,
    SGDClassifier, SGDRegressor
    )
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
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set(style='darkgrid', palette="GnBu_d")


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
        prlog('Elapsed time: {:.1f} seconds'.format(end-start), args[0], 1)
        return result

    return wrapper


def convert_to_df(X):
    ''' Convert X to pd.Dataframe '''

    if not isinstance(X, pd.DataFrame):
        columns = ['Feature ' + str(i) for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=columns)

    return X


def prlog(string, cl, level=0, time=False):

    '''
    DESCRIPTION -----------------------------------

    Print and save output to log file.

    PARAMETERS -------------------------------------

    cl     --> class of the element
    string --> string to output
    level  --> minimum verbosity level to print
    time   --> Wether to add the timestamp to the log

    '''

    if cl.verbose > level:
        print(string)

    if cl.log is not None:
        with open(cl.log, 'a+') as file:
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

class ATOM(object):

    def __init__(self, models=None, metric=None,
                 impute='median', features=None,
                 ratio=0.3, max_iter=15, max_time=np.inf, eps=1e-08,
                 batch_size=1, init_points=5, plot_bo=False, cv=True,
                 n_splits=4, log=None, n_jobs=1, verbose=0):

        '''
        DESCRIPTION -----------------------------------

        Initialize class.

        PARAMETERS -------------------------------------

        models      --> list of models to use
        metric      --> metric to perform evaluation on
        impute      --> strategy of the imputer to use (if None, no imputing)
        features    --> select best K (or fraction) features (None for all)
        ratio       --> train/test split ratio
        max_iter    --> maximum number of iterations of the BO
        max_time    --> maximum time for the BO (in seconds)
        eps         --> minimum distance between two consecutive x's
        batch_size  --> size of the batch in which the objective is evaluated
        init_points --> initial number of random tests of the BO
        plot_bo     --> boolean to plot the BO's progress
        cv          --> perform kfold cross validation
        n_splits    --> number of splits for the stratified kfold
        log         --> keep log file
        n_jobs      --> number of cores to use for parallel processing
        verbose     --> verbosity level (0, 1 or 2)

        '''

        # Set attributes to class (set to default if input is invalid)
        self.models = models
        self.metric = metric
        self.impute = impute
        self.features = features
        self.ratio = ratio if 0 < ratio < 1 else 0.3
        self.max_iter = int(max_iter) if max_iter > 0 else 15
        self.max_time = float(max_time) if max_time > 0 else np.inf
        self.eps = float(eps) if eps >= 0 else 1e-08
        self.batch_size = int(batch_size) if batch_size > 0 else 1
        self.init_points = int(init_points) if init_points > 0 else 5
        self.plot_bo = bool(plot_bo)
        self.cv = bool(cv)
        self.n_splits = int(n_splits) if n_splits > 0 else 4
        self.n_jobs = int(n_jobs)  # Gets checked later
        self.verbose = verbose if verbose in (0, 1, 2, 3) else 0

        # Set log file name
        self.log = log if log is None or log.endswith('.txt') else log + '.txt'

        # Save model erros (if any)
        self.errors = ''

    def check_features(self, X, output=False):
        ''' Remove non-numeric features with unhashable type '''

        if output:
            prlog('Checking feature types...', self, 1)

        for column in X.columns:
            X[column] = X[column].astype(str).str.strip()
            dtype = str(X[column].dtype)
            if dtype in ('datetime64', 'timedelta[ns]', 'category'):
                if output:
                    prlog(' --> Dropping feature {}. Unhashable type: {}.'
                          .format(column, dtype), self, 2)
                X.drop(column, axis=1)
            elif dtype == 'object':  # Strip str features from blank spaces
                X[column] = X[column].astype(str).str.strip()

        return X

    def imputer(self, X, strategy='median', max_frac_missing=0.5,
                missing=[np.inf, -np.inf, '', '?', 'NA', None]):

        '''
        DESCRIPTION -----------------------------------

        Impute missing values in feature df. Non-numeric
        features are always imputed with the most frequent
        strategy. Removes columns with too many missing values.

        PARAMETERS -------------------------------------

        X                  --> data features: pd.Dataframe or array
        strategy           --> impute strategy. Choose from:
                                   mean
                                   median
                                   most_frequent
        max_frac_missing   --> maximum fraction of NaN values in column
        missing            --> list of values to impute, besides NaN and None

        RETURNS ----------------------------------------

        Imputed dataframe.

        '''

        strats = ['mean', 'median', 'most_frequent']
        if strategy not in strats:
            raise ValueError(f'Unkwown impute strategy. Try one of {strats}.')

        # Convert array to dataframe (can be called independent of fit method)
        X = convert_to_df(X)
        X = self.check_features(X)  # Needed if ran alone

        prlog('Imputing missing values...', self, 1)

        # Make list for enumeration
        if not isinstance(missing, list):
            missing = [missing]

        # None must always be imputed
        if None not in missing:
            missing.append(None)

        X = X.replace(missing, np.NaN)  # Replace missing values with NaN

        imp_numeric = SimpleImputer(strategy=strategy)
        imp_nonnumeric = SimpleImputer(strategy='most_frequent')
        for i in X.columns:
            if X[i].isna().any():  # Check if any value is missing in column
                # Drop columns with too many NaN values
                nans = X[i].isna().sum()
                pnans = int(nans/len(X[i])*100)
                if nans > max_frac_missing * len(X[i]):
                    prlog(f' --> Feature {i} was removed since it contained ' +
                          f'{nans} ({pnans}%) missing values.', self, 2)
                    X.drop(i, axis=1, inplace=True)
                    continue

                # Check if column is numeric (int, float, unsigned int)
                imp = X[i].values.reshape(-1, 1)
                if X[i].dtype.kind in 'ifu':
                    X[i] = imp_numeric.fit_transform(imp)
                else:
                    X[i] = imp_nonnumeric.fit_transform(imp)

        return X

    def encoder(self, X, max_number_onehot=10):

        '''
        DESCRIPTION -----------------------------------

        Perform encoding on categorical features. The encoding
        type depends on the number of unique values in the column.
        Also removes columns with only one unique category.

        PARAMETERS -------------------------------------

        X                 --> data features: pd.Dataframe or array
        max_number_onehot --> threshold between onehot and label encoding

        RETURNS ----------------------------------------

        Encoded dataframe.

        '''

        # Convert array to dataframe (can be called independent of fit method)
        X = convert_to_df(X)
        X = self.check_features(X)  # Needed if ran alone

        prlog('Encoding categorical features...', self, 1)

        for i in X.columns:
            # Check if column is non-numeric (thus categorical)
            if X[i].dtype.kind not in 'ifu':
                # Count number of unique values in the column
                n_unique = len(np.unique(X[i]))

                # Perform encoding type dependent on number of unique values
                if 2 < n_unique <= max_number_onehot:
                    prlog(' --> One-hot-encoding feature {}. Contains {} '
                          .format(i, n_unique) + 'unique categories.', self, 2)
                    X = pd.concat([X, pd.get_dummies(X[i], prefix=i)], axis=1)
                    X.drop(i, axis=1, inplace=True)
                else:
                    prlog(' --> Label-encoding feature {}. Contains {} '
                          .format(i, n_unique) + 'unique categories.', self, 2)
                    X[i] = LabelEncoder().fit_transform(X[i])

        return X

    def feature_selection(self, X, Y,
                          strategy='univariate',
                          max_features=0.9,
                          threshold=-np.inf,
                          frac_variance=1.,
                          max_correlation=0.98):

        '''
        DESCRIPTION -----------------------------------

        Select the best features of the set. Ties between
        features with equal scores will be broken in an
        unspecified way.

        PARAMETERS -------------------------------------

        X            --> data features: pd.Dataframe or array
        Y            --> data targets: pd.Series or array
        strategy     --> strategy for feature selection:
                             'univariate': perform a univariate F-test
                             model with coef_ or feature_importances_ attribute
        max_features --> if < 1: fraction of features to select
                         if >= 1: number of features to select
        threshold    --> threshold value to use for selection. Only for model.
                         Choose from: float, 'mean', 'median'.
                            float
        frac_variance   --> minimum value of the Pearson correlation
                            cofficient to identify correlated features
        max_correlation --> remove features with constant instances in at
                            least this fraction of the total

        RETURNS ----------------------------------------

        Dataframe of the selected features.

        '''

        def remove_collinear(X, limit):

            '''
            DESCRIPTION -----------------------------------

            Finds pairs of collinear features based on the Pearson
            correlation coefficient. For each pair above the specified
            limit (in terms of absolute value), it removes one of the two.
            Using code adapted from:
                https://chrisalbon.com/machine_learning/
                    feature_selection/drop_highly_correlated_features/

            PARAMETERS -------------------------------------

            X     --> data features: pd.Dataframe or array
            limit --> minimum value of the Pearson correlation cofficient
                      to identify correlated features

            RETURNS ----------------------------------------

            Dataframe after removing one of the two correlated features.

            '''

            mtx = X.corr()  # Pearson correlation coefficient matrix

            # Extract the upper triangle of the correlation matrix
            upper = mtx.where(np.triu(np.ones(mtx.shape).astype(np.bool), k=1))

            # Select the features with correlations above the threshold
            to_drop = [i for i in upper.columns if any(abs(upper[i]) > limit)]

            # Dataframe to hold correlated pairs
            columns = ['drop_feature', 'corr_feature', 'corr_value']
            self.collinear = pd.DataFrame(columns=columns)

            # Iterate to record pairs of correlated features
            for column in to_drop:

                # Find the correlated features
                corr_features = list(upper.index[abs(upper[column]) > limit])

                # Find the correlated values
                corr_values = list(upper[column][abs(upper[column]) > limit])
                drop_features = [column for _ in corr_features]

                # Record the information in a temp dataframe
                temp = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                               'corr_feature': corr_features,
                                               'corr_value': corr_values})

                # Add to class attribute
                self.collinear = self.collinear.append(temp, ignore_index=True)

                prlog(f' --> Feature {column} was removed due to ' +
                      'collinearity with another feature.', self, 2)

            return X.drop(to_drop, axis=1)

        def remove_low_variance(X, frac_variance):

            '''
            DESCRIPTION -----------------------------------

            Removes featrues with too low variance.

            PARAMETERS -------------------------------------

            X             --> data features: pd.Dataframe or array
            frac_variance --> remove features with constant instances in at
                              least this fraction of the total

            RETURNS ----------------------------------------

            Dataframe after removing the low variance features.

            '''

            threshold = frac_variance * (1. - frac_variance)
            var = VarianceThreshold(threshold=threshold).fit(X)
            mask = var.get_support()  # Get boolean mask of selected features

            for n, column in enumerate(X.columns):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed due to' +
                          f' low variance: {var.variances_[n]:.2f}.', self, 2)
                    X.drop(column, axis=1, inplace=True)

            return X

        prlog('Performing feature selection...', self, 1)

        # First, drop features with too high correlation
        X = remove_collinear(X, limit=max_correlation)
        # Then, remove features with too low variance
        X = remove_low_variance(X, frac_variance=frac_variance)

        # Perform selection from model or univariate
        if strategy == 'univariate':
            # Set function dependent on goal
            f = f_classif if self.goal != 'regression' else f_regression
            if max_features < 1:
                # Not using fit_transform because it returns an array
                fs = SelectPercentile(f, percentile=max_features*100).fit(X, Y)
            else:
                fs = SelectKBest(f, k=max_features).fit(X, Y)

            mask = fs.get_support()
            for n, column in enumerate(X.columns):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed after the ' +
                          f'univariate F-test (score: {fs.scores_[n]:.2f}  ' +
                          f'p-value: {fs.pvalues_[n]:.2f}).', self, 2)
                    X.drop(column, axis=1, inplace=True)

        else:
            if max_features < 1:  # Set fraction of features
                max_features = int(max_features * X.shape[1])

            sfm = SelectFromModel(estimator=strategy,
                                  threshold=threshold,
                                  max_features=max_features).fit(X, Y)
            mask = sfm.get_support()
            for n, column in enumerate(X.columns):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed by the ' +
                          'recursive feature eliminator.', self, 2)
                    X.drop(column, axis=1, inplace=True)
        '''
        # Print selected features
        prlog(' --> List of selected features: ', self, 2)
        string = '   '
        for n, column in enumerate(X.columns):
            string += f'  {n+1}.{column}'
        prlog(string, self, 2)
        '''
        return X

    def fit(self, X, Y, percentage=100):

        '''
        DESCRIPTION -----------------------------------

        Run the pipeline.

        PARAMETERS -------------------------------------

        X          --> data features: pd.Dataframe or array
        Y          --> data targets: pd.Series or array
        percentage --> percentage of data to use

        '''

        # << ============ Inner Function ============ >>

        def not_regression(final_models):
            ''' Remove classification-only models from pipeline '''

            class_models = ['BNB', 'GNB', 'MNB', 'LogReg', 'LDA', 'QDA']
            for model in class_models:
                if model in final_models:
                    prlog(f"{model} can't perform regression tasks."
                          + " Removing model from pipeline.", self)
                    final_models.remove(model)

            return final_models

        def shuffle(a, b):
            ''' Shuffles pd.Dataframe a and pd.Series b in unison '''

            if len(a) != len(b):
                raise ValueError("X and Y don't have the same number of rows!")
            p = np.random.permutation(len(a))
            return a.ix[p], b[p]

        # << ============ Initialize ============ >>

        t_init = time()  # To measure the time the whole pipeline takes
        prlog('\n<================ ATOM ================>\n', self, 0, True)

        # << ============ Handle input ============ >>

        # Convert array to dataframe
        X = convert_to_df(X)

        # Convert target column to pandas series
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y, name='target')

        # Delete rows with NaN in target
        idx = pd.isnull(Y).any().nonzero()
        X.drop(X.index[idx], inplace=True)
        Y.drop(Y.index[idx], inplace=True)

        X, Y = shuffle(X, Y)  # Shuffle before selecting percentage
        X = X.head(int(len(X)*percentage/100))
        Y = Y.head(int(len(Y)*percentage/100))

        # << ============ Parameters tests ============ >>

        # Check number of cores for multiprocessing
        n_cores = multiprocessing.cpu_count()

        if self.n_jobs > n_cores:
            prlog('\nWarning! No {} cores available. n_jobs reduced to {}.'
                  .format(self.n_jobs, n_cores), self)
            self.n_jobs = n_cores

        elif self.n_jobs == 0:
            prlog("\nWarning! Value of n_jobs can't be {}. Using 1 core."
                  .format(self.n_jobs), self)
            self.n_jobs = 1

        else:
            if self.n_jobs <= -1:
                self.n_jobs = n_cores + 1 + self.n_jobs

            # Final check
            if self.n_jobs < 1 or self.n_jobs > n_cores:
                raise ValueError('Invalid value for n_jobs!')
            elif self.n_jobs != 1:
                prlog(f'Parallel processing with {self.n_jobs} cores.', self)

        # Set algorithm goal (regression, binaryclass or multiclass)
        classes = set(Y)
        if len(classes) < 2:
            raise ValueError(f'Only found one target value: {classes}!')
        elif len(classes) == 2:
            prlog('Algorithm set to binary classification.', self)
            self.goal = 'binary classification'
        elif 2 < len(classes) < 0.1*len(Y):
            prlog('Algorithm set to multiclass classification.' +
                  f' Number of classes: {len(classes)}', self)
            self.goal = 'multiclass classification'
        else:
            prlog('Algorithm set to regression.', self)
            self.goal = 'regression'

        # Check validity models
        model_list = ['BNB', 'GNB', 'MNB', 'GP', 'LinReg', 'LogReg', 'LDA',
                      'QDA', 'KNN', 'Tree', 'ET', 'RF', 'AdaBoost', 'GBM',
                      'XGBoost', 'lSVM', 'kSVM', 'PA', 'SGD', 'MLP']

        # Final list of models to be used
        # Class attribute because needed for boxplot
        self.final_models = []
        if self.models is None:  # Use all possible models (default)
            self.final_models = model_list.copy()
        else:
            # If only one model, make list for enumeration
            if not isinstance(self.models, list):
                self.models = [self.models]

            # Remove duplicates keeping same order
            # Use and on the None output of set.add to call the function
            self.models = [not set().add(x.lower()) and x
                           for x in self.models if x.lower() not in set()]

            # Set models to right name
            for m in self.models:
                # Compare strings case insensitive
                if m.lower() not in map(str.lower, model_list):
                    prlog(f"Unknown model {m}. Removed from pipeline.", self)
                else:
                    for n in model_list:
                        if m.lower() == n.lower():
                            self.final_models.append(n)
                            break

        # Check if XGBoost is available
        if 'XGBoost' in self.final_models and not xgb_import:
            prlog("Unable to import XGBoost. Model removed from pipeline.",
                  self)
            self.final_models.remove('XGBoost')

        # Linear regression can't perform classification
        if 'LinReg' in self.final_models and self.goal != 'regression':
            prlog("Linear Regression can't perform classification tasks."
                  + " Removing model from pipeline.", self)
            self.final_models.remove('LinReg')

        # Remove classification-only models from pipeline
        if self.goal == 'regression':
            self.final_models = not_regression(self.final_models)

        # Check if there are still valid models
        if len(self.final_models) == 0:
            raise ValueError(f"No models found in pipeline. Try {model_list}")

        prlog(f'Models in pipeline: {self.final_models}', self)

        # Set default metric
        if self.metric is None and self.goal == 'binary classification':
            self.metric = 'F1'
        elif self.metric is None:
            self.metric = 'MSE'

        # Check validity metric
        metric_class = ['Precision', 'Recall', 'Accuracy', 'F1', 'AUC',
                        'LogLoss', 'Jaccard']
        mreg = ['R2', 'max_error', 'MAE', 'MSE', 'MSLE']
        for m in metric_class + mreg:
            # Compare strings case insensitive
            if self.metric.lower() == m.lower():
                self.metric = m

        if self.metric not in metric_class + mreg:
            raise ValueError('Unknown metric. Try one of {}.'
                             .format(metric_class if self.goal ==
                                     'binary classification' else mreg))
        elif self.metric not in mreg and self.goal != 'binary classification':
            raise ValueError("{} is an invalid metric for {}. Try one of {}."
                             .format(self.metric, self.goal, mreg))

        # << ============ Save ATOM's parameters to log ============ >>

        parameters = 'Parameters: {metric: ' + str(self.metric) + \
                     ', impute: ' + str(self.impute) + \
                     ', features: ' + str(self.features) + \
                     ', ratio: ' + str(self.ratio) + \
                     ', max_iter: ' + str(self.max_iter) + \
                     ', max_time: ' + str(self.max_time) + \
                     ', eps: ' + str(self.eps) + \
                     ', batch_size: ' + str(self.batch_size) + \
                     ', init_points: ' + str(self.init_points) + \
                     ', plot_bo: ' + str(self.plot_bo) + \
                     ', cv: ' + str(self.cv) + \
                     ', n_splits: ' + str(self.n_splits) + \
                     ', n_jobs: ' + str(self.n_jobs) + \
                     ', verbose: ' + str(self.verbose) + '}'

        prlog(parameters, self, 5)  # Never print (only write to log)

        # << ============ Data preprocessing ============ >>

        prlog('\nData preprocessing =============>', self, 1)

        X = self.check_features(X, output=True)

        # Impute values
        if self.impute is not None:
            X = self.imputer(X, strategy=self.impute)

        X = self.encoder(X)  # Perform encoding on features

        # Perform feature selection
        if self.features is not None and self.features != 0:
            X = self.feature_selection(X, Y, k=self.features)

        # Count target values before encoding to numerical (for later print)
        unique, counts = np.unique(Y, return_counts=True)

        # Make sure the target categories are numerical
        if Y.dtype.kind not in 'ifu':
            Y = pd.Series(LabelEncoder().fit_transform(Y), name=Y.name)

        # << ============ Data preparation ============ >>

        data = {}  # Dictionary of data (complete, train, test and scaled)
        data['X'] = X
        data['Y'] = Y

        # Split train and test for the BO on percentage of data
        data['X_train'], data['X_test'], data['Y_train'], data['Y_test'] = \
            train_test_split(X, Y, test_size=self.ratio, random_state=1)

        # Check if features need to be scaled
        scaling_models = ['LinReg', 'LogReg', 'KNN', 'XGBoost',
                          'lSVM', 'kSVM', 'PA', 'SGD', 'MLP']

        if any(model in self.final_models for model in scaling_models):
            prlog('Scaling data...', self, 1)

            # Normalize features to mean=0, std=1
            data['X_scaled'] = StandardScaler().fit_transform(data['X'])
            scaler = StandardScaler().fit(data['X_train'])
            data['X_train_scaled'] = scaler.transform(data['X_train'])
            data['X_test_scaled'] = scaler.transform(data['X_test'])

        # Save data to class attribute for later use or for the user
        self.data = data
        self.X, self.Y = X, Y
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            data['X_train'], data['X_test'], data['Y_train'], data['Y_test']
        self.dataset = X.merge(Y.to_frame(), left_index=True, right_index=True)

        # << ============ Print data stats ============ >>

        prlog('\nData stats =====================>', self, 1)
        prlog('Number of features: {}\nNumber of instances: {}'
              .format(data['X'].shape[1], data['X'].shape[0]), self, 1)
        prlog('Size of the training set: {}\nSize of the validation set: {}'
              .format(len(data['X_train']), len(data['X_test'])), self, 1)

        # Print count of target values
        if self.goal != 'regression':
            lenx = max(max([len(str(i)) for i in unique]), len(Y.name))
            prlog('Number of instances per target class:', self, 2)
            prlog(f'{Y.name:{lenx}} --> Count', self, 2)
            for i in range(len(unique)):
                prlog(f'{unique[i]:<{lenx}} --> {counts[i]}', self, 2)

        # << =================== Core ==================== >>

        prlog('\n\nRunning pipeline =====================>', self)

        # If verbose=1, use tqdm to evaluate process
        if self.verbose == 1:
            loop = tqdm(self.final_models)
        else:
            loop = self.final_models

        # Loop over every independent model
        for model in loop:
            # Set model class
            setattr(self, model, eval(model)(data,
                                             self.metric,
                                             self.goal,
                                             self.log,
                                             self.verbose))

            try:  # If errors occure, just skip the model
                if model not in ('GNB', 'GP'):  # No hyperparameters to tune
                    getattr(self, model).BayesianOpt(self.max_iter,
                                                     self.max_time,
                                                     self.eps,
                                                     self.batch_size,
                                                     self.init_points,
                                                     self.plot_bo,
                                                     self.n_jobs)
                if self.cv:
                    getattr(self, model).cross_val_evaluation(self.n_splits,
                                                              self.n_jobs)

            except Exception as ex:
                prlog('Exception encountered while running '
                      + f'the {model} model. Removing model from pipeline.'
                      + f'\n{type(ex).__name__}: {ex}', self, 1, True)

                # Save the exception to model attribute
                exception = type(ex).__name__ + ': ' + str(ex)
                getattr(self, model).error = exception

                # Append exception to ATOM errors
                self.errors += (model + ' --> ' + exception + u'\n')

                # Replace model with value X for later removal
                # Can't remove at once to not disturb list order
                self.final_models = \
                    ['X' if x == model else x for x in self.final_models]

        # Set model attributes for lowercase as well
        for model in self.final_models:
            setattr(self, model.lower(), getattr(self, model))

        # Remove faulty models (replaced with X)
        while 'X' in self.final_models:
            self.final_models.remove('X')

        if self.cv:
            try:  # Check that at least one model worked
                lenx = max([len(getattr(self, m).name)
                            for m in self.final_models])
                max_mean = max([getattr(self, m).results.mean()
                                for m in self.final_models])
            except ValueError:
                raise ValueError('It appears all models failed to run...')

            # Print final results (summary of cross-validation)
            t = time() - t_init  # Total time in seconds
            h = int(t/3600.)
            m = int(t/60.) - h*60
            s = int(t - h*3600 - m*60)
            prlog('\n\nFinal stats ================>>', self)
            prlog(f'Total duration: {h:02}h:{m:02}m:{s:02}s', self)
            prlog(f'Target metric: {self.metric}', self)
            prlog('------------------------------------', self)

            for m in self.final_models:
                name = getattr(self, m).name
                mean = getattr(self, m).results.mean()
                std = getattr(self, m).results.std()

                # Highlight best score (if more than one)
                if mean == max_mean and len(self.final_models) > 1:
                    prlog(u'{0:{1}s} --> {2:.3f} \u00B1 {3:.3f} !!'
                          .format(name, lenx, mean, std), self)
                else:
                    prlog(u'{0:{1}s} --> {2:.3f} \u00B1 {3:.3f}'
                          .format(name, lenx, mean, std), self)

        # <====================== End fit function ======================>

    def boxplot(self, figsize=None, filename=None):
        ''' Plot a boxplot of the found metric results '''

        results, names = [], []
        try:  # Can't make plot before running fit!
            for m in self.final_models:
                results.append(getattr(self, m).results)
                names.append(getattr(self, m).shortname)
        except AttributeError:
            raise Exception('You need to fit the class before plotting!')

        if figsize is None:
            figsize = (int(8+len(names)/2), 6)

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
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

    def plot_correlation(self, X=None, Y=None,
                         figsize=(10, 10), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot the feature's correlation matrix. Ignores non-numeric columns.

        PARAMETERS -------------------------------------

        X        --> feature's dataframe (None if ATOM is already fitted)
        Y        --> target (None if ATOM is already fitted)
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

        if X is None and Y is None:
            try:
                df = self.dataset
            except AttributeError:
                raise ValueError('Provide data for the plot or fit the class!')
        elif X is None:
            df = self.data['X']
        else:
            df = convert_to_df(X)

        if Y is not None:
            if not isinstance(Y, pd.Series):
                Y = pd.Series(Y, name='target')
            df = df.merge(Y.to_frame(), left_index=True, right_index=True)

        # Compute the correlation matrix
        corr = df.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        sns.set_style('white')
        fig, ax = plt.subplots(figsize=figsize)

        # Draw the heatmap with the mask and correct aspect ratio
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Feature correlation matrix', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()


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

        # List of metrics where the goal is to minimize
        # (Largely same as the regression metrics)
        self.metric_min = ['max_error', 'MAE', 'MSE', 'MSLE']

        # List of tree-based models
        self.tree = ['Tree', 'Extra-Trees', 'RF', 'AdaBoost', 'GBM', 'XGBoost']

        # Set attributes to child class
        self.__dict__.update(kwargs)

    @timing
    def BayesianOpt(self, max_iter=15, max_time=np.inf, eps=1e-08,
                    batch_size=1, init_points=5, plot_bo=False, n_jobs=1):

        '''
        DESCRIPTION -----------------------------------

        Run the bayesian optmization algorithm.

        PARAMETERS -------------------------------------

        max_iter    --> maximum number of iterations
        max_time    --> maximum time for the BO (in seconds)
        eps         --> minimum distance between two consecutive x's
        batch_size  --> size of the batch in which the objective is evaluated
        init_points --> number of initial random tests of the BO
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
            ''' Function to optimize '''

            params = self.get_params(x)
            self.BO['params'].append(params)
            prlog(f'Parameters --> {params}', self, 2, True)

            alg = self.get_model(params).fit(self.X_train, self.Y_train)
            self.prediction = alg.predict(self.X_test)

            out = getattr(self, self.metric)

            self.BO['score'].append(out())
            prlog(f'Evaluation --> {self.metric}: {out():.4f}', self, 2)

            if plot_bo:
                # Start to fill NaNs with encountered metric values
                if np.isnan(self.y1).any():
                    for i, value in enumerate(self.y1):
                        if math.isnan(value):
                            self.y1[i] = out()
                            if i > 0:  # The first value must remain empty
                                self.y2[i] = abs(self.y1[i] - self.y1[i-1])
                            break
                else:  # If no NaNs anymore, continue deque
                    self.x.append(max(self.x)+1)
                    self.y1.append(out())
                    self.y2.append(abs(self.y1[-1] - self.y1[-2]))

                self.line1, self.line2, self.ax1, self.ax2 = \
                    animate_plot(self.x, self.y1, self.y2,
                                 self.line1, self.line2, self.ax1, self.ax2)

            return out()

        # << ============ Running optimization ============ >>
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
        prlog('', self, 2)  # Print extra line
        prlog('Final statistics for {}:{:9s}'.format(self.name, ' '), self, 1)
        prlog('Optimal parameters BO: {}'.format(self.best_params), self, 1)
        prlog('Optimal {} score: {:.4f}'.format(self.metric, score), self, 1)

    @timing
    def cross_val_evaluation(self, n_splits=5, n_jobs=1):

        '''
        DESCRIPTION -----------------------------------

        Run kfold cross-validation.

        PARAMETERS -------------------------------------

        n_splits    --> number of splits for the cross-validation
        n_jobs      --> number of cores for parallel processing

        '''

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

        # GNB and GP have no best_model yet cause no BO was performed
        if self.shortname in ('GNB', 'GP'):
            # Print stats
            prlog('\n', self, 1)
            prlog(f"Final Statistics for {self.name}:{' ':9s}", self, 1)

            self.best_model = self.get_model()
            self.model_fit = self.best_model.fit(self.X_train, self.Y_train)
            self.prediction = self.model_fit.predict(self.X_test)

        # Run cross-validation
        self.results = cross_val_score(self.best_model,
                                       self.X,
                                       self.Y,
                                       cv=kfold,
                                       scoring=scoring,
                                       n_jobs=n_jobs)

        # cross_val_score returns negative loss for minimizing metrics
        if self.metric in self.metric_min:
            self.results = -self.results

        prlog('--------------------------------------------------', self, 1)
        prlog('Cross_val {} score --> Mean: {:.4f}   Std: {:.4f}'
              .format(self.metric, self.results.mean(), self.results.std()),
              self, 1)

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

    def R2(self):
        return r2_score(self.Y_test, self.prediction)

    def max_error(self):
        return max_error(self.Y_test, self.prediction)

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
            raise ValueError('This method only works for ' +
                             'classification problems.')

        # From dataframe to array (fix for not scaled models)
        X = np.array(self.X_train)
        Y = np.array(self.Y_train)

        # Models without predict_proba() method
        if self.shortname in ('XGBoost', 'lSVM', 'kSVM'):
            # Calculate new probabilities more accurate with cv
            mod = CalibratedClassifierCV(self.best_model, cv=3).fit(X, Y)
        else:
            mod = self.model_fit

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        classes = list(set(Y))
        colors = ['r', 'b', 'g']
        for n in range(len(classes)):
            # Get features per class
            cl = X[np.where(Y == classes[n])]
            pred = mod.predict_proba(cl)[:, target_class]

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
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_feature_importance(self, figsize=(10, 15), filename=None):
        ''' Plot a (Tree based) model's feature importance '''

        if self.shortname not in self.tree:
            raise ValueError('This method only works for tree-based ' +
                             f'models. Try one of the following: {self.tree}')

        features = list(self.X)

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
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

    def plot_ROC(self, figsize=(10, 6), filename=None):
        ''' Plot Receiver Operating Characteristics curve '''

        if self.goal != 'binary classification':
            raise ValueError('This method only works for binary ' +
                             'classification problems.')

        # Calculate new probabilities more accurate with cv
        clf = CalibratedClassifierCV(self.best_model, cv=3).fit(self.X, self.Y)
        pred = clf.predict_proba(self.X)[:, 1]

        # Get False (True) Positive Rate
        fpr, tpr, thresholds = roc_curve(self.Y, pred)

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
        cm = confusion_matrix(self.Y_test, self.prediction)

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

    def plot_decision_tree(self, num_trees=0, max_depth=None,
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
                estimator = self.model_fit.estimators_[num_trees]
            else:
                estimator = self.model_fit

            sklearn.tree.plot_tree(estimator,
                                   max_depth=max_depth,
                                   rotate=rotate,
                                   rounded=True,
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


class GP(BaseModel):

    def __init__(self, *args):
        ''' Class initializer '''

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Gaussian Process', 'GP'

    def get_params(self, x):
        ''' Gp has no hyperparameters to optimize '''

        return False

    def get_model(self):
        ''' Returns the sklearn model '''

        if self.goal != 'regression':
            return GaussianProcessClassifier()
        else:
            return GaussianProcessRegressor()

    def get_domain(self):
        return False

    def get_init_values(self):
        return False


class GNB(BaseModel):

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


class MNB(BaseModel):

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


class BNB(BaseModel):

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


class LinReg(BaseModel):

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


class LogReg(BaseModel):

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


class LDA(BaseModel):

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


class QDA(BaseModel):

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


class KNN(BaseModel):

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


class Tree(BaseModel):

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


class ET(BaseModel):

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


class RF(BaseModel):

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


class AdaBoost(BaseModel):

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


class GBM(BaseModel):

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


class XGBoost(BaseModel):

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


class lSVM(BaseModel):

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


class kSVM(BaseModel):

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


class PA(BaseModel):

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


class SGD(BaseModel):

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


class MLP(BaseModel):

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

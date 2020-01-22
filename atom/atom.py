# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Module containing the main ATOM class

'''

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm
from time import time
import multiprocessing
import warnings
import importlib

# Sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
     f_classif, f_regression, mutual_info_classif, mutual_info_regression,
     chi2, SelectKBest, VarianceThreshold, SelectFromModel, RFE
    )

# Own package modules
from .basemodel import BaseModel, prlog
from .metrics import BaseMetric
from .models import *

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', palette='GnBu_d')


# << ============ Global variables ============ >>

# List of all the available models
model_list = ['BNB', 'GNB', 'MNB', 'GP', 'OLS', 'Ridge', 'Lasso', 'EN',
              'BR', 'LR', 'LDA', 'QDA', 'KNN', 'Tree', 'Bag', 'ET',
              'RF', 'AdaB', 'GBM', 'XGB', 'LGB', 'CatB', 'lSVM', 'kSVM',
              'PA', 'SGD', 'MLP']

# Tuple of models that need to import an extra package
optional_packages = (('XGB', 'xgboost'),
                     ('LGB', 'lightgbm'),
                     ('CatB', 'catboost'))

# List of models that need feature scaling
scaling_models = ['OLS', 'Ridge', 'Lasso', 'EN', 'BR', 'LR', 'KNN',
                  'XGB', 'LGB', 'CatB', 'lSVM', 'kSVM', 'PA', 'SGD', 'MLP']

# List of models that only work for regression/classification tasks
only_classification = ['BNB', 'GNB', 'MNB', 'LR', 'LDA', 'QDA']
only_regression = ['OLS', 'Lasso', 'EN', 'BR']

# List of pre-set binary classification metrics
mbin = ['tn', 'fp', 'fn', 'tp', 'ap']

# List of pre-set classification metrics
mclass = ['accuracy', 'auc', 'mcc', 'f1', 'hamming', 'jaccard', 'logloss',
          'precision', 'recall']

# List of pre-set regression metrics
mreg = ['mae', 'max_error', 'mse', 'msle', 'r2']


# << ============ Functions ============ >>

def params_to_log(f):
    ''' Decorator to save function's params to log file '''

    def wrapper(*args, **kwargs):
        # args[0]=class instance
        prlog('Function "' + str(f.__name__) + f'" parameters: {kwargs}',
              args[0], 5)
        result = f(*args, **kwargs)
        return result

    return wrapper


def conv_to_df(data, columns=None, pca=False):

    '''
    DESCRIPTION -----------------------------------

    Convert data to pd.Dataframe.

    PARAMETERS -------------------------------------

    data    --> dataset to convert
    columns --> name of the columns in the dataset. If None, autofilled.
    pca     --> wether the columns need to be called Features or Components

    '''
    if not isinstance(data, pd.DataFrame):
        if columns is None and not pca:
            columns = ['Feature ' + str(i) for i in range(len(data[0]))]
        elif columns is None:
            columns = ['Component ' + str(i) for i in range(len(data[0]))]
        return pd.DataFrame(data, columns=columns)
    else:
        return data


def conv_to_series(data, name=None):

    '''
    DESCRIPTION -----------------------------------

    Convert data to pd.Series.

    PARAMETERS -------------------------------------

    data --> dataset to convert
    name --> name of the target column. If None, autofilled.

    '''

    return pd.Series(data, name=name if name is not None else 'target')


def merge(X, y):
    ''' Merge pd.DataFrame and pd.Series into one df '''

    return X.merge(y.to_frame(), left_index=True, right_index=True)


def raise_TypeError(param, value):
    ''' Raise TypeError for wrong parameter type '''

    raise TypeError(f'Invalid type for {param} parameter: {type(value)}')


def raise_ValueError(param, value):
    ''' Raise ValueError for invalid parameter value '''

    raise ValueError(f'Invalid value for {param} parameter: {value}')


# << ============ Classes ============ >>

class ATOM(object):

    # Define class variables for plot settings
    style = 'darkgrid'
    palette = 'GnBu_d'
    title_fs = 20
    label_fs = 16
    tick_fs = 12

    # << ================= Methods ================= >>

    def __init__(self, X, y=None, percentage=100, test_size=0.3, log=None,
                 n_jobs=1, warnings=False, verbose=0, random_state=None):

        '''
        DESCRIPTION -----------------------------------

        Transform the input data into pandas objects and set the data class
        attributes. Also performs standard data cleaning steps.

        PARAMETERS -------------------------------------

        X            --> dataset as list, np.array or pd.DataFrame
        y            --> target column as string, list, np.array or pd.Series
        percentage   --> percentage of data to use
        test_size    --> fraction test/train size
        log          --> name of log file
        n_jobs       --> number of cores to use for parallel processing
        warnings     --> wether to show warnings when fitting the models
        verbose      --> verbosity level (0, 1, 2 or 3)
        random_state --> int seed for the RNG

        '''

        # << ============ Handle input data ============ >>

        # Convert X to pd.DataFrame (first to np.array for dimensions)
        if not isinstance(X, pd.DataFrame):
            if not isinstance(X, (np.ndarray, list)):
                raise_TypeError('X', X)
            X = conv_to_df(X)

        # Convert array to dataframe and target column to pandas series
        if isinstance(y, (list, np.ndarray, pd.Series)):
            if len(X) != len(y):
                raise ValueError("X and y don't have the same number of rows" +
                                 f': {len(X)}, {len(y)}.')

            # Convert y to pd.Series
            if not isinstance(y, pd.Series):
                if not isinstance(y, np.ndarray):
                    y = np.array(y)

                # Check that y is one-dimensional
                if y.ndim != 1:
                    raise ValueError('y should be a one-dimensional list, ' +
                                     f'array or pd.Series (y.ndim={y.ndim}).')
                y = conv_to_series(y)

            # Reset indices in case they are not in unison (else merge fails)
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)

            # Merge to one single dataframe with all data
            self.dataset = merge(X, y)
            self.target = y.name

        elif isinstance(y, str):
            if y not in X.columns:
                raise ValueError('Target column not found in X!')

            # Place target column last
            X = X[[col for col in X if col != y] + [y]]
            self.dataset = X.reset_index(drop=True)
            self.target = y

        elif y is None:
            self.dataset = conv_to_df(X.reset_index(drop=True))
            self.target = self.dataset.columns[-1]

        else:  # y is wrong type
            raise_TypeError('y', y)

        # << ============ Check input parameters ============ >>

        self._isfit = False  # Model has not been fitted yet
        if not isinstance(percentage, (int, float)):
            raise_TypeError('percentage', percentage)
        elif percentage <= 0 or percentage > 100:
            raise_ValueError('percentage', percentage)
        else:
            self.percentage = percentage
        if not isinstance(test_size, float):
            raise_TypeError('test_size', test_size)
        elif test_size <= 0 or test_size >= 1:
            raise_ValueError('test_size', test_size)
        else:
            self.test_size = test_size
        if not isinstance(log, (type(None), str)):
            raise_TypeError('log', log)
        elif log is None or log.endswith('.txt'):
            self.log = log
        else:
            self.log = log + '.txt'
        if not isinstance(warnings, bool) and warnings not in (0, 1):
            raise_TypeError('warnings', warnings)
        else:
            self.warnings = bool(warnings)
        if not isinstance(verbose, int):
            raise_TypeError('verbose', verbose)
        elif verbose < 0 or verbose > 3:
            raise_ValueError('verbose', verbose)
        else:
            self.verbose = verbose
        if not isinstance(random_state, (type(None), int)):
            raise_TypeError('random_state', random_state)
        elif random_state is not None:
            self.random_state = random_state
            np.random.seed(self.random_state)  # Set random seed
        else:
            self.random_state = random_state

        prlog('<<=============== ATOM ===============>>', self, time=True)

        # Check number of cores for multiprocessing
        if not isinstance(n_jobs, int):
            raise_TypeError('n_jobs', n_jobs)
        n_cores = multiprocessing.cpu_count()
        if n_jobs > n_cores:
            prlog('Warning! No {} cores available. n_jobs reduced to {}.'
                  .format(n_jobs, n_cores), self)
            self.n_jobs = n_cores

        elif n_jobs == 0:
            prlog("Warning! Value of n_jobs can't be 0. Using 1 core.", self)
            self.n_jobs = 1

        else:
            if n_jobs < 0:
                self.n_jobs = n_cores + 1 + n_jobs
            else:
                self.n_jobs = n_jobs

            # Final check for negative input
            if self.n_jobs < 1:
                raise_ValueError('n_jobs', n_jobs)
            elif self.n_jobs != 1:
                prlog(f'Parallel processing with {self.n_jobs} cores.', self)

        # << ============ Data cleaning ============ >>

        prlog('Initial data cleaning...', self, 1)

        for column in self.dataset:
            nunique = self.dataset[column].nunique(dropna=True)
            unique = self.dataset[column].unique()  # List of unique values

            # Drop features with incorrect column type
            dtype = str(self.dataset[column].dtype)
            if dtype in ('datetime64', 'datetime64[ns]', 'timedelta[ns]'):
                prlog(' --> Dropping feature {} due to unhashable type: {}.'
                      .format(column, dtype), self, 2)
                self.dataset.drop(column, axis=1, inplace=True)
                continue
            elif dtype in ('object', 'category'):
                # Strip categorical features from blank spaces
                self.dataset[column].astype(str).str.strip()

                # Drop features where all values are unique
                if nunique == len(self.dataset):
                    prlog(f' --> Dropping feature {column} due to maximum' +
                          ' cardinality.', self, 2)
                    self.dataset.drop(column, axis=1, inplace=True)

            # Drop features where all values are the same
            if nunique == 1:
                if column == self.target:
                    raise ValueError(f'Only found 1 target value: {unique[0]}')
                else:
                    prlog(f' --> Dropping feature {column}. Contains only 1 ' +
                          f'unique value: {unique[0]}', self, 2)
                    self.dataset.drop(column, axis=1, inplace=True)

        # Drop duplicate rows
        length = len(self.dataset)
        self.dataset.drop_duplicates(inplace=True)
        diff = length - len(self.dataset)  # Difference in length
        if diff > 0:
            prlog(f' --> Dropping {diff} duplicate rows.', self, 2)

        # Delete rows with NaN in target
        length = len(self.dataset)
        self.dataset.dropna(subset=[self.target], inplace=True)
        diff = length - len(self.dataset)  # Difference in length
        if diff > 0:
            prlog(f' --> Dropping {diff} rows with missing values ' +
                  'in target column.', self, 2)

        # << ============ Set algorithm task ============ >>

        # Get unique target values before encoding (for later print)
        self._unique = sorted(self.dataset[self.target].unique())

        if len(self._unique) == 2 and self.goal == 'classification':
            prlog('Algorithm task: binary classification.', self)
            self.task = 'binary classification'
        elif self.goal == 'classification':
            prlog('Algorithm task: multiclass classification.' +
                  f' Number of classes: {len(self._unique)}.', self)
            self.task = 'multiclass classification'
        elif self.goal == 'regression':
            prlog('Algorithm task: regression.', self)
            self.task = 'regression'

        # << ============ Map target column ============ >>

        # Make sure the target categories are numerical
        # Not strictly necessary for sklearn models, but cleaner
        if self.task != 'regression':
            if self.dataset[self.target].dtype.kind not in 'ifu':
                le = LabelEncoder()
                self.dataset[self.target] = \
                    le.fit_transform(self.dataset[self.target])
                self.mapping = {str(v): i for i, v in enumerate(le.classes_)}
            else:
                self.mapping = {str(i): v for i, v in enumerate(self._unique)}
        else:
            self.mapping = 'No target mapping for regression tasks'

        # << =========================================== >>

        self._split_dataset(self.dataset, self.percentage)  # Train/test split
        self.reset_attributes()  # Define data subsets class attributes
        self.stats(1)  # Print out data stats

    def _split_dataset(self, dataset, percentage=100):

        '''
        DESCRIPTION -----------------------------------

        Split a percentage of the dataset into a train and test set.

        PARAMETERS -------------------------------------

        dataset    --> dataset to use for splitting
        percentage --> percentage of the data to use

        '''

        # Get percentage of data (for successive halving)
        self.dataset = dataset.sample(frac=1)  # Shuffle first
        self.dataset = self.dataset.head(int(len(self.dataset)*percentage/100))
        self.dataset.reset_index(drop=True, inplace=True)

        # Split train and test sets on percentage of data
        self.train, self.test = train_test_split(self.dataset,
                                                 test_size=self.test_size,
                                                 shuffle=False)

    def reset_attributes(self, truth='dataset'):

        '''
        DESCRIPTION -----------------------------------

        Reset class attributes for the user based on the "ground truth"
        variable (usually the last one that changed).

        PARAMETERS -------------------------------------

        truth --> variable with the correct values to start from. If 'all',
                  it assumes dataset, train and test are all correct.

        '''

        # Check input parameters
        if not isinstance(truth, str):
            raise_TypeError('truth', truth)

        # Depending on truth, we change some attributes or others
        if truth == 'dataset':
            # Possible because the length of self.train doesn't change
            self.train = self.dataset[:len(self.train)]
            self.test = self.dataset[len(self.train):]

        elif truth in ('train_test', 'train', 'test'):
            # Join train and test on rows
            self.dataset = pd.concat([self.train, self.test], join='outer',
                                     ignore_index=False, copy=True)

        elif truth in ('X_train', 'y_train', 'X_test', 'y_test'):
            self.train = merge(self.X_train, self.y_train)
            self.test = merge(self.X_test, self.y_test)
            self.dataset = pd.concat([self.train, self.test], join='outer',
                                     ignore_index=False, copy=True)

        elif truth in ('X_y', 'X', 'y'):
            self.dataset = merge(self.X, self.y)
            self.train = self.dataset[:len(self.train)]
            self.test = self.dataset[len(self.train):]

        # Reset all indices
        for data in ['dataset', 'train', 'test']:
            getattr(self, data).reset_index(drop=True, inplace=True)

        self.X = self.dataset.drop(self.target, axis=1)
        self.y = self.dataset[self.target]
        self.X_train = self.train.drop(self.target, axis=1)
        self.y_train = self.train[self.target]
        self.X_test = self.test.drop(self.target, axis=1)
        self.y_test = self.test[self.target]

    def stats(self, _vb=None):
        ''' Print some information about the dataset '''

        # If the user calls the method, always print
        _vb = -2 if _vb is None else _vb

        prlog('\nDataset stats ===================>', self, _vb)
        prlog(f'Shape: {self.dataset.shape}', self, _vb)
        prlog('----------------------------------', self, _vb)
        prlog('Size of training set: {}\nSize of test set: {}'
              .format(len(self.train), len(self.test)), self, _vb)

        # Print count of target classes
        if self.task != 'regression':
            prlog('----------------------------------', self, _vb + 1)
            prlog(f'Instances in {self.target} per class:', self, _vb + 1)

            # Create dataframe with stats per target class
            index = []
            for key, value in self.mapping.items():
                if '0' not in self.mapping.keys():
                    index.append(str(value) + ': ' + key)
                else:
                    index.append(value)

            stats = pd.DataFrame(columns=[' total', ' train_set', ' test_set'])

            # Count number of occurrences in all sets
            uq_train, c_train = np.unique(self.y_train, return_counts=True)
            uq_test, c_test = np.unique(self.y_test, return_counts=True)
            for i, val in self.mapping.items():
                # If set has 0 instances of that class the array is empty
                idx_train = np.where(uq_train == val)[0]
                train = c_train[idx_train[0]] if len(idx_train) != 0 else 0
                idx_test = np.where(uq_test == val)[0]
                test = c_test[idx_test[0]] if len(idx_test) != 0 else 0
                stats = stats.append({' total': train + test,
                                      ' train_set': train,
                                      ' test_set': test}, ignore_index=True)

            stats.set_index(pd.Index(index), inplace=True)
            prlog(stats.to_string(header=True), self, _vb + 1)

        prlog('', self, 1)  # Insert an empty row

    def report(self, df='dataset', rows=None, filename=None):

        '''
        DESCRIPTION -----------------------------------

        Use pandas profiling on one of ATOM's data class attributes.

        PARAMETERS -------------------------------------

        df       --> name of the data class attribute to get the report from
        rows     --> number of rows to process (randomly picked)
        filename --> name of saved file

        '''

        try:
            from pandas_profiling import ProfileReport
        except ImportError:
            raise ModuleNotFoundError('Failed to import the pandas-profiling' +
                                      ' package. Install it before using the' +
                                      ' report method.')

        # Check input parameters
        if not isinstance(df, str):
            raise_TypeError('df', df)
        if not isinstance(rows, (type(None), int, float)):
            raise_TypeError('rows', rows)
        else:
            rows = getattr(self, df).shape[0] if rows is None else int(rows)
        if not isinstance(filename, (type(None), str)):
            raise_TypeError('filename', filename)

        prlog('Creating profile report...', self, 1)

        self.report = ProfileReport(getattr(self, df).sample(rows))
        try:  # Render if possible
            from IPython.display import display
            display(self.report)
        except Exception:
            pass

        if filename is not None:
            if not filename.endswith('.html'):
                filename = filename + '.html'
            self.report.to_file(filename)

    @params_to_log
    def impute(self, strat_num='remove', strat_cat='remove',
               max_frac_rows=0.5, max_frac_cols=0.5,
               missing=[np.inf, -np.inf, '', '?', 'NA', 'nan', 'inf']):

        '''
        DESCRIPTION -----------------------------------

        Handle missing values and removes columns with too many missing values.

        PARAMETERS -------------------------------------

        strat_num     --> impute strategy for numerical columns. Choose from:
                             'remove': remove row if any missing value
                             'mean': fill with mean of column
                             'median': fill with median of column
                             'most_frequent': fill with most frequent value
                             fill in with any other numerical value
        strat_cat     --> impute strategy for categorical columns. Choose from:
                             'remove': remove row if any missing value
                             'most_frequent': fill with most frequent value
                             fill in with any other string value
        min_frac_rows --> minimum fraction of non missing values in row
        min_frac_cols --> minimum fraction of non missing values in column
        missing       --> list of values to impute

        '''

        def fit_imputer(imputer):
            ''' Fit and transform the imputer class '''

            self.train[col] = imputer.fit_transform(
                                        self.train[col].values.reshape(-1, 1))
            self.test[col] = imputer.transform(
                                        self.test[col].values.reshape(-1, 1))

        # Check input parameters
        strats = ['remove', 'mean', 'median', 'knn', 'most_frequent']
        if not isinstance(strat_num, str):
            try:
                strat_num = float(strat_num)
            except TypeError:
                raise_TypeError('strat_num', strat_num)
        elif strat_num.lower() not in strats:
            raise_ValueError('strat_num', strat_num)
        if not isinstance(strat_cat, str):
            raise_TypeError('strat_cat', strat_cat)
        if not isinstance(max_frac_rows, float):
            raise_TypeError('max_frac_rows', max_frac_rows)
        elif max_frac_rows <= 0 or max_frac_rows >= 1:
            raise_ValueError('max_frac_rows', max_frac_rows)
        if not isinstance(max_frac_cols, float):
            raise_TypeError('max_frac_cols', max_frac_cols)
        elif max_frac_cols <= 0 or max_frac_cols >= 1:
            raise_ValueError('max_frac_cols', max_frac_cols)
        if not isinstance(missing, list):
            missing = [missing]  # Has to be an iterable for loop

        # Some values must always be imputed (but can be double)
        missing.extend([np.inf, -np.inf])
        missing = set(missing)

        prlog('Imputing missing values...', self, 1)

        # Replace missing values with NaN
        self.dataset.fillna(value=np.NaN, inplace=True)  # Replace None first
        for to_replace in missing:
            self.dataset.replace(to_replace, np.NaN, inplace=True)

        # Drop rows with too many NaN values
        max_frac_rows = int(max_frac_rows * self.dataset.shape[1])
        length = len(self.dataset)
        self.dataset.dropna(axis=0, thresh=max_frac_rows, inplace=True)
        diff = length - len(self.dataset)
        if diff > 0:
            prlog(f' --> Removing {diff} rows for containing too many ' +
                  f'missing values.', self, 2)

        self.reset_attributes('dataset')  # Fill train and test with NaNs

        # Loop over all columns to apply strategy dependent on type
        for col in self.dataset:
            series = self.dataset[col]

            # Drop columns with too many NaN values
            nans = series.isna().sum()  # Number of missing values in column
            pnans = int(nans/len(self.dataset) * 100)  # Percentage of NaNs
            if nans > max_frac_cols * len(self.dataset):
                prlog(f' --> Removing feature {col} for containing ' +
                      f'{nans} ({pnans}%) missing values.', self, 2)
                self.train.drop(col, axis=1, inplace=True)
                self.test.drop(col, axis=1, inplace=True)
                continue  # Skip to next column

            # Column is numerical and contains missing values
            if series.dtype.kind in 'ifu' and nans > 0:
                if not isinstance(strat_num, str):
                    prlog(f' --> Imputing {nans} missing values with number ' +
                          f'{str(strat_num)} in feature {col}.', self, 2)
                    imp = SimpleImputer(strategy='constant',
                                        fill_value=strat_num)
                    fit_imputer(imp)

                elif strat_num.lower() == 'remove':
                    self.train.dropna(subset=[col], axis=0, inplace=True)
                    self.test.dropna(subset=[col], axis=0, inplace=True)
                    prlog(f' --> Removing {nans} rows due to missing ' +
                          f'values in feature {col}.', self, 2)

                elif strat_num.lower() == 'knn':
                    prlog(f' --> Imputing {nans} missing values using the ' +
                          f'KNN imputer in feature {col}.', self, 2)
                    imp = KNNImputer()
                    fit_imputer(imp)

                elif strat_num.lower() in ('mean', 'median', 'most_frequent'):
                    prlog(f' --> Imputing {nans} missing values with ' +
                          f'{strat_num.lower()} in feature {col}.', self, 2)
                    imp = SimpleImputer(strategy=strat_num.lower())
                    fit_imputer(imp)

            # Column is non-numeric and contains missing values
            elif nans > 0:
                if strat_cat.lower() not in ['remove', 'most_frequent']:
                    prlog(f' --> Imputing {nans} missing values with ' +
                          f'{strat_cat} in feature {col}.', self, 2)
                    imp = SimpleImputer(strategy='constant',
                                        fill_value=strat_cat)
                    fit_imputer(imp)

                elif strat_cat.lower() == 'remove':
                    self.train.dropna(subset=[col], axis=0, inplace=True)
                    self.test.dropna(subset=[col], axis=0, inplace=True)
                    prlog(f' --> Removing {nans} rows due to missing ' +
                          f'values in feature {col}.', self, 2)

                elif strat_cat.lower() == 'most_frequent':
                    prlog(f' --> Imputing {nans} missing values with the mo' +
                          f'st frequent occurrence in feature {col}', self, 2)
                    imp = SimpleImputer(strategy=strat_cat)
                    fit_imputer(imp)

        self.reset_attributes('train_test')

    @params_to_log
    def encode(self, max_onehot=10, frac_to_other=0):

        '''
        DESCRIPTION -----------------------------------

        Perform encoding on categorical features. The encoding
        type depends on the number of unique values in the column.

        PARAMETERS -------------------------------------

        max_onehot    --> threshold between onehot and target-encoding
        frac_to_other --> classes with less instances than rows times
                          fraction_to_other are replaced with 'other'

        '''

        # Check parameters
        if not isinstance(max_onehot, (type(None), int)):
            raise_TypeError('max_onehot', max_onehot)
        elif max_onehot is None:
            max_onehot = 0
        elif max_onehot < 0:  # if 0, 1 or 2: it never uses one-hot encoding
            raise_ValueError('max_onehot', max_onehot)
        if not isinstance(frac_to_other, float):
            if frac_to_other not in (0, 1):  # Just cause doesn't fit in 1 line
                raise_TypeError('frac_to_other', frac_to_other)
        elif frac_to_other < 0 or frac_to_other > 1:
            raise_ValueError('frac_to_other', frac_to_other)

        prlog('Encoding categorical features...', self, 1)

        # Loop over all but last column (target is already encoded)
        for col in self.dataset.columns.values[:-1]:
            # Check if column is categorical
            if self.dataset[col].dtype.kind not in 'ifu':
                # Group uncommon classes into 'other'
                values = self.dataset[col].value_counts()
                for cls_, count in values.items():
                    if count < frac_to_other * len(self.dataset[col]):
                        self.dataset[col].replace(cls_, 'other', inplace=True)
                        self.reset_attributes('dataset')  # For target encoding

                # Count number of unique values in the column
                n_unique = len(self.dataset[col].unique())

                # Perform encoding type dependent on number of unique values
                if n_unique == 2:
                    prlog(f' --> Label-encoding feature {col}. Contains ' +
                          f'{n_unique} unique categories.', self, 2)
                    le = LabelEncoder()
                    self.dataset[col] = le.fit_transform(self.dataset[col])

                elif 2 < n_unique <= max_onehot:
                    prlog(f' --> One-hot-encoding feature {col}. Contains ' +
                          f'{n_unique} unique categories.', self, 2)
                    dummies = pd.get_dummies(self.dataset[col], prefix=col)
                    self.dataset = pd.concat([self.dataset, dummies], axis=1)
                    self.dataset.drop(col, axis=1, inplace=True)

                    # Place target column last
                    self.dataset = self.dataset[
                            [col for col in self.dataset if col != self.target]
                            + [self.target]]

                else:
                    prlog(f' --> Target-encoding feature {col}. Contains ' +
                          f'{n_unique} unique categories.', self, 2)

                    # Get mean of target in trainset for every category
                    means = self.train.groupby(col)[self.target].mean()

                    # Map the means over the complete dataset
                    # Test set is tranformed with the mapping of the trainset
                    self.dataset[col] = self.dataset[col].map(means)

        self.reset_attributes('dataset')  # Redefine new attributes

        # Check if mapping failed for the test set
        nans = self.dataset.isna().any()  # pd.Series of columns with nans
        cols = self.dataset.columns.values[nans.to_numpy().nonzero()]
        t = 's' if len(cols) > 1 else ''
        if nans.any():
            prlog('WARNING! It appears the target-encoding was not able to ' +
                  'map all categories in the test set. As a result, there wi' +
                  f"ll appear missing values in column{t}: {', '.join(cols)}" +
                  '. To solve this, try increasing the size of the dataset, ' +
                  'the frac_to_other and max_onehot parameters, or remove ' +
                  'features with high cardinality.', self)

    @params_to_log
    def outliers(self, max_sigma=3, include_target=False):

        '''
        DESCRIPTION -----------------------------------

        Remove outliers from the training set.

        PARAMETERS -------------------------------------

        max_sigma      --> maximum sigma accepted
        include_target --> include target column when checking for outliers

        '''

        # Check parameters
        if not isinstance(max_sigma, (int, float)):
            raise_TypeError('max_sigma', max_sigma)
        elif max_sigma <= 0:
            raise_ValueError('max_sigma', max_sigma)
        if not isinstance(include_target, bool):
            if include_target not in (0, 1):
                raise_TypeError('include_target', include_target)
            else:
                include_target = bool(include_target)

        prlog('Handling outliers...', self, 1)

        # Get z-score outliers index
        objective = self.train if include_target else self.X_train
        # Changes NaN to 0 to not let the algorithm crash
        ix = (np.nan_to_num(np.abs(zscore(objective))) < max_sigma).all(axis=1)

        delete = len(ix) - ix.sum()  # Number of False values in index
        if delete > 0:
            prlog(f' --> Dropping {delete} rows due to outliers.', self, 2)

        # Remove rows based on index and reset attributes
        self.train = self.train[ix]
        self.reset_attributes('train_test')

    @params_to_log
    def balance(self, oversample=None, undersample=None, n_neighbors=5):

        '''
        DESCRIPTION -----------------------------------

        Balance the number of instances per target class using oversampling
        (using a Adaptive Synthetic sampling approach) or undersampling
        (using NearMiss methods).

        PARAMETERS -------------------------------------

        oversample  --> oversampling strategy using ADASYN. Choose from:
                            None: don't oversample
                            float: fraction minority/majority (only for binary)
                            'minority': resample only the minority class
                            'not minority': resample all but minority class
                            'not majority': resample all but majority class
                            'all': resample all classes

        undersample --> undersampling strategy using NearMiss. Choose from:
                            None: don't undersample
                            float: fraction majority/minority (only for binary)
                            'majority': resample only the majority class
                            'not minority': resample all but minority class
                            'not majority': resample all but majority class
                            'all': resample all classes
        n_neighbors --> number of nearest neighbors for both algorithms

        '''

        def check_params(string, value):
            ''' Check the oversample and undersample parameters '''

            opts = ['majority', 'minority',
                    'not majority', 'not minority', 'all']
            if isinstance(value, str):
                if value not in opts:  # Not in one line to skip all elif
                    raise_ValueError(string, value)
            elif isinstance(value, float) or value == 1:
                if not self.task.startswith('binary'):
                    raise_TypeError(string, value)
                elif value <= 0 or value > 1:
                    raise_ValueError(string, value)
            elif value is not None:
                raise_TypeError(string, value)

        if self.task == 'regression':
            raise ValueError('This method only works for ' +
                             'classification tasks!')

        try:
            from imblearn.over_sampling import ADASYN
            from imblearn.under_sampling import NearMiss
        except ImportError:
            raise ModuleNotFoundError('Failed to import the imbalanced-learn' +
                                      ' package. Install it before using the' +
                                      ' balance method.')

        # Check parameters
        check_params('oversample', oversample)
        check_params('undersample', undersample)
        if not isinstance(n_neighbors, int):
            raise_TypeError('n_neighbors', n_neighbors)
        elif n_neighbors <= 0:
            raise_ValueError('n_neighbors', n_neighbors)

        if oversample is None and undersample is None:
            raise ValueError('Fill in a ratio for over or undersampling!')

        columns_x = self.X_train.columns  # Save name columns for later

        # Save number of instances per target class for counting
        counts = {}
        for key, value in self.mapping.items():
            counts[key] = (self.y_train == value).sum()

        # Oversample the minority class with SMOTE
        if oversample is not None:
            prlog('Performing oversampling...', self, 1)
            adasyn = ADASYN(sampling_strategy=oversample,
                            n_neighbors=n_neighbors,
                            n_jobs=self.n_jobs,
                            random_state=self.random_state)
            self.X_train, self.y_train = \
                adasyn.fit_resample(self.X_train, self.y_train)

            # Print changes
            for key, value in self.mapping.items():
                diff = (self.y_train == value).sum() - counts[key]
                if diff > 0:
                    prlog(f' --> Adding {diff} rows to class {key}.', self, 2)

        # Apply undersampling of majority class
        if undersample is not None:
            prlog('Performing undersampling...', self, 1)
            NM = NearMiss(sampling_strategy=undersample,
                          n_neighbors=n_neighbors,
                          n_jobs=self.n_jobs)
            self.X_train, self.y_train = NM.fit_resample(self.X_train,
                                                         self.y_train)

            # Print changes
            for key, value in self.mapping.items():
                diff = counts[key] - (self.y_train == value).sum()
                if diff > 0:
                    prlog(f' --> Removing {diff} rows from class {key}.',
                          self, 2)

        self.X_train = conv_to_df(self.X_train, columns=columns_x)
        self.y_train = conv_to_series(self.y_train, name=self.target)
        self.train = merge(self.X_train, self.y_train)
        self.reset_attributes('train_test')

    @params_to_log
    def feature_insertion(self, n_features=2, generations=20, population=500):

        '''
        DESCRIPTION -----------------------------------

        Use a genetic algorithm to create new combinations of existing
        features and add them to the original dataset in order to capture
        the non-linear relations between the original features. Implemented
        using the gplearn package (https://gplearn.readthedocs.io/en/stable/
        reference.html#symbolic-transformer).

        PARAMETERS -------------------------------------

        n_features  --> maximum number of newly generated features
        generations --> the number of generations to evolve
        population  --> The number of entities in each generation

        '''

        try:
            from gplearn.genetic import SymbolicTransformer
        except ImportError:
            raise ModuleNotFoundError('Failed to import the gplearn' +
                                      ' package. Install it before using the' +
                                      ' feature_insertion method.')

        # Check parameters
        if not isinstance(population, int):
            raise_TypeError('population', population)
        elif population < 100:
            raise ValueError('A minimum population of 100 is required!')
        if not isinstance(generations, int):
            raise_TypeError('generations', generations)
        elif generations < 1:
            raise_ValueError('generations', generations)
        if not isinstance(n_features, int):
            raise_TypeError('n_features', n_features)
        elif n_features <= 0:
            raise_ValueError('n_features', n_features)
        elif n_features > int(0.01 * population):
            raise ValueError("n_features can't be more than 1% of the " +
                             "population's size!")

        prlog('Running genetic algorithm...', self, 1)

        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs',
                        'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']

        self.genetic_algorithm = \
            SymbolicTransformer(generations=generations,
                                population_size=population,
                                hall_of_fame=int(0.1 * population),
                                n_components=int(0.01 * population),
                                init_depth=(1, 2),
                                function_set=function_set,
                                feature_names=self.X.columns,
                                max_samples=1.0,
                                verbose=0 if self.verbose < 3 else 1,
                                n_jobs=self.n_jobs)

        self.genetic_algorithm.fit(self.X_train, self.y_train)
        new_features = self.genetic_algorithm.transform(self.X)

        # ix = indicces of all new features that are not in the original set
        # descript = list of the operators applied to create the new features
        # fitness = list of fitness scores of the new features
        ix, descript, fitness = [], [], []
        for i, program in enumerate(self.genetic_algorithm):
            if str(program) not in self.X_train.columns:
                ix.append(i)
            descript.append(str(program))
            fitness.append(program.fitness_)

        # Remove all features that are identical to those in the dataset
        new_features = new_features[:, ix]
        descript = [descript[i] for i in range(len(descript)) if i in ix]
        fitness = [fitness[i] for i in range(len(fitness)) if i in ix]

        # Indices of all non duplicate elements in list
        ix = [ix for ix, v in enumerate(descript) if v not in descript[:ix]]

        # Remove all duplicate elements
        new_features = new_features[:, ix]
        descript = [descript[i] for i in range(len(descript)) if i in ix]
        fitness = [fitness[i] for i in range(len(fitness)) if i in ix]

        # Check if any new features remain in the loop
        if len(descript) == 0:
            prlog("WARNING! The genetic algorithm couldn't find any " +
                  'improving non-linear features!', self, 1)
            return None

        # Get indices of the best features
        if len(descript) > n_features:
            ix = np.argpartition(fitness, -n_features)[-n_features:]
        else:
            ix = range(len(descript))

        # Select best features only
        new_features = new_features[:, ix]
        descript = [descript[i] for i in range(len(descript)) if i in ix]
        fitness = [fitness[i] for i in range(len(fitness)) if i in ix]
        names = ['Feature ' + str(1 + i + len(self.X_train.columns))
                 for i in range(new_features.shape[1])]

        # Create dataframe attribute
        data = {'Name': names, 'Description': descript, 'Fitness': fitness}
        self.genetic_features = pd.DataFrame(data)
        prlog('-------------------------------------------------------------' +
              '-----------------------------', self, 2)

        for i in descript:
            prlog(' --> New feature ' + i + ' added to the dataset.', self, 1)

        self.X = pd.DataFrame(np.hstack((self.X, new_features)),
                              columns=self.X.columns.to_list() + names)

        self.reset_attributes('X')

    @params_to_log
    def feature_selection(self,
                          strategy=None,
                          solver=None,
                          max_features=None,
                          threshold=-np.inf,
                          min_variance_frac=1.,
                          max_correlation=0.98):

        '''
        DESCRIPTION -----------------------------------

        Select best features according to a univariate F-test or with a
        recursive feature selector (RFS). Ties between features with equal
        scores will be broken in an unspecified way. Also removes features
        with too low variance and too high collinearity.

        PARAMETERS -------------------------------------

        strategy          --> strategy for feature selection. Choose from:
                                 'univariate': perform a univariate F-test
                                 'PCA': perform principal component analysis
                                 'SFM': select best features from model
                                 'RFE': recursive feature eliminator
        solver            --> solver or model class for the strategy
        max_features      --> if < 1: fraction of features to select
                              if >= 1: number of features to select
                              None to select all (only for SFM)
        threshold         --> threshold value to use for selection. Only for
                              SFM. Choose from: float, 'mean', 'median'.
        min_variance_frac --> minimum value of the Pearson correlation
                              cofficient to identify correlated features
        max_correlation   --> remove features with constant instances in at
                              least this fraction of the total

        '''

        def remove_low_variance(min_variance_frac):

            '''
            DESCRIPTION -----------------------------------

            Removes featrues with too low variance.

            PARAMETERS -------------------------------------

            min_variance_frac --> remove features with same values in at
                                  least this fraction of the total

            '''

            threshold = min_variance_frac * (1. - min_variance_frac)
            var = VarianceThreshold(threshold=threshold).fit(self.X)
            mask = var.get_support()  # Get boolean mask of selected features

            for n, column in enumerate(self.X):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed due to' +
                          f' low variance: {var.variances_[n]:.2f}.', self, 2)
                    self.dataset.drop(column, axis=1, inplace=True)

        def remove_collinear(limit):

            '''
            DESCRIPTION -----------------------------------

            Finds pairs of collinear features based on the Pearson
            correlation coefficient. For each pair above the specified
            limit (in terms of absolute value), it removes one of the two.
            Using code adapted from:
                https://chrisalbon.com/machine_learning/
                    feature_selection/drop_highly_correlated_features/

            PARAMETERS -------------------------------------

            limit --> minimum value of the Pearson correlation cofficient
                      to identify correlated features

            '''

            mtx = self.X_train.corr()  # Pearson correlation coefficient matrix

            # Extract the upper triangle of the correlation matrix
            upper = mtx.where(np.triu(np.ones(mtx.shape).astype(np.bool), k=1))

            # Select the features with correlations above the threshold
            to_drop = [i for i in upper.columns if any(abs(upper[i]) > limit)]

            # Dataframe to hold correlated pairs
            self.collinear = pd.DataFrame(columns=['drop_feature',
                                                   'correlated_feature',
                                                   'correlation_value'])

            # Iterate to record pairs of correlated features
            for column in to_drop:
                # Find the correlated features
                corr_features = list(upper.index[abs(upper[column]) > limit])

                # Find the correlated values
                corr_values = list(round(
                                upper[column][abs(upper[column]) > limit], 5))
                drop_features = set([column for _ in corr_features])

                # Add to class attribute
                self.collinear = self.collinear.append(
                    {'drop_feature': ', '.join(drop_features),
                     'correlated_feature': ', '.join(corr_features),
                     'correlation_value': ', '.join(map(str, corr_values))},
                    ignore_index=True)

                prlog(f' --> Feature {column} was removed due to ' +
                      'collinearity with another feature.', self, 2)

            self.dataset.drop(to_drop, axis=1, inplace=True)

        # Check parameters
        if not isinstance(strategy, (type(None), str)):
            raise_TypeError('strategy', strategy)
        if not isinstance(max_features, (type(None), int, float)):
            raise_TypeError('max_features', max_features)
        elif max_features is not None and max_features <= 0:
            raise_ValueError('max_features', max_features)
        if not isinstance(threshold, (str, int, float)):
            raise_TypeError('threshold', threshold)
        if not isinstance(min_variance_frac, (type(None), int, float)):
            raise_TypeError('min_variance_frac', min_variance_frac)
        elif min_variance_frac is not None:
            if min_variance_frac < 0 or min_variance_frac > 1:
                raise_ValueError('min_variance_frac', min_variance_frac)
        if not isinstance(max_correlation, (type(None), int, float)):
            raise_TypeError('max_correlation', max_correlation)
        elif max_correlation is not None:
            if max_correlation < 0 or max_correlation > 1:
                raise_ValueError('max_correlation', max_correlation)

        prlog('Performing feature selection...', self, 1)

        # Then, remove features with too low variance
        if min_variance_frac is not None:
            remove_low_variance(min_variance_frac)

        # First, drop features with too high correlation
        if max_correlation is not None:
            remove_collinear(max_correlation)

        # The dataset is possibly changed
        self.reset_attributes('dataset')

        if strategy is None:
            return None  # Exit feature_selection

        # Set max_features as all or fraction of total
        if max_features is None:
            max_features = self.X_train.shape[1]
        elif max_features < 1:
            max_features = int(max_features * self.X_train.shape[1])

        # Perform selection based on strategy
        if strategy.lower() == 'univariate':
            # Set the solver
            solvers_dct = dict(f_classif=f_classif,
                               f_regression=f_regression,
                               mutual_info_classif=mutual_info_classif,
                               mutual_info_regression=mutual_info_regression,
                               chi2=chi2)
            if solver is None and self.task == 'regression':
                solver = f_regression
            elif solver is None:
                solver = f_classif
            elif solver in solvers_dct.keys():
                solver = solvers_dct[solver]
            elif isinstance(solver, str):
                raise ValueError('Unknown solver: Try one {}.'
                                 .format(', '.join(solvers_dct.keys())))

            self.univariate = SelectKBest(solver, k=max_features)
            self.univariate.fit(self.X, self.y)
            mask = self.univariate.get_support()
            for n, column in enumerate(self.X):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed after the ' +
                          'univariate test (score: {:.2f}  p-value: {:.2f}).'
                          .format(self.univariate.scores_[n],
                                  self.univariate.pvalues_[n]), self, 2)
                    self.dataset.drop(column, axis=1, inplace=True)
            self.reset_attributes('dataset')

        elif strategy.lower() == 'pca':
            prlog(f' --> Applying Principal Component Analysis... ', self, 2)

            # Define PCA
            solver = 'auto' if solver is None else solver
            self.PCA = PCA(n_components=max_features, svd_solver=solver)

            # Create and apply pipeline with scaled features
            pipe = make_pipeline(StandardScaler(), self.PCA).fit(self.X_train)
            self.X_train = conv_to_df(pipe.transform(self.X_train), pca=True)
            self.X_test = conv_to_df(pipe.transform(self.X_test), pca=True)
            self.reset_attributes('X_train')

        elif strategy.lower() == 'sfm':
            if solver is None:
                raise ValueError('Select a model for the solver!')

            try:  # Model already fitted
                self.SFM = SelectFromModel(estimator=solver,
                                           threshold=threshold,
                                           max_features=max_features,
                                           prefit=True)
                mask = self.SFM.get_support()

            except Exception:
                self.SFM = SelectFromModel(estimator=solver,
                                           threshold=threshold,
                                           max_features=max_features)
                self.SFM.fit(self.X, self.y)
                mask = self.SFM.get_support()

            for n, column in enumerate(self.X):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed by the ' +
                          f'{solver.__class__.__name__}.', self, 2)
                    self.dataset.drop(column, axis=1, inplace=True)
            self.reset_attributes('dataset')

        elif strategy.lower() == 'rfe':
            # Recursive feature eliminator
            if solver is None:
                raise ValueError('Select a model for the solver!')

            self.RFE = RFE(estimator=solver, n_features_to_select=max_features)
            self.RFE.fit(self.X, self.y)
            mask = self.RFE.support_

            for n, column in enumerate(self.X):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed by the ' +
                          'recursive feature eliminator.', self, 2)
                    self.dataset.drop(column, axis=1, inplace=True)
            self.reset_attributes('dataset')

        else:
            raise ValueError('Invalid value for strategy parameter. Choose ' +
                             "from: 'univariate', 'PCA', 'SFM' or 'RFE'.")

    @params_to_log
    def fit(self, models, metric, greater_is_better=True, needs_proba=False,
            successive_halving=False, skip_iter=0,
            max_iter=15, max_time=np.inf, eps=1e-08, batch_size=1,
            init_points=5, plot_bo=False, cv=3, bagging=None):

        '''
        DESCRIPTION -----------------------------------

        Fit class to the selected models. The optimal hyperparameters per
        model are selectred using a Bayesian Optimization (BO) algorithm
        with gaussian process as kernel. The resulting score of each step
        of the BO is either computed by cross-validation on the complete
        training set or by creating a validation set from the training set.
        This process will create some minimal leakage but ensures a maximal
        use of the provided data. The test set, however, does not contain any
        leakage and will be used to determine the final score of every model
        . Note that the best score on the BO can be consistently lower than
        the final score on the test set (despite the leakage) due to the
        considerable fewer instances on which it is trained. At the end of
        te pipeline, you can choose to test the robustness of the model
        applying a bagging algorithm, providing a distribution of the
        models' performance.

        PARAMETERS -------------------------------------

        models             --> list of models to use
        metric             --> metric to perform the estimator's evaluation on
        greater_is_better  --> metric is a score function or a loss function
        needs_proba        --> wether the metric needs a probability or score
        successive_halving --> wether to perform successive halving
        skip_iter          --> skip last n steps of successive halving
        max_iter           --> maximum number of iterations of the BO
        max_time           --> maximum time for the BO (in seconds)
        eps                --> minimum distance between two consecutive x's
        batch_size         --> batch size in which the objective is evaluated
        init_points        --> initial number of random tests of the BO
        plot_bo            --> boolean to plot the BO's progress
        cv                 --> splits for the cross validation
        bagging            --> number of splits for bagging

        '''

        # << ============ Inner Function ============ >>

        def run_iteration(self):
            ''' Core iterations, multiple needed for successive halving '''

            # In case there is no cross-validation
            results = 'No cross-validation performed'

            # If verbose=1, use tqdm to evaluate process
            if self.verbose == 1:
                loop = tqdm(self.models, desc='Processing')
            else:
                loop = self.models

            # Loop over every independent model
            for model in loop:
                # If PCA was applied, the data doesn't need to be scaled
                pca = True if hasattr(self, 'PCA') else False
                # Define model class
                setattr(self, model, eval(model)(self.data,
                                                 self.mapping,
                                                 self.metric,
                                                 pca,
                                                 self.task,
                                                 self.log,
                                                 self.n_jobs,
                                                 self.verbose,
                                                 self.random_state))

                try:  # If errors occure, just skip the model
                    with warnings.catch_warnings():
                        if not self.warnings:
                            warnings.simplefilter("ignore")
                        getattr(self, model).BayesianOpt(self.test_size,
                                                         self.max_iter,
                                                         self.max_time,
                                                         self.eps,
                                                         self.batch_size,
                                                         self.init_points,
                                                         self.cv,
                                                         self.plot_bo)

                        if self.bagging is not None:
                            getattr(self, model).bagging(self.bagging)

                except Exception as ex:
                    if self.max_iter == 0:
                        prlog('\n', self, 1)  # Add two empty lines
                    prlog('Exception encountered while running the '
                          + f'{model} model. Removing model from pipeline.'
                          + f'\n{type(ex).__name__}: {ex}', self, 1, True)

                    # Save the exception to model attribute
                    exception = type(ex).__name__ + ': ' + str(ex)
                    getattr(self, model).error = exception

                    # Append exception to ATOM errors dictionary
                    self.errors[model] = exception

                    # Replace model with value X for later removal
                    # Can't remove at once to not disturb list order
                    self.models[self.models.index(model)] = 'X'

                # Set model attributes for lowercase as well
                setattr(self, model.lower(), getattr(self, model))

            # Remove faulty models (replaced with X)
            while 'X' in self.models:
                self.models.remove('X')

            try:  # Fails if all models encountered errors
                # Get max length of the models' names
                len_ = [len(getattr(self, m).longname) for m in self.models]
                lenx = max(len_)

                # Get list of scores
                if self.bagging is None:
                    x = [getattr(self, m).score_test for m in self.models]
                else:
                    x = []
                    for m in self.models:
                        x.append(getattr(self, m).bagging_scores.mean())

                # Get length of best scores and index of longest
                len_ = [len(str(round(score, 3))) for score in x]
                idx = len_.index(max(len_))

                # Set number of decimals
                decimals = self.metric.dec if self.bagging is None else 3

                # Decide width of numbers to print (account for point of float)
                extra = decimals + 1 if decimals > 0 else 0
                width = len(str(int(x[idx]))) + extra

                # Take into account if score or loss function
                best = min(x) if not self.gib else max(x)

            except (ValueError, AttributeError):
                raise ValueError('It appears all models failed to run...')

            # Print final results
            t = time() - t_init  # Total time in seconds
            h = int(t/3600.)
            m = int(t/60.) - h*60
            s = int(t - h*3600 - m*60)
            prlog('\n\nFinal results ================>>', self)
            prlog(f'Duration: {h:02}h:{m:02}m:{s:02}s', self)
            prlog(f"Metric: {self.metric.longname}", self)
            prlog('--------------------------------', self)

            # Create dataframe with final results
            results = pd.DataFrame(columns=['model',
                                            'score_train',
                                            'score_test',
                                            'time'])
            if self.bagging is not None:
                pd.concat([results, pd.DataFrame(columns=['bagging_mean',
                                                          'bagging_std',
                                                          'bagging_time'])])

            for m in self.models:
                name = getattr(self, m).name
                longname = getattr(self, m).longname
                score_train = getattr(self, m).score_train
                score_test = getattr(self, m).score_test
                time_bo = getattr(self, m).time_bo

                if self.bagging is None:
                    results = results.append({'model': name,
                                              'score_train': score_train,
                                              'score_test': score_test,
                                              'time': time_bo},
                                             ignore_index=True)

                    # Highlight best score (if more than one)
                    if score_test == best and len(self.models) > 1:
                        prlog(u'{0:{1}s} --> {2:>{3}.{4}f} !!'
                              .format(longname, lenx, score_test,
                                      width, decimals), self)
                    else:
                        prlog(u'{0:{1}s} --> {2:>{3}.{4}f}'
                              .format(longname, lenx, score_test,
                                      width, decimals), self)

                else:
                    bs_mean = getattr(self, m).bagging_scores.mean()
                    bs_std = getattr(self, m).bagging_scores.std()
                    time_bag = getattr(self, m).time_bag
                    results = results.append({'model': name,
                                              'score_train': score_train,
                                              'score_test': score_test,
                                              'time': time_bo,
                                              'bagging_mean': bs_mean,
                                              'bagging_std': bs_std,
                                              'bagging_time': time_bag},
                                             ignore_index=True)

                    # Highlight best score (if more than one)
                    if bs_mean == best and len(self.models) > 1:
                        prlog(u'{0:{1}s} --> {2:>{3}.{4}f} \u00B1 {5:.3f} !!'
                              .format(longname, lenx, bs_mean, width,
                                      decimals, bs_std), self)
                    else:
                        prlog(u'{0:{1}s} --> {2:>{3}.{4}f} \u00B1 {5:.3f}'
                              .format(longname, lenx, bs_mean, width,
                                      decimals, bs_std), self)

            return results

        def data_preparation():
            ''' Make a dct of the data (complete, train, test and scaled) '''

            data = {}
            for set_ in ['X', 'y', 'X_train', 'y_train', 'X_test', 'y_test']:
                data[set_] = getattr(self, set_)

            # Check if any scaling models in final_models
            scale = any(model in self.models for model in scaling_models)
            # If PCA was performed, features are already scaled
            if scale and not hasattr(self, 'PCA'):
                # Normalize features to mean=0, std=1
                scaler = StandardScaler()
                data['X_train_scaled'] = scaler.fit_transform(data['X_train'])
                data['X_test_scaled'] = scaler.transform(data['X_test'])
                data['X_scaled'] = np.concatenate((data['X_train_scaled'],
                                                   data['X_test_scaled']))

                # Convert np.array to pd.DataFrame for all scaled features
                for set_ in ['X_scaled', 'X_train_scaled', 'X_test_scaled']:
                    data[set_] = conv_to_df(data[set_], self.X.columns)

            return data

        # << ======================== Initialize ======================== >>

        t_init = time()  # To measure the time the whole pipeline takes

        prlog('\nRunning pipeline =================>', self)

        # Set args to class attributes and correct inputs
        if not isinstance(models, list):
            if not isinstance(models, str):
                raise_TypeError('models', models)
            else:
                models = [models]
        gib = greater_is_better  # Shorter variable name for easy access
        if not isinstance(gib, bool) and gib not in (0, 1):
            raise_TypeError('greater_is_better', greater_is_better)
        else:
            self.gib = bool(greater_is_better)
        if not isinstance(needs_proba, bool) and needs_proba not in (0, 1):
            raise_TypeError('needs_proba', needs_proba)
        else:
            self.needs_proba = bool(needs_proba)
        sh = successive_halving
        if not isinstance(sh, bool) and sh not in (0, 1):
            raise_TypeError('successive_halving', successive_halving)
        else:
            self.successive_halving = bool(sh)
        if not isinstance(skip_iter, int):
            raise_TypeError('skip_iter', skip_iter)
        elif skip_iter < 0:
            raise_ValueError('skip_iter', skip_iter)
        else:
            self.skip_iter = skip_iter
        if not isinstance(max_iter, int):
            raise_TypeError('max_iter', max_iter)
        elif max_iter < 0:
            raise_ValueError('max_iter', max_iter)
        else:
            self.max_iter = max_iter
        if not isinstance(max_time, (int, float)):
            raise_TypeError('max_time', max_time)
        elif max_time < 0:
            raise_ValueError('max_time', max_time)
        else:
            self.max_time = max_time
        if not isinstance(eps, (int, float)):
            raise_TypeError('eps', eps)
        elif eps < 0:
            raise_ValueError('eps', eps)
        else:
            self.eps = eps
        if not isinstance(batch_size, int):
            raise_TypeError('batch_size', batch_size)
        elif batch_size < 1:
            raise_ValueError('batch_size', batch_size)
        else:
            self.batch_size = batch_size
        if not isinstance(init_points, int):
            raise_TypeError('init_points', init_points)
        elif init_points < 1:
            raise_ValueError('init_points', init_points)
        else:
            self.init_points = init_points
        if not isinstance(plot_bo, bool) and plot_bo not in (0, 1):
            raise_TypeError('plot_bo', plot_bo)
        else:
            self.plot_bo = bool(plot_bo)
        if not isinstance(cv, int):
            raise_TypeError('cv', cv)
        elif batch_size < 1:
            raise_ValueError('cv', cv)
        else:
            self.cv = cv
        if not isinstance(bagging, (type(None), int)):
            raise_TypeError('bagging', bagging)
        elif bagging is None or bagging == 0:
            self.bagging = None
        elif bagging < 0:
            raise_ValueError('bagging', bagging)
        else:
            self.bagging = bagging

        # Save model erros (if any) in dictionary
        self.errors = {}

        # << ============ Check validity models ============ >>

        # Final list of models to be used
        final_models = []
        if models == 'all':  # Use all possible models
            final_models = model_list.copy()
        else:
            # Remove duplicates keeping same order
            # Use and on the None output of set.add to call the function
            models = [not set().add(x.lower()) and x
                      for x in models if x.lower() not in set()]

            # Set models to right name
            for m in models:
                # Compare strings case insensitive
                if m.lower() not in map(str.lower, model_list):
                    prlog(f"Unknown model {m}. Removed from pipeline.", self)
                else:
                    for n in model_list:
                        if m.lower() == n.lower():
                            final_models.append(n)
                            break

        # Check if packages for not-sklearn models are available
        for model, package in optional_packages:
            if model in final_models:
                try:
                    importlib.import_module(package)
                except ImportError:
                    prlog(f'Unable to import {package}. Removing ' +
                          'model from pipeline.', self)
                    final_models.remove(model)

        # Remove regression/classification-only models from pipeline
        if self.task != 'regression':
            for model in only_regression:
                if model in final_models:
                    prlog(f"{model} can't perform classification tasks."
                          + " Removing model from pipeline.", self)
                    final_models.remove(model)
        else:
            for model in only_classification:
                if model in final_models:
                    prlog(f"{model} can't perform regression tasks."
                          + " Removing model from pipeline.", self)
                    final_models.remove(model)

        # Check if there are still valid models
        if len(final_models) == 0:
            raise ValueError("No models found in pipeline. Choose from: {}"
                             .format({', '.join(model_list)}))

        # Update model list attribute with correct values
        self.models = final_models

        if not self.successive_halving:
            prlog('Model{} in pipeline: {}'
                  .format('s' if len(self.models) > 1 else '',
                          ', '.join(self.models)), self)

        # << ============ Check validity metric ============ >>

        # Create dictionary of all the pre-defined metrics
        metrics = dict(
                tn=BaseMetric('tn', True, False, self.task),
                fp=BaseMetric('fp', False, False, self.task),
                fn=BaseMetric('fn', False, False, self.task),
                tp=BaseMetric('tp', True, False, self.task),
                accuracy=BaseMetric('accuracy', True, False, self.task),
                ap=BaseMetric('ap', True, True, self.task),
                auc=BaseMetric('auc', True, True, self.task),
                mae=BaseMetric('mae', False, False, self.task),
                max_error=BaseMetric('max_error', False, False, self.task),
                mcc=BaseMetric('mcc', True, False, self.task),
                mse=BaseMetric('mse', False, False, self.task),
                msle=BaseMetric('msle', False, False, self.task),
                f1=BaseMetric('f1', True, False, self.task),
                hamming=BaseMetric('hamming', False, False, self.task),
                jaccard=BaseMetric('jaccard', True, False, self.task),
                logloss=BaseMetric('logloss', False, True, self.task),
                precision=BaseMetric('precision', True, False, self.task),
                r2=BaseMetric('r2', True, False, self.task),
                recall=BaseMetric('recall', True, False, self.task),
                )

        if isinstance(metric, str):
            t = mreg if self.task == 'regression' else mclass
            if metric.lower() not in mbin + mclass + mreg:
                raise ValueError(f"Unknown metric: {metric}. " +
                                 "Try one of {', '.join(t)}.")
            elif metric.lower() in mbin and not self.task.startswith('binary'):
                raise ValueError(f'Invalid metric for {self.task} tasks. ' +
                                 f"Try one of: {', '.join(t)}.")
            elif metric.lower() not in mreg and self.task == 'regression':
                raise ValueError(f'{metric} is an invalid metric for regre' +
                                 f"ssion tasks. Try one of: {', '.join(t)}.")
            else:
                self.metric = metrics[metric.lower()]

        elif callable(metric):
            self.metric = BaseMetric(metric, self.gib,
                                     self.needs_proba, self.task)
        else:
            raise_TypeError('metric', metric)

        # Add all metrics as subclasses of the BaseMetric class
        for key, value in metrics.items():
            setattr(self.metric, key, value)

        prlog(f"Metric: {self.metric.longname}", self)

        # << =================== Core ==================== >>

        if self.successive_halving:
            self.results = []  # Save the cv's results in list of dataframes
            iteration = 0
            original_data = self.dataset.copy()
            while len(self.models) > 2**self.skip_iter - 1:
                # Select 1/N of data to use for this iteration
                self._split_dataset(original_data, 100./len(self.models))
                self.reset_attributes()
                self.data = data_preparation()
                prlog('\n\n<<================ Iteration {} ================>>'
                      .format(iteration), self)
                prlog('Model{} in pipeline: {}'
                      .format('s' if len(self.models) > 1 else '',
                              ', '.join(self.models)), self)
                self.stats(1)

                # Run iteration and append to the results list
                results = run_iteration(self)
                self.results.append(results)

                # Select best models for halving
                col = 'score_test' if self.bagging is None else 'bagging_mean'
                lx = results.nlargest(n=int(len(self.models)/2),
                                      columns=col,
                                      keep='all')

                # Keep the models in the same order
                n = []  # List of new models
                [n.append(m) for m in self.models if m in list(lx.model)]
                self.models = n.copy()
                iteration += 1

            self._isfit = True

        else:
            self.data = data_preparation()
            self.results = run_iteration(self)
            self._isfit = True

    # <====================== Utility methods ======================>

    def plot_bagging(self, iteration=-1,
                     title=None, figsize=None, filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot a boxplot of the bagging's results.

        PARAMETERS -------------------------------------

        iteration --> iteration of the successive halving to plot
        title     --> plot's title. None for default title
        figsize   --> figure size: format as (x, y)
        filename  --> name of the file to save

        '''

        def raise_exception():
            raise AttributeError('You need to fit the class using bagging ' +
                                 'before calling the boxplot method!')
        if not self._isfit:
            raise_exception()

        if self.bagging is None:
            raise_exception()

        results, names = [], []
        if self.successive_halving:
            df = self.results[iteration]
        else:
            df = self.results
        for m in df.model:
            results.append(getattr(self, m).bagging_scores)
            names.append(getattr(self, m).name)

        if figsize is None:  # Default figsize depends on number of models
            figsize = (int(8 + len(names)/2), 6)

        fig, ax = plt.subplots(figsize=figsize)
        plt.boxplot(results)

        title = 'Bagging results' if title is None else title
        plt.title(title, fontsize=ATOM.title_fs, pad=12)
        plt.xlabel('Model', fontsize=ATOM.label_fs, labelpad=12)
        plt.ylabel(self.metric.longname,
                   fontsize=ATOM.label_fs,
                   labelpad=12)
        ax.set_xticklabels(names)
        plt.xticks(fontsize=ATOM.tick_fs)
        plt.yticks(fontsize=ATOM.tick_fs)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_successive_halving(self, title=None,
                                figsize=(10, 6), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot the successive halving scores.

        PARAMETERS -------------------------------------

        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

        def raise_exception():
            raise AttributeError('You need to fit the class using a ' +
                                 'successive halving approach before ' +
                                 'calling the plot_successive_halving method!')

        if not self._isfit:
            raise_exception()

        if not self.successive_halving:
            raise_exception()

        models = self.results[0].model  # List of models in first iteration
        col = 'score' if self.bagging is None else 'bagging_mean'
        linx = [[] for m in models]
        liny = [[] for m in models]
        for m, df in enumerate(self.results):
            for n, model in enumerate(models):
                if model in df.model.values:  # If model in iteration
                    linx[n].append(m)
                    liny[n].append(
                            df[col][df.model == model].values[0])

        fig, ax = plt.subplots(figsize=figsize)
        for x, y, label in zip(linx, liny, models):
            plt.plot(x, y, lw=2, marker='o', label=label)
        plt.xlim(-0.1, len(self.results)-0.9)

        title = 'Successive halving results' if title is None else title
        plt.title(title, fontsize=ATOM.title_fs, pad=12)
        plt.legend(frameon=False, fontsize=ATOM.label_fs)
        plt.xlabel('Iteration', fontsize=ATOM.label_fs, labelpad=12)
        plt.ylabel(self.metric.longname,
                   fontsize=ATOM.label_fs,
                   labelpad=12)
        ax.set_xticks(range(len(self.results)))
        plt.xticks(fontsize=ATOM.tick_fs)
        plt.yticks(fontsize=ATOM.tick_fs)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_correlation(self, title=None, figsize=(10, 10), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot the feature's correlation matrix. Ignores non-numeric columns.

        PARAMETERS -------------------------------------

        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

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
        plt.title(title, fontsize=ATOM.title_fs, pad=12)
        fig.tight_layout()
        sns.set_style(ATOM.style)  # Set back to originals style
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_ROC(self, title=None, figsize=(10, 6), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot Receiver Operating Characteristics curve.

        PARAMETERS -------------------------------------

        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

        if self.task != 'binary classification':
            raise ValueError('This method only works for binary ' +
                             'classification tasks!')

        if not self._isfit:
            raise AttributeError('You need to fit the class before calling ' +
                                 'the plot_ROC method!')

        fig, ax = plt.subplots(figsize=figsize)
        for model in self.models:
            # Get False (True) Positive Rate
            Y_test = getattr(self, model).Y_test
            predict_proba = getattr(self, model).predict_proba_test[:, 1]
            auc = getattr(self, model).auc
            fpr, tpr, _ = roc_curve(Y_test, predict_proba)
            plt.plot(fpr, tpr, lw=2, label=f'{model} (AUC={auc:.3f})')

        plt.plot([0, 1], [0, 1], lw=2, color='black', linestyle='--')

        title = 'ROC curve' if title is None else title
        plt.title(title, fontsize=ATOM.title_fs, pad=12)
        plt.legend(loc='lower right',
                   frameon=False,
                   fontsize=ATOM.label_fs)
        plt.xlabel('FPR', fontsize=ATOM.label_fs, labelpad=12)
        plt.ylabel('TPR', fontsize=ATOM.label_fs, labelpad=12)
        plt.xticks(fontsize=ATOM.tick_fs)
        plt.yticks(fontsize=ATOM.tick_fs)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_PRC(self, title=None, figsize=(10, 6), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot precision-recall curve.

        PARAMETERS -------------------------------------

        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

        if self.task != 'binary classification':
            raise ValueError('This method only works for binary ' +
                             'classification tasks!')

        if not self._isfit:
            raise AttributeError('You need to fit the class before calling ' +
                                 'the plot_PRC method!')

        fig, ax = plt.subplots(figsize=figsize)
        for model in self.models:
            # Get precision-recall pairs for different probability thresholds
            Y_test = getattr(self, model).Y_test
            predict_proba = getattr(self, model).predict_proba_test[:, 1]
            ap = getattr(self, model).ap
            prec, recall, _ = precision_recall_curve(Y_test, predict_proba)
            plt.plot(recall, prec, lw=2, label=f'{model} (AP={ap:.3f})')

        title = 'Precision-recall curve' if title is None else title
        plt.title(title, fontsize=ATOM.title_fs, pad=12)
        plt.legend(loc='lower left',
                   frameon=False,
                   fontsize=ATOM.label_fs)
        plt.xlabel('Recall', fontsize=ATOM.label_fs, labelpad=12)
        plt.ylabel('Precision', fontsize=ATOM.label_fs, labelpad=12)
        plt.xticks(fontsize=ATOM.tick_fs)
        plt.yticks(fontsize=ATOM.tick_fs)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_PCA(self, show=None, title=None, figsize=None, filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot the explained variance ratio of the components. Only if PCA
        was applied on the dataset through the feature_selection method.

        PARAMETERS -------------------------------------

        show     --> number of components to show in the plot. None for all
        title    --> plot's title. None for default title
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

        if not hasattr(self, 'PCA'):
            raise ValueError('This plot is only availbale if you apply ' +
                             'PCA on the dataset through the ' +
                             'feature_selection method!')

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
            ax.text(v + 0.005, i - 0.08, f'{v:.3f}', fontsize=ATOM.tick_fs)

        plt.title('Explained variance ratio', fontsize=ATOM.title_fs, pad=12)
        plt.legend(loc='lower right', fontsize=ATOM.label_fs)
        plt.xlabel('Variance ratio', fontsize=ATOM.label_fs, labelpad=12)
        plt.ylabel('Components', fontsize=ATOM.label_fs, labelpad=12)
        plt.xticks(fontsize=ATOM.tick_fs)
        plt.yticks(fontsize=ATOM.tick_fs)
        plt.tight_layout()
        plt.show()

    # <====================== Metric methods ======================>

    def _final_results(self, metric):

        '''
        DESCRIPTION -----------------------------------

        Print final results for a specific metric.

        PARAMETERS -------------------------------------

        metric --> string of the metric attribute in self.metric

        '''

        if not self._isfit:
            raise AttributeError('You need to fit the class before calling ' +
                                 'for a metric method!')

        try:
            # Get max length of the models' names
            lenx = max([len(getattr(self, m).longname) for m in self.models])

            # Get list of scores
            x = [getattr(getattr(self, m), metric.name) for m in self.models]

            # Get length of best scores and index of longest
            len_ = [len(str(round(score, 3))) for score in x]
            idx = len_.index(max(len_))

            # Decide width of numbers to print (account for point of float)
            extra = metric.dec + 1 if metric.dec > 0 else 0
            width = len(str(int(x[idx]))) + extra

            # Take into account if score or loss function
            best = min(x) if not metric.gib else max(x)

        except Exception:
            raise ValueError(f'Invalid metric for {self.task} tasks!')

        prlog('\nFinal results ================>>', self)
        prlog(f'Metric: {metric.longname}', self)
        prlog('--------------------------------', self)

        for m in self.models:
            name = getattr(self, m).longname
            score = getattr(getattr(self, m), metric.name)

            # Highlight best score (if more than one)
            if score == best and len(self.models) > 1:
                prlog(u'{0:{1}s} --> {2:>{3}.{4}f} !!'
                      .format(name, lenx, score, width, metric.dec), self)
            else:
                prlog(u'{0:{1}s} --> {2:>{3}.{4}f}'
                      .format(name, lenx, score, width, metric.dec), self)

    def tn(self):
        self._final_results(self.metric.tn)

    def fp(self):
        self._final_results(self.metric.fp)

    def fn(self):
        self._final_results(self.metric.fn)

    def tp(self):
        self._final_results(self.metric.tp)

    def accuracy(self):
        self._final_results(self.metric.accuracy)

    def ap(self):
        self._final_results(self.metric.ap)

    def auc(self):
        self._final_results(self.metric.auc)

    def mae(self):
        self._final_results(self.metric.mae)

    def max_error(self):
        self._final_results(self.metric.max_error)

    def mcc(self):
        self._final_results(self.metric.mcc)

    def mse(self):
        self._final_results(self.metric.mse)

    def msle(self):
        self._final_results(self.metric.msle)

    def f1(self):
        self._final_results(self.metric.f1)

    def hamming(self):
        self._final_results(self.metric.hamming)

    def jaccard(self):
        self._final_results(self.metric.jaccard)

    def logloss(self):
        self._final_results(self.metric.logloss)

    def precision(self):
        self._final_results(self.metric.precision)

    def r2(self):
        self._final_results(self.metric.r2)

    def recall(self):
        self._final_results(self.metric.recall)

    # <============ Classmethods for plot settings ============>

    @classmethod
    def set_style(cls, style='darkgrid'):
        ''' Change the seaborn plotting style '''

        cls.style = style
        BaseModel.style = style
        sns.set_style(style)

    @classmethod
    def set_palette(cls, palette='GnBu_d'):
        ''' Change the seaborn color palette '''

        cls.palette = palette
        BaseModel.palette = palette
        sns.set_palette(palette)

    @classmethod
    def set_title_fontsize(cls, fontsize=20):
        ''' Change the fontsize of the plot's title '''

        cls.title_fs = fontsize
        BaseModel.title_fs = fontsize

    @classmethod
    def set_label_fontsize(cls, fontsize=16):
        ''' Change the fontsize of the plot's labels and legends '''

        cls.label_fs = fontsize
        BaseModel.label_fs = fontsize

    @classmethod
    def set_tick_fontsize(cls, fontsize=12):
        ''' Change the fontsize of the plot's ticks '''

        cls.tick_fs = fontsize
        BaseModel.tick_fs = fontsize


class ATOMClassifier(ATOM):
    ''' ATOM object for classificatin tasks '''

    def __init__(self, *args, **kwargs):
        ''' Initialize class '''

        self.goal = 'classification'
        super().__init__(*args, **kwargs)


class ATOMRegressor(ATOM):
    ''' ATOM object for regression tasks '''

    def __init__(self, *args, **kwargs):
        ''' Initialize class '''

        self.goal = 'regression'
        super().__init__(*args, **kwargs)

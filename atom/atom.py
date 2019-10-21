# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom

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
from .basemodel import prlog

# Sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
     f_classif, f_regression, mutual_info_classif, mutual_info_regression,
     chi2, SelectKBest, VarianceThreshold, SelectFromModel
    )
from sklearn.model_selection import train_test_split

# Models
from .models import (
        BNB, GNB, MNB, GP, LinReg, LogReg, LDA, QDA, KNN, Tree, Bag, ET, RF,
        AdaBoost, GBM, XGB, LGBM, lSVM, kSVM, PA, SGD, MLP
        )

# Others
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    imblearn_import = False
else:
    imblearn_import = True

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', palette="GnBu_d")


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


def convert_to_pd(data, columns=None):
    ''' Convert data to pd.Dataframe or pd.Series '''

    dim = data.ndim  # Get number of dimensions
    if dim > 1 and not isinstance(data, pd.DataFrame):
        if columns is None:
            columns = ['Feature ' + str(i) for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=columns)

    elif dim == 1 and not isinstance(data, pd.Series):
        name = columns if columns is not None else 'target'
        data = pd.Series(data, name=name)

    return data


def merge(X, Y):
    ''' Merge pd.DataFrame and pd.Series into one df '''
    return X.merge(Y.to_frame(), left_index=True, right_index=True)


# << ============ Classes ============ >>

class ATOM(object):

    def __init__(self, X, Y=None, target=None, percentage=100,
                 test_size=0.3, log=None, n_jobs=1,
                 warnings=False, verbose=0, random_state=None):

        '''
        DESCRIPTION -----------------------------------

        Initialize class.

        PARAMETERS -------------------------------------

        X            --> dataset as pd.DataFrame or np.array
        Y            --> target column as pd.Series or np.array
        target       --> name of target column in X (if Y not provided)
        percentage   --> percentage of data to use
        test_size    --> fraction test/train size
        log          --> name of log file
        n_jobs       --> number of cores to use for parallel processing
        warnings     --> wether to show warnings when fitting the models
        verbose      --> verbosity level (0, 1, 2 or 3)
        random_state --> int seed for the RNG

        '''

        # << ============ Handle input ============ >>

        # Convert array to dataframe and target column to pandas series
        if Y is not None:
            if len(X) != len(Y):
                raise ValueError("X and Y don't have the same number of rows" +
                                 f': {len(X)}, {len(Y)}.')
            X = convert_to_pd(X)
            Y = convert_to_pd(Y)
            self.dataset = merge(X, Y)

        elif target is not None:  # If target is filled, X has to be a df
            if target not in X.columns.values:
                raise ValueError('Target column not found in the dataset!')
            self.dataset = X

        else:
            self.dataset = convert_to_pd(X)

        # << ============ Parameters tests ============ >>

        # Set parameters to default if input is invalid
        self.target = self.dataset.columns[-1]
        self.percentage = percentage if 0 < percentage < 100 else 100
        self.test_size = test_size if 0 < test_size < 1 else 0.3
        self.log = log if log is None or log.endswith('.txt') else log + '.txt'
        self.warnings = bool(warnings)
        self.verbose = verbose if verbose in range(4) else 0
        if random_state is not None:
            self.random_state = int(random_state)
            np.random.seed(self.random_state)  # Set random seed
        else:
            self.random_state = None

        prlog('<<=============== ATOM ===============>>', self, time=True)

        # Check number of cores for multiprocessing
        self.n_jobs = int(n_jobs)
        n_cores = multiprocessing.cpu_count()
        if n_jobs > n_cores:
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

        # << ============ Set algorithm goal ============ >>

        classes = set(self.dataset.iloc[:, -1])
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

        # << ============ Data cleaning ============ >>

        # Drop features with incorrect column type and trim strings
        for column in self.dataset:
            dtype = str(self.dataset[column].dtype)
            if dtype in ('datetime64', 'timedelta[ns]', 'category'):
                prlog(' --> Dropping feature {} due to unhashable type: {}.'
                      .format(column, dtype), self, 2)
                self.dataset.drop(column, axis=1, inplace=True)
            elif dtype == 'object':  # Strip str features from blank spaces
                self.dataset[column].astype(str).str.strip()

        # Get unqiue target values before encoding (for later print)
        self.unique = np.unique(self.dataset[self.target])

        # Make sure the target categories are numerical
        if self.dataset[self.target].dtype.kind not in 'ifu':
            le = LabelEncoder()
            self.dataset[self.target] = pd.Series(
                le.fit_transform(self.dataset[self.target]), name=self.target
            )
            self.target_mapping = {l: i for i, l in enumerate(le.classes_)}

        self._split_dataset(self.dataset, percentage)  # Make train/test split
        self._reset_attributes()  # Define data subsets class attributes
        self.stats()  # Print out data stats

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

        # Split train and test for the BO on percentage of data
        self.train, self.test = train_test_split(self.dataset,
                                                 test_size=self.test_size,
                                                 shuffle=False)

    def _reset_attributes(self, truth='both'):

        '''
        DESCRIPTION -----------------------------------

        Reset class attributes for the user based on the "ground truth"
        variable (usually the last one that changed).

        PARAMETERS -------------------------------------

        truth --> variable with the correct values to start from.
                  Choose from: dataset, train_test or both.

        '''

        if truth == 'dataset':  # Split again (no shuffling to keep same data)
            # Possible because the length of self.train doesn't change
            self.train = self.dataset[:len(self.train)]
            self.test = self.dataset[len(self.train):]

        elif truth == 'train_test':  # Join train and test on rows
            self.dataset = pd.concat([self.train, self.test], join='outer',
                                     ignore_index=False, copy=True)

        # Reset all indices
        for data in ['dataset', 'train', 'test']:
            getattr(self, data).reset_index(drop=True, inplace=True)

        self.X = self.dataset.drop(self.target, axis=1)
        self.Y = self.dataset[self.target]
        self.X_train = self.train.drop(self.target, axis=1)
        self.Y_train = self.train[self.target]
        self.X_test = self.test.drop(self.target, axis=1)
        self.Y_test = self.test[self.target]

    def stats(self):
        ''' Print some statistics on the dataset '''

        prlog('\nData stats =====================>', self, 1)
        prlog('Number of features: {}\nNumber of instances: {}'
              .format(self.X.shape[1], self.X.shape[0]), self, 1)
        prlog('Size of training set: {}\nSize of test set: {}'
              .format(len(self.train), len(self.test)), self, 1)

        # Print count of target values
        if self.goal != 'regression':
            _, counts = np.unique(self.Y, return_counts=True)
            lx = max(max([len(str(i)) for i in self.unique]), len(self.Y.name))

            prlog('Instances per target class:', self, 2)
            prlog(f"{self.Y.name:{lx}} --> Count", self, 2)

            # Check if there was LabelEncoding for the target varaible or not
            for i in range(len(self.unique)):
                if hasattr(self, 'target_mapping'):
                    prlog('{0}: {1:<{2}} --> {3}'
                          .format(self.target_mapping[self.unique[i]],
                                  self.unique[i], lx - 3, counts[i]), self, 2)
                else:
                    prlog('{0:<{1}} --> {2}'
                          .format(self.unique[i], lx, counts[i]), self, 2)

        prlog('', self, 2)  # Insert an empty row

    @params_to_log
    def impute(self, strat_num='remove', strat_cat='remove', max_frac=0.5,
               missing=[np.inf, -np.inf, '', '?', 'NA', 'nan', 'NaN', None]):

        '''
        DESCRIPTION -----------------------------------

        Handle missing values and removes columns with too many missing values.

        PARAMETERS -------------------------------------

        strat_num  --> impute strategy for numerical columns. Choose from:
                           'remove': remove row if any missing value
                           'mean': fill with mean of column
                           'median': fill with median of column
                           'most_frequent': fill with most frequent value
                           fill in with any other numerical value
        strat_cat  --> impute strategy for categorical columns. Choose from:
                           'remove': remove row if any missing value
                           'most_frequent': fill with most frequent value
                           fill in with any other string value
        max_frac   --> maximum fraction of missing values in column
        missing    --> list of values to impute

        '''

        def fit_imputer(imputer):
            ''' Fit and transform the imputer class '''

            self.train[col] = imputer.fit_transform(
                                        self.train[col].values.reshape(-1, 1))
            self.test[col] = imputer.transform(
                                        self.test[col].values.reshape(-1, 1))

        # Check parameters
        max_frac = float(max_frac) if 0 < max_frac <= 1 else 0.5
        missing = [missing]  # Has to be an iterable for loop
        if None not in missing:
            missing.append(None)  # None must always be imputed

        prlog('Handling missing values...', self, 1)

        # Replace missing values with NaN
        self.train.replace(missing, np.NaN, inplace=True)
        self.test.replace(missing, np.NaN, inplace=True)

        # Delete rows with NaN in target
        self.train.dropna(subset=[self.target], inplace=True)
        self.test.dropna(subset=[self.target], inplace=True)
        new_len = len(self.train) + len(self.test)
        diff = len(self.dataset) - new_len
        if diff > 0:
            prlog(f' --> Removing {diff} rows due to missing values at ' +
                  'target column.', self, 2)

        # Loop pver all columns to apply strategy dependent on type
        strats = ['remove', 'mean', 'median', 'most_frequent']
        for col in self.train:
            series = self.train[col]

            # Drop columns with too many NaN values
            nans = series.isna().sum()  # Number of missing values in column
            pnans = int(nans/len(self.train)*100)  # Percentage of NaNs
            if nans > max_frac * len(self.train):
                prlog(f' --> Removing feature {col} since it contains ' +
                      f'{nans} ({pnans}%) missing values.', self, 2)
                self.train.drop(col, axis=1, inplace=True)
                self.test.drop(col, axis=1, inplace=True)
                continue  # Skip to next column

            # Column is numerical and contains missing values
            if series.dtype.kind in 'ifu' and nans > 0:
                if strat_num not in strats:
                    try:
                        strat_num = float(strat_num)
                    except ValueError:
                        raise ValueError('Invalid value for strat_num!')

                    prlog(f' --> Imputing {nans} values with ' +
                          str(strat_num) + f' in feature {col}.', self, 2)
                    imp = SimpleImputer(strategy='constant',
                                        fill_value=strat_num)
                    fit_imputer(imp)

                elif strat_num.lower() == 'remove':
                    self.train.dropna(subset=[col], axis=0, inplace=True)
                    self.test.dropna(subset=[col], axis=0, inplace=True)
                    prlog(f' --> Removing {nans} rows due to missing ' +
                          f'values in feature {col}.', self, 2)

                else:
                    prlog(f' --> Imputing {nans} values with ' +
                          strat_num.lower() + f' in feature {col}.', self, 2)
                    imp = SimpleImputer(strategy=strat_num.lower())
                    fit_imputer(imp)

            # Column is categorical and contains missing values
            elif nans > 0:
                if strat_cat not in ['remove', 'most_frequent']:
                    if not isinstance(strat_cat, str):
                        raise ValueError('Invalid value for strat_cat!')

                    prlog(f' --> Imputing {nans} values with ' +
                          strat_cat.lower() + f' in feature {col}.', self, 2)
                    imp = SimpleImputer(strategy='constant',
                                        fill_value=strat_cat)
                    fit_imputer(imp)

                elif strat_cat.lower() == 'remove':
                    self.train.dropna(subset=[col], axis=0, inplace=True)
                    self.test.dropna(subset=[col], axis=0, inplace=True)
                    prlog(f' --> Removing {nans} rows due to missing ' +
                          f'values in feature {col}.', self, 2)

                else:
                    prlog(f' --> Imputing {nans} values with ' +
                          strat_cat.lower() + f' in feature {col}', self, 2)
                    imp = SimpleImputer(strategy=strat_cat.lower())
                    fit_imputer(imp)

        self._reset_attributes('train_test')  # Redefine new attributes

    @params_to_log
    def encode(self, max_onehot=10):

        '''
        DESCRIPTION -----------------------------------

        Perform encoding on categorical features. The encoding
        type depends on the number of unique values in the column.

        PARAMETERS -------------------------------------

        max_onehot --> threshold between onehot and label encoding

        '''

        prlog('Encoding categorical features...', self, 1)

        # Check parameter (if 0, 1 or 2: it never uses one_hot)
        max_onehot = int(max_onehot) if max_onehot >= 0 else 10

        for col in self.dataset:
            # Check if column is categorical
            if self.dataset[col].dtype.kind not in 'ifu':
                # Count number of unique values in the column
                n_unique = len(np.unique(self.dataset[col]))

                # Perform encoding type dependent on number of unique values
                if 2 < n_unique <= max_onehot:
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
                    prlog(f' --> Label-encoding feature {col}. Contains ' +
                          f'{n_unique} unique categories.', self, 2)
                    enc = LabelEncoder()
                    self.dataset[col] = enc.fit_transform(self.dataset[col])

        self._reset_attributes('dataset')  # Redefine new attributes

    @params_to_log
    def outliers(self, max_sigma=3, include_target=False):

        '''
        DESCRIPTION -----------------------------------

        Remove outliers from the dataset.

        PARAMETERS -------------------------------------

        max_sigma      --> maximum sigma accepted
        include_target --> include target column when deleting outliers

        '''

        # Check parameters
        max_sigma = float(max_sigma) if max_sigma > 0 else 3
        include_target = bool(include_target)

        prlog('Handling outliers...', self, 1)

        # Get z-score outliers index
        objective = self.train if include_target else self.X_train
        idx = (np.abs(zscore(objective)) < max_sigma).all(axis=1)

        delete = len(idx) - idx.sum()  # Number of False values in idx
        if delete > 0:
            prlog(f' --> Dropping {delete} rows due to outliers.', self, 2)

        # Remove rows based on index and reset attributes
        self.train = self.train[idx]
        self._reset_attributes('train_test')

    @params_to_log
    def balance(self, oversample=None, neighbors=5, undersample=None):

        '''
        DESCRIPTION -----------------------------------

        Balance the number of instances per target class.

        PARAMETERS -------------------------------------

        oversample  --> oversampling strategy using SMOTE. Choose from:
                            None: don't oversample
                            float: fraction minority/majority (only for binary)
                            'minority': resample only the minority class
                            'not minority': resample all but minority class
                            'not majority': resample all but majority class
                            'all': resample all classes

        neighbors   --> number of nearest neighbors for SMOTE
        undersample --> undersampling strategy using RandomUndersampler.
                        Choose from:
                            float: fraction majority/minority (only for binary)
                            'minority': resample only the minority class
                            'not minority': resample all but minority class
                            'not majority': resample all but majority class
                            'all': resample all classes

        '''

        # Check parameters
        oversample = float(oversample) if oversample > 0 else 0
        undersample = float(undersample) if undersample > 0 else 1
        neighbors = int(neighbors) if neighbors > 0 else 5

        if not imblearn_import:
            prlog("Unable to import imblearn. Skipping balancing the data...",
                  self)

        columns_x = self.X_train.columns  # Save name columns for later
        length = len(self.X_train)

        # Oversample the minority class with SMOTE
        if oversample is not None:
            prlog('Performing oversampling...', self, 1)
            smote = SMOTE(sampling_strategy=oversample,
                          k_neighbors=neighbors,
                          n_jobs=self.n_jobs)
            self.X_train, self.Y_train = smote.fit_resample(self.X_train,
                                                            self.Y_train)
            diff = len(self.X_train) - length  # Difference in length
            prlog(f' --> Adding {diff} rows to minority class.', self, 2)

        # Apply undersampling of majority class
        if undersample is not None:
            prlog('Performing undersampling...', self, 1)
            RUS = RandomUnderSampler(sampling_strategy=undersample)
            self.X_train, self.Y_train = RUS.fit_resample(self.X_train,
                                                          self.Y_train)
            diff = length - len(self.X_train)  # Difference in length
            prlog(f' --> Removing {diff} rows from majority class.', self, 2)

        self.X_train = convert_to_pd(self.X_train, columns=columns_x)
        self.Y_train = convert_to_pd(self.Y_train, columns=self.target)
        self.train = merge(self.X_train, self.Y_train)
        self._reset_attributes('train_test')

    @params_to_log
    def feature_selection(self,
                          strategy='univariate',
                          solver=None,
                          max_features=None,
                          threshold=-np.inf,
                          frac_variance=1.,
                          max_correlation=0.98):

        '''
        DESCRIPTION -----------------------------------

        Select best features according to a univariate F-test or with a
        recursive feature selector (RFS). Ties between features with equal
        scores will be broken in an unspecified way. Also removes features
        with too low variance and too high collinearity.

        PARAMETERS -------------------------------------

        X            --> data features: pd.Dataframe or array
        Y            --> data targets: pd.Series or array
        strategy     --> strategy for feature selection. Choose from:
                             'univariate': perform a univariate F-test
                             'PCA': perform principal component analysis
                             'SFM': select best features from model
        solver       --> solver or model class for the strategy
        max_features --> if < 1: fraction of features to select
                         if >= 1: number of features to select
                         None to select all (only for SFM)
        threshold    --> threshold value to use for selection. Only for model.
                         Choose from: float, 'mean', 'median'.
        frac_variance   --> minimum value of the Pearson correlation
                            cofficient to identify correlated features
        max_correlation --> remove features with constant instances in at
                            least this fraction of the total

        '''

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
            columns = ['drop_feature', 'corr_feature', 'corr_value']
            self.collinear = pd.DataFrame(columns=columns)

            # Iterate to record pairs of correlated features
            for column in to_drop:
                # Find the correlated features
                corr_features = list(upper.index[abs(upper[column]) > limit])

                # Find the correlated values
                corr_values = list(upper[column][abs(upper[column]) > limit])
                drop_features = [column for _ in corr_features]

                # Add to class attribute
                self.collinear = \
                    self.collinear.append({'drop_feature': drop_features,
                                           'corr_feature': corr_features,
                                           'corr_value': corr_values},
                                          ignore_index=True)

                prlog(f' --> Feature {column} was removed due to ' +
                      'collinearity with another feature.', self, 2)

            self.dataset.drop(to_drop, axis=1)

        def remove_low_variance(frac_variance):

            '''
            DESCRIPTION -----------------------------------

            Removes featrues with too low variance.

            PARAMETERS -------------------------------------

            frac_variance --> remove features with constant instances in at
                              least this fraction of the total

            '''

            threshold = frac_variance * (1. - frac_variance)
            var = VarianceThreshold(threshold=threshold).fit(self.X)
            mask = var.get_support()  # Get boolean mask of selected features

            for n, column in enumerate(self.X):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed due to' +
                          f' low variance: {var.variances_[n]:.2f}.', self, 2)
                    self.dataset.drop(column, axis=1, inplace=True)

        # Check parameters
        self.strategy = str(strategy) if strategy is not None else None
        self.solver = solver
        self.max_features = max_features
        self.threshold = threshold
        self.frac_variance = float(frac_variance)
        self.max_correlation = float(max_correlation)

        prlog('Performing feature selection...', self, 1)

        # First, drop features with too high correlation
        remove_collinear(limit=self.max_correlation)
        # Then, remove features with too low variance
        remove_low_variance(frac_variance=self.frac_variance)
        # Dataset is possibly changed so need to reset attributes
        self._reset_attributes('dataset')

        # Set max_features as all or fraction of total
        if self.max_features is None:
            self.max_features = self.X_train.shape[1]
        elif self.max_features < 1:
            self.max_features = int(self.max_features * self.X_train.shape[1])

        # Perform selection based on strategy
        if self.strategy.lower() == 'univariate':
            if max_features is None:
                max_features = self.dataset.shape[1]

            # Set the solver
            solvers = ['f_classif', 'f_regression', 'mutual_info_classif',
                       'mutual_info_regression', 'chi2']
            if self.solver is None and self.goal == 'regression':
                func = f_regression
            elif self.solver is None:
                func = f_classif
            elif self.solver in solvers:
                func = eval(self.solver)
            else:
                raise ValueError('Unknown value for the univariate solver.' +
                                 f'Try one of : {solvers}')

            self.univariate = SelectKBest(func, k=self.max_features)
            self.univariate.fit(self.X, self.Y)
            mask = self.univariate.get_support()
            for n, column in enumerate(self.X):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed after the ' +
                          'univariate test (score: {:.2f}  p-value: {:.2f}).'
                          .format(self.univariate.scores_[n],
                                  self.univariate.pvalues_[n]), self, 2)
                    self.dataset.drop(column, axis=1, inplace=True)
            self._reset_attributes('dataset')

        elif self.strategy.lower() == 'pca':
            prlog(f' --> Applying Principal Component Analysis... ', self, 2)

            # Scale features first
            self.dataset = StandardScaler().fit_transform(self.X_train)
            self.solver = 'auto' if self.solver is None else self.solver
            self.PCA = PCA(n_components=max_features, svd_solver=solver)
            self.X_train = convert_to_pd(self.PCA.fit_transform(self.X_train))
            self.X_test = convert_to_pd(self.PCA.transform(self.X_test))
            self.train = merge(self.X_train, self.Y_train)
            self.test = merge(self.X_test, self.Y_test)
            self._reset_attributes('train_test')

        elif self.strategy.lower() == 'sfm':
            if self.solver is None:
                raise ValueError('Select a model for the solver!')

            self.SFM = SelectFromModel(estimator=self.solver,
                                       threshold=self.threshold,
                                       max_features=self.max_features)
            self.SFM.fit(self.X, self.Y)
            mask = self.SFM.get_support()

            for n, column in enumerate(self.X):
                if not mask[n]:
                    prlog(f' --> Feature {column} was removed by the ' +
                          'recursive feature eliminator.', self, 2)
                    self.dataset.drop(column, axis=1, inplace=True)
            self._reset_attributes('dataset')

        elif self.strategy is not None:
            raise ValueError('Invalid feature selection strategy selected.'
                             "Choose from: 'univariate', 'PCA' or 'SFM'")

    @params_to_log
    def fit(self, models=None, metric=None, successive_halving=False,
            skip_iter=0, max_iter=15, max_time=3600, eps=1e-08,
            batch_size=1, init_points=5, plot_bo=False, cv=3, bootstrap=None):

        '''
        DESCRIPTION -----------------------------------

        Initialize class.

        PARAMETERS -------------------------------------

        models             --> list of models to use
        metric             --> metric to perform evaluation on
        successive_halving --> wether to perform successive halving
        skip_iter          --> skip last n steps of successive halving
        max_iter           --> maximum number of iterations of the BO
        max_time           --> maximum time for the BO (in seconds)
        eps                --> minimum distance between two consecutive x's
        batch_size         --> batch size in which the objective is evaluated
        init_points        --> initial number of random tests of the BO
        plot_bo            --> boolean to plot the BO's progress
        cv                 --> splits for the cross validation
        bootstrap          --> number of splits for bootstrapping (0 for None)

        '''

        # << ============ Inner Function ============ >>

        def run_iteration(self):
            ''' Core iterations, multiple needed for successive halving '''

            # In case there is no cross-validation
            results = 'No cross-validation performed'

            # If verbose=1, use tqdm to evaluate process
            loop = tqdm(self.models) if self.verbose == 1 else self.models

            # Loop over every independent model
            for model in loop:
                # Define model class
                setattr(self, model, eval(model)(self.data,
                                                 self.metric,
                                                 self.goal,
                                                 self.log,
                                                 self.verbose))

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
                                                         self.plot_bo,
                                                         self.n_jobs)

                        if self.bootstrap is not None:
                            getattr(self, model).bootstrap(self.bootstrap)

                except Exception as ex:
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
                lenx = max([len(getattr(self, m).name) for m in self.models])

                # Get best score (min or max dependent on metric)
                metric_to_min = ['max_error', 'MAE', 'MSE', 'MSLE']
                if self.bootstrap is None:
                    x = [getattr(self, m).score for m in self.models]
                else:
                    x = [getattr(self, m).results.mean() for m in self.models]
                max_ = min(x) if self.metric in metric_to_min else max(x)

            except (ValueError, AttributeError):
                raise ValueError('It appears all models failed to run...')

            # Print final results
            t = time() - t_init  # Total time in seconds
            h = int(t/3600.)
            m = int(t/60.) - h*60
            s = int(t - h*3600 - m*60)
            prlog('\n\nFinal stats ================>>', self)
            prlog(f'Duration: {h:02}h:{m:02}m:{s:02}s', self)
            prlog(f'Target metric: {self.metric}', self)
            prlog('--------------------------------', self)

            # Create dataframe with final results
            if self.bootstrap is None:
                results = pd.DataFrame(columns=['model', 'score'])
            else:
                results = pd.DataFrame(columns=['model',
                                                'score',
                                                'bootstrap_mean',
                                                'bootstrap_std'])

            for m in self.models:
                name = getattr(self, m).name
                shortname = getattr(self, m).shortname
                score = getattr(self, m).score

                if self.bootstrap is None:
                    results = results.append({'model': shortname,
                                              'score': score},
                                             ignore_index=True)

                    # Highlight best score (if more than one)
                    if score == max_ and len(self.models) > 1:
                        prlog(u'{0:{1}s} --> {2:.3f} !!'
                              .format(name, lenx, score), self)
                    else:
                        prlog(u'{0:{1}s} --> {2:.3f}'
                              .format(name, lenx, score), self)

                else:
                    bs_mean = getattr(self, m).results.mean()
                    bs_std = getattr(self, m).results.std()
                    results = results.append({'model': shortname,
                                              'score': score,
                                              'bootstrap_mean': bs_mean,
                                              'bootstrap_std': bs_std},
                                             ignore_index=True)

                    # Highlight best score (if more than one)
                    if bs_mean == max_ and len(self.models) > 1:
                        prlog(u'{0:{1}s} --> {2:.3f} \u00B1 {3:.3f} !!'
                              .format(name, lenx, bs_mean, bs_std), self)
                    else:
                        prlog(u'{0:{1}s} --> {2:.3f} \u00B1 {3:.3f}'
                              .format(name, lenx, bs_mean, bs_std), self)

            return results

        def data_preparation():
            ''' Make a dct of the data (complete, train, test and scaled) '''

            data = {}
            for i in ['X', 'Y', 'X_train', 'Y_train', 'X_test', 'Y_test']:
                data[i] = eval('self.' + i)

            # List of models that need scaling
            scaling_models = ['LinReg', 'LogReg', 'KNN', 'XGB', 'LGBM',
                              'lSVM', 'kSVM', 'PA', 'SGD', 'MLP']
            # Check if any scaling models in final_models
            scale = any(model in self.models for model in scaling_models)
            # If PCA was performed, features are already scaled
            # Made string in case it is None
            if hasattr(self, 'strategy'):
                pca = True if str(self.strategy).lower() == 'pca' else False
            else:
                pca = False
            if scale and not pca:
                # Normalize features to mean=0, std=1
                data['X_scaled'] = StandardScaler().fit_transform(data['X'])
                scaler = StandardScaler().fit(data['X_train'])
                data['X_train_scaled'] = scaler.transform(data['X_train'])
                data['X_test_scaled'] = scaler.transform(data['X_test'])

            return data

        # << ============ Initialize ============ >>

        t_init = time()  # To measure the time the whole pipeline takes

        prlog('\nRunning pipeline =================>', self)

        # Set args to class attributes and correct inputs
        if not isinstance(models, list) and models is not None:
            self.models = list(models)
        else:
            self.models = models
        self.metric = str(metric) if metric is not None else None
        self.successive_halving = bool(successive_halving)
        self.skip_iter = int(skip_iter)
        self.max_iter = int(max_iter) if max_iter > 0 else 15
        self.max_time = int(max_time) if max_time > 0 else np.inf
        self.eps = float(eps)
        self.batch_size = int(batch_size) if batch_size > 0 else 1
        self.init_points = int(init_points) if init_points > 0 else 5
        self.plot_bo = bool(plot_bo)
        self.cv = int(cv) if cv > 0 else 3
        if bootstrap is None or bootstrap == 0:
            self.bootstrap = None
        else:
            self.bootstrap = int(bootstrap)

        # Save model erros (if any) in dictionary
        self.errors = {}

        # Save the cv's results in array of dataframes
        if self.successive_halving:
            self.results = []

        # << ============ Check validity models ============ >>

        model_list = ['BNB', 'GNB', 'MNB', 'GP', 'LinReg', 'LogReg', 'LDA',
                      'QDA', 'KNN', 'Tree', 'Bag', 'ET', 'RF', 'AdaBoost',
                      'GBM', 'XGB', 'LGBM', 'lSVM', 'kSVM', 'PA', 'SGD', 'MLP']

        # Final list of models to be used
        final_models = []
        if self.models is None:  # Use all possible models (default)
            final_models = model_list.copy()
        else:
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
                            final_models.append(n)
                            break

        # Check if XGBoost and lightgbm are available
        for model, package in zip(['XGB', 'LGBM'], ['xgboost', 'lightgbm']):
            if model in final_models:
                try:
                    importlib.import_module(package)
                except ImportError:
                    prlog(f'Unable to import {package}. Removing ' +
                          'model from pipeline.', self)
                    final_models.remove(model)

        # Linear regression can't perform classification
        if 'LinReg' in final_models and self.goal != 'regression':
            prlog("Linear Regression can't perform classification tasks."
                  + " Removing model from pipeline.", self)
            final_models.remove('LinReg')

        # Remove classification-only models from pipeline
        if self.goal == 'regression':
            class_models = ['BNB', 'GNB', 'MNB', 'LogReg', 'LDA', 'QDA']
            for model in class_models:
                if model in final_models:
                    prlog(f"{model} can't perform regression tasks."
                          + " Removing model from pipeline.", self)
                    final_models.remove(model)

        # Check if there are still valid models
        if len(final_models) == 0:
            raise ValueError("No models found in pipeline. Try one of {}"
                             .format(model_list))

        # Update model list attribute with correct values
        self.models = final_models
        if not self.successive_halving:
            prlog(f'Models in pipeline: {self.models}', self)

        # Set default metric
        if self.metric is None and self.goal == 'binary classification':
            self.metric = 'F1'
        elif self.metric is None:
            self.metric = 'MSE'

        # Check validity metric
        metric_class = ['Precision', 'Recall', 'Accuracy',
                        'F1', 'AUC', 'Jaccard']
        mreg = ['R2', 'max_error', 'MAE', 'MSE', 'MSLE']
        for m in metric_class + mreg:
            # Compare strings case insensitive
            if self.metric.lower() == m.lower():
                self.metric = m

        if self.metric not in metric_class + mreg:
            temp = mreg if self.goal == 'regression' else metric_class
            raise ValueError(f'Unknown metric. Try one of {temp}.')
        elif self.metric == 'AUC' and self.goal != 'binary classification':
            raise ValueError('AUC only works for binary classification tasks.')
        elif self.metric not in mreg and self.goal == 'regression':
            raise ValueError("{} is an invalid metric for {}. Try one of {}."
                             .format(self.metric, self.goal, mreg))

        # << =================== Core ==================== >>

        if self.successive_halving:
            iteration = 0
            original_data = self.dataset.copy()
            while len(self.models) > 2**self.skip_iter - 1:
                # Select 1/N of data to use for this iteration
                self._split_dataset(original_data, 100./len(self.models))
                self._reset_attributes()
                self.data = data_preparation()
                prlog('\n\n<<================ Iteration {} ================>>'
                      .format(iteration), self)
                prlog(f'Models in pipeline: {self.models}', self)
                self.stats()

                # Run iteration
                results = run_iteration(self)
                self.results.append(results)

                # Select best models for halving
                col = 'score' if self.bootstrap is None else 'bootstrap_mean'
                lx = results.nlargest(n=int(len(self.models)/2),
                                      columns=col,
                                      keep='all')

                # Keep the models in the same order
                n = []  # List of new models
                [n.append(m) for m in self.models if m in list(lx.model)]
                self.models = n.copy()
                iteration += 1

        else:
            self.data = data_preparation()
            self.results = run_iteration(self)

        # <====================== End fit function ======================>

    def boxplot(self, i=-1, figsize=None, filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot a boxplot of the found metric results.

        PARAMETERS -------------------------------------

        i        --> iteration of the successive halving to plot (default last)
        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

        results, names = [], []
        try:  # Can't make plot before running fit!
            df = self.results[i] if self.successive_halving else self.results
            for m in df.model:
                results.append(getattr(self, m).results)
                names.append(getattr(self, m).shortname)
        except (IndexError, AttributeError):
            raise Exception('You need to fit ATOM using bootstrap first!')

        if figsize is None:
            figsize = (int(8+len(names)/2), 6)

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.xlabel('Model', fontsize=16, labelpad=12)
        plt.ylabel(self.metric, fontsize=16, labelpad=12)
        plt.title('Model comparison', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_successive_halving(self, figsize=(10, 6), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot the successive halving scores.

        PARAMETERS -------------------------------------

        figsize  --> figure size: format as (x, y)
        filename --> name of the file to save

        '''

        if not self.successive_halving:
            raise ValueError('This plot is only available if the class was ' +
                             'fitted using a successive halving approach!')

        models = self.results[0].model  # List of models in first iteration
        col = 'score' if self.bootstrap is None else 'bootstrap_mean'
        linx = [[] for m in models]
        liny = [[] for m in models]
        try:  # Can't make plot before running fit!
            for m, df in enumerate(self.results):
                for n, model in enumerate(models):
                    if model in df.model.values:  # If model in iteration
                        linx[n].append(m)
                        liny[n].append(
                                df[col][df.model == model].values[0])
        except (IndexError, AttributeError):
            raise Exception('You need to fit the class before plotting!')

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        for x, y, label in zip(linx, liny, models):
            plt.plot(x, y, lw=2, marker='o', label=label)
        plt.xlim(-0.1, len(self.results)-0.9)
        plt.xlabel('Iteration', fontsize=16, labelpad=12)
        plt.ylabel(self.metric, fontsize=16, labelpad=12)
        plt.title('Successive halving scores', fontsize=16)
        plt.legend(frameon=False, fontsize=14)
        ax.set_xticks(range(len(self.results)))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_correlation(self, figsize=(10, 10), filename=None):

        '''
        DESCRIPTION -----------------------------------

        Plot the feature's correlation matrix. Ignores non-numeric columns.

        PARAMETERS -------------------------------------

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

        sns.set_style('white')
        fig, ax = plt.subplots(figsize=figsize)

        # Draw the heatmap with the mask and correct aspect ratio
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Feature correlation matrix', fontsize=16)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()
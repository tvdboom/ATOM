# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom
Description: Module containing the main ATOM class

"""

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import multiprocessing
import warnings as warn
import importlib
from scipy.stats import zscore
from typeguard import typechecked
from typing import Union, Optional, Sequence, List, Tuple

# Sklearn
from sklearn.metrics import SCORERS, get_scorer, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
     f_classif, f_regression, mutual_info_classif, mutual_info_regression,
     chi2, SelectKBest, VarianceThreshold, SelectFromModel, RFE
    )

# Own package modules
from .utils import (
     composed, crash, params_to_log, time_to_string,
     to_df, to_series, merge, check_is_fitted
     )
from .plots import (
        save, plot_correlation, plot_PCA, plot_ROC, plot_PRC, plot_bagging,
        plot_successive_halving, plot_permutation_importance,
        plot_feature_importance, plot_confusion_matrix, plot_threshold,
        plot_probabilities
        )
from .models import (
     GaussianProcess, GaussianNaïveBayes, MultinomialNaïveBayes,
     BernoulliNaïveBayes, OrdinaryLeastSquares, Ridge, Lasso, ElasticNet,
     BayesianRegression, LogisticRegression, LinearDiscriminantAnalysis,
     QuadraticDiscriminantAnalysis, KNearestNeighbors, DecisionTree,
     Bagging, ExtraTrees, RandomForest, AdaBoost, GradientBoostingMachine,
     XGBoost, LightGBM, CatBoost, LinearSVM, KernelSVM, PassiveAggressive,
     StochasticGradientDescent, MultilayerPerceptron
     )

# Plotting
import seaborn as sns
sns.set(style='darkgrid', palette='GnBu_d')


# << ============ Global variables ============ >>

# Variable types
scalar = Union[int, float]

# List of all the available models
model_list = dict(GP=GaussianProcess,
                  GNB=GaussianNaïveBayes,
                  MNB=MultinomialNaïveBayes,
                  BNB=BernoulliNaïveBayes,
                  OLS=OrdinaryLeastSquares,
                  Ridge=Ridge,
                  Lasso=Lasso,
                  EN=ElasticNet,
                  BR=BayesianRegression,
                  LR=LogisticRegression,
                  LDA=LinearDiscriminantAnalysis,
                  QDA=QuadraticDiscriminantAnalysis,
                  KNN=KNearestNeighbors,
                  Tree=DecisionTree,
                  Bag=Bagging,
                  ET=ExtraTrees,
                  RF=RandomForest,
                  AdaB=AdaBoost,
                  GBM=GradientBoostingMachine,
                  XGB=XGBoost,
                  LGB=LightGBM,
                  CatB=CatBoost,
                  lSVM=LinearSVM,
                  kSVM=KernelSVM,
                  PA=PassiveAggressive,
                  SGD=StochasticGradientDescent,
                  MLP=MultilayerPerceptron)

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


# << ================= Classes ================= >>

class ATOM(object):

    # Define class variables for plot aesthetics
    style = 'darkgrid'
    palette = 'GnBu_d'
    title_fontsize = 20
    label_fontsize = 16
    tick_fontsize = 12

    # << ================= Methods ================= >>

    @composed(crash, params_to_log, typechecked)
    def __init__(self,
                 X: Union[dict, Sequence[Sequence], np.ndarray, pd.DataFrame],
                 y: Union[None, str, list, tuple, dict, np.ndarray, pd.Series],
                 percentage: scalar,
                 test_size: float,
                 n_jobs: int,
                 warnings: bool,
                 verbose: int,
                 random_state: Optional[int]):

        """
        Transform the input data into pandas objects and set the data class
        attributes. Also performs standard data cleaning steps.

        PARAMETERS
        ----------
        X: dict, iterable, np.array or pd.DataFrame
            Dataset containing the features, with shape=(n_samples, n_features)

        y: string, iterable, np.array or pd.Series, optional (default=None)
            - If None: the last column of X is selected as target column
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        percentage: int or float, optional (default=100)
            Percentage of the data to use in the pipeline.

        test_size: float, optional (default=0.3)
            Split fraction of the train and test set.

        n_jobs: int, optional (default=1)
            Number of cores to use for parallel processing.
                - If -1, use all available cores
                - If <-1, use available_cores - 1 + n_jobs

            Beware that using multiple processes on the same machine may cause
            memory issues for large datasets.

        warnings: bool, optional (default=False)
            Wether to show warnings when fitting the models.

        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything
                - 1 to print minimum information
                - 2 to print average information
                - 3 to print maximum information

        random_state: int or None, optional (default=None)
            Seed used by the random number generator. If None, the random
            number generator is the RandomState instance used by `np.random`.

        """

        # << ============ Handle input data ============ >>

        # Convert X to pd.DataFrame
        if not isinstance(X, pd.DataFrame):
            X = to_df(X)

        # Convert array to dataframe and target column to pandas series
        if isinstance(y, (list, tuple, dict, np.ndarray, pd.Series)):
            if len(X) != len(y):
                raise ValueError("X and y don't have the same number of " +
                                 f"rows: {len(X)}, {len(y)}.")

            # Convert y to pd.Series
            if not isinstance(y, pd.Series):
                if not isinstance(y, np.ndarray):
                    y = np.array(y)

                # Check that y is one-dimensional
                if y.ndim != 1:
                    raise ValueError("y should be one-dimensional, got " +
                                     f"y.ndim={y.ndim}.")
                y = to_series(y)

            # Reset indices in case they are not in unison (else merge fails)
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)

            # Merge to one single dataframe with all data
            self.dataset = merge(X, y)
            self.target = y.name

        elif isinstance(y, str):
            if y not in X.columns:
                raise ValueError("Target column not found in X!")

            # Place target column last
            X = X[[col for col in X if col != y] + [y]]
            self.dataset = X.reset_index(drop=True)
            self.target = y

        elif y is None:
            self.dataset = X.reset_index(drop=True)
            self.target = self.dataset.columns[-1]

        # << ============ Check input parameters ============ >>

        if percentage <= 0 or percentage > 100:
            raise ValueError("Invalid value for the percentage parameter." +
                             "Value should be between 0 and 100, got {}."
                             .format(percentage))

        if test_size <= 0 or test_size >= 1:
            raise ValueError("Invalid value for the test_size parameter." +
                             "Value should be between 0 and 1, got {}."
                             .format(test_size))

        if verbose < 0 or verbose > 3:
            raise ValueError("Invalid value for the verbose parameter." +
                             "Value should be between 0 and 3, got {}."
                             .format(verbose))

        # Update attributes witrh params
        self.percentage = percentage
        self.test_size = test_size
        self.warnings = warnings
        if not warnings:  # Ignore all warnings
            warn.filterwarnings('ignore')
        self.verbose = verbose
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)  # Set random seed

        # Check if features are already scaled
        mean = self.dataset.mean(axis=1).mean()
        std = self.dataset.std(axis=1).mean()
        self._is_scaled = True if mean < 0.05 and 0.5 < std < 1.5 else False
        self._is_fitted = False  # Model has not been fitted yet

        self._log("<<=============== ATOM ===============>>")

        # Check number of cores for multiprocessing
        n_cores = multiprocessing.cpu_count()
        if n_jobs > n_cores:
            self._log("Warning! No {} cores available. n_jobs reduced to {}."
                      .format(n_jobs, n_cores))
            self.n_jobs = n_cores

        elif n_jobs == 0:
            self._log("Warning! Value of n_jobs can't be 0. Using 1 core.")
            self.n_jobs = 1

        else:
            if n_jobs < 0:
                self.n_jobs = n_cores + 1 + n_jobs
            else:
                self.n_jobs = n_jobs

            # Final check for negative input
            if self.n_jobs < 1:
                raise ValueError("Invalid value for the n_jobs parameter, " +
                                 f"got {n_jobs}.")

            elif self.n_jobs != 1:
                self._log(f"Parallel processing with {self.n_jobs} cores.")

        # << ============ Data cleaning ============ >>

        self._log("Initial data cleaning...", 1)

        for column in self.dataset:
            nunique = self.dataset[column].nunique(dropna=True)
            unique = self.dataset[column].unique()  # List of unique values

            # Drop features with incorrect column type
            dtype = str(self.dataset[column].dtype)
            if dtype in ('datetime64', 'datetime64[ns]', 'timedelta[ns]'):
                self._log(f" --> Dropping feature {column} due to " +
                          f"unhashable type: {dtype}.", 2)
                self.dataset.drop(column, axis=1, inplace=True)
                continue
            elif dtype in ('object', 'category'):
                # Strip categorical features from blank spaces
                self.dataset[column].astype(str).str.strip()

                # Drop features where all values are unique
                if nunique == len(self.dataset):
                    self._log(f" --> Dropping feature {column} due to " +
                              "maximum cardinality.", 2)
                    self.dataset.drop(column, axis=1, inplace=True)

            # Drop features where all values are the same
            if nunique == 1:
                if column == self.target:
                    raise ValueError(f"Only found 1 target value: {unique[0]}")
                else:
                    self._log(f" --> Dropping feature {column}. Contains " +
                              f"only one unique value: {unique[0]}", 2)
                    self.dataset.drop(column, axis=1, inplace=True)

        # Drop duplicate rows
        length = len(self.dataset)
        self.dataset.drop_duplicates(inplace=True)
        diff = length - len(self.dataset)  # Difference in length
        if diff > 0:
            self._log(f" --> Dropping {diff} duplicate rows.", 2)

        # Delete rows with NaN in target
        length = len(self.dataset)
        self.dataset.dropna(subset=[self.target], inplace=True)
        diff = length - len(self.dataset)  # Difference in length
        if diff > 0:
            self._log(f" --> Dropping {diff} rows with missing values " +
                      "in target column.", 2)

        # << ============ Set algorithm task ============ >>

        # Get unique target values before encoding (for later print)
        self._unique = sorted(self.dataset[self.target].unique())

        if len(self._unique) == 2 and self.goal == 'classification':
            self._log("Algorithm task: binary classification.")
            self.task = 'binary classification'
        elif self.goal == 'classification':
            self._log("Algorithm task: multiclass classification." +
                      f" Number of classes: {len(self._unique)}.")
            self.task = 'multiclass classification'
        elif self.goal == 'regression':
            self._log("Algorithm task: regression.")
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
            self.mapping = "No target mapping for regression tasks"

        # << =========================================== >>

        self._split_dataset(self.dataset, self.percentage)  # Train/test split
        self.update()  # Define data subsets class attributes
        self.stats(1)  # Print out data stats

    # << ======================= Utility methods ======================= >>

    def _log(self, string, level=0):

        """
        Print and save output to log file.

        PARAMETERS
        ----------
        string: string
            Message to save to log and print to stdout.

        level: int
            Minimum verbosity level in order to print the message to stdout.

        """

        if self.verbose > level:
            print(string)

        if self.log is not None:
            while string.startswith('\n'):  # Insert empty lines for clean view
                self.log.info('')
                string = string[1:]
            self.log.info(string)

    def _split_dataset(self, dataset, percentage=100):

        """
        Split a percentage of the dataset into a train and test set.

        PARAMETERS
        ----------
        dataset: pd.DataFrame
            Dataset to use for splitting.

        percentage: int or float, optional (default=100)
            Percentage of the data to use.

        """

        # Get percentage of data (for successive halving)
        self.dataset = dataset.sample(frac=1)  # Shuffle first
        self.dataset = self.dataset.head(int(len(self.dataset)*percentage/100))
        self.dataset.reset_index(drop=True, inplace=True)

        # Split train and test sets on percentage of data
        self.train, self.test = train_test_split(self.dataset,
                                                 test_size=self.test_size,
                                                 shuffle=False)

    @composed(crash, params_to_log)
    def stats(self, _verbose: int = -2):

        """
        Print some information about the dataset.

        PARAMETERS
        ----------
        _verbose: int, optional (default=-2)
            Internal parameter to always print if the user calls this method.

        """

        self._log("\nDataset stats ===================>", _verbose)
        self._log(f"Shape: {self.dataset.shape}", _verbose)

        nans = self.dataset.isna().sum().sum()
        if nans > 0:
            self._log(f"Missing values: {nans}", _verbose)
        categ = self.X.select_dtypes(include=['category', 'object']).shape[1]
        if categ > 0:
            self._log(f"Categorical columns: {categ}", _verbose)

        self._log(f"Scaled: {self._is_scaled}", _verbose)
        self._log("----------------------------------", _verbose)
        self._log(f"Size of training set: {len(self.train)}", _verbose)
        self._log(f"Size of test set: {len(self.test)}", _verbose)

        # Print count of target classes
        if self.task != 'regression':
            self._log("----------------------------------", _verbose + 1)
            self._log(f"Instances in {self.target} per class:", _verbose + 1)

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
            self._log(stats.to_markdown(), _verbose + 1)

        self._log('', 1)  # Insert an empty row

    @composed(crash, params_to_log)
    def scale(self, _print: bool = True):

        """
        Scale features to mean=0 and std=1.

        PARAMETERS
        ----------
        _print: bool, optional (default=True)
            Internal parameter to know if printing is needed.

        """

        columns_x = self.X_train.columns

        # Check if features are already scaled
        if not self._is_scaled:
            if _print:
                self._log("Scaling features...", 1)

            scaler = StandardScaler()
            self.X_train = to_df(scaler.fit_transform(self.X_train), columns_x)
            self.X_test = to_df(scaler.transform(self.X_test), columns_x)
            self.update('X_train')
            self._is_scaled = True

        elif _print:  # Inform the user
            self._log("The features are already scaled!")

    @composed(crash, typechecked)
    def update(self, df: str = 'dataset'):

        """
        If you change any of the class' data attributes (dataset, X, y,
        train, test, X_train, X_test, y_train, y_test) in between the
        pipeline, you should call this method to change all other data
        attributes to their correct values. Independent attributes are
        updated in unison, that is, setting df='X_train' will also
        update X_test, y_train and y_test, or df='train' will also
        update the test set, etc...

        PARAMETERS
        ----------
        df: str, optional (default='dataset')
            Data attribute (as string) that has been changed.

        """

        if df not in ['train_test', 'X_y'] and not hasattr(self, df):
            raise ValueError("Invalid value for the df parameter." +
                             "Value should be a data attribute of the ATOM " +
                             f"class, got {df}. See the documentation " +
                             "for the available options.")

        # Depending on df, we change some attributes or others
        if df == 'dataset':
            # Possible because the length of self.train doesn't change
            self.train = self.dataset[:len(self.train)]
            self.test = self.dataset[len(self.train):]

        elif df in ('train_test', 'train', 'test'):
            # Join train and test on rows
            self.dataset = pd.concat([self.train, self.test], join='outer',
                                     ignore_index=False, copy=True)

        elif df in ('X_train', 'y_train', 'X_test', 'y_test'):
            self.train = merge(self.X_train, self.y_train)
            self.test = merge(self.X_test, self.y_test)
            self.dataset = pd.concat([self.train, self.test], join='outer',
                                     ignore_index=False, copy=True)

        elif df in ('X_y', 'X', 'y'):
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

    @composed(crash, params_to_log, typechecked)
    def report(self,
               df: str = 'dataset',
               rows: Optional[scalar] = None,  # float for 1e3, etc.
               filename: Optional[str] = None):

        """
        Get an extensive profile analysis of the data. The report is rendered
        in HTML5 and CSS3. Note that this method can be slow for rows>10k.
        Dependency: pandas-profiling.

        PARAMETERS
        ----------
        df: str, optional(default='dataset')
            Name of the data class attribute to get the report from.

        rows: int or None, optional(default=None)
            Number of rows to process (randomly picked). None for all rows.

        filename: str or None, optional (default=None)
            Name of the file when saved (as .html). None to not save anything.

        """

        try:
            from pandas_profiling import ProfileReport
        except ImportError:
            raise ModuleNotFoundError("Failed to import the pandas-profiling" +
                                      " package. Install it before using the" +
                                      " report method.")

        # If rows=None, select all rows in the dataframe
        rows = getattr(self, df).shape[0] if rows is None else int(rows)

        self._log("Creating profile report...", 1)

        ProfileReport(getattr(self, df).sample(rows))
        try:  # Render if possible (for jupyter notebook)
            from IPython.display import display
            display(self.report)
        except Exception:
            pass

        if filename is not None:
            if not filename.endswith('.html'):
                filename = filename + '.html'
            self.report.to_file(filename)

    @composed(crash, params_to_log, typechecked)
    def results(self, metric: Optional[str] = None):

        """
        Print the pipeline's final results for a specific metric.

        PARAMETERS
        ----------
        metric: string or None, optional (default=None)
            String of one of sklearn's predefined metrics. If None, the metric
            used to fit the pipeline is selected.

        """

        check_is_fitted(self._is_fitted)  # Raise error is class not fitted
        if metric is None:
            metric = self.metric.name
        elif metric not in SCORERS.keys():
            raise ValueError("Unknown value for the metric parameter, " +
                             f"got {metric}. Try one of {SCORERS.keys()}.")

        # Get max length of the models' names
        maxlen = max([len(getattr(self, m).longname) for m in self.models])

        # Get list of scores
        scores = [getattr(getattr(self, m), metric) for m in self.models]

        self._log("\nFinal results ================>>")
        self._log(f"Metric: {metric}")
        self._log("--------------------------------")

        for m in self.models:
            # Check that the metric is valid for this task
            if isinstance(getattr(getattr(self, m), metric), str):
                raise ValueError(f"Invalid metric for {self.task} tasks!")

            longname = getattr(self, m).longname
            score_test = getattr(getattr(self, m), metric)

            # Create string of the score
            print_ = "{:{}s} --> {:.3f}".format(longname, maxlen, score_test)

            # Highlight best score (if more than one model in pipeline)
            if score_test == max(scores) and len(self.models) > 1:
                print_ += ' !!'

            self._log(print_, -2)  # Always print

    @composed(crash, params_to_log, typechecked)
    def save(self, filename: Optional[str] = None):

        """
        Save the ATOM class to a pickle file.

        Parameters
        ----------
        filename: str or None, optional (default=None)
            Name to save the file with. None to save with default name.

        """

        save(self, self.__class__.__name__ if filename is None else filename)
        self._log("ATOM class saved successfully!", 1)

    # << ================ Data pre-procesing methods ================ >>

    @composed(crash, params_to_log, typechecked)
    def impute(self,
               strat_num: Union[scalar, str] = 'remove',
               strat_cat: str = 'remove',
               max_frac_rows: float = 0.5,
               max_frac_cols: float = 0.5,
               missing: Optional[Union[scalar, str, list]] = None):

        """
        Handle missing values according to the selected strategy. Also
        removes rows and columns with too many missing values.

        PARAMETERS
        ----------
        strat_num: str, int or float, optional (default='remove')
            Imputing strategy for numerical columns. Choose from:
                - 'remove': remove row if any missing value
                - 'mean': impute with mean of column
                - 'median': impute with median of column
                - 'knn': impute using a K-Nearest Neighbors approach
                - 'most_frequent': impute with most frequent value
                - int or float: impute with provided numerical value

        strat_cat: str, optional (default='remove')
            Imputing strategy for categorical columns. Choose from:
                - 'remove': remove row if any missing value
                - 'most_frequent': impute with most frequent value
                - string: impute with provided string

        min_frac_rows: float, optional (default=0.5)
            Minimum fraction of non missing values in row. If less,
            the row is removed.

        min_frac_cols: float, optional (default=0.5)
            Minimum fraction of non missing values in column. If less,
            the column is removed.

        missing: int, float or list, optional (default=None)
            List of values to impute. None for default list: [None, np.NaN,
            np.inf, -np.inf, '', '?', 'NA', 'nan', 'inf']

        """

        def fit_imputer(imputer):
            """ Fit and transform the imputer class """

            self.train[col] = imputer.fit_transform(
                                        self.train[col].values.reshape(-1, 1))
            self.test[col] = imputer.transform(
                                        self.test[col].values.reshape(-1, 1))

        # Check input parameters
        strats = ['remove', 'mean', 'median', 'knn', 'most_frequent']
        if isinstance(strat_num, str) and strat_num.lower() not in strats:
            raise ValueError("Unknown strategy for the strat_num parameter" +
                             ", got {}. Choose from: {}."
                             .format(strat_num, ', '.join(strats)))
        if max_frac_rows <= 0 or max_frac_rows >= 1:
            raise ValueError("Invalid value for the max_frac_rows parameter." +
                             "Value should be between 0 and 1, got {}."
                             .format(max_frac_rows))
        if max_frac_cols <= 0 or max_frac_cols >= 1:
            raise ValueError("Invalid value for the max_frac_cols parameter." +
                             "Value should be between 0 and 1, got {}."
                             .format(max_frac_cols))

        # Set default missing list
        if missing is None:
            missing = [np.inf, -np.inf, '', '?', 'NA', 'nan', 'inf']
        elif not isinstance(missing, list):
            missing = [missing]  # Has to be an iterable for loop

        # Some values must always be imputed (but can be double)
        missing.extend([np.inf, -np.inf])
        missing = set(missing)

        self._log("Imputing missing values...", 1)

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
            self._log(f" --> Removing {diff} rows for containing too many " +
                      "missing values.", 2)

        self.update('dataset')  # Fill train and test with NaNs

        # Loop over all columns to apply strategy dependent on type
        for col in self.dataset:
            series = self.dataset[col]

            # Drop columns with too many NaN values
            nans = series.isna().sum()  # Number of missing values in column
            pnans = int(nans/len(self.dataset) * 100)  # Percentage of NaNs
            if nans > max_frac_cols * len(self.dataset):
                self._log(f" --> Removing feature {col} for containing " +
                          f"{nans} ({pnans}%) missing values.", 2)
                self.train.drop(col, axis=1, inplace=True)
                self.test.drop(col, axis=1, inplace=True)
                continue  # Skip to next column

            # Column is numerical and contains missing values
            if series.dtype.kind in 'ifu' and nans > 0:
                if not isinstance(strat_num, str):
                    self._log(f" --> Imputing {nans} missing values with " +
                              f"number {str(strat_num)} in feature {col}.", 2)
                    imp = SimpleImputer(strategy='constant',
                                        fill_value=strat_num)
                    fit_imputer(imp)

                elif strat_num.lower() == 'remove':
                    self.train.dropna(subset=[col], axis=0, inplace=True)
                    self.test.dropna(subset=[col], axis=0, inplace=True)
                    self._log(f" --> Removing {nans} rows due to missing " +
                              f"values in feature {col}.", 2)

                elif strat_num.lower() == 'knn':
                    self._log(f" --> Imputing {nans} missing values using " +
                              f"the KNN imputer in feature {col}.", 2)
                    imp = KNNImputer()
                    fit_imputer(imp)

                elif strat_num.lower() in ('mean', 'median', 'most_frequent'):
                    self._log(f" --> Imputing {nans} missing values with " +
                              f"{strat_num.lower()} in feature {col}.", 2)
                    imp = SimpleImputer(strategy=strat_num.lower())
                    fit_imputer(imp)

            # Column is non-numeric and contains missing values
            elif nans > 0:
                if strat_cat.lower() not in ['remove', 'most_frequent']:
                    self._log(f" --> Imputing {nans} missing values with " +
                              f"{strat_cat} in feature {col}.", 2)
                    imp = SimpleImputer(strategy='constant',
                                        fill_value=strat_cat)
                    fit_imputer(imp)

                elif strat_cat.lower() == 'remove':
                    self.train.dropna(subset=[col], axis=0, inplace=True)
                    self.test.dropna(subset=[col], axis=0, inplace=True)
                    self._log(f" --> Removing {nans} rows due to missing " +
                              f"values in feature {col}.", 2)

                elif strat_cat.lower() == 'most_frequent':
                    self._log(f" --> Imputing {nans} missing values with" +
                              f"most_frequent in feature {col}", 2)
                    imp = SimpleImputer(strategy=strat_cat)
                    fit_imputer(imp)

        self.update('train_test')

    @composed(crash, params_to_log, typechecked)
    def encode(self, max_onehot: Optional[int] = 10, frac_to_other: float = 0):

        """
        Perform encoding of categorical features. The encoding type depends
        on the number of unique values in the column: label-encoding for
        n_unique=2, one-hot-encoding for 2 < n_unique <= max_onehot and
        target-encoding for n_unique > max_onehot. It also replaces classes
        with low occurences with the value 'other' in order to prevent too
        high cardinality.

        PARAMETERS
        ----------
        max_onehot: int or None, optional (default=10)
            Maximum number of unique values in a feature to perform
            one-hot-encoding. If None, it will never perform one-hot-encoding.

        frac_to_other: float, optional (default=0)
            Classes with less instances than n_rows * fraction_to_other
            are replaced with 'other'.

        """

        # Check parameters
        if max_onehot is None:
            max_onehot = 0
        elif max_onehot < 0:  # if 0, 1 or 2: it never uses one-hot encoding
            raise ValueError("Invalid value for the max_onehot parameter." +
                             f"Value should be >= 0, got {max_onehot}.")
        if frac_to_other < 0 or frac_to_other > 1:
            raise ValueError("Invalid value for the frac_to_other parameter." +
                             "Value should be between 0 and 1, got {}."
                             .format(frac_to_other))

        self._log("Encoding categorical features...", 1)

        # Loop over all but last column (target is already encoded)
        for col in self.dataset.columns.values[:-1]:
            # Check if column is categorical
            if self.dataset[col].dtype.kind not in 'ifu':
                # Group uncommon classes into 'other'
                values = self.dataset[col].value_counts()
                for cls_, count in values.items():
                    if count < frac_to_other * len(self.dataset[col]):
                        self.dataset[col].replace(cls_, 'other', inplace=True)
                        self.update('dataset')  # For target encoding

                # Count number of unique values in the column
                n_unique = len(self.dataset[col].unique())

                # Perform encoding type dependent on number of unique values
                if n_unique == 2:
                    self._log(f" --> Label-encoding feature {col}. " +
                              f"Contains {n_unique} unique categories.", 2)
                    le = LabelEncoder()
                    self.dataset[col] = le.fit_transform(self.dataset[col])

                elif 2 < n_unique <= max_onehot:
                    self._log(f" --> One-hot-encoding feature {col}. " +
                              f"Contains {n_unique} unique categories.", 2)
                    dummies = pd.get_dummies(self.dataset[col], prefix=col)
                    self.dataset = pd.concat([self.dataset, dummies], axis=1)
                    self.dataset.drop(col, axis=1, inplace=True)

                    # Place target column last
                    self.dataset = self.dataset[
                            [col for col in self.dataset if col != self.target]
                            + [self.target]]

                else:
                    self._log(f" --> Target-encoding feature {col}.  " +
                              f"Contains {n_unique} unique categories.", 2)

                    # Get mean of target in trainset for every category
                    means = self.train.groupby(col)[self.target].mean()

                    # Map the means over the complete dataset
                    # Test set is tranformed with the mapping of the trainset
                    self.dataset[col] = self.dataset[col].map(means)

        self.update('dataset')  # Redefine new attributes

        # Check if mapping failed for the test set
        nans = self.dataset.isna().any()  # pd.Series of columns with nans
        cols = self.dataset.columns.values[nans.to_numpy().nonzero()]
        t = 's' if len(cols) > 1 else ''
        if nans.any():
            self._log("WARNING! It appears the target-encoding was not " +
                      "able to map all categories in the test set. As a" +
                      "result, there will appear missing values in " +
                      f"column{t}: {', '.join(cols)}. To solve this, try " +
                      "increasing the size of the dataset, the " +
                      "frac_to_other and max_onehot parameters, or remove " +
                      "features with high cardinality.")

    @composed(crash, params_to_log, typechecked)
    def outliers(self,
                 max_sigma: scalar = 3,
                 include_target: bool = False):

        """
        Remove rows from the training set where at least one value lies further
        than `max_sigma` * standard_deviation away from the mean of the column.

        PARAMETERS
        ----------
        max_sigma: int or float, optional (default=3)
            Maximum allowed standard deviations from the mean.

        include_target: bool, optional (default=False)
            Wether to include the target column when searching for outliers.

        """

        # Check parameters
        if max_sigma <= 0:
            raise ValueError("Invalid value for the max_sigma parameter." +
                             f"Value should be > 0, got {max_sigma}.")

        self._log('Handling outliers...', 1)

        # Get z-score outliers index
        objective = self.train if include_target else self.X_train
        # Changes NaN to 0 to not let the algorithm crash
        ix = (np.nan_to_num(np.abs(zscore(objective))) < max_sigma).all(axis=1)

        delete = len(ix) - ix.sum()  # Number of False values in index
        if delete > 0:
            self._log(f" --> Dropping {delete} rows due to outliers.", 2)

        # Remove rows based on index and reset attributes
        self.train = self.train[ix]
        self.update('train_test')

    @composed(crash, params_to_log, typechecked)
    def balance(self,
                oversample: Optional[Union[scalar, str]] = None,
                undersample: Optional[Union[scalar, str]] = None,
                n_neighbors: int = 5):

        """
        Balance the number of instances per target class. Only for
        classification tasks. Dependency: imbalanced-learn.

        PARAMETERS
        ----------
        oversample: float, string or None, optional (default=None)
            Oversampling strategy using ADASYN. Choose from:
                - None: don't oversample
                - float: fraction minority/majority (only for binary classif.)
                - 'minority': resample only the minority class
                - 'not minority': resample all but minority class
                - 'not majority': resample all but majority class
                - 'all': resample all classes

        undersample: float, string or None, optional (default=None)
            Undersampling strategy using NearMiss. Choose from:
                - None: don't undersample
                - float: fraction majority/minority (only for binary)
                - 'majority': resample only the majority class
                - 'not minority': resample all but minority class
                - 'not majority': resample all but majority class
                - 'all': resample all classes

        n_neighbors: int, optional (default=5)
            Number of nearest neighbors used for any of the algorithms.

        """

        def check_params(name, value):

            """
            Check the oversample and undersample parameters.

            Parameters
            ----------
            name: string
                Name of the parameter.

            value: float or string
                Value of the parameter.

            """

            # List of admitted string values
            strategies = ['majority', 'minority',
                          'not majority', 'not minority', 'all']
            if isinstance(value, str) and value not in strategies:
                raise ValueError(f"Unknown value for the {name} parameter," +
                                 " got {}. Choose from: {}."
                                 .format(value, ', '.join(strategies)))
            elif isinstance(value, float) or value == 1:
                if not self.task.startswith('binary'):
                    raise TypeError(f"Invalid type for the {name} param" +
                                    "eter, got {}. Choose from: {}."
                                    .format(value, ', '.join(strategies)))
                elif value <= 0 or value > 1:
                    raise ValueError(f"Invalid value for the {name} param" +
                                     "eter. Value should be between 0 and 1," +
                                     f" got {value}.")

        if self.task == 'regression':
            raise ValueError("This method is only available for " +
                             "classification tasks!")

        try:
            from imblearn.over_sampling import ADASYN
            from imblearn.under_sampling import NearMiss
        except ImportError:
            raise ModuleNotFoundError("Failed to import the imbalanced-learn" +
                                      " package. Install it before using the" +
                                      " balance method.")

        # Check parameters
        check_params('oversample', oversample)
        check_params('undersample', undersample)
        if n_neighbors <= 0:
            raise ValueError("Invalid value for the n_neighbors parameter." +
                             "Value should be >0, got {percentage}.")

        # At least one of the two strategies needs to be applied
        if oversample is None and undersample is None:
            raise ValueError("Oversample and undersample cannot be both None!")

        columns_x = self.X_train.columns  # Save name columns for later

        # Save number of instances per target class for counting
        counts = {}
        for key, value in self.mapping.items():
            counts[key] = (self.y_train == value).sum()

        # Oversample the minority class with SMOTE
        if oversample is not None:
            self._log("Performing oversampling...", 1)
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
                    self._log(f" --> Adding {diff} rows to class {key}.", 2)

        # Apply undersampling of majority class
        if undersample is not None:
            self._log("Performing undersampling...", 1)
            NM = NearMiss(sampling_strategy=undersample,
                          n_neighbors=n_neighbors,
                          n_jobs=self.n_jobs)
            self.X_train, self.y_train = NM.fit_resample(self.X_train,
                                                         self.y_train)

            # Print changes
            for k, value in self.mapping.items():
                diff = counts[key] - (self.y_train == value).sum()
                if diff > 0:
                    self._log(f" --> Removing {diff} rows from class {k}.", 2)

        self.X_train = to_df(self.X_train, columns=columns_x)
        self.y_train = to_series(self.y_train, name=self.target)
        self.train = merge(self.X_train, self.y_train)
        self.update('train_test')

    @composed(crash, params_to_log, typechecked)
    def feature_insertion(self,
                          n_features: int = 2,
                          generations: int = 20,
                          population: int = 500):

        """
        Use a genetic algorithm to create new combinations of existing
        features and add them to the original dataset in order to capture
        the non-linear relations between the original features. A dataframe
        containing the description of the newly generated features and their
        scores can be accessed through the `genetic_features` attribute. The
        algorithm is implemented using the Symbolic Transformer method, which
        can be accessed through the `genetic_algorithm` attribute. It is
        adviced to only use this method when fitting linear models.
        Dependency: gplearn.

        PARAMETERS -------------------------------------

        n_features: int, optional (default=2)
            Maximum number of newly generated features (no more than 1%
            of the population).

        generations: int, optional (default=20)
            Number of generations to evolve.

        population: int, optional (default=500)
            Number of programs in each generation.

        """

        try:
            from gplearn.genetic import SymbolicTransformer
        except ImportError:
            raise ModuleNotFoundError("Failed to import the gplearn" +
                                      " package. Install it before using " +
                                      "the feature_insertion method.")

        # Check parameters
        if population < 100:
            raise ValueError("Invalid value for the population parameter." +
                             f"Value should be >100, got {population}.")
        if generations < 1:
            raise ValueError("Invalid value for the generations parameter." +
                             f"Value should be >100, got {generations}.")
        if n_features <= 0:
            raise ValueError("Invalid value for the n_features parameter." +
                             f"Value should be >0, got {n_features}.")
        elif n_features > int(0.01 * population):
            raise ValueError("Invalid value for the n_features parameter." +
                             "Value should be <1% of the population, " +
                             f"got {n_features}.")

        self._log("Running genetic algorithm...", 1)

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
            self._log("WARNING! The genetic algorithm couldn't find any " +
                      "improving non-linear features!", 1)
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
        self._log("---------------------------------------------------" +
                  "---------------------------------------", 2)

        for feature in descript:
            self._log(f" --> New feature {feature} added to the dataset.", 1)

        self.X = pd.DataFrame(np.hstack((self.X, new_features)),
                              columns=self.X.columns.to_list() + names)

        self.update('X')

    @composed(crash, params_to_log, typechecked)
    def feature_selection(self,
                          strategy: Optional[str] = None,
                          solver: Optional[Union[str, callable]] = None,
                          max_features: Optional[scalar] = None,
                          min_variance_frac: Optional[scalar] = 1.,
                          max_correlation: Optional[float] = 0.98,
                          **kwargs):

        """
        Remove features according to the selected strategy. Ties between
        features with equal scores will be broken in an unspecified way. Also
        removes features with too low variance and too high collinearity.

        PARAMETERS
        ----------
        strategy: string or None, optional (default=None)
            Feature selection strategy to use. Choose from:
                - None: do not perform any feature selection algorithm
                - 'univariate': perform a univariate F-test
                - 'PCA': perform principal component analysis
                - 'SFM': select best features from model
                - 'RFE': recursive feature eliminator

        solver: string, callable or None, optional (default=None)
            Solver or model to use for the feature selection strategy. See the
            sklearn documentation for an extended descrition of the choices.
            Select None for the default option per strategy (not applicable
            for SFM and RFE).
                - for 'univariate', choose from:
                    + 'f_classif' (default for classification tasks)
                    + 'f_regression' (default for regression tasks)
                    + 'mutual_info_classif'
                    + 'mutual_info_regression'
                    + 'chi2'
                    + Any function taking two arrays (X, y), and returning
                      arrays (scores, pvalues). See the documentation.
                - for 'PCA', choose from:
                    + 'auto' (default)
                    + 'full'
                    + 'arpack'
                    + 'randomized'
                - for 'SFM': choose a base estimator from which the
                             transformer is built. The estimator must have
                             either a feature_importances_ or coef_ attribute
                             after fitting. No default option.
                - for 'RFE': choose a supervised learning estimator. The
                             estimator must have either a feature_importances_
                             or coef_ attribute after fitting. No default
                             option.

        max_features: int, float or None, optional (default=None)
            Number of features to select.
                - if < 1: fraction of features to select
                - if >= 1: number of features to select
                - None to select all

        min_variance_frac: float or None, optional (default=1.)
            Remove features with the same value in at least this fraction of
            the total. The default is to keep all features with non-zero
            variance, i.e. remove the features that have the same value in all
            samples. None to skip this step.

        max_correlation: float or None, optional (default=0.98)
            Minimum value of the Pearson correlation cofficient to identify
            correlated features. A dataframe of the removed features and their
            correlation values can be accessed through the collinear attribute.
            None to skip this step.

        **kwargs
            Any extra parameter for the PCA, SFM or RFE. See the sklearn
            documentation for the available options.

        """

        def remove_low_variance(min_variance_frac):

            """
            Removes features with too low variance. Will automatically scale
            the features if threshold variance > 0.

            PARAMETERS
            ----------
            min_variance_frac: float
                Remove features with same values in at least this fraction
                of the total.

            """

            # Calculate threshold variance as p*(1-p)
            threshold = min_variance_frac * (1. - min_variance_frac)

            if threshold > 0:  # In this case: normalize the features
                self.scale(0)

            self.var = VarianceThreshold(threshold=threshold).fit(self.X)
            mask = self.var.get_support()  # Get bool mask of selected features

            for n, column in enumerate(self.X):
                if not mask[n]:
                    self._log(f" --> Feature {column} was removed due to low" +
                              f" variance: {self.var.variances_[n]:.2f}.", 2)
                    self.dataset.drop(column, axis=1, inplace=True)

        def remove_collinear(limit):

            """
            Finds pairs of collinear features based on the Pearson
            correlation coefficient. For each pair above the specified
            limit (in terms of absolute value), it removes one of the two.
            Using code adapted from: https://chrisalbon.com/machine_learning/
            feature_selection/drop_highly_correlated_features

            PARAMETERS
            ----------
            limit: float
                Minimum value of the Pearson correlation cofficient to
                identify correlated features.

            """

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

                self._log(f" --> Feature {column} was removed due to " +
                          "collinearity with another feature.", 2)

            self.dataset.drop(to_drop, axis=1, inplace=True)

        # Check parameters
        if max_features is not None and max_features <= 0:
            raise ValueError("Invalid value for the max_features parameter." +
                             f"Value should be >0, got {max_features}.")
        if min_variance_frac is not None and not 0 <= min_variance_frac <= 1:
            raise ValueError("Invalid value for the min_variance_frac param" +
                             "eter. Value should be between 0 and 1, got {}."
                             .format(min_variance_frac))
        if max_correlation is not None and not 0 <= max_correlation <= 1:
            raise ValueError("Invalid value for the max_correlation param" +
                             "eter. Value should be between 0 and 1, got {}."
                             .format(max_correlation))

        self._log("Performing feature selection...", 1)

        # Then, remove features with too low variance
        if min_variance_frac is not None:
            remove_low_variance(min_variance_frac)

        # First, drop features with too high correlation
        if max_correlation is not None:
            remove_collinear(max_correlation)

        # The dataset is possibly changed
        self.update('dataset')

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
                raise ValueError("Unknown solver: Try one {}."
                                 .format(', '.join(solvers_dct.keys())))

            self.univariate = SelectKBest(solver, k=max_features)
            self.univariate.fit(self.X, self.y)
            mask = self.univariate.get_support()
            for n, col in enumerate(self.X):
                if not mask[n]:
                    self._log(f" --> Feature {col} was removed after the uni" +
                              "variate test (score: {:.2f}  p-value: {:.2f})."
                              .format(self.univariate.scores_[n],
                                      self.univariate.pvalues_[n]), 2)
                    self.dataset.drop(col, axis=1, inplace=True)
            self.update('dataset')

        elif strategy.lower() == 'pca':
            self._log(f" --> Applying Principal Component Analysis... ", 2)

            self.scale(0)  # Scale features (if not done already)

            # Define PCA
            solver = 'auto' if solver is None else solver
            self.PCA = PCA(n_components=max_features,
                           svd_solver=solver,
                           **kwargs)
            self.PCA.fit(self.X_train)
            self.X_train = to_df(self.PCA.transform(self.X_train), pca=True)
            self.X_test = to_df(self.PCA.transform(self.X_test), pca=True)
            self.update('X_train')

        elif strategy.lower() == 'sfm':
            if solver is None:
                raise ValueError("Select a model for the solver!")

            try:  # Model already fitted
                self.SFM = SelectFromModel(estimator=solver,
                                           max_features=max_features,
                                           prefit=True,
                                           **kwargs)
                mask = self.SFM.get_support()

            except Exception:
                self.SFM = SelectFromModel(estimator=solver,
                                           max_features=max_features,
                                           **kwargs)
                self.SFM.fit(self.X, self.y)
                mask = self.SFM.get_support()

            for n, column in enumerate(self.X):
                if not mask[n]:
                    self._log(f" --> Feature {column} was removed by the " +
                              f"{solver.__class__.__name__}.", 2)
                    self.dataset.drop(column, axis=1, inplace=True)
            self.update('dataset')

        elif strategy.lower() == 'rfe':
            # Recursive feature eliminator
            if solver is None:
                raise ValueError('Select a model for the solver!')

            self.RFE = RFE(estimator=solver,
                           n_features_to_select=max_features,
                           **kwargs)
            self.RFE.fit(self.X, self.y)
            mask = self.RFE.support_

            for n, column in enumerate(self.X):
                if not mask[n]:
                    self._log(f" --> Feature {column} was removed by the " +
                              "recursive feature eliminator.", 2)
                    self.dataset.drop(column, axis=1, inplace=True)
            self.update('dataset')

        else:
            raise ValueError("Invalid value for the strategy parameter. Cho" +
                             "ose from: 'univariate', 'PCA', 'SFM' or 'RFE'.")

    # << ======================== Pipeline ======================== >>

    @composed(crash, params_to_log, typechecked)
    def pipeline(self,
                 models: Union[str, List[str], Tuple[str]],
                 metric: Optional[Union[str, callable]] = None,
                 greater_is_better: bool = True,
                 needs_proba: bool = False,
                 successive_halving: bool = False,
                 skip_iter: int = 0,
                 max_iter: Union[int, Sequence[int]] = 10,
                 max_time: Union[scalar, Sequence[scalar]] = np.inf,
                 init_points: Union[int, Sequence[int]] = 5,
                 cv: Union[int, Sequence[int]] = 3,
                 plot_bo: bool = False,
                 bagging: Optional[int] = None):

        """
        The pipeline method is where the models are fitted to the data and
        their performance is evaluated according to the selected metric. For
        every model, the pipeline applies the following steps:

            1. The optimal hyperparameters are selectred using a Bayesian
               Optimization (BO) algorithm with gaussian process as kernel.
               The resulting score of each step of the BO is either computed
               by cross-validation on the complete training set or by randomly
               splitting the training set every iteration into a (sub) training
               set and a validation set. This process can create some data
               leakage but ensures a maximal use of the provided data. The test
               set, however, does not contain any leakage and will be used to
               determine the final score of every model. Note that, if the
               dataset is relatively small, the best score on the BO can
               consistently be lower than the final score on the test set
               (despite the leakage) due to the considerable fewer instances on
               which it is trained.

            2. Once the best hyperparameters are found, the model is trained
               again, now using the complete training set. After this,
               predictions are made on the test set.

            3. You can choose to evaluate the robustness of each model's
            applying a bagging algorithm, i.e. the model will be trained
            multiple times on a bootstrapped training set, returning a
            distribution of its performance on the test set.

        PARAMETERS
        ----------
        models: string, list or tuple
            List of models to fit on the data. Use the predefined acronyms
            to select the models. Possible values are (case insensitive):
                - 'GNB' for Gaussian Naïve Bayes (no hyperparameter tuning)
                - 'MNB' for Multinomial Naïve Bayes
                - 'BNB' for Bernoulli Naïve Bayes
                - 'GP' for Gaussian Process (no hyperparameter tuning)
                - 'OLS' for Ordinary Least Squares (no hyperparameter tuning)
                - 'Ridge' for Ridge Linear
                - 'Lasso' for Lasso Linear Regression
                - 'EN' for ElasticNet Linear Regression
                - 'BR' for Bayesian Regression (with ridge regularization)
                - 'LR' for Logistic Regression
                - 'LDA' for Linear Discriminant Analysis
                - 'QDA' for Quadratic Discriminant Analysis
                - 'KNN' for K-Nearest Neighbors
                - 'Tree' for a single Decision Tree
                - 'Bag' for Bagging (with decision tree as base estimator)
                - 'ET' for Extra-Trees
                - 'RF' for Random Forest
                - 'AdaB' for AdaBoost (with decision tree as base estimator)
                - 'GBM' for Gradient Boosting Machine
                - 'XGB' for XGBoost (if package is available)
                - 'LGB' for LightGBM (if package is available)
                - 'CatB' for CatBoost (if package is available)
                - 'lSVM' for Linear Support Vector Machine
                - 'kSVM' for Kernel (non-linear) Support Vector Machine
                - 'PA' for Passive Aggressive
                - 'SGD' for Stochastic Gradient Descent
                - 'MLP' for Multilayer Perceptron

        metric: string or callable, optional (default=None)
            Metric on which the pipeline fits the models. Choose from any of
            the string scorers predefined by sklearn, use a score (or loss)
            function with signature metric(y, y_pred, **kwargs) or use a
            scorer object. If None, the default metric per task is selected:
                - 'f1' for binary classification
                - 'f1_weighted' for multiclas classification
                - 'r2' for regression

        greater_is_better: bool, optional (default=True)
            Wether the metric is a score function or a loss function,
            i.e. if True, a higher score is better and if False, lower is
            better. Will be ignored if the metric is a string or a scorer.

        needs_proba: bool, optional (default=False)
            Whether the metric function requires probability estimates out of a
            classifier. If True, make sure that every model in the pipeline has
            a `predict_proba` method! Will be ignored if the metric is a string
            or a scorer.

        successive_halving: bool, optional (default=False)
            Wether to use a successive halving approach when running the
            pipeline. This technique fits N models to 1/N of the data. The best
            half are selected to go to the next iteration where the process is
            repeated. This continues until only one model remains, which is
            fitted on the complete dataset. Beware that a model's performance
            can depend greatly on the amount of data on which it is trained.
            For this reason we recommend only to use this technique with
            similar models, e.g. only using tree-based models.

        skip_iter: int, optional (default=0)
            Skip last `skip_iter` iterations of the successive halving. Will be
            ignored if successive_halving=False.

        max_iter: int or iterable, optional (default=10)
            Maximum number of iterations of the BO. If 0, skip the BO and fit
            the model on its default parameters. If iterable, the n-th value
            will apply to the n-th model in the pipeline.

        max_time: int, float or iterable, optional (default=np.inf)
            Maximum time allowed for the BO per model (in seconds). If 0, skip
            the BO and fit the model on its default parameters. If iterable,
            the n-th value will apply to the n-th model in the pipeline.

        init_points: int or iterable, optional (default=5)
            Initial number of tests of the BO before fitting the surrogate
            function. If iterable, the n-th value will apply to the n-th model
            in the pipeline.

        cv: int or iterable, optional (default=3)
            Strategy to fit and score the model selected after every step
            of the BO.
                - if 1, randomly split into a train and validation set
                - if >1, perform a k-fold cross validation on the training set

        plot_bo: bool, optional (default=False)
            Wether to plot the BO's progress as it runs. Creates a canvas with
            two plots: the first plot shows the score of every trial and the
            second shows the distance between the last consecutive steps. Don't
            forget to call `%matplotlib` at the start of the cell if you are
            using jupyter notebook!

        bagging: int or None, optional (default=None)
            Number of data sets (bootstrapped from the training set) to use in
            the bagging algorithm. If None or 0, no bagging is performed.

        """

        # << ================= Inner Function ================= >>

        def check_params(name, value, len_models):

            """
            Validate the length of the parameter. Mut be equal to len(models).

            Parameters
            ----------
            name: string
                Name of the parameter.

            value: list or tuple
                Value of the parameter provided by the user.

            len_models: int
                Length of the list of models in the pipeline.

            """

            if len(value) != len(models):
                raise ValueError(f"Invalid value for the {name} parameter. " +
                                 "Length should be equal to the number of " +
                                 f"models, got len(models)={len_models} " +
                                 f"and len({name})={len(value)}.")

        def data_preparation():

            """
            Make a dict of the data (complete, train, test and scaled)

            Returns
            -------
            data: dictionary
                Dictionary of the data. Can differ per model due to scaling.

            """

            data = {}
            for set_ in ['X', 'y', 'X_train', 'y_train', 'X_test', 'y_test']:
                data[set_] = getattr(self, set_)

            # Check if any scaling models in final_models
            scale = any(model in self.models for model in scaling_models)
            if scale and not self._is_scaled:
                # Normalize features to mean=0, std=1
                scaler = StandardScaler()
                data['X_train_scaled'] = scaler.fit_transform(data['X_train'])
                data['X_test_scaled'] = scaler.transform(data['X_test'])
                data['X_scaled'] = np.concatenate((data['X_train_scaled'],
                                                   data['X_test_scaled']))

                # Convert np.array to pd.DataFrame for all scaled features
                for set_ in ['X_scaled', 'X_train_scaled', 'X_test_scaled']:
                    data[set_] = to_df(data[set_], self.X.columns)

            return data

        def run_iteration():

            """
            Core iterations of the pipeline, where models are created and
            fitted. Multiple needed for successive halving.

            Returns
            -------
            scores: pd.DataFrame
                Dataframe of the scores for this iteration of the pipeline.

            """

            # If verbose=1, use tqdm to evaluate process
            if self.verbose == 1:
                loop = tqdm(self.models, desc='Processing')
                loop.clear()  # Prevent starting a progress bar before the loop
            else:
                loop = self.models

            # Loop over every independent model
            iter_ = zip(loop, max_iter, max_time, init_points, cv)
            for model, max_iter_, max_time_, init_points_, cv_ in iter_:
                model_time = time()

                # Define model class
                setattr(self, model, model_list[model](self))

                try:  # If errors occure, just skip the model
                    # Run Bayesian Optimization
                    getattr(self, model).bayesian_optimization(
                            max_iter_, max_time_, init_points_, cv_, plot_bo)

                    # Fit the model to the test set
                    getattr(self, model).fit()

                    # Perform bagging
                    getattr(self, model).bagging(self.bagging)

                    # Get the total time spend on this model
                    total_time = time_to_string(model_time)
                    setattr(getattr(self, model), 'total_time', total_time)
                    self._log('-' * 49, 1)
                    self._log(f'Total time: {total_time}', 1)

                except Exception as ex:
                    if max_iter_ > 0 and max_time_ > 0:
                        self._log('', 1)  # Add extra line
                    self._log("Exception encountered while running the "
                              + f"{model} model. Removing model from pipeline."
                              + f"\n{type(ex).__name__}: {ex}", 1)

                    # Save the exception to model attribute
                    exception = type(ex).__name__ + ': ' + str(ex)
                    getattr(self, model).error = exception

                    # Append exception to ATOM errors dictionary
                    if not isinstance(self.errors, dict):
                        self.errors = {}
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
                maxlen = max(len_)

                # Get list of scors on the test set
                scrs = []
                for m in self.models:
                    if bagging is None:
                        scrs.append(getattr(self, m).score_test)
                    else:
                        scrs.append(getattr(self, m).bagging_scores.mean())

            except (ValueError, AttributeError):
                raise ValueError('It appears all models failed to run...')

            # Print final results
            self._log("\n\nFinal results ================>>")
            self._log(f"Duration: {time_to_string(t_init)}")
            self._log(f"Metric: {self.metric.name}")
            self._log("--------------------------------")

            # Create dataframe with final results
            scores = pd.DataFrame(columns=['model',
                                           'total_time',
                                           'score_train',
                                           'score_test',
                                           'fit_time'])

            if self.bagging is not None:
                pd.concat([scores, pd.DataFrame(columns=['bagging_mean',
                                                         'bagging_std',
                                                         'bagging_time'])])

            for m in self.models:
                name = getattr(self, m).name
                longname = getattr(self, m).longname
                total_time = getattr(self, m).total_time
                score_train = getattr(self, m).score_train
                score_test = getattr(self, m).score_test
                fit_time = getattr(self, m).fit_time

                if bagging is None:
                    scores = scores.append({'model': name,
                                            'total_time': total_time,
                                            'score_train': score_train,
                                            'score_test': score_test,
                                            'fit_time': fit_time},
                                           ignore_index=True)

                    # Create string of the score
                    print_ = "{0:{1}s} --> {2:.3f}".format(
                                                longname, maxlen, score_test)

                    # Highlight best score and assign winner attribute
                    if score_test == max(scrs) and len(self.models) > 1:
                        self.winner = getattr(self, m)
                        print_ += ' !!'

                else:
                    bs_mean = getattr(self, m).bagging_scores.mean()
                    bs_std = getattr(self, m).bagging_scores.std()
                    bs_time = getattr(self, m).bs_time
                    scores = scores.append({'model': name,
                                            'total_time': total_time,
                                            'score_train': score_train,
                                            'score_test': score_test,
                                            'fit_time': fit_time,
                                            'bagging_mean': bs_mean,
                                            'bagging_std': bs_std,
                                            'bagging_time': bs_time},
                                           ignore_index=True)

                    # Create string of the score
                    print1 = f"{longname:{maxlen}s} --> {bs_mean:.3f}"
                    print2 = f"{bs_std:.3f}"
                    print_ = print1 + u" \u00B1 " + print2

                    # Highlight best score and assign winner attribute
                    if bs_mean == max(scrs) and len(self.models) > 1:
                        self.winner = getattr(self, m)
                        print_ += ' !!'

                # Annotate if model overfitted when train 20% > test
                if score_train - 0.2 * score_train > score_test:
                    print_ += ' ~'

                self._log(print_)  # Print the score

            return scores

        # << ======================== Initialize ======================== >>

        t_init = time()  # To measure the time the whole pipeline takes

        self._log('\nRunning pipeline =================>')

        # Check parameters
        if isinstance(models, str):
            models = [models]
        if skip_iter < 0:
            raise ValueError("Invalid value for the skip_iter parameter." +
                             f"Value should be >=0, got {skip_iter}.")
        if isinstance(max_iter, (list, tuple)):
            check_params('max_iter', max_iter, len(models))
        else:
            max_iter = [max_iter for _ in models]
        if isinstance(max_time, (list, tuple)):
            check_params('max_time', max_time, len(models))
        else:
            max_time = [max_time for _ in models]
        if isinstance(init_points, (list, tuple)):
            check_params('init_points', init_points, len(models))
        else:
            init_points = [init_points for _ in models]
        if isinstance(cv, (list, tuple)):
            check_params('cv', cv, len(models))
        else:
            cv = [cv for _ in models]
        if bagging is None or bagging == 0:
            bagging = None

        # Make attributes of some parameters to use them in plot functions
        self.successive_halving = successive_halving
        self.bagging = bagging

        # Save model erros (if any) to attribute
        self.errors = "No exceptions encountered!"

        # << =================== Check validity models ================== >>

        # Final list of models to be used
        self.models = []

        # Set models to right name
        for m in models:
            # Compare strings case insensitive
            if m.lower() not in map(str.lower, model_list.keys()):
                raise ValueError("Unknown model: {}! Choose from: {}"
                                 .format(m, ', '.join(model_list.keys())))
            else:
                for n in model_list.keys():
                    if m.lower() == n.lower():
                        self.models.append(n)
                        break

        # Check for duplicates
        if len(self.models) != len(set(self.models)):
            raise ValueError("Duplicate models found in pipeline!")

        # Check if packages for not-sklearn models are available
        for model, package in optional_packages:
            if model in self.models:
                try:
                    importlib.import_module(package)
                except ImportError:
                    raise ValueError(f"Unable to import {package}!")

        # Remove regression/classification-only models from pipeline
        if self.task != 'regression':
            for model in only_regression:
                if model in self.models:
                    raise ValueError(f"The {model} model can't perform " +
                                     "classification tasks!")
        else:
            for model in only_classification:
                if model in self.models:
                    raise ValueError(f"The {model} model can't perform " +
                                     "regression tasks!")

        if not successive_halving:
            self._log("Model{} in pipeline: {}"
                      .format('s' if len(self.models) > 1 else '',
                              ', '.join(self.models)))

        # << ================== Check validity metric =================== >>

        if metric is None:
            if self.task.startswith('binary'):
                self.metric = get_scorer('f1')
                self.metric.name = 'f1'
            elif self.task.startswith('multiclass'):
                self.metric = get_scorer('f1_weighted')
                self.metric.name = 'f1_weighted'
            else:
                self.metric = get_scorer('r2')
                self.metric.name = 'r2'
        elif isinstance(metric, str):
            if metric not in SCORERS.keys():
                raise ValueError("Unknown value for the metric parameter, " +
                                 f"got {metric}. Try one of {SCORERS.keys()}.")
            self.metric = get_scorer(metric)
            self.metric.name = metric
        elif hasattr(metric, '_score_func'):  # Provided metric is scoring
            self.metric = metric
            self.metric.name = self.metric._score_func.__name__
        else:  # Metric is a metric function with signature metric(y, y_pred)
            self.metric = make_scorer(metric, greater_is_better, needs_proba)
            self.metric.name = self.metric._score_func.__name__

        # Add all metrics as subclasses of the BaseMetric class
        for key, value in SCORERS.items():
            setattr(self.metric, key, value)
            setattr(getattr(self.metric, key), 'name', key)

        self._log(f"Metric: {self.metric.name}")

        # << ======================== Core ======================== >>

        if self.successive_halving:
            self.scores = []  # Save the cv's scores in list of dataframes
            iteration = 0
            original_data = self.dataset.copy()
            while len(self.models) > 2**skip_iter - 1:
                # Select 1/N of data to use for this iteration
                self._split_dataset(original_data, 100./len(self.models))
                self.update()
                self.data = data_preparation()
                self._log("\n\n<<=============== Iteration {} ==============>>"
                          .format(iteration))
                self._log("Model{} in pipeline: {}"
                          .format('s' if len(self.models) > 1 else '',
                                  ', '.join(self.models)))
                self.stats(1)

                # Run iteration and append to the scores list
                scores = run_iteration()
                self.scores.append(scores)

                # Select best models for halving
                col = 'score_test' if self.bagging is None else 'bagging_mean'
                lx = scores.nlargest(n=int(len(self.models)/2),
                                     columns=col,
                                     keep='all')

                # Keep the models in the same order
                n = []  # List of new models
                [n.append(m) for m in self.models if m in list(lx.model)]
                self.models = n.copy()
                iteration += 1

            self._is_fitted = True

        else:
            self.data = data_preparation()
            self.scores = run_iteration()
            self._is_fitted = True

    # ======================== Plot methods =======================>

    @composed(params_to_log, typechecked)
    def plot_correlation(self,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 10),
                         filename: Optional[str] = None,
                         display: bool = True):

        """ Correlation maxtrix plot of the data """

        plot_correlation(self, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_PCA(self,
                 show: Optional[int] = None,
                 title: Optional[str] = None,
                 figsize: Optional[Tuple[int, int]] = None,
                 filename: Optional[str] = None,
                 display: bool = True):

        """
        Plot the explained variance ratio of the components. Only if PCA
        was applied on the dataset through the feature_selection method.

        """

        plot_PCA(self, show, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_bagging(self,
                     models: Union[None, str, Sequence[str]] = None,
                     title: Optional[str] = None,
                     figsize: Optional[Tuple[int, int]] = None,
                     filename: Optional[str] = None,
                     display: bool = True):

        """ Plot a boxplot of the bagging's results """

        plot_bagging(self, models, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_successive_halving(self,
                                models: Union[None, str, Sequence[str]] = None,
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 6),
                                filename: Optional[str] = None,
                                display: bool = True):

        """ Plot the models' scores per iteration of the successive halving """

        plot_successive_halving(self, models,
                                title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_ROC(self,
                 models: Union[None, str, Sequence[str]] = None,
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):

        """ Plot the Receiver Operating Characteristics curve """

        plot_ROC(self, models, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_PRC(self,
                 models: Union[None, str, Sequence[str]] = None,
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):

        """ Plot the precision-recall curve """

        plot_PRC(self, models, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_permutation_importance(
                            self,
                            models: Union[None, str, Sequence[str]] = None,
                            show: Optional[int] = None,
                            n_repeats: int = 10,
                            title: Optional[str] = None,
                            figsize: Optional[Tuple[int, int]] = None,
                            filename: Optional[str] = None,
                            display: bool = True):

        """ Plot the feature permutation importance of models """

        plot_permutation_importance(self, models, show, n_repeats,
                                    title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_feature_importance(self,
                                models: Union[None, str, Sequence[str]] = None,
                                show: Optional[int] = None,
                                title: Optional[str] = None,
                                figsize: Optional[Tuple[int, int]] = None,
                                filename: Optional[str] = None,
                                display: bool = True):

        """ Plot tree-based model's normalized feature importances """

        plot_feature_importance(self, models, show,
                                title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_confusion_matrix(self,
                              models: Union[None, str, Sequence[str]] = None,
                              normalize: bool = False,
                              title: Optional[str] = None,
                              figsize: Tuple[int, int] = (8, 8),
                              filename: Optional[str] = None,
                              display: bool = True):

        """
        For 1 model: plot it's confusion matrix in a heatmap.
        For >1 models: compare TP, FP, FN and TN in a barplot.

        """

        plot_confusion_matrix(self, models, normalize,
                              title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_threshold(self,
                       models: Union[None, str, Sequence[str]] = None,
                       metric: Optional[Union[str, callable]] = None,
                       steps: int = 100,
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       filename: Optional[str] = None,
                       display: bool = True):

        """

        Plot performance metric(s) against threshold values.

        """

        plot_threshold(self, models, metric, steps,
                       title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_probabilities(self,
                           models: Union[None, str, Sequence[str]] = None,
                           target: Union[int, str] = 1,
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           filename: Optional[str] = None,
                           display: bool = True):

        """
        Plot a function of the probability of the classes
        of being the target class.

        """

        plot_probabilities(self, models, target,
                           title, figsize, filename, display)

    # <============ Classmethods for plot settings ============>

    @composed(classmethod, typechecked)
    def set_style(cls, style: str = 'darkgrid'):

        """
        Change the seaborn plotting style.
        See https://seaborn.pydata.org/tutorial/aesthetics.html

        Parameters
        ----------
        style: string, optional (default='darkgrid')
            Name of the plotting style.

        """

        sns.set_style(style)
        cls.style = style

    @composed(classmethod, typechecked)
    def set_palette(cls, palette: str = 'GnBu_d'):

        """
        Change the seaborn color palette.
        See https://seaborn.pydata.org/tutorial/color_palettes.html

        Parameters
        ----------
        palette: string, optional(default='GnBu_d')
            Name of the palette.

        """

        sns.set_palette(palette)
        cls.palette = palette

    @composed(classmethod, typechecked)
    def set_title_fontsize(cls, fontsize: int = 20):

        """
        Change the fontsize of the plot's title.

        Parameters
        ----------
        fontsize: int
            Size of the font.

        """

        cls.title_fontsize = fontsize

    @composed(classmethod, typechecked)
    def set_label_fontsize(cls, fontsize: int = 16):

        """
        Change the fontsize of the plot's labels and legends.

        Parameters
        ----------
        fontsize: int
            Size of the font.

        """

        cls.label_fontsize = fontsize

    @composed(classmethod, typechecked)
    def set_tick_fontsize(cls, fontsize: int = 12):

        """
        Change the fontsize of the plot's ticks.

        Parameters
        ----------
        fontsize: int
            Size of the font.

        """

        cls.tick_fontsize = fontsize

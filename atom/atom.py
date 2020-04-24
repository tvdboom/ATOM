# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

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
from typeguard import typechecked
from typing import Union, Optional, Sequence, List, Tuple

# Sklearn
from sklearn.metrics import SCORERS, get_scorer, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Own modules
from .models import model_list
from .data_cleaning import (
    StandardCleaner, Scaler, Imputer, Encoder, Outliers, Balancer
    )
from .feature_selection import FeatureGenerator, FeatureSelector
from .utils import (
    scalar, X_types, y_types, train_types, optional_packages, scaling_models,
    only_classification, only_regression, composed, crash, params_to_log,
    time_to_string, merge, check_scaling, check_is_fitted
    )
from .plots import (
    save, plot_correlation, plot_PCA, plot_components, plot_RFECV,
    plot_ROC, plot_PRC, plot_bagging, plot_successive_halving,
    plot_learning_curve, plot_permutation_importance, plot_feature_importance,
    plot_confusion_matrix, plot_threshold, plot_probabilities,
    plot_calibration, plot_gains, plot_lift
    )

# Plotting
import seaborn as sns
sns.set(style='darkgrid', palette='GnBu_d')


# << ================= Classes ================= >>

class ATOM(object):
    """ATOM base class."""

    # Define class variables for plot aesthetics
    style = 'darkgrid'
    palette = 'GnBu_d'
    title_fontsize = 20
    label_fontsize = 16
    tick_fontsize = 12

    # << ================= Methods ================= >>

    @composed(crash, params_to_log, typechecked)
    def __init__(self,
                 X: X_types,
                 y: y_types,
                 percentage: scalar,
                 test_size: float,
                 n_jobs: int,
                 warnings: bool,
                 verbose: int,
                 random_state: Optional[int]):
        """Class initializer.

        Applies standard data cleaning. These steps include:
            - Transforming the input data into a pd.DataFrame (if it wasn't one
              already) that can be accessed through the class' data attributes.
            - Removing columns with prohibited data types ('datetime64',
             'datetime64[ns]', 'timedelta[ns]').
            - Removing categorical columns with maximal cardinality (the number
              of unique values is equal to the number of instances. Usually the
              case for names, IDs, etc...).
            - Removing columns with minimum cardinality (all values are equal).
            - Removing rows with missing values in the target column.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=-1)
            - If int: index of the column of X which is selected as target
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
            If False, it suppresses all warnings.

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
        # << ============= Attribute references ============= >>

        # Data attributes
        self.dataset = None
        self.train = None
        self.test = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Method attributes
        self.scaler = None
        self.profile = None
        self.imputer = None
        self.encoder = None
        self.outlier = None
        self.balancer = None
        self.feature_generator = None
        self.genetic_features = None
        self.feature_selector = None
        self.collinear = None

        # Model attributes
        self.errors = None
        self.winner = None
        self.scores = None

        # << ============ Check input Parameters ============ >>

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

        # Update attributes with params
        self.percentage = percentage
        self.test_size = test_size
        self.warnings = warnings
        if not warnings:  # Ignore all warnings
            warn.filterwarnings('ignore')
        self.verbose = verbose
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)  # Set random seed

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

            if self.n_jobs != 1:
                self._log(f"Parallel processing with {self.n_jobs} cores.")

        # << ============ Data cleaning ============ >>

        # List of data types ATOM can't handle
        prohibited_types = ['datetime64', 'datetime64[ns]', 'timedelta[ns]']

        # Apply the standard cleaning steps
        self.standard_cleaner = \
            StandardCleaner(prohibited_types=prohibited_types, atom=self)
        X, y = self.standard_cleaner.transform(X, y)

        self.dataset = merge(X, y)  # Merge to one dataframe with all data
        self.target = y.name

        self._is_scaled = check_scaling(self.dataset)
        self._is_fitted = False  # Model has not been fitted yet

        # << ============ Set algorithm task ============ >>

        # Get unique target values before encoding (for later print)
        self._unique = sorted(self.dataset[self.target].unique())
        if len(self._unique) == 1:
            raise ValueError(f"Only found 1 target value: {self._unique[0]}")
        elif len(self._unique) == 2 and self.goal == 'classification':
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
            le = LabelEncoder()
            self.dataset[self.target] = \
                le.fit_transform(self.dataset[self.target])
            self.mapping = {str(v): i for i, v in enumerate(le.classes_)}

        # << =========================================== >>

        self._split_dataset(self.dataset, percentage)  # Train/test split
        self.update()  # Define data subsets class attributes
        self.stats(1)  # Print out data stats

    # << ======================= Utility methods ======================= >>

    def _log(self, string, level=0):
        """Print and save output to log file.

        Parameters
        ----------
        string: str
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
        """Split a percentage of the dataset into a train and test set.

        Parameters
        ----------
        dataset: pd.DataFrame
            Dataset to use for splitting.

        percentage: int or float, optional (default=100)
            Percentage of the data to use.

        """
        # Shuffle the dataset and get percentage of data
        self.dataset = dataset.sample(frac=percentage/100.)
        self.dataset.reset_index(drop=True, inplace=True)

        # Split train and test sets on percentage of data
        self.train, self.test = train_test_split(self.dataset,
                                                 test_size=self.test_size,
                                                 shuffle=False)

    @composed(crash, params_to_log)
    def stats(self, _verbose: int = -2):
        """Print some information about the dataset.

        Parameters
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
            # Create dataframe with stats per target class
            index = []
            for key, value in self.mapping.items():
                try:
                    list_ = list(map(int, self.mapping.keys()))
                    if list_ != list(self.mapping.values()):
                        index.append(str(value) + ': ' + key)
                    else:
                        index.append(value)
                except ValueError:
                    index.append(str(value) + ': ' + key)

            stats = pd.DataFrame(columns=[' total', ' train_set', ' test_set'])

            # Count number of occurrences in all sets
            uq_train, c_train = np.unique(self.y_train, return_counts=True)
            uq_test, c_test = np.unique(self.y_test, return_counts=True)
            keys, values = '', []
            for key, value in self.mapping.items():
                # If set has 0 instances of that class the array is empty
                idx_train = np.where(uq_train == value)[0]
                train = c_train[idx_train[0]] if len(idx_train) != 0 else 0
                idx_test = np.where(uq_test == value)[0]
                test = c_test[idx_test[0]] if len(idx_test) != 0 else 0
                stats = stats.append({' total': train + test,
                                      ' train_set': train,
                                      ' test_set': test}, ignore_index=True)

                keys += key + ':'
                values.append(train + test)

            stats.set_index(pd.Index(index), inplace=True)

            string = ''  # Class balance string for values
            for i in values:
                string += str(round(i/(train+test), 1)) + ':'

            self._log("----------------------------------", _verbose + 1)
            if len(self.mapping.keys()) < 5:  # Gets ugly for too many classes
                self._log("Class balance: {} <==> {}"  # [-1] to remove last :
                          .format(keys[:-1], string[:-1]), _verbose + 1)
            self._log(f"Instances in {self.target} per class:", _verbose + 1)
            self._log(stats.to_markdown(), _verbose + 1)

        self._log('', 1)  # Insert an empty row

    @composed(crash, params_to_log)
    def scale(self):
        """Scale features to mean=0 and std=1."""
        self.scaler = Scaler(atom=self).fit(self.X_train)
        self.X = self.scaler.transform(self.X)
        self._is_scaled = True
        self.update('X')

    @composed(crash, typechecked)
    def update(self, df: str = 'dataset'):
        """Update data attributes.

        If you change any of the class' data attributes (dataset, X, y,
        train, test, X_train, X_test, y_train, y_test) in between the
        pipeline, you should call this method to change all other data
        attributes to their correct values. Independent attributes are
        updated in unison, that is, setting df='X_train' will also
        update X_test, y_train and y_test, or df='train' will also
        update the test set, etc...

        Parameters
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
        idx = self.train.index
        if df == 'dataset':
            self.train = self.dataset[self.dataset.index.isin(idx)]
            self.test = self.dataset[~self.dataset.index.isin(idx)]

        elif df in ('train_test', 'train', 'test'):
            # Join train and test on rows
            self.dataset = pd.concat([self.train, self.test],
                                     join='outer',
                                     ignore_index=False,
                                     copy=True)

        elif df in ('X_train', 'y_train', 'X_test', 'y_test'):
            self.train = merge(self.X_train, self.y_train)
            self.test = merge(self.X_test, self.y_test)
            self.dataset = pd.concat([self.train, self.test],
                                     join='outer',
                                     ignore_index=False,
                                     copy=True)

        elif df in ('X_y', 'X', 'y'):
            self.dataset = merge(self.X, self.y)
            self.train = self.dataset[self.dataset.index.isin(idx)]
            self.test = self.dataset[~self.dataset.index.isin(idx)]

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
        """Create an extensive profile analysis of the data.

        The profile report is rendered in HTML5 and CSS3. Note that this
        method can be slow for rows>10k. Dependency: pandas-profiling.

        Parameters
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

        self.profile = ProfileReport(getattr(self, df).sample(rows))
        try:  # Render if possible (for jupyter notebook)
            from IPython.display import display
            display(self.profile)
        except Exception:
            pass

        if filename is not None:
            if not filename.endswith('.html'):
                filename = filename + '.html'
            self.profile.to_file(filename)

    @composed(crash, params_to_log, typechecked)
    def results(self, metric: Optional[str] = None):
        """Print the pipeline's final results for a specific metric.

        If a model shows a `XXX`, it means the metric failed for that specific
        model. This can happen if either the metric is unavailable for the task
        or if the model does not have a `predict_proba` method while the metric
        needs it.

        Parameters
        ----------
        metric: string or None, optional (default=None)
            String of one of sklearn's predefined metrics. If None, the metric
            used to fit the pipeline is selected and the bagging results will
            be showed (if used).

        """
        check_is_fitted(self._is_fitted)  # Raise error is class not fitted
        _isNone = False  # If the metric parameter is called with None or not
        if metric is None:
            _isNone = True
            metric = self.metric.name

        if metric not in SCORERS.keys():
            raise ValueError("Unknown value for the metric parameter, " +
                             f"got {metric}. Try one of {SCORERS.keys()}.")

        # Get max length of the models' names
        maxlen = max([len(getattr(self, m).longname) for m in self.models])

        # Get list of scores
        scrs = []
        for m in self.models:
            if _isNone and self._has_bag:
                scrs.append(getattr(self, m).bagging_scores.mean())

            # If invalid metric, don't append to scores
            elif not isinstance(getattr(getattr(self, m), metric), str):
                scrs.append(getattr(getattr(self, m), metric))

        if len(scrs) == 0:  # All metrics failed
            raise ValueError("Invalid metric!")

        self._log("\nFinal results ================>>", -2)
        self._log(f"Metric: {metric}", -2)
        self._log("--------------------------------", -2)

        for model in self.models:
            m = getattr(self, model)

            # If metric is None and bagging, print out bagging results
            if _isNone and self._has_bag:
                score = m.bagging_scores.mean()

                # Create string of the score
                print1 = f"{m.longname:{maxlen}s} --> {score:.3f}"
                print2 = f"{m.bagging_scores.std():.3f}"
                print_ = print1 + u" \u00B1 " + print2
            else:
                score = getattr(m, metric)

                # Create string of the score (if wrong metric for model -> XXX)
                if isinstance(score, str):
                    print_ = "{:{}s} --> XXX".format(m.longname, maxlen)
                else:
                    print_ = "{:{}s} --> {:.3f}".format(m.longname,
                                                        maxlen,
                                                        score)

            # Highlight best score (if more than one model in pipeline)
            if score == max(scrs) and len(self.models) > 1:
                print_ += ' !!'

            # Annotate if model overfitted when train 20% > test
            if metric == self.metric.name:
                if m.score_train - 0.2 * m.score_train > m.score_test:
                    print_ += ' ~'

            self._log(print_, -2)  # Always print

    @composed(crash, params_to_log, typechecked)
    def save(self, filename: Optional[str] = None):
        """Save the ATOM class to a pickle file.

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
               min_frac_rows: float = 0.5,
               min_frac_cols: float = 0.5,
               missing: Optional[Union[scalar, str, list]] = None):
        """Handle missing values in the dataset.

        Impute or remove missing values according to the selected strategy.
        Also removes rows and columns with too many missing values.

        Parameters
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
            Minimum fraction of non missing values in a row. If less,
            the row is removed.

        min_frac_cols: float, optional (default=0.5)
            Minimum fraction of non missing values in a column. If less,
            the column is removed.

        missing: int, float or list, optional (default=None)
            List of values to impute. None for default list: [None, np.NaN,
            np.inf, -np.inf, '', '?', 'NA', 'nan', 'inf']

        """
        self.imputer = Imputer(strat_num=strat_num,
                               strat_cat=strat_cat,
                               min_frac_rows=min_frac_rows,
                               min_frac_cols=min_frac_cols,
                               missing=missing,
                               atom=self).fit(self.X_train, self.y_train)
        self.X, self.y = self.imputer.transform(self.X, self.y)

        self.update('X_y')

    @composed(crash, params_to_log, typechecked)
    def encode(self,
               max_onehot: Optional[int] = 10,
               encode_type: str = 'Target',
               frac_to_other: float = 0,
               **kwargs):
        """Perform encoding of categorical features.

        The encoding type depends on the number of unique values in the column:
            - label-encoding for n_unique=2
            - one-hot-encoding for 2 < n_unique <= max_onehot
            - 'encode_type' for n_unique > max_onehot

        It also replaces classes with low occurrences with the value 'other' in
        order to prevent too high cardinality.

        Parameters
        ----------
        max_onehot: int or None, optional (default=10)
            Maximum number of unique values in a feature to perform
            one-hot-encoding. If None, it will never perform one-hot-encoding.

        encode_type: str, optional (default='Target')
            Type of encoding to use for high cardinality features. Choose from
            one of the encoders available from the category_encoders package.

        frac_to_other: float, optional (default=0)
            Classes with less instances than n_rows * fraction_to_other
            are replaced with 'other'.

        **kwargs
            Additional keyword arguments passed to the encoder type.

        """
        self.encoder = Encoder(max_onehot=max_onehot,
                               encode_type=encode_type,
                               frac_to_other=frac_to_other,
                               atom=self,
                               **kwargs).fit(self.X_train, self.y_train)
        self.X = self.encoder.transform(self.X)

        self.update('X')

    @composed(crash, params_to_log, typechecked)
    def outliers(self,
                 strategy: Union[scalar, str] = 'remove',
                 max_sigma: scalar = 3,
                 include_target: bool = False):
        """Remove or replace outliers in the training set.

        Outliers are defined as values that lie further than
        `max_sigma` * standard_deviation away from the mean of the column.

        Parameters
        ----------
        strategy: int, float or str, optional (default='remove')
            Which strategy to apply on the outliers. Choose from:
                - 'remove' to drop any row with outliers from the dataset
                - 'min_max' to replace it with the min or max of the column
                - Any numerical value with which to replace the outliers

        max_sigma: int or float, optional (default=3)
            Maximum allowed standard deviations from the mean.

        include_target: bool, optional (default=False)
            Whether to include the target column when searching for outliers.
            Can be useful for regression tasks.

        """
        self.outlier = Outliers(strategy=strategy,
                                max_sigma=max_sigma,
                                atom=self)
        if not include_target:
            self.X_train = self.outlier.transform(self.X_train)
        else:
            self.X_train, self.y_train = \
                self.outlier.transform(self.X_train, self.y_train)

        self.update('X_train')

    @composed(crash, params_to_log, typechecked)
    def balance(self,
                oversample: Optional[Union[scalar, str]] = None,
                undersample: Optional[Union[scalar, str]] = None,
                n_neighbors: int = 5):
        """Balance the training set.

        Balance the number of instances per target class in the training set.
        If both oversampling and undersampling are used, they will be applied
        in that order. Only for classification tasks.
        Dependency: imbalanced-learn.

        Parameters
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
                - float: fraction minority/majority (only for binary classif.)
                - 'majority': resample only the majority class
                - 'not minority': resample all but minority class
                - 'not majority': resample all but majority class
                - 'all': resample all classes

        n_neighbors: int, optional (default=5)
            Number of nearest neighbors used for any of the algorithms.

        """
        try:
            import imblearn
        except ImportError:
            raise ModuleNotFoundError("Failed to import the imbalanced-learn" +
                                      " package. Install it before using the" +
                                      " balance method.")

        self.balancer = Balancer(oversample=oversample,
                                 undersample=undersample,
                                 n_neighbors=n_neighbors,
                                 atom=self)
        self.X_train, self.y_train = self.balancer.transform(self.X_train,
                                                             self.y_train)

        self.update('X_train')

    @composed(crash, params_to_log, typechecked)
    def feature_generation(self,
                           n_features: int = 2,
                           generations: int = 20,
                           population: int = 500):
        """Create new non-linear features.

        Use a genetic algorithm to create new combinations of existing
        features and add them to the original dataset in order to capture
        the non-linear relations between the original features. A dataframe
        containing the description of the newly generated features and their
        scores can be accessed through the `genetic_features` attribute. The
        algorithm is implemented using the Symbolic Transformer method, which
        can be accessed through the `genetic_algorithm` attribute. It is
        advised to only use this method when fitting linear models.
        Dependency: gplearn.

        Parameters -------------------------------------

        n_features: int, optional (default=2)
            Maximum number of newly generated features (no more than 1%
            of the population).

        generations: int, optional (default=20)
            Number of generations to evolve.

        population: int, optional (default=500)
            Number of programs in each generation.

        """
        try:
            import gplearn
        except ImportError:
            raise ModuleNotFoundError("Failed to import the gplearn" +
                                      " package. Install it before using " +
                                      "the feature_generation method.")

        self.feature_generator = \
            FeatureGenerator(n_features=n_features,
                             generations=generations,
                             population=population,
                             atom=self)
        self.feature_generator.fit(self.X_train, self.y_train)

        self.X, self.y = self.feature_generator.transform(self.X, self.y)

        # Attribute for the newly generated features
        if self.feature_generator.genetic_features is not None:
            self.genetic_features = self.feature_generator.genetic_features

        self.update('X')

    @composed(crash, params_to_log, typechecked)
    def feature_selection(self,
                          strategy: Optional[str] = None,
                          solver: Optional[Union[str, callable]] = None,
                          n_features: Optional[scalar] = None,
                          max_frac_repeated: Optional[scalar] = 1.,
                          max_correlation: Optional[float] = 0.98,
                          **kwargs):
        """Apply feature selection techniques.

        Remove features according to the selected strategy. Ties between
        features with equal scores will be broken in an unspecified way.
        Also removes features with too low variance and finds pairs of
        collinear features based on the Pearson correlation coefficient. For
        each pair above the specified limit (in terms of absolute value), it
        removes one of the two.

        Note that the RFE and RFECV strategies don't work when the solver is a
        CatBoost model due to incompatibility of the APIs. If the pipeline has
        already ran before running the RFECV, the scoring parameter will be set
        to the selected metric (if scoring=None).

        Parameters
        ----------
        strategy: string or None, optional (default=None)
            Feature selection strategy to use. Choose from:
                - None: do not perform any feature selection algorithm
                - 'univariate': perform a univariate F-test
                - 'PCA': perform principal component analysis
                - 'SFM': select best features from model
                - 'RFE': recursive feature eliminator
                - 'RFECV': RFE with cross-validated selection

            The sklearn objects can be found under the univariate, PCA, SFM,
            RFE or RFECV attributes of the class.

        solver: string, callable or None, optional (default=None)
            Solver or model to use for the feature selection strategy. See the
            sklearn documentation for an extended description of the choices.
            Select None for the default option per strategy (not applicable
            for SFM, RFE and RFECV).
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
                             after fitting. No default option. You can use a
                             model from the ATOM pipeline. No default option.
                - for 'RFE': choose a supervised learning estimator. The
                             estimator must have either a feature_importances_
                             or coef_ attribute after fitting. You can use a
                             model from the ATOM pipeline. No default option.
                - for 'RFECV': choose a supervised learning estimator. The
                               estimator must have either feature_importances_
                               or coef_ attribute after fitting. You can use a
                               model from the ATOM pipeline. No default option.

        n_features: int, float or None, optional (default=None)
            Number of features to select (except for RFECV, where it's the
            minimum number of features to select).
                - if < 1: fraction of features to select
                - if >= 1: number of features to select
                - None to select all

        max_frac_repeated: float or None, optional (default=1.)
            Remove features with the same value in at least this fraction of
            the total rows. The default is to keep all features with non-zero
            variance, i.e. remove the features that have the same value in all
            samples. None to skip this step.

        max_correlation: float or None, optional (default=0.98)
            Minimum value of the Pearson correlation coefficient to identify
            correlated features. A dataframe of the removed features and their
            correlation values can be accessed through the collinear attribute.
            None to skip this step.

        **kwargs
            Any extra parameter for the PCA, SFM, RFE or RFECV. See the
            sklearn documentation for the available options.

        """
        self.feature_selector = \
            FeatureSelector(strategy=strategy,
                            solver=solver,
                            n_features=n_features,
                            max_frac_repeated=max_frac_repeated,
                            max_correlation=max_correlation,
                            atom=self,
                            **kwargs)
        self.feature_selector.fit(self.X_train, self.y_train)

        self.X, self.y = self.feature_selector.transform(self.X, self.y)

        # Attribute for the newly generated features
        if self.feature_selector.collinear is not None:
            self.collinear = self.feature_selector.collinear

        self.update('X')

    # << ======================== Pipeline ======================== >>

    def _run_pipeline(self,
                      models: Union[str, List[str], Tuple[str]],
                      metric: Optional[Union[str, callable]] = None,
                      greater_is_better: bool = True,
                      needs_proba: bool = False,
                      needs_threshold: bool = False,
                      successive_halving: bool = False,
                      skip_iter: int = 0,
                      train_sizing: bool = False,
                      train_sizes: train_types = np.linspace(0.1, 1.0, 10),
                      max_iter: Union[int, Sequence[int]] = 0,
                      max_time: Union[scalar, Sequence[scalar]] = np.inf,
                      init_points: Union[int, Sequence[int]] = 5,
                      cv: Union[int, Sequence[int]] = 3,
                      plot_bo: bool = False,
                      bagging: Optional[int] = None):
        """Fit the models to the dataset.

        The pipeline method is where the models are fitted to the data and
        their performance is evaluated according to the selected metric. For
        every model, the pipeline applies the following steps:

            1. The optimal hyperParameters are selected using a Bayesian
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

            2. Once the best hyperParameters are found, the model is trained
               again, now using the complete training set. After this,
               predictions are made on the test set.

            3. You can choose to evaluate the robustness of each model's
            applying a bagging algorithm, i.e. the model will be trained
            multiple times on a bootstrapped training set, returning a
            distribution of its performance on the test set.

        A couple of things to take into account:
            - The metric implementation follows sklearn's API. This means that
              the implementation always tries to maximize the scorer, i.e. loss
              functions will be made negative.
            - If an exception is encountered while fitting a model, the
              pipeline will automatically jump to the next model and save the
              exception in the `errors` attribute.
            - When showing the final results, a `!!` indicates the highest
              score and a `~` indicates that the model is possibly overfitting
              (training set has a score at least 20% higher than the test set).
            - The winning model subclass will be attached to the `winner`
              attribute.

        Parameters
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
            whether the metric is a score function or a loss function,
            i.e. if True, a higher score is better and if False, lower is
            better. Will be ignored if the metric is a string or a scorer.

        needs_proba: bool, optional (default=False)
            Whether the metric function requires probability estimates out of a
            classifier. If True, make sure that every model in the pipeline has
            a `predict_proba` method! Will be ignored if the metric is a string
            or a scorer.

        needs_threshold: bool, optional (default=False)
            Whether the metric function takes a continuous decision certainty.
            This only works for binary classification using estimators that
            have either a `decision_function` or `predict_proba` method. Will
            be ignored if the metric is a string or a scorer.

        successive_halving: bool, optional (default=False)
            Whether to use successive halving on the pipeline. Only recommended
            when fitting similar models, e.g. only using tree-based models.

        skip_iter: int, optional (default=0)
            Skip last `skip_iter` iterations of the successive halving. Will be
            ignored if successive_halving=False.

        train_sizing: bool, optional (default=False)
            whether to use train sizing on the pipeline.

        train_sizes: sequence, optional (default=np.linspace(0.1, 1.0, 10))
            Relative or absolute numbers of training examples that will be used
            to generate the learning curve. If the dtype is float, it is
            regarded as a fraction of the maximum size of the training set.
            Otherwise it is interpreted as absolute sizes of the training sets.
            Will be ignored if train_sizing=False.

        max_iter: int or sequence, optional (default=10)
            Maximum number of iterations of the BO. If 0, skip the BO and fit
            the model on its default Parameters. If sequence, the n-th value
            will apply to the n-th model in the pipeline.

        max_time: int, float or sequence, optional (default=np.inf)
            Maximum time allowed for the BO per model (in seconds). If 0, skip
            the BO and fit the model on its default Parameters. If sequence,
            the n-th value will apply to the n-th model in the pipeline.

        init_points: int or sequence, optional (default=5)
            Initial number of tests of the BO before fitting the surrogate
            function. If sequence, the n-th value will apply to the n-th model
            in the pipeline.

        cv: int or sequence, optional (default=3)
            Strategy to fit and score the model selected after every step
            of the BO.
                - if 1, randomly split into a train and validation set
                - if >1, perform a k-fold cross validation on the training set

        plot_bo: bool, optional (default=False)
            whether to plot the BO's progress as it runs. Creates a canvas with
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
            """Validate the length of the parameter (equal to len(models)).

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
            """Create scaled data attributes if needed."""
            # Check if any scaling models in final_models
            scale = any(model in self.models for model in scaling_models)
            if scale and not self._is_scaled:
                # Scale features to mean=0 and std=1
                scaler = Scaler().fit(self.X_train)
                self.X_train_scaled = scaler.transform(self.X_train)
                self.X_test_scaled = scaler.transform(self.X_test)

        def run_iteration():
            """Core iterations of the pipeline.

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

                try:  # If errors occurs, just skip the model
                    # Run Bayesian Optimization
                    getattr(self, model).bayesian_optimization(
                        max_iter_, max_time_, init_points_, cv_, plot_bo)

                    # Fit the model to the test set
                    getattr(self, model).fit()

                    # Perform bagging
                    getattr(self, model).bagging(bagging)

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

            if self._has_bag:
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
                    if score_test == max(scrs):
                        self.winner = getattr(self, m)
                        if len(self.models) > 1:
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
                    if bs_mean == max(scrs):
                        self.winner = getattr(self, m)
                        if len(self.models) > 1:
                            print_ += ' !!'

                # Annotate if model overfitted when train 20% > test
                if score_train - 0.2 * score_train > score_test:
                    print_ += ' ~'

                self._log(print_)  # Print the score

            return scores

        # << ======================== Initialize ======================== >>

        t_init = time()  # To measure the time the whole pipeline takes

        self._log('\nRunning pipeline =================>')

        # Check Parameters
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

        # Make attributes of some Parameters to use them in plot functions
        self._has_sh = successive_halving
        self._has_ts = train_sizing
        self._has_bag = False if bagging is None else True

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

        if metric is None and not hasattr(self, 'metric'):
            if self.task.startswith('binary'):
                self.metric = get_scorer('f1')
                self.metric.name = 'f1'
            elif self.task.startswith('multiclass'):
                self.metric = get_scorer('f1_weighted')
                self.metric.name = 'f1_weighted'
            else:
                self.metric = get_scorer('r2')
                self.metric.name = 'r2'
        elif metric is None and hasattr(self, 'metric'):
            pass  # Metric is already defined
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
            self.metric = make_scorer(metric,
                                      greater_is_better,
                                      needs_proba,
                                      needs_threshold)
            self.metric.name = self.metric._score_func.__name__

        # Add all metrics as subclasses of the BaseMetric class
        for key, value in SCORERS.items():
            setattr(self.metric, key, value)
            setattr(getattr(self.metric, key), 'name', key)

        self._log(f"Metric: {self.metric.name}")

        # << ======================== Core ======================== >>

        if successive_halving:
            self.scores = []  # Save the cv's scores in list of dataframes
            original_df = self.dataset.copy()
            iteration = 0
            while len(self.models) > 2**skip_iter - 1:
                # Select 1/N of data to use for this iteration
                percentage = 100./len(self.models)
                self._split_dataset(original_df, percentage)
                self.update()
                data_preparation()
                self._log("\n\n<<=============== Iteration {} ==============>>"
                          .format(iteration))
                self._log("Model{} in pipeline: {}"
                          .format('s' if len(self.models) > 1 else '',
                                  ', '.join(self.models)))
                self._log(f"Percentage of data: {round(percentage, 1)}%")
                self._log(f"Size of training set: {len(self.train)}")
                self._log(f"Size of test set: {len(self.test)}")

                # Run iteration and append to the scores list
                scores = run_iteration()
                self.scores.append(scores)

                # Select best models for halving
                col = 'score_test' if not self._has_bag else 'bagging_mean'
                lx = scores.nlargest(n=int(len(self.models)/2),
                                     columns=col,
                                     keep='all')

                # Keep the models in the same order
                n = []  # List of new models
                [n.append(m) for m in self.models if m in list(lx.model)]
                self.models = n.copy()
                iteration += 1

        elif train_sizing:
            self.scores = []  # Save the cv's scores in list of dataframes
            self._sizes = []  # Number of training samples for plot
            original_df = self.dataset.copy()
            for iteration, size in enumerate(train_sizes):
                # Set size to percentage of data
                size = size*100 if size <= 1 else size*100/len(original_df)

                # Select fraction of data to use for this iteration
                self._split_dataset(original_df, size)
                self.update()
                self._sizes.append(len(self.X_train))
                data_preparation()
                self._log("\n\n<<=============== Iteration {} ==============>>"
                          .format(iteration))
                self._log(f"Percentage of data: {round(size, 1)}%")
                self._log(f"Size of training set: {len(self.train)}")
                self._log(f"Size of test set: {len(self.test)}")

                # Run iteration and append to the scores list
                scores = run_iteration()
                self.scores.append(scores)

        else:
            data_preparation()
            self.scores = run_iteration()

        self._is_fitted = True

    # =================== API pipeline methods ====================>

    @composed(crash, params_to_log, typechecked)
    def pipeline(self,
                 models: Union[str, List[str], Tuple[str]],
                 metric: Optional[Union[str, callable]] = None,
                 greater_is_better: bool = True,
                 needs_proba: bool = False,
                 needs_threshold: bool = False,
                 max_iter: Union[int, Sequence[int]] = 0,
                 max_time: Union[scalar, Sequence[scalar]] = np.inf,
                 init_points: Union[int, Sequence[int]] = 5,
                 cv: Union[int, Sequence[int]] = 3,
                 plot_bo: bool = False,
                 bagging: Optional[int] = None):
        """Fit the models to the dataset in a direct fashion."""
        self._run_pipeline(models, metric, greater_is_better, needs_proba,
                           needs_threshold, False, 0, False, [0], max_iter,
                           max_time, init_points, cv, plot_bo, bagging)

    @composed(crash, params_to_log, typechecked)
    def successive_halving(self,
                           models: Union[str, List[str], Tuple[str]],
                           metric: Optional[Union[str, callable]] = None,
                           greater_is_better: bool = True,
                           needs_proba: bool = False,
                           needs_threshold: bool = False,
                           skip_iter: int = 0,
                           max_iter: Union[int, Sequence[int]] = 0,
                           max_time: Union[scalar, Sequence[scalar]] = np.inf,
                           init_points: Union[int, Sequence[int]] = 5,
                           cv: Union[int, Sequence[int]] = 3,
                           plot_bo: bool = False,
                           bagging: Optional[int] = None):
        """Fit the models to the dataset in a successive halving fashion.

        If you want to compare similar models, you can choose to use a
        successive halving approach when running the pipeline. This technique
        fits N models to 1/N of the data. The best half are selected to go to
        the next iteration where the process is repeated. This continues until
        only one model remains, which is fitted on the complete dataset. Beware
        that a model's performance can depend greatly on the amount of data on
        which it is trained. For this reason we recommend only to use this
        technique with similar models, e.g. only using tree-based models.
        """
        self._run_pipeline(models, metric, greater_is_better, needs_proba,
                           needs_threshold, True, skip_iter, False, [0],
                           max_iter, max_time, init_points, cv, plot_bo,
                           bagging)

    @composed(crash, params_to_log, typechecked)
    def train_sizing(self,
                     models: Union[str, List[str], Tuple[str]],
                     metric: Optional[Union[str, callable]] = None,
                     greater_is_better: bool = True,
                     needs_proba: bool = False,
                     needs_threshold: bool = False,
                     train_sizes: train_types = np.linspace(0.1, 1.0, 10),
                     max_iter: Union[int, Sequence[int]] = 0,
                     max_time: Union[scalar, Sequence[scalar]] = np.inf,
                     init_points: Union[int, Sequence[int]] = 5,
                     cv: Union[int, Sequence[int]] = 3,
                     plot_bo: bool = False,
                     bagging: Optional[int] = None):
        """Fit the models to the dataset in a training sizing fashion.

        If you want to compare how different models perform when training on
        varying dataset sizes, you can choose to use the train_sizing approach
        when running the pipeline.
        """
        self._run_pipeline(models, metric, greater_is_better, needs_proba,
                           needs_threshold, False, 0, True, train_sizes,
                           max_iter, max_time, init_points, cv, plot_bo,
                           bagging)

    @composed(crash, params_to_log, typechecked)
    def transform(self,
                  X: X_types,
                  standard_cleaner: bool = True,
                  scale: bool = True,
                  impute: bool = True,
                  encode: bool = True,
                  outliers: bool = False,
                  balance: bool = False,
                  feature_generation: bool = True,
                  feature_selection: bool = True,
                  verbose: int = None):
        """Apply all data transformations in ATOM to new data.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        standard_cleaner: bool, optional (default=True)
            Whether to apply the standard cleaning step in the transformer.

        scale: bool, optional (default=True)
            Whether to apply the scaler step in the transformer.

        impute: bool, optional (default=True)
            Whether to apply the imputer step in the transformer.

        encode: bool, optional (default=True)
            Whether to apply the encoder step in the transformer.

        outliers: bool, optional (default=False)
            Whether to apply the outlier step in the transformer.

        balance: bool, optional (default=False)
            Whether to apply the balancer step in the transformer.

        feature_generation: bool, optional (default=True)
            Whether to apply the feature generator step in the transformer.

        feature_selection: bool, optional (default=True)
            Whether to apply the feature selector step in the transformer.

        verbose: int, optional (default=None)
            Verbosity level of the output. If None, it uses the ATOM verbosity.

        Returns
        -------
        X: pd.DataFrame
            Transformed dataset.

        """
        steps = dict(standard_cleaner='standard_cleaner',
                     scale='scaler',
                     impute='imputer',
                     encode='encoder',
                     outliers='outlier',
                     balance='balancer',
                     feature_generation='feature_generator',
                     feature_selection='feature_selector')

        # Check parameters
        if verbose is not None and (verbose < 0 or verbose > 3):
            raise ValueError("Invalid value for the verbose parameter." +
                             "Value should be between 0 and 3, got {}."
                             .format(verbose))

        for key, value in steps.items():
            if eval(key) and getattr(self, value) is not None:
                # If verbose is specified, change the class verbosity
                if verbose is not None:
                    getattr(self, value).verbose = verbose
                X = getattr(self, value).transform(X)

        return X

    def _pipeline_methods(self, X, attribute, **kwargs):
        """Apply pipeline methods on new data.

        First transform the new data and apply the attribute on the winning
        model. The model has to have the provided attribute.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        attribute: str
            Attribute of the model to be applied.

        **kwargs
            Additional parameters for the transform method.

        Returns
        -------
        np.array
            Return of the attribute.

        """
        check_is_fitted(self._is_fitted)
        if not hasattr(self.winner.best_model_fit, attribute):
            raise AttributeError("The winning model doesn't have a " +
                                 f"{attribute} attribute!")

        X_transformed = self.transform(X, **kwargs)
        return getattr(self.winner.best_model_fit, attribute)(X_transformed)

    @composed(crash, params_to_log, typechecked)
    def predict(self, X: X_types, **kwargs):
        """Get predictions on new data."""
        return self._pipeline_methods(X, 'predict', **kwargs)

    @composed(crash, params_to_log, typechecked)
    def predict_proba(self, X: X_types, **kwargs):
        """Get probability predictions on new data."""
        return self._pipeline_methods(X, 'predict_proba', **kwargs)

    @composed(crash, params_to_log, typechecked)
    def predict_log_proba(self, X: X_types, **kwargs):
        """Get log probability predictions on new data."""
        return self._pipeline_methods(X, 'predict_log_proba', **kwargs)

    @composed(crash, params_to_log, typechecked)
    def decision_function(self, X: X_types, **kwargs):
        """Get the decision function on new data."""
        return self._pipeline_methods(X, 'decision_function', **kwargs)

    # ======================== Plot methods =======================>

    @composed(crash, params_to_log, typechecked)
    def plot_correlation(self,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 10),
                         filename: Optional[str] = None,
                         display: bool = True):
        """Plot the data's correlation maxtrix."""
        plot_correlation(self, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_PCA(self,
                 title: Optional[str] = None,
                 figsize: Optional[Tuple[int, int]] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the explained variance ratio vs the number of component.

        Only if PCA was applied on the dataset through the feature_selection
        method.

        """
        plot_PCA(self, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_components(self,
                        show: Optional[int] = None,
                        title: Optional[str] = None,
                        figsize: Optional[Tuple[int, int]] = None,
                        filename: Optional[str] = None,
                        display: bool = True):
        """Plot the explained variance ratio per component.

        Only if PCA was applied on the dataset through the feature_selection
        method.

        """
        plot_components(self, show, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_RFECV(self,
                   title: Optional[str] = None,
                   figsize: Optional[Tuple[int, int]] = (10, 6),
                   filename: Optional[str] = None,
                   display: bool = True):
        """Plot the RFECV results.

        Plot the scores obtained by the estimator fitted on every subset of
        the data. Only if RFECV was applied on the dataset through the
        feature_selection method.

        """
        plot_RFECV(self, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_bagging(self,
                     models: Union[None, str, Sequence[str]] = None,
                     title: Optional[str] = None,
                     figsize: Optional[Tuple[int, int]] = None,
                     filename: Optional[str] = None,
                     display: bool = True):
        """Boxplot of the bagging's results."""
        plot_bagging(self, models, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_successive_halving(self,
                                models: Union[None, str, Sequence[str]] = None,
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 6),
                                filename: Optional[str] = None,
                                display: bool = True):
        """Plot the models' scores per iteration of the successive halving."""
        plot_successive_halving(self, models,
                                title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_learning_curve(
                    self,
                    models: Union[None, str, Sequence[str]] = None,
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    filename: Optional[str] = None,
                    display: bool = True):
        """Plot the model's learning curve: score vs training samples."""
        plot_learning_curve(self, models, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_ROC(self,
                 models: Union[None, str, Sequence[str]] = None,
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the Receiver Operating Characteristics curve."""
        plot_ROC(self, models, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_PRC(self,
                 models: Union[None, str, Sequence[str]] = None,
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the precision-recall curve."""
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
        """Plot the feature permutation importance of models."""
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
        """Plot a tree-based model's normalized feature importances."""
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
        """Plot the confusion matrix.

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
        """Plot performance metric(s) against threshold values."""
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
        """Plot the distribution of predicted probabilities."""
        plot_probabilities(self, models, target,
                           title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_calibration(self,
                         n_bins: int = 10,
                         models: Union[None, str, Sequence[str]] = None,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 10),
                         filename: Optional[str] = None,
                         display: bool = True):
        """Plot the calibration curve for a binary classifier."""
        plot_calibration(self, models, n_bins,
                         title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_gains(self,
                   models: Union[None, str, Sequence[str]] = None,
                   title: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 6),
                   filename: Optional[str] = None,
                   display: bool = True):
        """Plot the cumulative gains curve."""
        plot_gains(self, models, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_lift(self,
                  models: Union[None, str, Sequence[str]] = None,
                  title: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 6),
                  filename: Optional[str] = None,
                  display: bool = True):
        """Plot the lift curve."""
        plot_lift(self, models, title, figsize, filename, display)

    # <============ Classmethods for plot settings ============>

    @composed(classmethod, typechecked)
    def set_style(cls, style: str = 'darkgrid'):
        """Change the seaborn plotting style.

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
        """Change the seaborn color palette.

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
        """Change the fontsize of the plot's title.

        Parameters
        ----------
        fontsize: int
            Size of the font.

        """
        cls.title_fontsize = fontsize

    @composed(classmethod, typechecked)
    def set_label_fontsize(cls, fontsize: int = 16):
        """Change the fontsize of the plot's labels and legends.

        Parameters
        ----------
        fontsize: int
            Size of the font.

        """
        cls.label_fontsize = fontsize

    @composed(classmethod, typechecked)
    def set_tick_fontsize(cls, fontsize: int = 12):
        """Change the fontsize of the plot's ticks.

        Parameters
        ----------
        fontsize: int
            Size of the font.

        """
        cls.tick_fontsize = fontsize

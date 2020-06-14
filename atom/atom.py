# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the main ATOM class

"""

# << ============ Import Packages ============ >>

# Standard packages
import os
import warnings as warn
import numpy as np
import pandas as pd
import multiprocessing
from typeguard import typechecked
from typing import Union, Optional, Sequence, List, Tuple

# Sklearn
from sklearn.model_selection import train_test_split

# Own modules
from .models import get_model_name
from .data_cleaning import (
    BaseCleaner, StandardCleaner, Scaler,
    Imputer, Encoder, Outliers, Balancer
    )
from .feature_selection import FeatureGenerator, FeatureSelector
from .training import (
    TrainerClassifier, TrainerRegressor,
    SuccessiveHalvingClassifier, SuccessiveHalvingRegressor,
    TrainSizingClassifier, TrainSizingRegressor
    )
from .plots import BasePlotter, plot_correlation
from .utils import (
    X_TYPES, Y_TYPES, TRAIN_TYPES, composed, crash, params_to_log, merge,
    check_property, check_scaling, catch_return,  variable_return, get_metric,
    get_metric_name, get_best_score, infer_task, attach_methods, clear, save
    )


# << ================= Classes ================= >>

class ATOM(BasePlotter):
    """ATOM base class."""

    @composed(crash, params_to_log, typechecked)
    def __init__(self,
                 X: X_TYPES,
                 y: Y_TYPES,
                 n_rows: Union[int, float],
                 test_size: float,
                 n_jobs: int,
                 verbose: int,
                 warnings: Union[bool, str],
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

        n_rows: int or float, optional (default=1)
            if <=1: fraction of the data to use.
            if >1: number of rows to use.

        test_size: float, optional (default=0.3)
            Split fraction of the train and test set.

        n_jobs: int, optional (default=1)
            Number of cores to use for parallel processing.
                - If -1, use all available cores
                - If <-1, use available_cores - 1 + value

            Beware that using multiple processes on the same machine may
            cause memory issues for large datasets.

        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything.
                - 1 to print basic information.
                - 2 to print extended information.

        warnings: bool or str, optional (default=True)
            - If boolean: True for showing all warnings (equal to 'default'
                          when string) and False for suppressing them (equal
                          to 'ignore' when string).
            - If string: one of python's actions in the warnings environment.

            Note that this changing this parameter will affect the
            `PYTHONWARNINGS` environment.

        random_state: int or None, optional (default=None)
            Seed used by the random number generator. If None, the random
            number generator is the RandomState instance used by `np.random`.

        """
        # << ============= Attribute references ============= >>

        # Data attributes
        self._train_idx = None
        self._test_idx = None

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

        # Trainer attributes
        self.trainer = None
        self.models = []
        self.metric = None
        self.errors = {}
        self.winner = None
        self.results = None

        # << ============ Check input Parameters ============ >>

        if n_rows <= 0:
            raise ValueError("Invalid value for the n_rows parameter. " +
                             f"Value should be >0, got {n_rows}.")
        elif n_rows > len(X):
            n_rows = len(X)

        if test_size <= 0 or test_size >= 1:
            raise ValueError("Invalid value for the test_size parameter. " +
                             "Value should be between 0 and 1, got {}."
                             .format(test_size))

        # Update attributes and properties with params
        self._test_size = test_size
        self.verbose = verbose  # First verbose cause of log in n_jobs.setter
        self.warnings = warnings
        self.random_state = random_state

        self.log("<<=============== ATOM ===============>>")

        # << ============ Set algorithm task ============ >>

        X, y = BaseCleaner._prepare_input(X, y)

        # Assign the algorithm's task
        self.task = infer_task(y, self.goal)
        self.log(f"Algorithm task: {self.task}.")

        self.n_jobs = n_jobs  # Later, for print at right location

        # << ============ Data cleaning ============ >>

        # List of data types ATOM can't handle
        prohibited_types = ['datetime64', 'datetime64[ns]', 'timedelta[ns]']

        # Whether to map the target column to numerical values
        map_target = False if self.goal.startswith('reg') else True

        # Apply the standard cleaning steps
        self.standard_cleaner = \
            StandardCleaner(prohibited_types=prohibited_types,
                            map_target=map_target,
                            verbose=self.verbose,
                            logger=self.logger)
        X_y = merge(*self.standard_cleaner.transform(X, y))
        self.mapping = self.standard_cleaner.mapping

        # << ============ Split dataset ============= >>

        # Shuffle the dataset and get number of rows
        kwargs = {'frac': n_rows} if n_rows <= 1 else {'n': int(n_rows)}
        self.dataset = X_y.sample(random_state=self.random_state, **kwargs)
        self.dataset.reset_index(drop=True, inplace=True)

        # Get train and test indices
        train, test = train_test_split(self.dataset,
                                       test_size=self._test_size,
                                       shuffle=False)
        self._train_idx = train.index
        self._test_idx = test.index
        self.stats(1)  # Print out data stats

    # << ===================== Parameter properties ==================== >>

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    @typechecked
    def n_jobs(self, n_jobs: int):
        # Check number of cores for multiprocessing
        n_cores = multiprocessing.cpu_count()
        if n_jobs > n_cores:
            self.log("Warning! No {} cores available. n_jobs reduced to {}."
                     .format(n_jobs, n_cores))
            n_jobs = n_cores

        elif n_jobs == 0:
            self.log("Warning! Value of n_jobs can't be 0. Using 1 core.")
            n_jobs = 1

        else:
            n_jobs = n_cores + 1 + n_jobs if n_jobs < 0 else n_jobs

            # Final check for negative input
            if n_jobs < 1:
                raise ValueError("Invalid value for the n_jobs parameter, " +
                                 f"got {n_jobs}.")

            if n_jobs != 1:
                self.log(f"Parallel processing with {n_jobs} cores.")

        self._n_jobs = n_jobs

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    @typechecked
    def verbose(self, verbose: int):
        if verbose < 0 or verbose > 2:
            raise ValueError("Invalid value for the verbose parameter. Value" +
                             f" should be between 0 and 2, got {verbose}.")
        self._verbose = verbose

    @property
    def warnings(self):
        return self._warnings

    @warnings.setter
    @typechecked
    def warnings(self, warnings: Union[bool, str]):
        if isinstance(warnings, bool):
            self._warnings = 'default' if warnings else 'ignore'
        else:
            opts = ['error', 'ignore', 'always', 'default', 'module', 'once']
            if warnings not in opts:
                raise ValueError(
                    "Invalid value for the warnings parameter, got "
                    f"{warnings}. Choose from: {', '.join(opts)}.")
            self._warnings = warnings

        warn.simplefilter(self._warnings)  # Change the filter in this process
        os.environ['PYTHONWARNINGS'] = self._warnings  # Affects subprocesses

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    @typechecked
    def random_state(self, random_state: Optional[int]):
        if random_state and random_state < 0:
            raise ValueError("Invalid value for the random_state parameter. " +
                             f"Value should be >0, got {random_state}.")
        np.random.seed(random_state)  # Set random seed
        self._random_state = random_state

    # << ======================= Data properties ======================= >>

    @property
    def dataset(self):
        return self._data

    @dataset.setter
    @typechecked
    def dataset(self, dataset: X_TYPES):
        df = check_property(dataset, 'dataset')
        self._data = df

    @property
    def train(self):
        return self._data[self._data.index.isin(self._train_idx)]

    @train.setter
    @typechecked
    def train(self, train: X_TYPES):
        df = check_property(train, 'train', under=self.test, under_name='test')
        self._train_idx = df.index
        self._data = pd.concat([df, self.test])

    @property
    def test(self):
        return self._data[self._data.index.isin(self._test_idx)]

    @test.setter
    @typechecked
    def test(self, test: X_TYPES):
        df = check_property(test, 'test', under=self.train, under_name='train')
        self._test_idx = df.index
        self._data = pd.concat([self.train, df])

    @property
    def X(self):
        return self._data.drop(self.target, axis=1)

    @X.setter
    @typechecked
    def X(self, X: X_TYPES):
        df = check_property(X, 'X', side=self.y, side_name='y')
        self._data = merge(df, self.y)

    @property
    def y(self):
        return self._data[self.target]

    @y.setter
    @typechecked
    def y(self, y: Y_TYPES):
        series = check_property(y, 'y', side=self.X, side_name='X')
        self._data = merge(self._data.drop(self.target, axis=1), series)

    @property
    def X_train(self):
        return self.train.drop(self.target, axis=1)

    @X_train.setter
    def X_train(self, X_train: X_TYPES):
        df = check_property(X_train, 'X_train',
                            side=self.y_train, side_name='y_train',
                            under=self.X_test, under_name='X_test')
        self._data = pd.concat([merge(df, self.train[self.target]), self.test])

    @property
    def X_test(self):
        return self.test.drop(self.target, axis=1)

    @X_test.setter
    @typechecked
    def X_test(self, X_test: X_TYPES):
        df = check_property(X_test, 'X_test',
                            side=self.y_test, side_name='y_test',
                            under=self.X_train, under_name='X_train')
        self._data = pd.concat([self.train, merge(df, self.test[self.target])])

    @property
    def y_train(self):
        return self.train[self.target]

    @y_train.setter
    @typechecked
    def y_train(self, y_train: Y_TYPES):
        series = check_property(y_train, 'y_train',
                                side=self.X_train, side_name='X_train',
                                under=self.y_test, under_name='y_test')
        self._data = pd.concat([merge(self.X_train, series), self.test])

    @property
    def y_test(self):
        return self.test[self.target]

    @y_test.setter
    @typechecked
    def y_test(self, y_test: Y_TYPES):
        series = check_property(y_test, 'y_test',
                                side=self.X_test, side_name='X_test',
                                under=self.y_train, under_name='y_train')
        self._data = pd.concat([self.train, merge(self.X_test, series)])

    @property
    def target(self):
        return self._data.columns[-1]

    # << ======================= Utility methods ======================= >>

    @composed(crash, typechecked)
    def log(self, msg: Union[int, float, str], level: int = 0):
        """Print and save output to log file.

        Parameters
        ----------
        msg: int, float or str
            Message to save to the logger and print to stdout.

        level: int
            Minimum verbosity level in order to print the message.

        """
        if self.verbose >= level:
            print(msg)

        if self.logger is not None:
            if isinstance(msg, str):
                while msg.startswith('\n'):  # Insert empty lines
                    self.logger.info('')
                    msg = msg[1:]
            self.logger.info(str(msg))

    @crash
    def stats(self, _vb: int = -2):
        """Print some information about the dataset.

        Parameters
        ----------
        _vb: int, optional (default=-2)
            Internal parameter to always print if the user calls this method.

        """
        self.log("\nDataset stats ===================>", _vb)
        self.log(f"Shape: {self.dataset.shape}", _vb)

        nans = self.dataset.isna().sum().sum()
        if nans > 0:
            self.log(f"Missing values: {nans}", _vb)
        categ = self.X.select_dtypes(include=['category', 'object']).shape[1]
        if categ > 0:
            self.log(f"Categorical columns: {categ}", _vb)

        self.log(f"Scaled: {check_scaling(self.X)}", _vb)
        self.log("----------------------------------", _vb)
        self.log(f"Size of training set: {len(self.train)}", _vb)
        self.log(f"Size of test set: {len(self.test)}", _vb)

        # Print count of target classes
        if self.task != 'regression':
            # Create dataframe with stats per target class
            index = []
            for key, value in self.mapping.items():
                try:
                    list_ = list(map(int, self.mapping))
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

            self.log("----------------------------------", _vb + 1)
            if len(self.mapping) < 5:  # Gets ugly for too many classes
                self.log("Class balance: {} <==> {}"  # [-1] to remove last :
                         .format(keys[:-1], string[:-1]), _vb + 1)
            self.log(f"Instances in {self.target} per class:", _vb + 1)
            self.log(stats.to_markdown(), _vb + 1)

        self.log('', 1)  # Insert an empty row

    @composed(crash, params_to_log, typechecked)
    def report(self,
               df: str = 'dataset',
               n_rows: Optional[Union[int, float]] = None,  # float for 1e3...
               filename: Optional[str] = None):
        """Create an extensive profile analysis of the data.

        The profile report is rendered in HTML5 and CSS3. Note that this
        method can be slow for rows>10k. Dependency: pandas-profiling.

        Parameters
        ----------
        df: str, optional(default='dataset')
            Name of the data class property to get the report from.

        n_rows: int or None, optional(default=None)
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
        rows = getattr(self, df).shape[0] if n_rows is None else int(n_rows)

        self.log("Creating profile report...", 1)

        self.profile = ProfileReport(getattr(self, df).sample(rows))
        try:  # Render if possible (for jupyter notebook)
            from IPython.display import display
            display(self.profile)
        except ModuleNotFoundError:
            pass

        if filename:
            if not filename.endswith('.html'):
                filename = filename + '.html'
            self.profile.to_file(filename)
            self.log("Report saved successfully!", 1)

    @composed(crash, params_to_log, typechecked)
    def clear(self, models: Union[str, Sequence[str]] = 'all'):
        """Clear models from the trainer.

        If the winning model is removed. The next best model (through
        score_test or mean_bagging if available) is selected as winner.

        Parameters
        ----------
        models: str, or sequence, optional (default='all')
            Name of the models to clear from the pipeline. If 'all', clear
            all models.

        """
        # Prepare the models parameter
        if models == 'all':
            keyword = 'Pipeline'
            models = self.models.copy()
        elif isinstance(models, str):
            keyword = 'Model'
            models = [get_model_name(models)]
        else:
            keyword = 'Models'
            models = [get_model_name(m) for m in models]

        clear(self, models)

        if not self.models:
            self.trainer = None  # Clear the trainer attribute

        self.log(keyword + " cleared successfully!", 1)

    @composed(crash, params_to_log, typechecked)
    def save(self, filename: Optional[str] = None):
        """Save the ATOM class to a pickle file.

        Parameters
        ----------
        filename: str or None, optional (default=None)
            Name to save the file with. None to save with default name.

        """
        save(self, self.__class__.__name__ if filename is None else filename)
        self.log("ATOM class saved successfully!", 1)

    # << ================ Data pre-processing methods ================ >>

    @composed(crash, params_to_log)
    def scale(self):
        """Scale features to mean=0 and std=1."""
        self.scaler = Scaler(verbose=self.verbose, logger=self.logger)
        self.scaler.fit(self.X_train)

        self.X = self.scaler.transform(self.X)

    @composed(crash, params_to_log, typechecked)
    def impute(self,
               strat_num: Union[int, float, str] = 'remove',
               strat_cat: str = 'remove',
               min_frac_rows: float = 0.5,
               min_frac_cols: float = 0.5,
               missing: Optional[Union[int, float, str, list]] = None):
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
                               verbose=self.verbose,
                               logger=self.logger)
        self.imputer.fit(self.X_train, self.y_train)

        X, y = self.imputer.transform(self.X, self.y)
        self.dataset = merge(X, y)
        self.dataset.reset_index(drop=True, inplace=True)

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
                               verbose=self.verbose,
                               logger=self.logger,
                               **kwargs)
        self.encoder.fit(self.X_train, self.y_train)

        self.X = self.encoder.transform(self.X)

    @composed(crash, params_to_log, typechecked)
    def outliers(self,
                 strategy: Union[int, float, str] = 'remove',
                 max_sigma: Union[int, float] = 3,
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
                                include_target=include_target,
                                verbose=self.verbose,
                                logger=self.logger)

        X_train, y_train = self.outlier.transform(self.X_train, self.y_train)
        self.train = merge(X_train, y_train)
        self.dataset.reset_index(drop=True, inplace=True)

    @composed(crash, params_to_log, typechecked)
    def balance(self,
                oversample: Optional[Union[int, float, str]] = None,
                undersample: Optional[Union[int, float, str]] = None,
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

        if self.task == 'regression':
            raise ValueError("This method is only available for " +
                             "classification tasks!")

        self.balancer = Balancer(oversample=oversample,
                                 undersample=undersample,
                                 n_neighbors=n_neighbors,
                                 n_jobs=self.n_jobs,
                                 verbose=self.verbose,
                                 logger=self.logger,
                                 random_state=self.random_state)

        # Add mapping from atom for better printing
        self.balancer.mapping = self.mapping

        X_train, y_train = self.balancer.transform(self.X_train, self.y_train)
        self.train = merge(X_train, y_train)
        self.dataset.reset_index(drop=True, inplace=True)

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
                             n_jobs=self.n_jobs,
                             verbose=self.verbose,
                             logger=self.logger,
                             random_state=self.random_state)
        self.feature_generator.fit(self.X_train, self.y_train)

        self.X = self.feature_generator.transform(self.X)

        # Attach attributes to the ATOM class
        for attr in ['genetic_algorithm', 'genetic_features']:
            setattr(self, attr, getattr(self.feature_selector, attr))

    @composed(crash, params_to_log, typechecked)
    def feature_selection(self,
                          strategy: Optional[str] = None,
                          solver: Optional[Union[str, callable]] = None,
                          n_features: Optional[Union[int, float]] = None,
                          max_frac_repeated: Optional[Union[int, float]] = 1.,
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
        if strategy.lower() == 'univariate' and solver is None:
            if self.task.startswith('reg'):
                solver = 'f_regression'
            else:
                solver = 'f_classif'
        elif strategy.lower() in ['sfm', 'rfe', 'rfecv']:
            if solver is None and self.winner:
                solver = self.winner.best_model_fit
            elif isinstance(solver, str):
                solver += '_reg' if self.task.startswith('reg') else '_class'

        # If pipeline ran already, use selected metric for RFECV
        if strategy.lower() == 'rfecv':
            if self.metric and 'scoring' not in kwargs:
                kwargs['scoring'] = self.metric

        self.feature_selector = \
            FeatureSelector(strategy=strategy,
                            solver=solver,
                            n_features=n_features,
                            max_frac_repeated=max_frac_repeated,
                            max_correlation=max_correlation,
                            n_jobs=self.n_jobs,
                            verbose=self.verbose,
                            logger=self.logger,
                            random_state=self.random_state,
                            **kwargs)
        self.feature_selector.fit(self.X_train, self.y_train)

        # Attach used attributes to the ATOM class
        for attr in ['collinear', 'univariate',
                     'scaler', 'PCA', 'SFM', 'RFE', 'RFECV']:
            if getattr(self.feature_selector, attr) is not None:
                setattr(self, attr, getattr(self.feature_selector, attr))

        # Attach plot methods to the ATOM class
        attach_methods(self, self.feature_selector.__class__, 'plot')

        self.X = self.feature_selector.transform(self.X)

    # << ======================== Training ======================== >>

    @composed(crash, params_to_log, typechecked)
    def run(self,
            models: Union[str, List[str], Tuple[str]],
            metric: Optional[Union[str, callable]] = None,
            greater_is_better: bool = True,
            needs_proba: bool = False,
            needs_threshold: bool = False,
            n_calls: Union[int, Sequence[int]] = 0,
            n_random_starts: Union[int, Sequence[int]] = 5,
            bo_kwargs: dict = {},
            bagging: Optional[Union[int, Sequence[int]]] = None):
        """Fit the models to the dataset in a direct fashion.

        Contrary to the Trainer class in training.py, this method allows
        subsequent runs and stores all results as attributes (only the model
        subclasses are overwritten if an instance of a model is rerun).

        """
        # If this is the first direct run, clear all previous results
        if type(self.trainer).__name__ != 'Trainer':
            clear(self, models=self.models)
            if metric is None:  # Assign default metric
                if self.task.startswith('bin'):
                    metric = get_metric('f1', True, False, False)
                elif self.task.startswith('multi'):
                    metric = get_metric('f1_weighted', True, False, False)
                else:
                    metric = get_metric('r2', True, False, False)

        elif metric is None:  # Assign the existing metric
            metric = self.metric

        else:  # Check that the selected metric is the same as previous run
            metric = get_metric(
                metric, greater_is_better, needs_proba, needs_threshold)
            metric_name = get_metric_name(metric)
            if metric_name != self.metric.name:
                raise ValueError("Invalid metric parameter! Metric " +
                                 f"{self.metric.name} is already in use," +
                                 f" got {metric_name}. Use the clear method " +
                                 "before selecting a new metric.")

        params = (models, metric, greater_is_better, needs_proba,
                  needs_threshold, n_calls, n_random_starts, bo_kwargs,
                  bagging, self.n_jobs, self.verbose, self.logger,
                  self.warnings, self.random_state)

        if self.goal.startswith('class'):
            self.trainer = TrainerClassifier(*params)
        else:
            self.trainer = TrainerRegressor(*params)

        try:
            self.trainer.run(self.train, self.test)
        except ValueError as error:
            # Save errors to ATOM object before the exception
            for model, error in self.trainer.errors.items():
                self.errors[model] = error
            raise ValueError(error)
        else:  # Update attributes
            if self.results is None:
                self.results = pd.DataFrame(
                    columns=self.trainer.results.columns)

            # Attach plot and transformation methods to the ATOM class
            methods = ['plot', '_pipe', 'predict',
                       'decision', 'score', 'outcome']
            attach_methods(self, self.trainer.__class__, methods)
            self.trainer.mapping = self.mapping

            models = [m for m in self.trainer.models if m not in self.models]
            self.models += models
            self.metric = self.trainer.metric
            for idx, row in self.trainer.results.iterrows():
                self.errors.pop(idx, None)  # Remove model from errors if there
                setattr(self, idx, getattr(self.trainer, idx))
                setattr(self, idx.lower(), getattr(self.trainer, idx.lower()))
                self.results.loc[idx] = row

                # Add transform method to model subclasses
                setattr(getattr(self.trainer, idx), 'transform', self.transform)

            # Assign winning model
            best = self.results.apply(lambda row: get_best_score(row), axis=1)
            self.winner = getattr(self, str(best.idxmax()))

            for model, error in self.trainer.errors.items():
                self.errors[model] = error

    @composed(crash, params_to_log, typechecked)
    def successive_halving(
            self,
            models: Union[str, List[str], Tuple[str]],
            metric: Optional[Union[str, callable]] = None,
            greater_is_better: bool = True,
            needs_proba: bool = False,
            needs_threshold: bool = False,
            skip_iter: int = 0,
            n_calls: Union[int, Sequence[int]] = 0,
            n_random_starts: Union[int, Sequence[int]] = 5,
            bo_kwargs: dict = {},
            bagging: Optional[Union[int, Sequence[int]]] = None):
        """Fit the models to the dataset in a successive halving fashion.

        If you want to compare similar models, you can choose to use a
        successive halving approach when running the pipeline. This technique
        fits N models to 1/N of the data. The best half are selected to go to
        the side iteration where the process is repeated. This continues until
        only one model remains, which is fitted on the complete dataset. Beware
        that a model's performance can depend greatly on the amount of data on
        which it is trained. For this reason we recommend only to use this
        technique with similar models, e.g. only using tree-based models.
        """
        clear(self, models=self.models)

        params = (models, metric, greater_is_better, needs_proba,
                  needs_threshold, skip_iter, n_calls, n_random_starts,
                  bo_kwargs, bagging, self.n_jobs, self.verbose, self.logger,
                  self.warnings, self.random_state)

        if self.goal.startswith('class'):
            self.trainer = SuccessiveHalvingClassifier(*params)
        else:
            self.trainer = SuccessiveHalvingRegressor(*params)

        try:
            self.trainer.run(self.train, self.test)
        except ValueError as error:
            # Save errors to ATOM object before the exception
            for model, error in self.trainer.errors.items():
                self.errors[model] = error
            raise ValueError(error)
        else:
            attach_methods(self, self.trainer.__class__, 'plot')
            self.trainer.mapping = self.mapping
            for attr in ['models', 'metric', 'results', 'winner']:
                setattr(self, attr, getattr(self.trainer, attr))

            for idx in self.results[0].index:
                setattr(getattr(self.trainer, idx), 'transform', self.transform)

            for model, error in self.trainer.errors.items():
                self.errors[model] = error

    @composed(crash, params_to_log, typechecked)
    def train_sizing(self,
                     models: Union[str, List[str], Tuple[str]],
                     metric: Optional[Union[str, callable]] = None,
                     greater_is_better: bool = True,
                     needs_proba: bool = False,
                     needs_threshold: bool = False,
                     train_sizes: TRAIN_TYPES = np.linspace(0.2, 1.0, 5),
                     n_calls: Union[int, Sequence[int]] = 0,
                     n_random_starts: Union[int, Sequence[int]] = 5,
                     bo_kwargs: dict = {},
                     bagging: Optional[Union[int, Sequence[int]]] = None):
        """Fit the models to the dataset in a training sizing fashion.

        If you want to compare how different models perform when training on
        varying dataset sizes, you can choose to use the train_sizing approach
        when running the pipeline.
        """
        clear(self, models=self.models)

        params = (models, metric, greater_is_better, needs_proba,
                  needs_threshold, train_sizes, n_calls, n_random_starts,
                  bo_kwargs, bagging, self.n_jobs, self.verbose, self.logger,
                  self.warnings, self.random_state)

        if self.goal.startswith('class'):
            self.trainer = TrainSizingClassifier(*params)
        else:
            self.trainer = TrainSizingRegressor(*params)

        try:
            self.trainer.run(self.train, self.test)
        except ValueError as error:
            # Save errors to ATOM object before the exception
            for model, error in self.trainer.errors.items():
                self.errors[model] = error
            raise ValueError(error)
        else:
            attach_methods(self, self.trainer.__class__, 'plot')
            self.trainer.mapping = self.mapping
            for attr in ['models', 'metric', 'results', 'winner']:
                setattr(self, attr, getattr(self.trainer, attr))

            for idx in self.trainer.results[0].index:
                setattr(getattr(self.trainer, idx), 'transform', self.transform)

            for model, error in self.trainer.errors.items():
                self.errors[model] = error

    # << ================== Transformation methods ================== >>

    @composed(crash, params_to_log, typechecked)
    def transform(self,
                  X: X_TYPES,
                  y: Y_TYPES = None,
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

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            - If None, y is not used in the transformation
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

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

        y: pd.Series
            Transformed target column. Only returned if provided.

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
            if eval(key) and getattr(self, value):
                # If verbose is specified, change the class verbosity
                if verbose is not None:
                    getattr(self, value).verbose = verbose
                X, y = catch_return(getattr(self, value).transform(X, y))

        return variable_return(X, y)

    # << ======================= Plot methods ======================= >>

    @composed(crash, params_to_log, typechecked)
    def plot_correlation(self,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 10),
                         filename: Optional[str] = None,
                         display: bool = True):
        """Plot the data's correlation matrix."""
        plot_correlation(self, title, figsize, filename, display)

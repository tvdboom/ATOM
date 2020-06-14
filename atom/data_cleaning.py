# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the data cleaning estimators.

"""

# << ============ Import Packages ============ >>

# Standard packages
import os
import numpy as np
import pandas as pd
import multiprocessing
import warnings as warn
from copy import deepcopy
from typeguard import typechecked
from typing import Union, Optional, Sequence

# Sklearn
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# Other packages
from scipy.stats import zscore
from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.basen import BaseNEncoder
from category_encoders.binary import BinaryEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.polynomial import PolynomialEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.woe import WOEEncoder
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss

# Own modules
from .utils import (
    X_TYPES, Y_TYPES, prepare_logger, variable_return, to_df, to_series, merge,
    check_is_fitted, infer_task, composed, crash
    )


# << ================ Classes ================= >>

class BaseCleaner(object):
    """Base estimator for the data cleaning classes."""

    def __init__(self, **kwargs):
        """Checks standard parameters and convert to attributes.

        Parameters
        ----------
        **kwargs
            Standard keyword arguments for the classes. Can include:
                - n_jobs: number of cores to use for parallel processing.
                - verbose: Verbosity level of the output.
                - warnings: If False, suppresses all warnings.
                - logger: name of the logging file or Logger object.
                - random_state: seed used by the random number generator.

        """
        if 'n_jobs' in kwargs:
            n_jobs = kwargs['n_jobs']

            # Check number of cores for multiprocessing
            n_cores = multiprocessing.cpu_count()
            if n_jobs > n_cores:
                self.n_jobs = n_cores
            elif n_jobs == 0:
                self.n_jobs = 1
            else:
                self.n_jobs = n_cores + 1 + n_jobs if n_jobs < 0 else n_jobs

                # Final check for negative input
                if self.n_jobs < 1:
                    raise ValueError("Invalid value for the n_jobs " +
                                     f"parameter, got {n_jobs}.")

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
            if self.verbose < 0 or self.verbose > 2:
                raise ValueError("Invalid value for the verbose parameter. " +
                                 "Value should be between 0 and 2, got {}."
                                 .format(self.verbose))

        if not kwargs.get('warnings', True):
            # Ignore all warnings (changes python environment!)
            warn.simplefilter('ignore')  # Change the filter in this process
            os.environ['PYTHONWARNINGS'] = 'ignore'  # Also affect subprocesses

        if 'logger' in kwargs:
            self.logger = prepare_logger(logger=kwargs['logger'],
                                         class_name=self.__class__.__name__)

        if 'random_state' in kwargs:
            self.random_state = kwargs['random_state']
            if self.random_state and self.random_state < 0:
                raise ValueError("Invalid value for the random_state " +
                                 "parameter. Value should be >0, got {}."
                                 .format(self.random_state))
            np.random.seed(self.random_state)  # Set random seed

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

        if self.logger:
            if isinstance(msg, str):
                while msg.startswith('\n'):  # Insert empty lines
                    self.logger.info('')
                    msg = msg[1:]
            self.logger.info(str(msg))

    @staticmethod
    def _prepare_input(X, y):
        """Prepare the input data.

        Copy X and y and convert to pandas. If already in pandas frame, reset
        all indices for them to be able to merge.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            - If None, y is not used in the estimator
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Copy of the feature dataset.

        y: pd.Series
            Copy of the target column corresponding to X.

        """
        X = to_df(deepcopy(X))  # Copy to not overwrite mutable variables

        # Convert array to dataframe and target column to pandas series
        if isinstance(y, (list, tuple, dict, np.ndarray, pd.Series)):
            y = deepcopy(y)
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
                y = to_series(y, index=X.index)

            elif not X.index.equals(y.index):  # Compare indices
                raise ValueError("X and y don't have the same indices!")

            return X, y

        elif isinstance(y, str):
            if y not in X.columns:
                raise ValueError("Target column not found in X!")

            return X.drop(y, axis=1), X[y]

        elif isinstance(y, int):
            return X.drop(X.columns[y], axis=1), X[X.columns[y]]

        elif y is None:
            return X, y

    @composed(crash, typechecked)
    def fit_transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit and transform with (optionally) both X and y parameters.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            - If None, y is not used in the estimator
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Imputed dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        try:
            return self.fit(X, y).transform(X, y)
        except AttributeError:
            return self.transform(X, y)


class Scaler(BaseEstimator, BaseCleaner):
    """Scale features to mean=0 and std=1."""

    @typechecked
    def __init__(self,
                 verbose: int = 0,
                 logger: Optional[Union[str, callable]] = None):
        """Initialize class.

        Parameters
        ----------
        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything.
                - 1 to print basic information.

        logger: str or callable, optional (default=None)
            - If string: name of the logging file. 'auto' for default name
                         with timestamp. None to not save any log.
            - If callable: python Logger object.

        """
        super().__init__(verbose=verbose, logger=logger)
        self.standard_scaler = None

    @composed(crash, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit the scaler to the data.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            Does nothing. Only for continuity of API. Is returned unchanged if
            provided.

        Returns
        -------
        self: Scaler

        """
        X, y = self._prepare_input(X, y)

        # Check if features are already scaled
        self.standard_scaler = StandardScaler().fit(X)

        return self

    @composed(crash, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Scale the data.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            Does nothing. Only for continuity of API. Is returned unchanged if
            provided.

        Returns
        -------
        X: pd.DataFrame
            Scaled dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        check_is_fitted(self, 'standard_scaler')
        X, y = self._prepare_input(X, y)

        self.log("Scaling features...", 1)
        X = to_df(self.standard_scaler.transform(X), X.index, X.columns)
        return variable_return(X, y)


class StandardCleaner(BaseEstimator, BaseCleaner):
    """Applies standard data cleaning steps."""

    def __init__(self,
                 prohibited_types: Union[str, Sequence[str]] = [],
                 strip_categorical: bool = True,
                 maximum_cardinality: bool = True,
                 minimum_cardinality: bool = True,
                 missing_target: bool = True,
                 map_target: Optional[bool] = None,
                 verbose: int = 0,
                 logger: Optional[Union[str, callable]] = None,):
        """Class initializer.

        Parameters
        ----------
        prohibited_types: str or sequence, optional (default=[])
            Prohibited data type. Columns with any of these types will be
            removed from the dataset.

        strip_categorical: bool, optional (default=True)
            Whether to strip the spaces from values in the categorical columns.

        maximum_cardinality: bool, optional (default=True)
            Whether to remove categorical columns with maximum cardinality,
            i.e. the number of unique values is equal to the number of
            instances. Usually the case for names, IDs, etc...

        minimum_cardinality: bool, optional (default=True)
            Whether to remove columns with minimum cardinality, i.e. all values
            in the column are the same.

        missing_target: bool, optional (default=True)
            Whether to remove rows with missing values in the target column.

        map_target: bool or None, optional (default=None)
            Whether to map the target column to numerical values. Should only
            be used for classification tasks. If None, infer task from the
            provided target column.

        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything.
                - 1 to print basic information.
                - 2 to print extended information.

        logger: str, callable or None, optional (default=None)
            - If string: name of the logging file. 'auto' for default name
                         with timestamp. None to not save any log.
            - If callable: python Logger object.

        """
        super().__init__(verbose=verbose, logger=logger)

        # Define attributes
        if isinstance(prohibited_types, str):
            self.prohibited_types = [prohibited_types]
        else:
            self.prohibited_types = prohibited_types
        self.strip_categorical = strip_categorical
        self.maximum_cardinality = maximum_cardinality
        self.minimum_cardinality = minimum_cardinality
        self.missing_target = missing_target
        self.map_target = map_target

        self.mapping = {}

    @composed(crash, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply data cleaning steps to the data.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            - If None, y is not used in the estimator
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Feature dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        X, y = self._prepare_input(X, y)

        self.log("Standard data cleaning...", 1)

        for col in X:
            unique = X[col].unique()
            n_unique = X[col].nunique(dropna=True)

            # Drop features with invalid data type
            dtype = str(X[col].dtype)
            if dtype in self.prohibited_types:
                self.log(f" --> Dropping feature {col} due to " +
                         f"unhashable type: {dtype}.", 2)
                X.drop(col, axis=1, inplace=True)
                continue

            elif dtype in ('object', 'category') and self.strip_categorical:
                # Strip categorical features from blank spaces
                X[col] = X[col].astype(str).str.strip()

                # Drop features where all values are different
                if self.maximum_cardinality and n_unique == len(X):
                    self.log(f" --> Dropping feature {col} due to " +
                             "maximum cardinality.", 2)
                    X.drop(col, axis=1, inplace=True)

            # Drop features with minimum cardinality (all values are the same)
            if n_unique == 1 and self.minimum_cardinality:
                self.log(f" --> Dropping feature {col} due to minimum " +
                         f"cardinality. Contains only value: {unique[0]}", 2)
                X.drop(col, axis=1, inplace=True)

        if y is not None:
            # Delete rows with NaN in target
            if self.missing_target:
                length = len(y)
                y.dropna(inplace=True)
                X = X[X.index.isin(y.index)]  # Select only indices that remain
                diff = length - len(y)  # Difference in side
                if diff > 0:
                    # Reset indices for the merger
                    X.reset_index(drop=True, inplace=True)
                    y.reset_index(drop=True, inplace=True)
                    self.log(f" --> Dropping {diff} rows with missing " +
                             "values in target column.", 2)

            task = infer_task(y) if self.map_target is None else 'reg'
            # Map the target column to numerical values
            if self.map_target or not task.startswith('reg'):
                le = LabelEncoder()
                y = to_series(le.fit_transform(y), index=y.index, name=y.name)
                self.mapping = {str(v): i for i, v in enumerate(le.classes_)}

        return variable_return(X, y)


class Imputer(BaseEstimator, BaseCleaner):
    """Handle missing values in the dataset."""

    def __init__(self,
                 strat_num: Union[int, float, str] = 'remove',
                 strat_cat: str = 'remove',
                 min_frac_rows: float = 0.5,
                 min_frac_cols: float = 0.5,
                 missing: Optional[Union[int, float, str, list]] = None,
                 verbose: int = 0,
                 logger: Optional[Union[str, callable]] = None,):
        """Initialize class.

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

        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything.
                - 1 to print basic information.
                - 2 to print extended information.

        logger: str, callable or None, optional (default=None)
            - If string: name of the logging file. 'auto' for default name
                         with timestamp. None to not save any log.
            - If callable: python Logger object.

        """
        super().__init__(verbose=verbose, logger=logger)

        # Check input Parameters
        strats = ['remove', 'mean', 'median', 'knn', 'most_frequent']
        if isinstance(strat_num, str) and strat_num.lower() not in strats:
            raise ValueError("Unknown strategy for the strat_num parameter" +
                             ", got {}. Choose from: {}."
                             .format(strat_num, ', '.join(strats)))
        if min_frac_rows <= 0 or min_frac_rows >= 1:
            raise ValueError("Invalid value for the min_frac_rows parameter." +
                             "Value should be between 0 and 1, got {}."
                             .format(min_frac_rows))
        if min_frac_cols <= 0 or min_frac_cols >= 1:
            raise ValueError("Invalid value for the min_frac_cols parameter." +
                             "Value should be between 0 and 1, got {}."
                             .format(min_frac_cols))

        # Set default missing list
        if missing is None:
            missing = [np.inf, -np.inf, '', '?', 'NA', 'nan', 'inf']
        elif not isinstance(missing, list):
            missing = [missing]  # Has to be an iterable for loop

        # Some values must always be imputed (but can be double)
        missing.extend([np.inf, -np.inf])
        missing = set(missing)

        # Define attributes
        self.strat_num = strat_num
        self.strat_cat = strat_cat
        self.min_frac_rows = min_frac_rows
        self.min_frac_cols = min_frac_cols
        self.missing = missing

        self._imputers = {}
        self._is_fitted = False

    @composed(crash, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit the individual imputers on each column.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            Does nothing. Only for continuity of API. Is returned unchanged if
            provided.

        Returns
        -------
        self: Imputer

        """
        X, y = self._prepare_input(X, y)

        # Replace missing values with NaN
        X.fillna(value=np.NaN, inplace=True)  # Replace None first
        for to_replace in self.missing:
            X.replace(to_replace, np.NaN, inplace=True)

        # Drop rows with too many NaN values
        min_frac_rows = int(self.min_frac_rows * X.shape[1])
        X.dropna(axis=0, thresh=min_frac_rows, inplace=True)

        # Loop over all columns to fit the impute classes
        for col in X:
            values = X[col].values.reshape(-1, 1)

            # Column is numerical
            if X[col].dtype.kind in 'ifu':
                if isinstance(self.strat_num, str):
                    if self.strat_num.lower() == 'knn':
                        self._imputers[col] = KNNImputer().fit(values)

                    # Strategies: mean, median or most_frequent.
                    elif self.strat_num.lower() != 'remove':
                        self._imputers[col] = SimpleImputer(
                            strategy=self.strat_num.lower()
                            ).fit(values)

            # Column is categorical
            elif self.strat_cat.lower() == 'most_frequent':
                self._imputers[col] = SimpleImputer(
                    strategy=self.strat_cat.lower()
                    ).fit(values)

        self._is_fitted = True
        return self

    @composed(crash, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the missing values transformations.

        Impute or remove missing values according to the selected strategy.
        Also removes rows and columns with too many missing values.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            - If None, y is not used in the estimator
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Imputed dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        check_is_fitted(self, '_is_fitted')
        X, y = self._prepare_input(X, y)

        self.log("Imputing missing values...", 1)

        # Replace missing values with NaN
        X.fillna(value=np.NaN, inplace=True)  # Replace None first
        for to_replace in self.missing:
            X.replace(to_replace, np.NaN, inplace=True)

        # Drop rows with too many NaN values
        min_frac_rows = int(self.min_frac_rows * X.shape[1])
        length = len(X)
        X.dropna(axis=0, thresh=min_frac_rows, inplace=True)
        if y is not None:
            y = y[y.index.isin(X.index)]  # Select only indices that remain
        diff = length - len(X)
        if diff > 0:
            self.log(f" --> Removing {diff} rows for containing less than " +
                     f"{int(self.min_frac_rows*100)}% non-missing values.", 2)

        # Loop over all columns to apply strategy dependent on type
        for col in X:
            values = X[col].values.reshape(-1, 1)

            # Drop columns with too many NaN values
            nans = X[col].isna().sum()  # Number of missing values in column
            pnans = int(nans/len(X) * 100)  # Percentage of NaNs
            if (len(X) - nans)/len(X) < self.min_frac_cols:
                self.log(f" --> Removing feature {col} for containing " +
                         f"{nans} ({pnans}%) missing values.", 2)
                X.drop(col, axis=1, inplace=True)
                continue  # Skip to side column

            # Column is numerical and contains missing values
            if X[col].dtype.kind in 'ifu' and nans > 0:
                if not isinstance(self.strat_num, str):
                    self.log(f" --> Imputing {nans} missing values with " +
                             f"number {str(self.strat_num)} in feature " +
                             f"{col}.", 2)
                    X[col].replace(np.NaN, self.strat_num, inplace=True)

                elif self.strat_num.lower() == 'remove':
                    X.dropna(subset=[col], axis=0, inplace=True)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self.log(f" --> Removing {nans} rows due to missing " +
                             f"values in feature {col}.", 2)

                elif self.strat_num.lower() == 'knn':
                    self.log(f" --> Imputing {nans} missing values using " +
                             f"the KNN imputer in feature {col}.", 2)
                    X[col] = self._imputers[col].transform(values)

                else:  # Strategies: mean, median or most_frequent.
                    self.log(f" --> Imputing {nans} missing values with " +
                             f"{self.strat_num.lower()} in feature {col}.", 2)
                    X[col] = self._imputers[col].transform(values)

            # Column is categorical and contains missing values
            elif nans > 0:
                if self.strat_cat.lower() not in ['remove', 'most_frequent']:
                    self.log(f" --> Imputing {nans} missing values with " +
                             f"{self.strat_cat} in feature {col}.", 2)
                    X[col].replace(np.NaN, self.strat_cat, inplace=True)

                elif self.strat_cat.lower() == 'remove':
                    X.dropna(subset=[col], axis=0, inplace=True)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self.log(f" --> Removing {nans} rows due to missing " +
                             f"values in feature {col}.", 2)

                elif self.strat_cat.lower() == 'most_frequent':
                    self.log(f" --> Imputing {nans} missing values with " +
                             f"most_frequent in feature {col}", 2)
                    X[col] = self._imputers[col].transform(values)

        return variable_return(X, y)


class Encoder(BaseEstimator, BaseCleaner):
    """Encode categorical features."""

    def __init__(self,
                 max_onehot: Optional[int] = 10,
                 encode_type: str = 'Target',
                 frac_to_other: float = 0,
                 verbose: int = 0,
                 logger: Optional[Union[str, callable]] = None,
                 **kwargs):
        """Initialize class.

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

        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything.
                - 1 to print basic information.
                - 2 to print extended information.

        logger: str, callable or None, optional (default=None)
            - If string: name of the logging file. 'auto' for default name
                         with timestamp. None to not save any log.
            - If callable: python Logger object.

        **kwargs
            Additional keyword arguments for the encoder selected by
            the `encoder_type` parameter.

        """
        super().__init__(verbose=verbose, logger=logger)

        types = dict(BackwardDifference=BackwardDifferenceEncoder,
                     BaseN=BaseNEncoder,
                     Binary=BinaryEncoder,
                     CatBoost=CatBoostEncoder,
                     # Hashing=HashingEncoder,
                     Helmert=HelmertEncoder,
                     JamesStein=JamesSteinEncoder,
                     LeaveOneOut=LeaveOneOutEncoder,
                     MEstimate=MEstimateEncoder,
                     # OneHot=OneHotEncoder,
                     Ordinal=OrdinalEncoder,
                     Polynomial=PolynomialEncoder,
                     Sum=SumEncoder,
                     Target=TargetEncoder,
                     WOE=WOEEncoder)

        # Check Parameters
        if max_onehot is None:
            max_onehot = 0
        elif max_onehot < 0:  # if 0, 1 or 2: it never uses one-hot encoding
            raise ValueError("Invalid value for the max_onehot parameter." +
                             f"Value should be >= 0, got {max_onehot}.")
        if encode_type.lower() not in [x.lower() for x in types]:
            raise ValueError("Invalid value for the encode_type parameter." +
                             f"Choose from: {', '.join(types)}.")
        if frac_to_other < 0 or frac_to_other > 1:
            raise ValueError("Invalid value for the frac_to_other parameter." +
                             "Value should be between 0 and 1, got {}."
                             .format(frac_to_other))

        self.max_onehot = max_onehot
        for key, value in types.items():
            if key.lower() == encode_type.lower():
                self.encode_type = key
                self._rest_encoder = value
                break
        self.frac_to_other = frac_to_other
        self.kwargs = kwargs

        self._to_other = {}
        self._col_to_type = {}
        self._encoders = {}
        self._is_fitted = False

    @composed(crash, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit the individual encoders on each column.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            - If None, y is not used in the estimator
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        Returns
        -------
        self: Encoder

        """
        X, y = self._prepare_input(X, y)

        for col in X:
            self._to_other[col] = []
            if X[col].dtype.kind not in 'ifu':  # If column is categorical
                # Group uncommon categories into 'other'
                for category, count in X[col].value_counts().items():
                    if count < self.frac_to_other * len(X[col]):
                        self._to_other[col].append(category)
                        X[col].replace(category, 'other', inplace=True)

                # Check column on missing values
                if X[col].isna().any():
                    raise ValueError(f"The column {col} encountered missing " +
                                     "values. Impute them using the impute " +
                                     "method before encoding!")

                # Count number of unique values in the column
                n_unique = len(X[col].unique())

                # Convert series to dataframe for ingestion of package
                values = pd.DataFrame(X[col])

                # Perform encoding type dependent on number of unique values
                if n_unique == 2:
                    self._col_to_type[col] = 'Ordinal'
                    self._encoders[col] = \
                        OrdinalEncoder(handle_unknown='error').fit(values)

                elif 2 < n_unique <= self.max_onehot:
                    self._col_to_type[col] = 'One-hot'
                    self._encoders[col] = \
                        OneHotEncoder(handle_unknown='error',
                                      use_cat_names=True).fit(values)

                else:
                    self._col_to_type[col] = self.encode_type
                    self._encoders[col] = \
                        self._rest_encoder(handle_unknown='error',
                                           **self.kwargs).fit(values, y)

        self._is_fitted = True
        return self

    @composed(crash, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the encoding transformations.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            Does nothing. Only for continuity of API. Is returned unchanged if
            provided.

        Returns
        -------
        X: pd.DataFrame
            Imputed dataframe.

        """
        check_is_fitted(self, '_is_fitted')
        X, y = self._prepare_input(X, y)

        self.log("Encoding categorical features...", 1)

        for col in X:
            if X[col].dtype.kind not in 'ifu':  # If column is categorical
                # Convert categories to 'other'
                for category in self._to_other[col]:
                    X[col].replace(category, 'other', inplace=True)

                # Check column on missing values
                if X[col].isna().any():
                    raise ValueError(f"The column {col} encountered missing " +
                                     "values. Impute them using the impute " +
                                     "method before encoding!")

                # Count number of unique values in the column
                n_unique = len(X[col].unique())

                # Convert series to dataframe for ingestion of package
                values = pd.DataFrame(X[col])

                # Get index of the column
                idx = X.columns.get_loc(col)

                # Perform encoding type dependent on number of unique values
                if self._col_to_type[col] == 'Ordinal':
                    self.log(f" --> Label-encoding feature {col}. " +
                             f"Contains {n_unique} unique categories.", 2)
                    X[col] = self._encoders[col].transform(values)

                elif self._col_to_type[col] == 'One-hot':
                    self.log(f" --> One-hot-encoding feature {col}. " +
                             f"Contains {n_unique} unique categories.", 2)
                    onehot_cols = self._encoders[col].transform(values)
                    # Insert the new columns at old location
                    for i, column in enumerate(onehot_cols):
                        X.insert(idx + i, column, onehot_cols[column])
                    # Drop the original and _nan column
                    X = X.drop([col, onehot_cols.columns[-1]], axis=1)

                else:
                    self.log(f" --> {self.encode_type}-encoding feature " +
                             f"{col}. Contains {n_unique} unique categories.",
                             2)
                    rest_cols = self._encoders[col].transform(values)
                    X = X.drop(col, axis=1)  # Drop the original column
                    # Insert the new columns at old location
                    for i, column in enumerate(rest_cols):
                        X.insert(idx+i, column, rest_cols[column])

        return variable_return(X, y)


class Outliers(BaseEstimator, BaseCleaner):
    """Remove or replace outliers in the dataset."""

    def __init__(self,
                 strategy: Union[int, float, str] = 'remove',
                 max_sigma: Union[int, float] = 3,
                 include_target: bool = False,
                 verbose: int = 0,
                 logger: Optional[Union[str, callable]] = None):
        """Class initializer.

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

        verbose: int, optional (default=0)
            Verbosity level of the class. Possible values are:
                - 0 to not print anything.
                - 1 to print basic information.
                - 2 to print extended information.

        logger: str, callable or None, optional (default=None)
            - If string: name of the logging file. 'auto' for default name
                         with timestamp. None to not save any log.
            - If callable: python Logger object.

        """
        super().__init__(verbose=verbose, logger=logger)

        # Check Parameters
        if isinstance(strategy, str):
            if strategy.lower() not in ['remove', 'min_max']:
                raise ValueError("Invalid value for the strategy parameter." +
                                 f"Choose from: 'remove', 'min_max'.")
        if max_sigma <= 0:
            raise ValueError("Invalid value for the max_sigma parameter." +
                             f"Value should be > 0, got {max_sigma}.")

        # Define attributes
        self.strategy = strategy
        self.max_sigma = max_sigma
        self.include_target = include_target

    @composed(crash, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the transformations on the data.

        If y is provided, it will include the target column in the search
        for outliers.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            - If None, y is not used in the estimator
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Imputed dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        X, y = self._prepare_input(X, y)

        self.log('Handling outliers...', 1)

        # Get z-scores
        objective = merge(X, y) if self.include_target and y is not None else X
        z_scores = zscore(objective, nan_policy='propagate')

        if not isinstance(self.strategy, str):
            cond = np.abs(z_scores) > self.max_sigma
            objective.mask(cond, self.strategy, inplace=True)
            if cond.sum() > 0:
                self.log(f" --> Replacing {cond.sum()} outliers with " +
                         f"value {self.strategy}.", 2)

        elif self.strategy.lower() == 'min_max':
            counts = 0
            for i, col in enumerate(objective):
                # Replace outliers with NaN and after that with max,
                # so that the max is not calculated with the outliers in it
                cond1 = z_scores[:, i] > self.max_sigma
                mask = objective[col].mask(cond1, np.NaN)
                objective[col] = mask.replace(np.NaN, mask.max(skipna=True))

                # Replace outliers with minimum
                cond2 = z_scores[:, i] < -self.max_sigma
                mask = objective[col].mask(cond2, np.NaN)
                objective[col] = mask.replace(np.NaN, mask.min(skipna=True))

                # Sum number of replacements
                counts += cond1.sum() + cond2.sum()

            if counts > 0:
                self.log(f" --> Replacing {counts} outliers with the min " +
                         "or max of the column.", 2)

        elif self.strategy.lower() == 'remove':
            ix = (np.abs(zscore(z_scores)) <= self.max_sigma).all(axis=1)
            delete = len(ix) - ix.sum()  # Number of False values in index
            if delete > 0:
                self.log(f" --> Dropping {delete} rows due to outliers.", 2)

            # Remove rows based on index
            objective = objective[ix]
            if y is not None:
                y = y[ix]

        if y is not None:
            if self.include_target:
                return objective.drop(y.name, axis=1), objective[y.name]
            else:
                return objective, y
        else:
            return objective


class Balancer(BaseEstimator, BaseCleaner):
    """Balance the classes in a dataset."""

    def __init__(self,
                 oversample: Optional[Union[int, float, str]] = None,
                 undersample: Optional[Union[int, float, str]] = None,
                 n_neighbors: int = 5,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        """Initialize class.

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

        logger: str, callable or None, optional (default=None)
            - If string: name of the logging file. 'auto' for default name
                         with timestamp. None to not save any log.
            - If callable: python Logger object.

        random_state: int or None, optional (default=None)
            Seed used by the random number generator. If None, the random
            number generator is the RandomState instance used by `np.random`.

        """
        super().__init__(n_jobs=n_jobs,
                         verbose=verbose,
                         logger=logger,
                         random_state=random_state)

        # Not both strategies can be applied at the same time
        if oversample and undersample:
            raise ValueError("Oversample and undersample cannot be " +
                             "applied both at the same time!")

        # At least one of the two strategies needs to be applied
        if not oversample and not undersample:
            raise ValueError("Oversample and undersample cannot be both None!")

        # List of admitted string values
        strats = ['not majority', 'not minority', 'all', 'auto']
        strat_under = strats + ['majority']
        strat_over = strats + ['minority']
        if isinstance(oversample, str) and oversample not in strat_over:
            raise ValueError(f"Unknown value for the oversample parameter," +
                             " got {}. Choose from: {}."
                             .format(oversample, ', '.join(strat_over)))
        if isinstance(undersample, str) and undersample not in strat_under:
            raise ValueError(f"Unknown value for the undersample parameter," +
                             " got {}. Choose from: {}."
                             .format(undersample, ', '.join(strat_under)))
        if n_neighbors <= 0:
            raise ValueError("Invalid value for the n_neighbors parameter." +
                             f"Value should be >0, got {n_neighbors}.")

        # Define attributes
        self.oversample = oversample
        self.undersample = undersample
        self.n_neighbors = n_neighbors

        self.mapping = {}
        self._cols = None

    @composed(crash, typechecked)
    def transform(self, X: X_TYPES, y: Y_TYPES = -1):
        """Apply the transformations on the data.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=-1)
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Balanced dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        X, y = self._prepare_input(X, y)

        # Save index and columns for later
        index = X.index
        columns = X.columns
        name = y.name

        # Create dict of category counts in y
        counts = {}
        if not self.mapping:
            self.mapping = {str(i): v for i, v in enumerate(y.unique())}
        for key, value in self.mapping.items():
            counts[key] = np.sum(y == value)

        # Oversampling using ADASYN
        if self.oversample is not None:
            self.log("Performing oversampling...", 1)
            adasyn = ADASYN(sampling_strategy=self.oversample,
                            n_neighbors=self.n_neighbors,
                            n_jobs=self.n_jobs,
                            random_state=self.random_state)
            X, y = adasyn.fit_resample(X, y)

        # Undersampling using NearMiss
        if self.undersample is not None:
            self.log("Performing undersampling...", 1)
            NM = NearMiss(sampling_strategy=self.undersample,
                          n_neighbors=self.n_neighbors,
                          n_jobs=self.n_jobs)
            X, y = NM.fit_resample(X, y)

        # Print changes
        for key, value in self.mapping.items():
            diff = counts[key] - np.sum(y == value)
            if diff > 0:
                self.log(f" --> Removing {diff} rows from category: {key}.", 2)
            elif diff < 0:
                # Add new indices to the total index
                index = list(index) + list(np.arange(len(index) + diff))
                self.log(f" --> Adding {-diff} rows to category: {key}.", 2)

        X = to_df(X, index=index, columns=columns)
        y = to_series(y, index=index, name=name)
        return X, y

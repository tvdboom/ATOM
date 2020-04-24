# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the data cleaning estimators.

"""

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
from scipy.stats import zscore
from typing import Union, Optional, Sequence
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Third party packages
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
from .utils import scalar, variable_return, to_df, to_series, merge


# << ================ Classes ================= >>

class DataCleanerEstimator(object):
    """Base estimator for the data cleaning classes."""

    def __init__(self, **kwargs):
        """Class initializer.

        Parameters
        ----------
        **kwargs
            Can contain the ATOM class from which to borrow parameters such
            as verbose and n_jobs or can contain those parameters directly.

        """
        self.kwargs = kwargs
        self.verbose = 0  # Default verbosity
        self._is_fitted = False

        # List of possible attributes needed by the estimators
        attributes = ['task', 'mapping', 'metric', 'winner',
                      'n_jobs', 'verbose', 'random_state']

        # Attach attributes to the class
        for attr in attributes:
            if 'atom' in self.kwargs.keys():
                try:  # Can fail with task
                    setattr(self, attr, getattr(self.kwargs['atom'], attr))
                except AttributeError:
                    pass

            # If the attribute is given directly as parameter, overwrite ATOM
            if attr in self.kwargs.keys():
                setattr(self, attr, self.kwargs[attr])
                del self.kwargs[attr]

        # Separate attribute for protected member for logger
        if 'atom' in self.kwargs.keys():
            self.log = self.kwargs['atom']._log
            del self.kwargs['atom']
        else:
            self.log = None

    def _check_is_fitted(self):
        """Check Whether the class has been fitted."""
        if not self._is_fitted:
            raise AttributeError("This instance is not fitted yet. Call " +
                                 "'fit' with appropriate arguments before " +
                                 "using this estimator.")

    def _log(self, string, level=0):
        """Print and save output to log file.

        Parameters
        ----------
        string: string
            Message to save to log and print to stdout.

        level: int
            Minimum verbosity level (of the ATOM object) in order to print the
            message to the ATOM logger.

        """
        if self.verbose > level:
            if callable(self.log):
                self.log(string, level)
            else:
                print(string)

    @staticmethod
    def prepare_input(X, y):
        """Prepare the input data.

        Copy X and y and convert to pandas. If already in pandas frame, reset
        all indices for them to be able to merge.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, dict, sequence, np.array, pd.Series or None
            Target column index, name or array-like, with shape=(n_samples,).

        Returns
        -------
        X: pd.DataFrame
            Copy of the feature dataset.

        y: pd.Series
            Copy of the target column corresponding to X.

        """
        X = X.copy()  # Make a copy to not overwrite mutable variables
        # Convert X to pd.DataFrame
        if not isinstance(X, pd.DataFrame):
            X = to_df(X)
        else:
            X.reset_index(drop=True, inplace=True)

        # Convert array to dataframe and target column to pandas series
        if isinstance(y, (list, tuple, dict, np.ndarray, pd.Series)):
            y = y.copy()
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
            else:
                y.reset_index(drop=True, inplace=True)

            return X, y

        elif isinstance(y, str):
            if y not in X.columns:
                raise ValueError("Target column not found in X!")

            return X.drop(y, axis=1), X[y]

        elif isinstance(y, int):
            return X.drop(X.columns[y], axis=1), X[X.columns[y]]

        elif y is None:
            return X, y

    def fit_transform(self, X, y=None):
        """Fit and transform with (optionally) both X and y parameters.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: sequence, np.array or pd.Series, optional (default=None)
            Data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Imputed dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        return self.fit(X, y).transform(X, y)


class Scaler(BaseEstimator, DataCleanerEstimator):
    """Scale features to mean=0 and std=1."""

    def __init__(self, **kwargs):
        """Initialize class.

        Parameters
        ----------
        **kwargs
            Additional parameters for the estimator.

        """
        super().__init__(**kwargs)
        self._cols = None
        self.standard_scaler = None

    def fit(self, X, y=None):
        """Fit the scaler to the data.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: None
            Does nothing. Only for continuity of API.

        Returns
        -------
        self: Scaler

        """
        X, y = self.prepare_input(X, y)
        self._cols = X.columns

        # Check if features are already scaled
        self.standard_scaler = StandardScaler().fit(X)

        return self

    def transform(self, X, y=None):
        """Scale the data.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: sequence, np.array or pd.Series, optional (default=None)
            Data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Scaled dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        X, y = self.prepare_input(X, y)

        self._log("Scaling features...", 1)

        X = to_df(self.standard_scaler.transform(X), self._cols)
        return variable_return(X, y)


class StandardCleaner(BaseEstimator, DataCleanerEstimator):
    """Applies standard data cleaning steps."""

    def __init__(self,
                 strip_categorical: bool = True,
                 missing_target: bool = True,
                 maximum_cardinality: bool = True,
                 minimum_cardinality: bool = True,
                 prohibited_types: Optional[Union[str, Sequence[str]]] = None,
                 **kwargs):
        """Class initializer.

        Parameters
        ----------
        strip_categorical: bool, optional (default=True)
            Whether to strip the spaces from values in the categorical columns.

        missing_target: bool, optional (default=True)
            Whether to remove rows with missing values in the target column.

        maximum_cardinality: bool, optional (default=True)
            Whether to remove categorical columns with maximum cardinality,
            i.e. the number of unique values is equal to the number of
            instances. Usually the case for names, IDs, etc...

        minimum_cardinality: bool, optional (default=True)
            Whether to remove columns with minimum cardinality, i.e. all values
            in the column are the same.

        prohibited_types: str, sequence or None, optional (default=None)
            Prohibited data type. Columns with any of these types will be
            removed from the dataset.

        **kwargs
            Additional parameters for the estimator.

        """
        super().__init__(**kwargs)

        # Define attributes
        self.strip_categorical = strip_categorical
        self.missing_target = missing_target
        self.maximum_cardinality = maximum_cardinality
        self.minimum_cardinality = minimum_cardinality
        if isinstance(prohibited_types, str):
            self.prohibited_types = [prohibited_types]
        else:
            self.prohibited_types = prohibited_types

    def transform(self, X, y=None):
        """Apply data cleaning steps to the dataset.

        These steps include:
            - Removing columns with prohibited data types.
            - Strip categorical features from white spaces.
            - Removing categorical columns with maximal cardinality (the number
              of unique values is equal to the number of instances. Usually the
              case for names, IDs, etc...).
            - Removing columns with minimum cardinality (all values are equal).
            - Removing rows with missing values in the target column.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: sequence, np.array or pd.Series, optional (default=None)
            Data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Feature dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        X, y = self.prepare_input(X, y)

        self._log("Standard data cleaning...", 1)

        for col in X:
            unique = X[col].unique()
            n_unique = X[col].nunique(dropna=True)

            # Drop features with invalid data type
            dtype = str(X[col].dtype)
            if dtype in self.prohibited_types:
                self._log(f" --> Dropping feature {col} due to " +
                          f"unhashable type: {dtype}.", 2)
                X.drop(col, axis=1, inplace=True)
                continue

            elif dtype in ('object', 'category'):
                # Strip categorical features from blank spaces
                if self.strip_categorical:
                    X[col].astype(str).str.strip()

                # Drop features where all values are different
                if self.maximum_cardinality and n_unique == len(X):
                    self._log(f" --> Dropping feature {col} due to " +
                              "maximum cardinality.", 2)
                    X.drop(col, axis=1, inplace=True)

            # Drop features where all values are the same
            if n_unique == 1 and self.minimum_cardinality:
                self._log(f" --> Dropping feature {col} due to minimum " +
                          f"cardinality. Contains only value: {unique[0]}", 2)
                X.drop(col, axis=1, inplace=True)

        # Delete rows with NaN in target (if y is provided)
        if y is not None:
            length = len(y)
            y.dropna(inplace=True)
            X = X[X.index.isin(y.index)]  # Select only indices that remain
            diff = length - len(y)  # Difference in length
            if diff > 0:
                self._log(f" --> Dropping {diff} rows with missing values " +
                          "in target column.", 2)

        return variable_return(X, y)


class Imputer(BaseEstimator, DataCleanerEstimator):
    """Handle missing values in the dataset."""

    def __init__(self,
                 strat_num: Union[scalar, str] = 'remove',
                 strat_cat: str = 'remove',
                 min_frac_rows: float = 0.5,
                 min_frac_cols: float = 0.5,
                 missing: Optional[Union[scalar, str, list]] = None,
                 **kwargs):
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

        **kwargs
            Additional parameters for the estimator.

        """
        super().__init__(**kwargs)

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

    def fit(self, X, y=None):
        """Fit the individual imputers on each column.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: None
            Does nothing. Only for continuity of API.

        Returns
        -------
        self: Imputer

        """
        X, y = self.prepare_input(X, y)

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

    def transform(self, X, y=None):
        """Apply the missing values transformations.

        Impute or remove missing values according to the selected strategy.
        Also removes rows and columns with too many missing values.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: sequence, np.array or pd.Series, optional (default=None)
            Data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Imputed dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        self._check_is_fitted()
        X, y = self.prepare_input(X, y)

        self._log("Imputing missing values...", 1)

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
            self._log(f" --> Removing {diff} rows for containing less than " +
                      f"{int(self.min_frac_rows*100)}% non-missing values.", 2)

        # Loop over all columns to apply strategy dependent on type
        for col in X:
            values = X[col].values.reshape(-1, 1)

            # Drop columns with too many NaN values
            nans = X[col].isna().sum()  # Number of missing values in column
            pnans = int(nans/len(X) * 100)  # Percentage of NaNs
            if (len(X) - nans)/len(X) < self.min_frac_cols:
                self._log(f" --> Removing feature {col} for containing " +
                          f"{nans} ({pnans}%) missing values.", 2)
                X.drop(col, axis=1, inplace=True)
                continue  # Skip to next column

            # Column is numerical and contains missing values
            if X[col].dtype.kind in 'ifu' and nans > 0:
                if not isinstance(self.strat_num, str):
                    self._log(f" --> Imputing {nans} missing values with " +
                              f"number {str(self.strat_num)} in feature " +
                              f"{col}.", 2)
                    X[col].replace(np.NaN, self.strat_num, inplace=True)

                elif self.strat_num.lower() == 'remove':
                    X.dropna(subset=[col], axis=0, inplace=True)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self._log(f" --> Removing {nans} rows due to missing " +
                              f"values in feature {col}.", 2)

                elif self.strat_num.lower() == 'knn':
                    self._log(f" --> Imputing {nans} missing values using " +
                              f"the KNN imputer in feature {col}.", 2)
                    X[col] = self._imputers[col].transform(values)

                else:  # Strategies: mean, median or most_frequent.
                    self._log(f" --> Imputing {nans} missing values with " +
                              f"{self.strat_num.lower()} in feature {col}.", 2)
                    X[col] = self._imputers[col].transform(values)

            # Column is categorical and contains missing values
            elif nans > 0:
                if self.strat_cat.lower() not in ['remove', 'most_frequent']:
                    self._log(f" --> Imputing {nans} missing values with " +
                              f"{self.strat_cat} in feature {col}.", 2)
                    X[col].replace(np.NaN, self.strat_cat, inplace=True)

                elif self.strat_cat.lower() == 'remove':
                    X.dropna(subset=[col], axis=0, inplace=True)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self._log(f" --> Removing {nans} rows due to missing " +
                              f"values in feature {col}.", 2)

                elif self.strat_cat.lower() == 'most_frequent':
                    self._log(f" --> Imputing {nans} missing values with" +
                              f"most_frequent in feature {col}", 2)
                    X[col] = self._imputers[col].transform(values)

        return variable_return(X, y)


class Encoder(BaseEstimator, DataCleanerEstimator):
    """Encode categorical features."""

    def __init__(self,
                 max_onehot: Optional[int] = 10,
                 encode_type: str = 'Target',
                 frac_to_other: float = 0,
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

        **kwargs
            Additional parameters for the estimator.

        """
        super().__init__(**kwargs)

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
        if encode_type.lower() not in [x.lower() for x in types.keys()]:
            raise ValueError("Invalid value for the encode_type parameter." +
                             f"Choose from: {', '.join(types.keys())}.")
        if frac_to_other < 0 or frac_to_other > 1:
            raise ValueError("Invalid value for the frac_to_other parameter." +
                             "Value should be between 0 and 1, got {}."
                             .format(frac_to_other))

        self.max_onehot = max_onehot
        for key in types.keys():
            if key.lower() == encode_type.lower():
                self.encode_type = key
                self._rest_encoder = types[key]
                break
        self.frac_to_other = frac_to_other

        self._to_other = {}
        self._col_to_type = {}
        self._encoders = {}

    def fit(self, X, y=None):
        """Fit the individual encoders on each column.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: sequence, np.array or pd.Series, optional (default=None)
            Data target column with shape=(n_samples,)

        Returns
        -------
        self: Encoder

        """
        X, y = self.prepare_input(X, y)

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

    def transform(self, X, y=None):
        """Apply the encoding transformations.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: None
            Does nothing. Only for continuity of API. Is returned unchanged if
            provided.

        Returns
        -------
        X: pd.DataFrame
            Imputed dataframe.

        """
        self._check_is_fitted()
        X, y = self.prepare_input(X, y)

        self._log("Encoding categorical features...", 1)

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
                    self._log(f" --> Label-encoding feature {col}. " +
                              f"Contains {n_unique} unique categories.", 2)
                    X[col] = self._encoders[col].transform(values)

                elif self._col_to_type[col] == 'One-hot':
                    self._log(f" --> One-hot-encoding feature {col}. " +
                              f"Contains {n_unique} unique categories.", 2)
                    onehot_cols = self._encoders[col].transform(values)
                    # Insert the new columns at old location
                    for i, column in enumerate(onehot_cols):
                        X.insert(idx + i, column, onehot_cols[column])
                    # Drop the original and _nan column
                    X = X.drop([col, onehot_cols.columns[-1]], axis=1)

                else:
                    self._log(f" --> {self.encode_type}-encoding feature " +
                              f"{col}. Contains {n_unique} unique categories.",
                              2)
                    rest_cols = self._encoders[col].transform(values)
                    # Drop the original column
                    X = X.drop(col, axis=1)
                    # Insert the new columns at old location
                    for i, column in enumerate(rest_cols):
                        X.insert(idx+i, column, rest_cols[column])

        return variable_return(X, y)


class Outliers(BaseEstimator, DataCleanerEstimator):
    """Remove or replace outliers in the dataset."""

    def __init__(self,
                 strategy: Union[scalar, str] = 'remove',
                 max_sigma: scalar = 3,
                 **kwargs):
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

        **kwargs
            Additional parameters for the estimator.

        """
        super().__init__(**kwargs)

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

    def transform(self, X, y=None):
        """Apply the transformations on the data.

        If y is provided, it will include the target column in the search
        for outliers.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: sequence, np.array, pd.Series or None, optional (default=None)
            Data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Imputed dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        X, y = self.prepare_input(X, y)

        self._log('Handling outliers...', 1)

        # Get z-scores
        objective = X if y is None else merge(X, y)
        z_scores = zscore(objective, nan_policy='propagate')

        if not isinstance(self.strategy, str):
            cond = np.abs(z_scores) > self.max_sigma
            objective.mask(cond, self.strategy, inplace=True)
            if cond.sum() > 0:
                self._log(f" --> Replacing {cond.sum()} outliers with " +
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
                self._log(f" --> Replacing {counts} outliers with the min " +
                          "or max of the column.", 2)

        elif self.strategy.lower() == 'remove':
            ix = (np.abs(zscore(z_scores)) <= self.max_sigma).all(axis=1)
            delete = len(ix) - ix.sum()  # Number of False values in index
            if delete > 0:
                self._log(f" --> Dropping {delete} rows due to outliers.", 2)

            # Remove rows based on index
            objective = objective[ix]

        if y is None:
            return objective
        else:
            return objective.drop(y.name, axis=1), objective[y.name]


class Balancer(BaseEstimator, DataCleanerEstimator):
    """Balance the classes in a dataset."""

    def __init__(self,
                 oversample: Optional[Union[scalar, str]] = None,
                 undersample: Optional[Union[scalar, str]] = None,
                 n_neighbors: int = 5,
                 **kwargs):
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

        **kwargs
            Additional parameters for the estimator.

        """
        def check_params(name, value):
            """Check the oversample and undersample parameters.

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

        super().__init__(**kwargs)

        if self.task == 'regression':
            raise ValueError("This method is only available for " +
                             "classification tasks!")

        # Check Parameters
        check_params('oversample', oversample)
        check_params('undersample', undersample)
        if n_neighbors <= 0:
            raise ValueError("Invalid value for the n_neighbors parameter." +
                             "Value should be >0, got {percentage}.")

        # At least one of the two strategies needs to be applied
        if oversample is None and undersample is None:
            raise ValueError("Oversample and undersample cannot be both None!")

        # Define attributes
        self.oversample = oversample
        self.undersample = undersample
        self.n_neighbors = n_neighbors

        self._cols = None

    def transform(self, X, y):
        """Apply the transformations on the data.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: sequence, np.array, pd.Series or None, optional (default=None)
            Data target column with shape=(n_samples,)

        Returns
        -------
        X: pd.DataFrame
            Balanced dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        X, y = self.prepare_input(X, y)

        # Save name columns for later
        col_names = X.columns
        target_name = y.name

        # Save number of instances per target class for counting
        counts = {}
        for key, value in self.mapping.items():
            counts[key] = (y == value).sum()

        # Oversample the minority class with SMOTE
        if self.oversample is not None:
            self._log("Performing oversampling...", 1)
            adasyn = ADASYN(sampling_strategy=self.oversample,
                            n_neighbors=self.n_neighbors,
                            n_jobs=self.n_jobs,
                            random_state=self.random_state)
            X, y = adasyn.fit_resample(X, y)

            # Print changes
            for key, value in self.mapping.items():
                diff = (y == value).sum() - counts[key]
                if diff > 0:
                    self._log(f" --> Adding {diff} rows to class {key}.", 2)

        # Apply undersampling of majority class
        if self.undersample is not None:
            self._log("Performing undersampling...", 1)
            NM = NearMiss(sampling_strategy=self.undersample,
                          n_neighbors=self.n_neighbors,
                          n_jobs=self.n_jobs)
            X, y = NM.fit_resample(X, y)

            # Print changes
            for k, value in self.mapping.items():
                diff = counts[key] - (y == value).sum()
                if diff < 0:  # diff is negative since it removes
                    self._log(f" --> Removing {-diff} rows from class {k}.", 2)

        return to_df(X, columns=col_names), to_series(y, name=target_name)
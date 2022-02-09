# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the data cleaning transformers.

"""

# Standard packages
import numpy as np
from inspect import signature
from scipy.stats import zscore
from typeguard import typechecked
from collections import defaultdict
from typing import Union, Optional, Dict, Any
from sklearn.base import BaseEstimator, clone
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    LabelEncoder,
)
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.one_hot import OneHotEncoder

# Own modules
from .basetransformer import BaseTransformer
from .utils import (
    SCALAR, SEQUENCE_TYPES, X_TYPES, Y_TYPES, SCALING_STRATS,
    ENCODING_STRATS, PRUNING_STRATS, BALANCING_STRATS, lst, it,
    variable_return, to_series, merge, check_is_fitted, composed,
    crash, method_to_log,
)


class TransformerMixin:
    """Mixin class for all transformers in ATOM.

    Different from sklearn, since it accounts for the transformation
    of y and a possible absence of the fit method.

    """

    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None, **fit_params):
        """Does nothing.

         Implemented for continuity of the API of transformers without
         a fit method.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        transformer
            Fitted instance of self.

        """
        return self

    @composed(crash, method_to_log, typechecked)
    def fit_transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None, **fit_params):
        """Fit to data, then transform it.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe.

        pd.Series
            Target column corresponding to X. Only returned if provided.

        """
        return self.fit(X, y, **fit_params).transform(X, y)


class FuncTransformer(BaseTransformer):
    """Custom transformer for functions."""

    def __init__(self, func, columns, args, verbose, logger, **kwargs):
        super().__init__(verbose=verbose, logger=logger)
        self.func = func
        self.columns = columns
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"FuncTransformer(func={self.func.__name__}, columns={self.columns})"

    def fit(self, X, y):
        """Does nothing. Implemented for continuity of the API."""
        return self

    def transform(self, X, y=None):
        """Apply function to the dataset.

        If the provided column is not in the dataset, a new
        column is created at the right. If the column already
        exists, the values are replaced.

        """
        self.log(f"Applying function {self.func.__name__} to the dataset...", 1)

        dataset = X if y is None else merge(X, y)
        X[self.columns] = self.func(dataset, *self.args, **self.kwargs)

        return variable_return(X, y)


class DropTransformer(BaseTransformer):
    """Custom transformer to drop columns."""

    def __init__(self, columns, verbose, logger):
        super().__init__(verbose=verbose, logger=logger)
        self.columns = columns

    def __repr__(self):
        return f"DropTransformer(columns={self.columns})"

    def fit(self, X, y):
        """Does nothing. Implemented for continuity of the API."""
        return self

    def transform(self, X, y=None):
        """Drop columns from the dataset."""
        self.log(f"Applying DropTransformer...", 1)
        for col in self.columns:
            self.log(f" --> Dropping column {col} from the dataset.", 2)
            X = X.drop(col, axis=1)

        return variable_return(X, y)


class Scaler(BaseEstimator, TransformerMixin, BaseTransformer):
    """Scale the data.

    Apply one of sklearn's scalers. Categorical columns are ignored.

    Parameters
    ----------
    strategy: str, optional (default="standard")
        Strategy with which to scale the data. Choose from:
            - standard
            - minmax
            - maxabs
            - robust

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    <strategy>: sklearn transformer
        Object with which the data is scaled.

    """

    @typechecked
    def __init__(
        self,
        strategy: str = "standard",
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
        **kwargs,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strategy = strategy
        self.kwargs = kwargs

        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Scaler
            Fitted instance of self.

        """
        X, y = self._prepare_input(X, y)

        if self.strategy.lower() in SCALING_STRATS:
            self._estimator = SCALING_STRATS[self.strategy.lower()](**self.kwargs)

            # Add the estimator as attribute to the instance
            setattr(self, self.strategy.lower(), self._estimator)
        else:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                "Choose from: standard, minmax, maxabs, robust."
            )

        self._estimator.fit(X.select_dtypes(include="number"))
        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Scaled dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Scaling features...", 1)
        X_numerical = X.select_dtypes(include=["number"])
        X_transformed = self._estimator.transform(X_numerical)

        # Replace the numerical columns with the transformed values
        for i, col in enumerate(X_numerical):
            X[col] = X_transformed[:, i]

        return X


class Gauss(BaseEstimator, TransformerMixin, BaseTransformer):
    """Transform the data to follow a Gaussian distribution.

    This transformation is useful for modeling issues related to
    heteroscedasticity (non-constant variance), or other situations
    where normality is desired. Missing values are disregarded in
    fit and maintained in transform. Categorical columns are ignored.

    Note that the yeo-johnson and box-cox strategies standardize the
    data after transforming. Use the kwargs to change this behaviour.

    Note that the quantile strategy performs a non-linear transformation.
    This may distort linear correlations between variables measured at
    the same scale but renders variables measured at different scales
    more directly comparable.

    Parameters
    ----------
    strategy: str, optional (default="yeo-johnson")
        The transforming strategy. Choose from:
            - yeo-johnson
            - box-cox (only works with strictly positive values)
            - quantile (non-linear transformation)

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    random_state: int or None, optional (default=None)
        Seed used by the quantile strategy. If None, the random
        number generator is the `RandomState` used by `np.random`.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    <strategy>: sklearn transformer
        Object with which the data is transformed.

    """

    @typechecked
    def __init__(
        self,
        strategy: str = "yeo-johnson",
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(verbose=verbose, logger=logger, random_state=random_state)
        self.strategy = strategy
        self.kwargs = kwargs

        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Scaler
            Fitted instance of self.

        """
        X, y = self._prepare_input(X, y)

        if self.strategy.lower() in ("yeo-johnson", "yeojohnson"):
            self.yeojohnson = self._estimator = PowerTransformer(
                method="yeo-johnson",
                **self.kwargs,
            )
        elif self.strategy.lower() in ("box-cox", "boxcox"):
            self.boxcox = self._estimator = PowerTransformer(
                method="box-cox",
                **self.kwargs,
            )
        elif self.strategy.lower() == "quantile":
            random_state = self.kwargs.pop("random_state", self.random_state)
            output_distribution = self.kwargs.pop("output_distribution", "normal")
            self.quantile = self._estimator = QuantileTransformer(
                output_distribution=output_distribution,
                random_state=random_state,
                **self.kwargs,
            )
        else:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                "Choose from: yeo-johnson, box-cox, quantile."
            )

        self._estimator.fit(X.select_dtypes(include="number"))
        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the transformations to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Scaled dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log(f"Making features Gaussian-like...", 1)

        X_numerical = X.select_dtypes(include=["number"])
        X_transformed = self._estimator.transform(X_numerical)

        # Replace the numerical columns with the transformed values
        for i, col in enumerate(X_numerical):
            X[col] = X_transformed[:, i]

        return X


class Cleaner(BaseEstimator, TransformerMixin, BaseTransformer):
    """Applies standard data cleaning steps on a dataset.

    Use the parameters to choose which transformations to perform.
    The available steps are:
        - Drop columns with specific data types.
        - Strip categorical features from white spaces.
        - Drop categorical columns with maximal cardinality.
        - Drop columns with minimum cardinality.
        - Drop duplicate rows.
        - Drop rows with missing values in the target column.
        - Encode the target column.

    Parameters
    ----------
    drop_types: str, sequence or None, optional (default=None)
        Columns with these data types are dropped from the dataset.

    strip_categorical: bool, optional (default=True)
        Whether to strip spaces from the categorical columns.

    drop_max_cardinality: bool, optional (default=True)
        Whether to drop categorical columns with maximum cardinality,
        i.e. the number of unique values is equal to the number of
        samples. Usually the case for names, IDs, etc...

    drop_min_cardinality: bool, optional (default=True)
        Whether to drop columns with minimum cardinality, i.e. all
        values in the column are the same.

    drop_duplicates: bool, optional (default=False)
        Whether to drop duplicate rows. Only the first occurrence of
        every duplicated row is kept.

    drop_missing_target: bool, optional (default=True)
        Whether to drop rows with missing values in the target column.
        This parameter is ignored if `y` is not provided.

    encode_target: bool, optional (default=True)
        Whether to Label-encode the target column. This parameter is
        ignored if `y` is not provided.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    Attributes
    ----------
    missing: list
        Values that are considered "missing". Default values are: "",
        "?", "None", "NA", "nan", "NaN" and "inf". Note that `None`,
        `NaN`, `+inf` and `-inf` are always considered missing since
        they are incompatible with sklearn estimators.

    mapping: dict
        Target values mapped to their respective encoded integer. Only
        available if encode_target=True.

    """

    @typechecked
    def __init__(
        self,
        drop_types: Optional[Union[str, SEQUENCE_TYPES]] = None,
        strip_categorical: bool = True,
        drop_max_cardinality: bool = True,
        drop_min_cardinality: bool = True,
        drop_duplicates: bool = False,
        drop_missing_target: bool = True,
        encode_target: bool = True,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.drop_types = drop_types
        self.strip_categorical = strip_categorical
        self.drop_max_cardinality = drop_max_cardinality
        self.drop_min_cardinality = drop_min_cardinality
        self.drop_duplicates = drop_duplicates
        self.drop_missing_target = drop_missing_target
        self.encode_target = encode_target

        self.mapping = {}
        self.missing = ["", "?", "NA", "nan", "NaN", "None", "inf"]
        self._is_fitted = True

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the data cleaning steps to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        pd.DataFrame
            Transformed dataframe.

        pd.Series
            Target column corresponding to X. Only returned if provided.

        """
        X, y = self._prepare_input(X, y)

        self.log("Applying data cleaning...", 1)

        # Replace all missing values with NaN
        X = X.replace(self.missing + [np.inf, -np.inf], np.NaN)

        for name, column in X.items():
            # Count occurrences in the column
            n_unique = column.nunique(dropna=True)

            # Drop features with invalid data type
            if column.dtype.name in lst(self.drop_types):
                self.log(
                    f" --> Dropping feature {name} for having a "
                    f"prohibited type: {column.dtype.name}.", 2
                )
                X = X.drop(name, axis=1)
                continue

            elif column.dtype.name in ("object", "category"):
                if self.strip_categorical:
                    # Strip strings from blank spaces
                    X[name] = column.apply(
                        lambda val: val.strip() if isinstance(val, str) else val
                    )

                # Drop features where all values are different
                if self.drop_max_cardinality and n_unique == len(X):
                    self.log(
                        f" --> Dropping feature {name} due to maximum cardinality.", 2
                    )
                    X = X.drop(name, axis=1)
                    continue

            # Drop features with minimum cardinality (all values are the same)
            if self.drop_min_cardinality:
                all_nan = column.isna().sum() == len(X)
                if n_unique == 1 or all_nan:
                    self.log(
                        f" --> Dropping feature {name} due to minimum "
                        f"cardinality. Contains only 1 class: "
                        f"{'NaN' if all_nan else column.unique()[0]}."
                    )
                    X = X.drop(name, axis=1)

        # Drop duplicate samples
        if self.drop_duplicates:
            X = X.drop_duplicates(ignore_index=True)

        if y is not None:
            # Delete samples with NaN in target
            if self.drop_missing_target:
                length = len(y)  # Save original length to count deleted rows later
                y = y.replace(self.missing + [np.inf, -np.inf], np.NaN).dropna()
                X = X[X.index.isin(y.index)]  # Select only indices that remain
                diff = length - len(y)  # Difference in size
                if diff > 0:
                    self.log(
                        f" --> Dropping {diff} samples with "
                        "missing values in target column.", 2
                    )

            # Label-encode the target column
            if self.encode_target:
                encoder = LabelEncoder()
                y = to_series(encoder.fit_transform(y), y.index, y.name)
                self.mapping = {str(it(v)): i for i, v in enumerate(encoder.classes_)}

                # Only print if the target column wasn't already encoded
                if any([key != str(value) for key, value in self.mapping.items()]):
                    self.log(" --> Label-encoding the target column.", 2)

        return variable_return(X, y)


class Imputer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Handle missing values in the data.

    Impute or remove missing values according to the selected strategy.
    Also removes rows and columns with too many missing values. Use
    the `missing` attribute to customize what are considered "missing
    values".

    Parameters
    ----------
    strat_num: str, int or float, optional (default="drop")
        Imputing strategy for numerical columns. Choose from:
            - "drop": Drop rows containing missing values.
            - "mean": Impute with mean of column.
            - "median": Impute with median of column.
            - "knn": Impute using a K-Nearest Neighbors approach.
            - "most_frequent": Impute with most frequent value.
            - int or float: Impute with provided numerical value.

    strat_cat: str, optional (default="drop")
        Imputing strategy for categorical columns. Choose from:
            - "drop": Drop rows containing missing values.
            - "most_frequent": Impute with most frequent value.
            - str: Impute with provided string.

    max_nan_rows: int, float or None, optional (default=None)
        Maximum number or fraction of missing values in a row
        (if more, the row is removed). If None, ignore this step.

    max_nan_cols: int, float, optional (default=None)
        Maximum number or fraction of missing values in a column
        (if more, the column is removed). If None, ignore this step.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    Attributes
    ----------
    missing: list
        Values that are considered "missing". Default values are: "",
        "?", "None", "NA", "nan", "NaN" and "inf". Note that `None`,
        `NaN`, `+inf` and `-inf` are always considered missing since
        they are incompatible with sklearn estimators.

    """

    @typechecked
    def __init__(
        self,
        strat_num: Union[SCALAR, str] = "drop",
        strat_cat: str = "drop",
        max_nan_rows: Optional[SCALAR] = None,
        max_nan_cols: Optional[Union[float]] = None,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strat_num = strat_num
        self.strat_cat = strat_cat
        self.max_nan_rows = max_nan_rows
        self.max_nan_cols = max_nan_cols

        self.missing = ["", "?", "None", "NA", "nan", "NaN", "inf"]
        self._max_nan_rows = None
        self._max_nan_cols = None
        self._imputers = {}
        self._num_cols = []
        self._drop_cols = []
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Imputer
            Fitted instance of self.

        """
        X, y = self._prepare_input(X, y)
        self._num_cols = list(X.select_dtypes(include="number").columns)

        # Check input Parameters
        strats = ["drop", "mean", "median", "knn", "most_frequent"]
        if isinstance(self.strat_num, str) and self.strat_num.lower() not in strats:
            raise ValueError(
                "Unknown strategy for the strat_num parameter, got "
                f"{self.strat_num}. Choose from: {', '.join(strats)}."
            )
        if self.max_nan_rows:
            if self.max_nan_rows < 0:
                raise ValueError(
                    "Invalid value for the max_nan_rows parameter. "
                    f"Value should be >0, got {self.max_nan_rows}."
                )
            elif self.max_nan_rows <= 1:
                self._max_nan_rows = int(len(X.columns) * self.max_nan_rows)
            else:
                self._max_nan_rows = self.max_nan_rows

        if self.max_nan_cols:
            if self.max_nan_cols < 0:
                raise ValueError(
                    "Invalid value for the max_nan_cols parameter. "
                    f"Value should be >0, got {self.max_nan_cols}."
                )
            elif self.max_nan_cols <= 1:
                self._max_nan_cols = int(len(X) * self.max_nan_cols)
            else:
                self._max_nan_cols = self.max_nan_cols

        self.log("Fitting Imputer...", 1)

        # Replace all missing values with NaN
        X = X.replace(self.missing + [np.inf, -np.inf], np.NaN)

        # Drop rows with too many NaN values
        if self._max_nan_rows:
            X = X.dropna(axis=0, thresh=self._max_nan_rows)

        # Reset internal attrs in case of repeated fit
        self._drop_cols = []
        self._imputers = {}

        # Assign an imputer to each column
        for name, column in X.items():
            # Remember columns with too many missing values
            if self._max_nan_cols and column.isna().sum() > self._max_nan_cols:
                self._drop_cols.append(name)
                continue  # Skip to side column

            # Column is numerical
            if name in self._num_cols:
                if isinstance(self.strat_num, str):
                    if self.strat_num.lower() == "knn":
                        self._imputers[name] = KNNImputer().fit(X[[name]])

                    elif self.strat_num.lower() == "most_frequent":
                        self._imputers[name] = SimpleImputer(
                            strategy="constant",
                            fill_value=column.mode()[0],
                        ).fit(X[[name]])

                    # Strategies mean or median
                    elif self.strat_num.lower() != "drop":
                        self._imputers[name] = SimpleImputer(
                            strategy=self.strat_num.lower()
                        ).fit(X[[name]])

            # Column is categorical
            elif self.strat_cat.lower() == "most_frequent":
                self._imputers[name] = SimpleImputer(
                    strategy="constant",
                    fill_value=column.mode()[0],
                ).fit(X[[name]])

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Impute the missing values.

        Note that leaving y=None can lead to inconsistencies in
        data length between X and y if rows are dropped during
        the transformation.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        pd.Series
            Target column corresponding to X.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Imputing missing values...", 1)

        # Replace all missing values with NaN
        X = X.replace(self.missing + [np.inf, -np.inf], np.NaN)

        # Drop rows with too many missing values
        if self._max_nan_rows:
            length = len(X)
            X = X.dropna(axis=0, thresh=self._max_nan_rows)
            if y is not None:
                y = y[y.index.isin(X.index)]  # Select only indices that remain
            diff = length - len(X)
            if diff > 0:
                self.log(
                    f" --> Dropping {diff} samples for containing more "
                    f"than {self._max_nan_rows} missing values.", 2
                )

        for name, column in X.items():
            nans = column.isna().sum()

            # Drop columns with too many missing values
            if name in self._drop_cols:
                self.log(
                    f" --> Dropping feature {name}. Contains {nans} "
                    f"({nans * 100 // len(X)}%) missing values.", 2
                )
                X = X.drop(name, axis=1)
                continue

            # Apply only if column is numerical and contains missing values
            if name in self._num_cols and nans > 0:
                if not isinstance(self.strat_num, str):
                    self.log(
                        f" --> Imputing {nans} missing values with number "
                        f"{str(self.strat_num)} in feature {name}.", 2
                    )
                    X[name] = column.replace(np.NaN, self.strat_num)

                elif self.strat_num.lower() == "drop":
                    X = X.dropna(subset=[name], axis=0)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self.log(
                        f" --> Dropping {nans} samples due to missing "
                        f"values in feature {name}.", 2
                    )

                elif self.strat_num.lower() == "knn":
                    self.log(
                        f" --> Imputing {nans} missing values using "
                        f"the KNN imputer in feature {name}.", 2
                    )
                    X[name] = self._imputers[name].transform(X[[name]])

                else:  # Strategies mean, median or most_frequent
                    mode = round(self._imputers[name].statistics_[0], 2)
                    self.log(
                        f" --> Imputing {nans} missing values with "
                        f"{self.strat_num.lower()} ({mode}) in feature {name}.", 2
                    )
                    X[name] = self._imputers[name].transform(X[[name]])

            # Column is categorical and contains missing values
            elif nans > 0:
                if self.strat_cat.lower() not in ("drop", "most_frequent"):
                    self.log(
                        f" --> Imputing {nans} missing values with "
                        f"{self.strat_cat} in feature {name}.", 2
                    )
                    X[name] = column.replace(np.NaN, self.strat_cat)

                elif self.strat_cat.lower() == "drop":
                    X = X.dropna(subset=[name], axis=0)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self.log(
                        f" --> Dropping {nans} samples due to "
                        f"missing values in feature {name}.", 2
                    )

                elif self.strat_cat.lower() == "most_frequent":
                    mode = self._imputers[name].statistics_[0]
                    self.log(
                        f" --> Imputing {nans} missing values with "
                        f"most_frequent ({mode}) in feature {name}.", 2
                    )
                    X[name] = self._imputers[name].transform(X[[name]])

        return variable_return(X, y)


class Encoder(BaseEstimator, TransformerMixin, BaseTransformer):
    """Perform encoding of categorical features.

    The encoding type depends on the number of classes in the column:
        - If n_classes=2 or ordinal feature, use Ordinal-encoding.
        - If 2 < n_classes <= `max_onehot`, use OneHot-encoding.
        - If n_classes > `max_onehot`, use `strategy`-encoding.

    Missing values are propagated to the output column. Unknown
    classes encountered during transforming are imputed according
    to the selected strategy. Classes with low occurrences can be
    replaced with the value `other` in order to prevent too high
    cardinality.

    Two category-encoders estimators are unavailable:
        - OneHotEncoder: Use the `max_onehot` parameter.
        - HashingEncoder: Incompatibility of APIs.

    Parameters
    ----------
    strategy: str or estimator, optional (default="LeaveOneOut")
        Type of encoding to use for high cardinality features. Choose
        from any of the estimators in the category-encoders package
        or provide a custom one.

    max_onehot: int or None, optional (default=10)
        Maximum number of unique values in a feature to perform
        one-hot encoding. If None, `strategy`-encoding is always
        used for columns with more than two classes.

    ordinal: dict or None, optional (default=None)
        Order of ordinal features, where the dict key is the feature's
        name and the value is the class order, e.g. {"salary": ["low",
        "medium", "high"]}.

    frac_to_other: int, float or None, optional (default=None)
        Classes with fewer occurrences than `fraction_to_other` (as
        total number or fraction of rows) are replaced with the string
        `other`. This transformation is done before the encoding of the
        column. If None, skip this step.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    """

    @typechecked
    def __init__(
        self,
        strategy: Union[str, Any] = "LeaveOneOut",
        max_onehot: Optional[int] = 10,
        ordinal: Optional[Dict[str, SEQUENCE_TYPES]] = None,
        frac_to_other: Optional[SCALAR] = None,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
        **kwargs,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strategy = strategy
        self.max_onehot = max_onehot
        self.ordinal = ordinal
        self.frac_to_other = frac_to_other
        self.kwargs = kwargs

        self.mapping = {}
        self._cat_cols = None
        self._max_onehot = None
        self._frac_to_other = None
        self._to_other = defaultdict(list)
        self._categories = {}
        self._encoders = {}
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Y_TYPES = None):
        """Fit to data.

        Note that leaving y=None can lead to errors if the `strategy`
        encoder requires target values.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        Encoder
            Fitted instance of self.

        """
        X, y = self._prepare_input(X, y)
        self._cat_cols = list(X.select_dtypes(exclude="number").columns)

        if isinstance(self.strategy, str):
            if self.strategy.lower().endswith("encoder"):
                self.strategy = self.strategy[:-7]  # Remove the Encoder at the end
            if self.strategy.lower() not in ENCODING_STRATS:
                raise ValueError(
                    f"Invalid value for the strategy parameter, got {self.strategy}. "
                    f"Choose from: {', '.join(ENCODING_STRATS)}."
                )
            estimator = ENCODING_STRATS[self.strategy.lower()](
                handle_missing="return_nan",
                handle_unknown="value",
                **self.kwargs,
            )
        elif not all([hasattr(self.strategy, attr) for attr in ("fit", "transform")]):
            raise TypeError(
                "Invalid type for the strategy parameter. A custom"
                "estimator must have a fit and transform method."
            )
        elif callable(self.strategy):
            estimator = self.strategy(**self.kwargs)
        else:
            estimator = self.strategy

        if self.max_onehot is None:
            self._max_onehot = 0
        elif self.max_onehot <= 2:  # If <=2, it would never use one-hot
            raise ValueError(
                "Invalid value for the max_onehot parameter."
                f"Value should be > 2, got {self.max_onehot}."
            )
        else:
            self._max_onehot = self.max_onehot

        if self.frac_to_other:
            if self.frac_to_other < 0:
                raise ValueError(
                    "Invalid value for the frac_to_other parameter. Value "
                    f"should be between >0, got {self.frac_to_other}."
                )
            elif self.frac_to_other < 1:
                self._frac_to_other = int(self.frac_to_other * len(X))
            else:
                self._frac_to_other = self.frac_to_other

        self.log("Fitting Encoder...", 1)

        # Reset internal attrs in case of repeated fit
        self._to_other = defaultdict(list)
        self._categories, self._encoders = {}, {}

        for name, column in X[self._cat_cols].items():
            # Group uncommon classes into "other"
            if self._frac_to_other:
                for category, count in column.value_counts().items():
                    if count <= self._frac_to_other:
                        self._to_other[name].append(category)
                        X[name] = column.replace(category, "other")

            # Get the unique categories before fitting
            self._categories[name] = sorted(column.unique().tolist())

            # Perform encoding type dependent on number of unique values
            ordi = self.ordinal or {}
            if name in ordi or len(self._categories[name]) == 2:
                # Create custom mapping from 0 to N - 1
                mapping = {
                    v: i for i, v in enumerate(ordi.get(name, self._categories[name]))
                }
                mapping.setdefault(np.NaN, -1)  # Encoder always needs mapping of NaN

                self._encoders[name] = OrdinalEncoder(
                    mapping=[{"col": name, "mapping": mapping}],
                    handle_missing="return_nan",
                    handle_unknown="value",
                ).fit(X[[name]])

            elif 2 < len(self._categories[name]) <= self._max_onehot:
                self._encoders[name] = OneHotEncoder(
                    use_cat_names=True,
                    handle_missing="return_nan",
                    handle_unknown="value",
                ).fit(X[[name]])

            else:
                args = [X[[name]]]
                if "y" in signature(estimator.fit).parameters:
                    args.append(y)
                self._encoders[name] = clone(estimator).fit(*args)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the transformations to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Encoded dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Encoding categorical columns...", 1)

        for name, column in X[self._cat_cols].items():
            # Convert classes to "other"
            X[name] = column.replace(self._to_other[name], "other")

            n_classes = len(column.unique())
            self.log(
                f" --> {self._encoders[name].__class__.__name__[:-7]}-encoding "
                f"feature {name}. Contains {n_classes} classes.", 2
            )

            # Count the propagated missing values
            n_nans = column.isna().sum()
            if n_nans:
                self.log(f"   >>> Propagating {n_nans} missing values.", 2)

            # Get the new encoded columns
            new_cols = self._encoders[name].transform(X[[name]])

            # Drop _nan columns (since missing values are propagated)
            new_cols = new_cols[[col for col in new_cols if not col.endswith("_nan")]]

            # Check for unknown classes
            uc = len([i for i in column.unique() if i not in self._categories[name]])
            if uc:
                self.log(f"   >>> Handling {uc} unknown classes.", 2)

            # Insert the new columns at old location
            for i, new_col in enumerate(sorted(new_cols)):
                if new_col in X:
                    X[new_col] = new_cols[new_col].values  # Replace existing column
                else:
                    # Drop the original column
                    if name in X:
                        idx = X.columns.get_loc(name)
                        X = X.drop(name, axis=1)

                    X.insert(idx + i, new_col, new_cols[new_col])

        return X


class Pruner(BaseEstimator, TransformerMixin, BaseTransformer):
    """Prune outliers from the data.

    Replace or remove outliers. The definition of outlier depends
    on the selected strategy and can greatly differ from one another.
    Ignores categorical columns.

    Parameters
    ----------
    strategy: str or sequence, optional (default="z-score")
        Strategy with which to select the outliers. If sequence of
        strategies, only samples marked as outliers by all chosen
        strategies are dropped. Choose from:
            - "z-score": Uses the z-score of each data value.
            - "iForest": Uses an Isolation Forest.
            - "EE": Uses an Elliptic Envelope.
            - "LOF": Uses a Local Outlier Factor.
            - "SVM": Uses a One-class SVM.
            - "DBSCAN": Uses DBSCAN clustering.
            - "OPTICS": Uses OPTICS clustering.

    method: int, float or str, optional (default="drop")
        Method to apply on the outliers. Only the z-score strategy
        accepts another method than "drop". Choose from:
            - "drop": Drop any sample with outlier values.
            - "min_max": Replace outlier with the min/max of the column.
            - Any numerical value with which to replace the outliers.

    max_sigma: int or float, optional (default=3)
        Maximum allowed standard deviations from the mean of the
        column. If more, it is considered an outlier. Only if
        strategy="z-score".

    include_target: bool, optional (default=False)
        Whether to include the target column in the search for
        outliers. This can be useful for regression tasks. Only
        if strategy="z-score".

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator. If
        sequence of strategies, the params should be provided in a dict
        with the strategy's name as key.

    Attributes
    ----------
    <strategy>: sklearn estimator
        Object (lowercase strategy) used to prune the data,
        e.g. `pruner.iforest` for the isolation forest strategy.

    """

    @typechecked
    def __init__(
        self,
        strategy: Union[str, SEQUENCE_TYPES] = "z-score",
        method: Union[SCALAR, str] = "drop",
        max_sigma: SCALAR = 3,
        include_target: bool = False,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
        **kwargs,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strategy = strategy
        self.method = method
        self.max_sigma = max_sigma
        self.include_target = include_target
        self.kwargs = kwargs

        self._is_fitted = True

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the outlier strategy on the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        pd.DataFrame
            Transformed dataframe.

        pd.Series
            Target column corresponding to X. Only returned if provided.

        """
        X, y = self._prepare_input(X, y)

        # Check Parameters
        for strat in lst(self.strategy):
            if strat.lower() not in ["z-score"] + list(PRUNING_STRATS):
                raise ValueError(
                    "Invalid value for the strategy parameter. Choose from: "
                    "z-score, iForest, EE, LOF, SVM, DBSCAN, OPTICS."
                )
            if str(self.method).lower() != "drop" and strat.lower() != "z-score":
                raise ValueError(
                    "Invalid value for the method parameter. Only the z-score "
                    f"strategy accepts another method than 'drop', got {self.method}."
                )

        if isinstance(self.method, str):
            if self.method.lower() not in ("drop", "min_max"):
                raise ValueError(
                    "Invalid value for the method parameter."
                    "Choose from: drop, min_max."
                )

        if self.max_sigma <= 0:
            raise ValueError(
                "Invalid value for the max_sigma parameter."
                f"Value should be > 0, got {self.max_sigma}."
            )

        # Allocate kwargs to every estimator
        kwargs = {}
        for strat in lst(self.strategy):
            kwargs[strat.lower()] = {}
            for key, value in self.kwargs.items():
                # Parameters for this estimator only
                if key.lower() == strat.lower():
                    kwargs[strat.lower()].update(value)
                # Parameters for all estimators
                elif key.lower() not in map(str.lower, lst(self.strategy)):
                    kwargs[strat.lower()].update({key: value})

        self.log("Pruning outliers...", 1)

        # Prepare dataset (merge with y and exclude categorical columns)
        objective = merge(X, y) if self.include_target and y is not None else X
        objective = objective.select_dtypes(include=["number"])

        outliers = []
        for strat in lst(self.strategy):
            if strat.lower() == "z-score":
                z_scores = np.array(zscore(objective, nan_policy="propagate"))

                if not isinstance(self.method, str):
                    cond = np.abs(z_scores) > self.max_sigma
                    objective = objective.mask(cond, self.method)
                    self.log(
                        f" --> Replacing {cond.sum()} outlier "
                        f"values with {self.method}.", 2
                    )

                elif self.method.lower() == "min_max":
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

                    self.log(
                        f" --> Replacing {counts} outlier values "
                        "with the min or max of the column.", 2
                    )

                elif self.method.lower() == "drop":
                    mask = (np.abs(zscore(z_scores)) <= self.max_sigma).all(axis=1)
                    outliers.append(mask)
                    if len(lst(self.strategy)) > 1:
                        self.log(
                            f" --> The z-score strategy detected "
                            f"{len(mask) - sum(mask)} outliers.", 2
                        )

            else:
                estimator = PRUNING_STRATS[strat.lower()](**kwargs[strat.lower()])
                mask = estimator.fit_predict(objective) != -1
                outliers.append(mask)
                if len(lst(self.strategy)) > 1:
                    self.log(
                        f" --> The {estimator.__class__.__name__} "
                        f"detected {len(mask) - sum(mask)} outliers.", 2
                    )

                # Add the estimator as attribute to the instance
                setattr(self, strat.lower(), estimator)

        if outliers:
            # Select outliers from intersection of strategies
            mask = [any([i for i in strats]) for strats in zip(*outliers)]
            self.log(f" --> Dropping {len(mask) - sum(mask)} outliers.", 2)

            # Keep only the non-outliers from the data
            X, objective = X[mask], objective[mask]
            if y is not None:
                y = y[mask]

        else:  # Replace the columns in X with the new values from objective
            for col in objective:
                if y is None or col != y.name:
                    X[col] = objective[col]

        if y is not None:
            if self.include_target:
                return X, objective[y.name]
            else:
                return X, y
        else:
            return X


class Balancer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Balance the number of samples per class in the target column.

    When oversampling, the newly created samples have an increasing
    integer index for numerical indices, and an index of the form
    [estimator]_N for non-numerical indices, where N stands for the
    N-th sample in the data set. Use only for classification tasks.

    Parameters
    ----------
    strategy: str or estimator, optional (default="ADASYN")
        Type of algorithm with which to balance the dataset. Choose
        from any of the estimators in the imbalanced-learn package or
        provide a custom one (has to have a `fit_resample` method).

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - value.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    <strategy>: imblearn estimator
        Object (lowercase strategy) used to balance the data,
        e.g. `balancer.adasyn` for the default strategy.

    mapping: dict
        Target values mapped to their respective encoded integer.

    """

    @typechecked
    def __init__(
        self,
        strategy: Union[str, Any] = "ADASYN",
        n_jobs: int = 1,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            logger=logger,
            random_state=random_state,
        )
        self.strategy = strategy
        self.kwargs = kwargs

        self.mapping = {}
        self._is_fitted = True

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Y_TYPES = -1):
        """Balance the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence, optional (default=-1)
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        pd.DataFrame
            Balanced dataframe.

        pd.Series
            Target column corresponding to X.

        """

        def log_changes(y):
            """Print the changes per target class."""
            for key, value in self.mapping.items():
                diff = counts[key] - np.sum(y == value)
                if diff > 0:
                    self.log(f" --> Removing {diff} samples from class: {key}.", 2)
                elif diff < 0:
                    self.log(f" --> Adding {-diff} samples to class: {key}.", 2)

        X, y = self._prepare_input(X, y)

        if isinstance(self.strategy, str):
            if self.strategy.lower() not in BALANCING_STRATS:
                raise ValueError(
                    f"Invalid value for the strategy parameter, got {self.strategy}. "
                    f"Choose from: {', '.join(BALANCING_STRATS)}."
                )
            estimator = BALANCING_STRATS[self.strategy.lower()](**self.kwargs)
        elif not hasattr(self.strategy, "fit_resample"):
            raise TypeError(
                "Invalid type for the strategy parameter. A "
                "custom estimator must have a fit_resample method."
            )
        elif callable(self.strategy):
            estimator = self.strategy(**self.kwargs)
        else:
            estimator = self.strategy

        # Create dict of class counts in y
        counts = {}
        if not self.mapping:
            self.mapping = {str(i): v for i, v in enumerate(y.unique())}
        for key, value in self.mapping.items():
            counts[key] = np.sum(y == value)

        # Add n_jobs or random_state if its one of the estimator's parameters
        for param in ("n_jobs", "random_state"):
            if param in estimator.get_params():
                estimator.set_params(**{param: getattr(self, param)})

        if "over_sampling" in estimator.__module__:
            self.log(f"Oversampling with {estimator.__class__.__name__}...", 1)

            index = X.index  # Save indices for later reassignment
            X, y = estimator.fit_resample(X, y)

            # Create indices for the new samples
            if index.dtype.kind in "ifu":
                new_index = range(max(index) + 1, max(index) + len(X) - len(index) + 1)
            else:
                new_index = [
                    f"{estimator.__class__.__name__.lower()}_{i}"
                    for i in range(1, len(X) - len(index) + 1)
                ]

            # Assign the old + new indices
            X.index = list(index) + list(new_index)
            y.index = list(index) + list(new_index)

            log_changes(y)

        elif "under_sampling" in estimator.__module__:
            self.log(f"Undersampling with {estimator.__class__.__name__}...", 1)

            estimator.fit_resample(X, y)

            # Select chosen rows (imblearn doesn't return them in order)
            samples = sorted(estimator.sample_indices_)
            X, y = X.iloc[samples, :], y.iloc[samples]

            log_changes(y)

        elif "combine" in estimator.__module__:
            self.log(f"Balancing with {estimator.__class__.__name__}...", 1)

            index = X.index
            X_new, y_new = estimator.fit_resample(X, y)

            # Select rows that were kept by the undersampler
            if estimator.__class__.__name__ == "SMOTEENN":
                samples = sorted(estimator.enn_.sample_indices_)
            elif estimator.__class__.__name__ == "SMOTETomek":
                samples = sorted(estimator.tomek_.sample_indices_)

            # Select the remaining samples from the old dataframe
            old_samples = [s for s in samples if s < len(X)]
            X, y = X.iloc[old_samples, :], y.iloc[old_samples]

            # Create indices for the new samples
            if index.dtype.kind in "ifu":
                new_index = range(max(index) + 1, max(index) + len(X_new) - len(X) + 1)
            else:
                new_index = [
                    f"{estimator.__class__.__name__.lower()}_{i}"
                    for i in range(1, len(X_new) - len(X) + 1)
                ]

            # Select the new samples and assign the new indices
            X_new = X_new.iloc[-len(X_new) + len(old_samples):, :]
            X_new.index = new_index
            y_new = y_new.iloc[-len(y_new) + len(old_samples):]
            y_new.index = new_index

            # First, output the samples created
            for key, value in self.mapping.items():
                diff = np.sum(y_new == value)
                if diff > 0:
                    self.log(f" --> Adding {diff} samples to class: {key}.", 2)

            # Then, output the samples dropped
            for key, value in self.mapping.items():
                diff = counts[key] - np.sum(y == value)
                if diff > 0:
                    self.log(f" --> Removing {diff} samples from class: {key}.", 2)

            # Add the new samples to the old dataframe
            X, y = X.append(X_new), y.append(y_new)

        # Add the estimator as attribute to the instance
        setattr(self, estimator.__class__.__name__.lower(), estimator)

        return X, y

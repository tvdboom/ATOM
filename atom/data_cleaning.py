# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the data cleaning estimators.

"""

# Standard packages
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Union, Optional, Sequence

# Sklearn
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# Other packages
from scipy.stats import zscore
from category_encoders.one_hot import OneHotEncoder

# Own modules
from .basetransformer import BaseTransformer
from .utils import (
    X_TYPES, Y_TYPES, ENCODER_TYPES, BALANCER_TYPES, variable_return, to_df,
    to_series, merge, check_is_fitted, composed, crash, method_to_log,
)


class BaseCleaner(object):
    """Base class for the data_cleaning and feature_engineering methods."""

    @composed(crash, method_to_log, typechecked)
    def fit_transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit and transform with (optionally) both X and y parameters.

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, list, tuple,  np.array or pd.Series
            - If None: y is ignored in the transformation.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Data target column with shape=(n_samples,).

        Returns
        -------
        X: pd.DataFrame
            Transformed dataframe.

        y: pd.Series
            Target column corresponding to X. Only returned if provided.

        """
        try:
            return self.fit(X, y).transform(X, y)
        except AttributeError:
            return self.transform(X, y)


class Scaler(BaseEstimator, BaseTransformer, BaseCleaner):
    """Scale data to mean=0 and std=1.

    Parameters
    ----------
    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If str: name of the logging file. "auto" for default name.
        - If class: python `Logger` object.

    """

    @typechecked
    def __init__(
        self, verbose: int = 0, logger: Optional[Union[bool, str, callable]] = None
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.standard_scaler = None

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit the scaler to the data.

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, list, tuple,  np.array or pd.Series, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        self: Scaler

        """
        X, y = self._prepare_input(X, y)
        self.standard_scaler = StandardScaler().fit(X)

        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Scale the data.

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, list, tuple,  np.array or pd.Series, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Scaled dataframe.

        """
        check_is_fitted(self, "standard_scaler")
        X, y = self._prepare_input(X, y)

        self.log("Scaling features...", 1)
        return to_df(self.standard_scaler.transform(X), X.index, X.columns)


class Cleaner(BaseEstimator, BaseTransformer, BaseCleaner):
    """Applies standard data cleaning steps on a dataset.

     These steps can include:
        - Strip categorical features from white spaces.
        - Removing columns with prohibited data types.
        - Removing categorical columns with maximal cardinality.
        - Removing columns with minimum cardinality.
        - Removing rows with missing values in the target column.
        - Encode the target column.

    Parameters
    ----------
    prohibited_types: str or iterable, optional (default=[])
        Columns with any of these types will be removed from the dataset.

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
        Ignored if y is not provided.

    encode_target: bool, optional (default=True)
        Whether to Label-encode the target column.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If str: name of the logging file. "auto" for default name.
        - If class: python `Logger` object.

    """

    def __init__(
        self,
        prohibited_types: Optional[Union[str, Sequence[str]]] = None,
        strip_categorical: bool = True,
        maximum_cardinality: bool = True,
        minimum_cardinality: bool = True,
        missing_target: bool = True,
        encode_target: bool = True,
        verbose: int = 0,
        logger: Optional[Union[bool, str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.prohibited_types = prohibited_types
        self.strip_categorical = strip_categorical
        self.maximum_cardinality = maximum_cardinality
        self.minimum_cardinality = minimum_cardinality
        self.missing_target = missing_target
        self.encode_target = encode_target

        self.mapping = {}

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply data cleaning steps to the data.

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, list, tuple,  np.array or pd.Series
            - If None: y is ignored in the transformation.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        X: pd.DataFrame
            Feature dataframe.

        y: pd.Series
            Target column corresponding to X. Only returned if provided.

        """
        X, y = self._prepare_input(X, y)

        # Prepare the type of prohibited_types
        if not self.prohibited_types:
            self.prohibited_types = []
        elif isinstance(self.prohibited_types, str):
            self.prohibited_types = [self.prohibited_types]

        self.log("Applying data cleaning...", 1)

        for col in X:
            unique = X[col].unique()
            n_unique = X[col].nunique(dropna=True)

            # Drop features with invalid data type
            dtype = str(X[col].dtype)
            if dtype in self.prohibited_types:
                self.log(
                    f" --> Dropping feature {col} for having a "
                    f"prohibited type: {dtype}.", 2
                )
                X.drop(col, axis=1, inplace=True)
                continue

            elif dtype in ("object", "category"):  # If non-numerical feature...
                if self.strip_categorical:
                    # Strip strings from blank spaces
                    X[col] = X[col].apply(
                        lambda val: val.strip() if isinstance(val, str) else val
                    )

                # Drop features where all values are different
                if self.maximum_cardinality and n_unique == len(X):
                    self.log(
                        f" --> Dropping feature {col} due to maximum cardinality.", 2
                    )
                    X.drop(col, axis=1, inplace=True)

            # Drop features with minimum cardinality (all values are the same)
            if n_unique == 1 and self.minimum_cardinality:
                self.log(
                    f" --> Dropping feature {col} due to minimum "
                    f"cardinality. Contains only 1 class: {unique[0]}.", 2
                )
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
                    self.log(
                        f" --> Dropping {diff} rows with "
                        "missing values in target column.", 2
                    )

            # Label-encode the target column
            if self.encode_target:
                self.log(f" --> Label-encoding the target column.", 2)
                encoder = LabelEncoder()
                y = to_series(encoder.fit_transform(y), index=y.index, name=y.name)
                self.mapping = {str(v): i for i, v in enumerate(encoder.classes_)}

        return variable_return(X, y)


class Imputer(BaseEstimator, BaseTransformer, BaseCleaner):
    """Handle missing values in the data.

    Impute or remove missing values according to the selected strategy.
    Also removes rows and columns with too many missing values.

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

    min_frac_rows: float, optional (default=0.5)
        Minimum fraction of non-missing values in a row. If less,
        the row is removed.

    min_frac_cols: float, optional (default=0.5)
        Minimum fraction of non-missing values in a column. If less,
        the column is removed.

    missing: int, float or list, optional (default=None)
        List of values to treat as "missing". None to use the default
        values: [None, np.NaN, np.inf, -np.inf, "", "?", "NA", "nan", "None", "inf"].
        Note that np.NaN, None, np.inf and -np.inf will always be imputed since they
        are incompatible with most estimators.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If str: name of the logging file. "auto" for default name.
        - If class: python `Logger` object.

    """

    def __init__(
        self,
        strat_num: Union[int, float, str] = "drop",
        strat_cat: str = "drop",
        min_frac_rows: float = 0.5,
        min_frac_cols: float = 0.5,
        missing: Optional[Union[int, float, str, list]] = None,
        verbose: int = 0,
        logger: Optional[Union[bool, str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strat_num = strat_num
        self.strat_cat = strat_cat
        self.min_frac_rows = min_frac_rows
        self.min_frac_cols = min_frac_cols
        self.missing = missing

        self._imputers = {}
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit the individual imputers on each column.

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, list, tuple,  np.array or pd.Series, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        self: Imputer

        """
        X, y = self._prepare_input(X, y)

        # Check input Parameters
        strats = ["drop", "mean", "median", "knn", "most_frequent"]
        if isinstance(self.strat_num, str) and self.strat_num.lower() not in strats:
            raise ValueError(
                "Unknown strategy for the strat_num parameter, got "
                f"{self.strat_num}. Choose from: {', '.join(strats)}."
            )
        if self.min_frac_rows <= 0 or self.min_frac_rows >= 1:
            raise ValueError(
                "Invalid value for the min_frac_rows parameter. Value "
                f"should be between 0 and 1, got {self.min_frac_rows}."
            )
        if self.min_frac_cols <= 0 or self.min_frac_cols >= 1:
            raise ValueError(
                "Invalid value for the min_frac_cols parameter. Value "
                f"should be between 0 and 1, got {self.min_frac_cols}."
            )

        # Set default missing list
        if self.missing is None:
            self.missing = [np.inf, -np.inf, "", "?", "NA", "nan", "None", "inf"]
        elif not isinstance(self.missing, list):
            self.missing = [self.missing]  # Has to be an iterable for loop

        # Some values must always be imputed (but can be double)
        self.missing.extend([np.inf, -np.inf])
        self.missing = set(self.missing)

        self.log("Fitting Imputer...", 1)

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
            if X[col].dtype.kind in "ifu":
                if isinstance(self.strat_num, str):
                    if self.strat_num.lower() == "knn":
                        self._imputers[col] = KNNImputer().fit(values)

                    # Strategies: mean, median or most_frequent.
                    elif self.strat_num.lower() != "drop":
                        self._imputers[col] = SimpleImputer(
                            strategy=self.strat_num.lower()
                        ).fit(values)

            # Column is categorical
            elif self.strat_cat.lower() == "most_frequent":
                self._imputers[col] = SimpleImputer(
                    strategy=self.strat_cat.lower()
                ).fit(values)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the missing values transformations.

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, list, tuple,  np.array or pd.Series
            - If None: y is ignored in the transformation.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        X: pd.DataFrame
            Imputed dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        check_is_fitted(self, "_is_fitted")
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
            self.log(
                f" --> Dropping {diff} rows for containing less than "
                f"{int(self.min_frac_rows*100)}% non-missing values.", 2
            )

        # Loop over all columns to apply strategy dependent on type
        for col in X:
            values = X[col].values.reshape(-1, 1)

            # Drop columns with too many NaN values
            nans = X[col].isna().sum()  # Number of missing values in column
            p_nans = int(nans / len(X) * 100)  # Percentage of NaNs
            if (len(X) - nans) / len(X) < self.min_frac_cols:
                self.log(
                    f" --> Dropping feature {col} for containing "
                    f"{nans} ({p_nans}%) missing values.", 2
                )
                X.drop(col, axis=1, inplace=True)
                continue  # Skip to side column

            # Column is numerical and contains missing values
            if X[col].dtype.kind in "ifu" and nans > 0:
                if not isinstance(self.strat_num, str):
                    self.log(
                        f" --> Imputing {nans} missing values with number "
                        f"{str(self.strat_num)} in feature {col}.", 2
                    )
                    X[col].replace(np.NaN, self.strat_num, inplace=True)

                elif self.strat_num.lower() == "drop":
                    X.dropna(subset=[col], axis=0, inplace=True)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self.log(
                        f" --> Dropping {nans} rows due to missing "
                        f"values in feature {col}.", 2
                    )

                elif self.strat_num.lower() == "knn":
                    self.log(
                        f" --> Imputing {nans} missing values using "
                        f"the KNN imputer in feature {col}.", 2
                    )
                    X[col] = self._imputers[col].transform(values)

                else:  # Strategies: mean, median or most_frequent.
                    self.log(
                        f" --> Imputing {nans} missing values with "
                        f"{self.strat_num.lower()} in feature {col}.", 2
                    )
                    X[col] = self._imputers[col].transform(values)

            # Column is categorical and contains missing values
            elif nans > 0:
                if self.strat_cat.lower() not in ["drop", "most_frequent"]:
                    self.log(
                        f" --> Imputing {nans} missing values with "
                        f"{self.strat_cat} in feature {col}.", 2
                    )
                    X[col].replace(np.NaN, self.strat_cat, inplace=True)

                elif self.strat_cat.lower() == "drop":
                    X.dropna(subset=[col], axis=0, inplace=True)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self.log(
                        f" --> Dropping {nans} rows due to missing "
                        f"values in feature {col}.", 2
                    )

                elif self.strat_cat.lower() == "most_frequent":
                    self.log(
                        f" --> Imputing {nans} missing values with "
                        f"most_frequent in feature {col}.", 2
                    )
                    X[col] = self._imputers[col].transform(values)

        return variable_return(X, y)


class Encoder(BaseEstimator, BaseTransformer, BaseCleaner):
    """Perform encoding of categorical features.

    The encoding type depends on the number of unique values in the column:
        - If n_unique=2, use Label-encoding.
        - If 2 < n_unique <= max_onehot, use OneHot-encoding.
        - If n_unique > max_onehot, use `strategy`-encoding.

    Also replaces classes with low occurrences with the value `other` in order to
    prevent too high cardinality. Categorical features are defined as all columns
    whose dtype.kind not in `ifu`. Will raise an error if it encounters missing
    values or unknown classes when transforming.

    Parameters
    ----------
    strategy: str, optional (default="LeaveOneOut")
        Type of encoding to use for high cardinality features. Choose from one of
        the estimators available in the category-encoders package except for:
            - OneHotEncoder: Use the `max_onehot` parameter.
            - HashingEncoder: Incompatibility of APIs.

    max_onehot: int or None, optional (default=10)
        Maximum number of unique values in a feature to perform one-hot-encoding.
        If None, it will always use `strategy` when n_unique > 2.

    frac_to_other: float, optional (default=None)
        Classes with less occurrences than n_rows * fraction_to_other are
        replaced with the string `other`. If None, this skip this step.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If str: name of the logging file. "auto" for default name.
        - If class: python `Logger` object.

    **kwargs
        Additional keyword arguments passed to the `strategy` estimator.

    """

    def __init__(
        self,
        strategy: str = "LeaveOneOut",
        max_onehot: Optional[int] = 10,
        frac_to_other: Optional[float] = None,
        verbose: int = 0,
        logger: Optional[Union[bool, str, callable]] = None,
        **kwargs,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strategy = strategy
        self.max_onehot = max_onehot
        self.frac_to_other = frac_to_other
        self.kwargs = kwargs

        self._to_other = {}
        self._encoders = {}
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Y_TYPES):
        """Fit the individual encoders on each column.

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, list, tuple,  np.array or pd.Series
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        self: Encoder

        """
        X, y = self._prepare_input(X, y)

        # Check Parameters
        if self.strategy.lower().endswith("encoder"):
            self.strategy = self.strategy[:-7]  # Remove the Encoder at the end
        if self.strategy.lower() not in ENCODER_TYPES:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                f"Choose from: {', '.join(ENCODER_TYPES)}."
            )
        strategy = ENCODER_TYPES[self.strategy.lower()]

        if self.max_onehot is None:
            self.max_onehot = 0
        elif self.max_onehot < 0:  # if 0, 1 or 2: it never uses one-hot encoding
            raise ValueError(
                "Invalid value for the max_onehot parameter."
                f"Value should be >= 0, got {self.max_onehot}."
            )
        if self.frac_to_other:
            if self.frac_to_other <= 0 or self.frac_to_other >= 1:
                raise ValueError(
                    "Invalid value for the frac_to_other parameter. Value "
                    f"should be between 0 and 1, got {self.frac_to_other}."
                )

        self.log("Fitting Encoder...", 1)

        for col in X:
            self._to_other[col] = []
            if X[col].dtype.kind not in "ifu":  # If column is categorical
                # Group uncommon classes into "other"
                if self.frac_to_other:
                    for category, count in X[col].value_counts().items():
                        if count < self.frac_to_other * len(X[col]):
                            self._to_other[col].append(category)
                            X[col].replace(category, "other", inplace=True)

                # Count number of unique values in the column
                n_unique = len(X[col].unique())

                # Convert series to dataframe for ingestion of package
                values = pd.DataFrame(X[col])

                # Perform encoding type dependent on number of unique values
                if n_unique == 2:
                    self._encoders[col] = LabelEncoder().fit(values)

                elif 2 < n_unique <= self.max_onehot:
                    self._encoders[col] = OneHotEncoder(
                        handle_missing="error",
                        handle_unknown="error",
                        use_cat_names=True,
                    ).fit(values)

                else:
                    self._encoders[col] = strategy(
                        handle_missing="error", handle_unknown="error", **self.kwargs
                    ).fit(values, y)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the encoding transformations.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Encoded dataframe.

        """
        check_is_fitted(self, "_is_fitted")
        X, y = self._prepare_input(X, y)

        self.log("Encoding categorical columns...", 1)

        for col in X:
            if X[col].dtype.kind not in "ifu":  # If column is categorical
                # Convert classes to "other"
                for category in self._to_other[col]:
                    X[col].replace(category, "other", inplace=True)

                # Count number of unique values in the column
                n_unique = len(X[col].unique())

                # Convert series to dataframe for ingestion of package
                values = pd.DataFrame(X[col])

                # Get index of the column
                idx = X.columns.get_loc(col)

                self.log(
                    f" --> {self._encoders[col].__class__.__name__[:-7]}-encoding"
                    f" feature {col}. Contains {n_unique} unique classes.", 2
                )
                # Perform encoding type dependent on number of unique values
                if self._encoders[col].__class__.__name__[:-7] == "Label":
                    X[col] = self._encoders[col].transform(values)

                elif self._encoders[col].__class__.__name__[:-7] == "OneHot":
                    onehot_cols = self._encoders[col].transform(values)
                    # Insert the new columns at old location
                    for i, column in enumerate(onehot_cols):
                        X.insert(idx + i, column, onehot_cols[column])
                    # Drop the original and _nan columns
                    X = X.drop([col, onehot_cols.columns[-1]], axis=1)

                else:
                    rest_cols = self._encoders[col].transform(values)
                    X = X.drop(col, axis=1)  # Drop the original column
                    # Insert the new columns at old location
                    for i, column in enumerate(rest_cols):
                        X.insert(idx + i, column, rest_cols[column])

        return X


class Outliers(BaseEstimator, BaseTransformer, BaseCleaner):
    """Remove or replace outliers in the data.

    Outliers are values that lie further than `max_sigma` * standard_deviation
    away from the mean of the column. Ignores categorical columns.

    Parameters
    ----------
    strategy: int, float or str, optional (default="drop")
        Strategy to apply on the outliers. Choose from:
            - "drop": Drop any row with outliers.
            - "min_max": Replace the outlier with the min or max of the column.
            - Any numerical value with which to replace the outliers.

    max_sigma: int or float, optional (default=3)
        Maximum allowed standard deviations from the mean of the column.
        If more, it is considered an outlier.

    include_target: bool, optional (default=False)
        Whether to include the target column in the transformation. This can
        be useful for regression tasks.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If str: name of the logging file. "auto" for default name.
        - If class: python `Logger` object.

    """

    def __init__(
        self,
        strategy: Union[int, float, str] = "drop",
        max_sigma: Union[int, float] = 3,
        include_target: bool = False,
        verbose: int = 0,
        logger: Optional[Union[bool, str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strategy = strategy
        self.max_sigma = max_sigma
        self.include_target = include_target

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the transformations on the data.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series
            - If None: y is ignored in the transformation.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        X: pd.DataFrame
            Imputed dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        X, y = self._prepare_input(X, y)

        # Check Parameters
        if isinstance(self.strategy, str):
            if self.strategy.lower() not in ["drop", "min_max"]:
                raise ValueError(
                    "Invalid value for the strategy parameter."
                    f"Choose from: 'drop', 'min_max'."
                )
        if self.max_sigma <= 0:
            raise ValueError(
                "Invalid value for the max_sigma parameter."
                f"Value should be > 0, got {self.max_sigma}."
            )

        self.log("Handling outliers...", 1)

        # Prepare dataset (merge with y and exclude categorical columns)
        objective = merge(X, y) if self.include_target and y is not None else X
        objective = objective.select_dtypes(exclude=["category", "object"])

        # Get z-scores
        z_scores = zscore(objective, nan_policy="propagate")

        if not isinstance(self.strategy, str):
            cond = np.abs(z_scores) > self.max_sigma
            objective.mask(cond, self.strategy, inplace=True)
            if cond.sum() > 0:
                self.log(
                    f" --> Replacing {cond.sum()} outliers with "
                    f"value {self.strategy}.", 2
                )

        elif self.strategy.lower() == "min_max":
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
                self.log(
                    f" --> Replacing {counts} outliers with the min "
                    "or max of the column.", 2
                )

        elif self.strategy.lower() == "drop":
            ix = (np.abs(zscore(z_scores)) <= self.max_sigma).all(axis=1)
            delete = len(ix) - ix.sum()  # Number of False values in index
            if delete > 0:
                self.log(f" --> Dropping {delete} rows due to outliers.", 2)

            # Drop rows based on index
            objective = objective[ix]
            X = X[ix]
            if y is not None:
                y = y[ix]

        # Replace the numerical columns in X with the new values from objective
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


class Balancer(BaseEstimator, BaseTransformer, BaseCleaner):
    """Balance the number of rows per target category.

    Use only for classification tasks.

    Parameters
    ----------
    strategy: str, optional (default="ADASYN")
        Type of algorithm to use for oversampling or undersampling. Choose from one
        of the estimators available in the imbalanced-learn package.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - value.

        Beware that using multiple processes on the same machine may
        cause memory issues for large datasets.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If str: name of the logging file. "auto" for default name.
        - If class: python `Logger` object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` instance used by `numpy.random`.

    **kwargs
        Additional keyword arguments passed to the `strategy` estimator.

    """

    def __init__(
        self,
        strategy: str = "ADASYN",
        n_jobs: int = 1,
        verbose: int = 0,
        logger: Optional[Union[bool, str, callable]] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            n_jobs=n_jobs, verbose=verbose, logger=logger, random_state=random_state
        )
        self.strategy = strategy
        self.kwargs = kwargs

        self.mapping = {}
        self._cols = None

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Y_TYPES = -1):
        """Apply the transformations on the data.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        X: pd.DataFrame
            Balanced dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        X, y = self._prepare_input(X, y)

        # Check Parameters
        if self.strategy.lower() not in BALANCER_TYPES:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                f"Choose from: {', '.join(BALANCER_TYPES)}."
            )
        strategy = BALANCER_TYPES[self.strategy.lower()]

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

        if "over_sampling" in strategy.__module__:
            self.log(f"Oversampling with {strategy.__name__}...", 1)
        else:
            self.log(f"Undersampling with {strategy.__name__}...", 1)
        estimator = strategy(**self.kwargs)

        # Add n_jobs or random_state if its one of the balancer's parameters
        for param in ["n_jobs", "random_state"]:
            if param in estimator.get_params():
                estimator.set_params(**{param: getattr(self, param)})

        X, y = estimator.fit_resample(X, y)

        # Add the estimator as attribute to Balancer
        setattr(self, strategy.__name__.lower(), estimator)

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

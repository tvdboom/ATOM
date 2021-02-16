# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the data cleaning estimators.

"""

# Standard packages
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Union, Optional
from scipy.stats import zscore
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from category_encoders.one_hot import OneHotEncoder

# Own modules
from .basetransformer import BaseTransformer
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, SCALING_STRATS, ENCODING_STRATS,
    PRUNING_STRATS, BALANCING_STRATS, it, variable_return, to_df,
    to_series, merge, check_is_fitted, composed, crash, method_to_log,
)


class TransformerMixin:
    """Mixin class for all transformers in ATOM.

    Different from sklearn, since it accounts for the transformation
    of y and a possible absence of the fit method.

    """

    @composed(crash, method_to_log, typechecked)
    def fit_transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None, **fit_params):
        """Fit to data, then transform it.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
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
        X: pd.DataFrame
            Transformed dataframe.

        y: pd.Series
            Target column corresponding to X. Only returned if provided.

        """
        try:
            return self.fit(X, y, **fit_params).transform(X, y)
        except AttributeError:
            return self.transform(X, y)


class Scaler(BaseEstimator, TransformerMixin, BaseTransformer):
    """Scale the data.

    This class applies one of sklearn's scalers. Additionally, it
    returns a dataframe when provided and it ignores non-numerical
    columns (instead of raising an exception).

    Parameters
    ----------
    strategy: str, optional (default="standard")
        Scaler object with which to scale the data. Options are:
            - standard: Scale with StandardScaler.
            - minmax: Scale with MinMaxScaler.
            - maxabs: Scale with MaxAbsScaler.
            - robust: Scale with RobustScaler.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. Use "auto" for default name.
        - If class: Python `Logger` object.

        The default name consists of the class' name followed by the
        timestamp of the logger's creation.

    Attributes
    ----------
    scaler: class
        Instance with which the data is scaled.

    """

    @typechecked
    def __init__(
            self,
            strategy: str = "standard",
            verbose: int = 0,
            logger: Optional[Union[str, callable]] = None
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strategy = strategy
        self.scaler = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit to data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        self: Scaler

        """
        X, y = self._prepare_input(X, y)

        if self.strategy.lower() in SCALING_STRATS:
            self.scaler = SCALING_STRATS[self.strategy.lower()]()
        else:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                "Available options are: standard, minmax, maxabs, robust."
            )

        self.scaler.fit(X.select_dtypes(include="number"))
        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Scaled dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Scaling features...", 1)
        X_numerical = X.select_dtypes(include=["int64", "float64"])
        X_transformed = self.scaler.transform(X_numerical)

        # Replace the numerical columns with the transformed values
        for i, col in enumerate(X_numerical):
            X[col] = X_transformed[:, i]

        return X


class Cleaner(BaseEstimator, TransformerMixin, BaseTransformer):
    """Applies standard data cleaning steps on a dataset.

    Use the parameters to choose which transformations to perform.
    The available steps are:
        - Remove columns with prohibited data types.
        - Remove categorical columns with maximal cardinality.
        - Remove columns with minimum cardinality.
        - Strip categorical features from white spaces.
        - Drop duplicate rows.
        - Drop rows with missing values in the target column.
        - Encode the target column.

    Parameters
    ----------
    prohibited_types: str, sequence or None, optional (default=None)
        Columns with these types will be removed from the dataset.

    maximum_cardinality: bool, optional (default=True)
        Whether to remove categorical columns with maximum cardinality,
        i.e. the number of unique values is equal to the number of
        instances. Usually the case for names, IDs, etc...

    minimum_cardinality: bool, optional (default=True)
        Whether to remove columns with minimum cardinality, i.e. all
        values in the column are the same.

    strip_categorical: bool, optional (default=True)
        Whether to strip spaces from the categorical columns.

    drop_duplicates: bool, optional (default=False)
        Whether to drop duplicate rows. Only the first occurrence of
        every duplicated row is kept.

    missing_target: bool, optional (default=True)
        Whether to drop rows with missing values in the target column.
        Is ignored if y is not provided.

    encode_target: bool, optional (default=True)
        Whether to Label-encode the target column. Is ignored if y is
        not provided.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. Use "auto" for default name.
        - If class: Python `Logger` object.

        The default name consists of the class' name followed by the
        timestamp of the logger's creation.

    Attributes
    ----------
    missing: list
        List of values that are considered "missing". Default values
        are: "", "?", "None", "NA", "nan", "NaN" and "inf". Note that
        `None`, `NaN`, `+inf` and `-inf` are always considered missing
        since they are incompatible with sklearn estimators.

    mapping: dict
        Dictionary of the target values mapped to their respective
        encoded integer. Only available if encode_target=True.

    """

    @typechecked
    def __init__(
        self,
        prohibited_types: Optional[Union[str, SEQUENCE_TYPES]] = None,
        maximum_cardinality: bool = True,
        minimum_cardinality: bool = True,
        strip_categorical: bool = True,
        drop_duplicates: bool = False,
        missing_target: bool = True,
        encode_target: bool = True,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.prohibited_types = prohibited_types
        self.maximum_cardinality = maximum_cardinality
        self.minimum_cardinality = minimum_cardinality
        self.strip_categorical = strip_categorical
        self.drop_duplicates = drop_duplicates
        self.missing_target = missing_target
        self.encode_target = encode_target

        self.mapping = {}
        self.missing = ["", "?", "NA", "nan", "NaN", "None", "inf"]

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the data cleaning steps to the data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        X: pd.DataFrame
            Transformed dataframe.

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
                X = X.drop(col, axis=1)
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
                    X = X.drop(col, axis=1)

            # Drop features with minimum cardinality (all values are the same)
            if n_unique == 1 and self.minimum_cardinality:
                self.log(
                    f" --> Dropping feature {col} due to minimum "
                    f"cardinality. Contains only 1 class: {unique[0]}.", 2
                )
                X = X.drop(col, axis=1)

        # Drop duplicate samples
        if self.drop_duplicates:
            X = X.drop_duplicates(ignore_index=True)

        if y is not None:
            # Delete samples with NaN in target
            if self.missing_target:
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
                y = to_series(encoder.fit_transform(y), index=y.index, name=y.name)
                self.mapping = {str(it(v)): i for i, v in enumerate(encoder.classes_)}

                # Only print if the target column wasn't already encoded
                if any([key != str(value) for key, value in self.mapping.items()]):
                    self.log(f" --> Label-encoding the target column.", 2)

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

    min_frac_rows: float, optional (default=0.5)
        Minimum fraction of non-missing values in a row. If less,
        the row is removed.

    min_frac_cols: float, optional (default=0.5)
        Minimum fraction of non-missing values in a column. If less,
        the column is removed.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. Use "auto" for default name.
        - If class: Python `Logger` object.

        The default name consists of the class' name followed by the
        timestamp of the logger's creation.

    Attributes
    ----------
    missing: list
        List of values that are considered "missing". Default values
        are: "", "?", "None", "NA", "nan", "NaN" and "inf". Note that
        `None`, `NaN`, `+inf` and `-inf` are always considered missing
        since they are incompatible with sklearn estimators.

    """

    @typechecked
    def __init__(
        self,
        strat_num: Union[int, float, str] = "drop",
        strat_cat: str = "drop",
        min_frac_rows: float = 0.5,
        min_frac_cols: float = 0.5,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strat_num = strat_num
        self.strat_cat = strat_cat
        self.min_frac_rows = min_frac_rows
        self.min_frac_cols = min_frac_cols

        self.missing = ["", "?", "None", "NA", "nan", "NaN", "inf"]
        self._imputers = {}
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit to data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
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

        self.log("Fitting Imputer...", 1)

        # Replace all missing values with NaN
        X = X.replace(self.missing + [np.inf, -np.inf], np.NaN)

        # Drop rows with too many NaN values
        X = X.dropna(axis=0, thresh=int(self.min_frac_rows * X.shape[1]))

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
        """Impute the missing values.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence
            - If None: y is ignored.
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
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Imputing missing values...", 1)

        # Replace missing values with NaN
        X = X.replace(self.missing + [np.inf, -np.inf], np.NaN)

        # Drop samples with too many NaN values
        length = len(X)
        X = X.dropna(axis=0, thresh=int(self.min_frac_rows * X.shape[1]))
        if y is not None:
            y = y[y.index.isin(X.index)]  # Select only indices that remain
        diff = length - len(X)
        if diff > 0:
            self.log(
                f" --> Dropping {diff} samples for containing less than "
                f"{int(self.min_frac_rows*100)}% non-missing values.", 2
            )

        # Loop over all columns to apply strategy dependent on type
        for col in X:
            values = X[col].values.reshape(-1, 1)

            # Drop columns with too many NaN values
            nans = X[col].isna().sum()  # Number of missing values in column
            p_nans = nans * 100 // len(X)  # Percentage of NaNs
            if (len(X) - nans) / len(X) < self.min_frac_cols:
                self.log(
                    f" --> Dropping feature {col} for containing "
                    f"{nans} ({p_nans}%) missing values.", 2
                )
                X = X.drop(col, axis=1)
                continue  # Skip to side column

            # Column is numerical and contains missing values
            if X[col].dtype.kind in "ifu" and nans > 0:
                if not isinstance(self.strat_num, str):
                    self.log(
                        f" --> Imputing {nans} missing values with number "
                        f"{str(self.strat_num)} in feature {col}.", 2
                    )
                    X[col] = X[col].replace(np.NaN, self.strat_num)

                elif self.strat_num.lower() == "drop":
                    X = X.dropna(subset=[col], axis=0)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self.log(
                        f" --> Dropping {nans} samples due to missing "
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
                    X[col] = X[col].replace(np.NaN, self.strat_cat)

                elif self.strat_cat.lower() == "drop":
                    X = X.dropna(subset=[col], axis=0)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self.log(
                        f" --> Dropping {nans} samples due to missing "
                        f"values in feature {col}.", 2
                    )

                elif self.strat_cat.lower() == "most_frequent":
                    self.log(
                        f" --> Imputing {nans} missing values with "
                        f"most_frequent in feature {col}.", 2
                    )
                    X[col] = self._imputers[col].transform(values)

        return variable_return(X, y)


class Encoder(BaseEstimator, TransformerMixin, BaseTransformer):
    """Perform encoding of categorical features.

    The encoding type depends on the number of classes in the column:
        - If n_unique=2, use Label-encoding.
        - If 2 < n_unique <= max_onehot, use OneHot-encoding.
        - If n_unique > max_onehot, use `strategy`-encoding.

    Also replaces classes with low occurrences with the value `other`
    in order to prevent too high cardinality. Categorical features are
    defined as all columns whose dtype.kind not in `ifu`. Will raise
    an error if it encounters missing values or unknown classes when
    transforming.

    Parameters
    ----------
    strategy: str, optional (default="LeaveOneOut")
        Type of encoding to use for high cardinality features. Choose
        from one of the estimators available in the category-encoders
        package except for:
            - OneHotEncoder: Use the `max_onehot` parameter.
            - HashingEncoder: Incompatibility of APIs.

    max_onehot: int or None, optional (default=10)
        Maximum number of unique values in a feature to perform
        one-hot-encoding. If None, it will always use `strategy`
        when n_unique > 2.

    frac_to_other: float, optional (default=None)
        Classes with less occurrences than n_rows * fraction_to_other
        are replaced with the string `other`. If None, this skip this
        step.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. Use "auto" for default name.
        - If class: Python `Logger` object.

        The default name consists of the class' name followed by the
        timestamp of the logger's creation.

    **kwargs
        Additional keyword arguments passed to the `strategy` estimator.

    """

    @typechecked
    def __init__(
        self,
        strategy: str = "LeaveOneOut",
        max_onehot: Optional[int] = 10,
        frac_to_other: Optional[float] = None,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
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
        """Fit to data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence
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
        if self.strategy.lower() not in ENCODING_STRATS:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                f"Choose from: {', '.join(ENCODING_STRATS)}."
            )
        strategy = ENCODING_STRATS[self.strategy.lower()]

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
                            X[col] = X[col].replace(category, "other")

                # Count number of unique values in the column
                n_unique = len(X[col].unique())

                # Perform encoding type dependent on number of unique values
                if n_unique == 2:
                    self._encoders[col] = LabelEncoder().fit(X[col])

                elif 2 < n_unique <= self.max_onehot:
                    self._encoders[col] = OneHotEncoder(
                        handle_missing="error",
                        handle_unknown="error",
                        use_cat_names=True,
                    ).fit(pd.DataFrame(X[col]))

                else:
                    self._encoders[col] = strategy(
                        handle_missing="error", handle_unknown="error", **self.kwargs
                    ).fit(pd.DataFrame(X[col]), y)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the transformations on the data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Encoded dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Encoding categorical columns...", 1)

        for idx, col in enumerate(X):
            if X[col].dtype.kind not in "ifu":  # If column is categorical
                # Convert classes to "other"
                X[col] = X[col].replace(self._to_other[col], "other")

                self.log(
                    f" --> {self._encoders[col].__class__.__name__[:-7]}-encoding "
                    f"feature {col}. Contains {len(X[col].unique())} unique classes.", 2
                )

                # Perform encoding type dependent on number of unique values
                if self._encoders[col].__class__.__name__[:-7] == "Label":
                    X[col] = self._encoders[col].transform(X[col])

                elif self._encoders[col].__class__.__name__[:-7] == "OneHot":
                    onehot_cols = self._encoders[col].transform(pd.DataFrame(X[col]))
                    # Insert the new columns at old location
                    for i, column in enumerate(onehot_cols):
                        X.insert(idx + i, column, onehot_cols[column])
                    # Drop the original and _nan columns
                    X = X.drop([col, onehot_cols.columns[-1]], axis=1)

                else:
                    rest_cols = self._encoders[col].transform(pd.DataFrame(X[col]))
                    X = X.drop(col, axis=1)  # Drop the original column
                    # Insert the new columns at old location
                    for i, column in enumerate(rest_cols):
                        X.insert(idx + i, column, rest_cols[column])

        return X


class Pruner(BaseEstimator, TransformerMixin, BaseTransformer):
    """Prune outliers from the data.

    Replace or remove outliers. The definition of outlier depends
    on the selected strategy and can greatly differ from one each
    other. Ignores categorical columns.

    Parameters
    ----------
    strategy: str, optional (default="z-score")
        Strategy with which to select the outliers. Choose from:
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
        Whether to include the target column in the transformation.
        This can be useful for regression tasks.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. Use "auto" for default name.
        - If class: Python `Logger` object.

        The default name consists of the class' name followed by the
        timestamp of the logger's creation.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    """

    @typechecked
    def __init__(
        self,
        strategy: str = "z-score",
        method: Union[int, float, str] = "drop",
        max_sigma: Union[int, float] = 3,
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

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the outlier strategy on the data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        X: pd.DataFrame
            Transformed dataframe.

        y: pd.Series
            Target column corresponding to X.

        """
        X, y = self._prepare_input(X, y)

        # Check Parameters
        if self.strategy.lower() not in ["z-score"] + list(PRUNING_STRATS):
            raise ValueError(
                "Invalid value for the strategy parameter. Choose from: "
                f"z-score, {', '.join(PRUNING_STRATS)}."
            )
        if isinstance(self.method, str):
            if self.method.lower() not in ["drop", "min_max"]:
                raise ValueError(
                    "Invalid value for the method parameter."
                    f"Choose from: 'drop', 'min_max'."
                )
        if self.strategy.lower() != "z-score" and str(self.method).lower() != "drop":
            raise ValueError(
                "Invalid value for the method parameter. Only the z-score "
                f"strategy accepts another method than 'drop', got {self.method}."
            )
        if self.max_sigma <= 0:
            raise ValueError(
                "Invalid value for the max_sigma parameter."
                f"Value should be > 0, got {self.max_sigma}."
            )

        self.log("Pruning outliers...", 1)

        # Prepare dataset (merge with y and exclude categorical columns)
        objective = merge(X, y) if self.include_target and y is not None else X
        objective = objective.select_dtypes(exclude=["category", "object"])

        if self.strategy.lower() == "z-score":
            z_scores = zscore(objective, nan_policy="propagate")

            if not isinstance(self.method, str):
                cond = np.abs(z_scores) > self.max_sigma
                objective = objective.mask(cond, self.method)
                self.log(
                    f" --> Replacing {cond.sum()} outlier values with {self.method}.", 2
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
                ix = (np.abs(zscore(z_scores)) <= self.max_sigma).all(axis=1)
                delete = len(ix) - ix.sum()  # Number of False values in index
                self.log(f" --> Dropping {delete} samples due to outlier values.", 2)

                # Drop rows based on index
                X, objective = X[ix], objective[ix]
                if y is not None:
                    y = y[ix]

        else:
            estimator = PRUNING_STRATS[self.strategy.lower()](**self.kwargs)
            mask = estimator.fit_predict(X) != -1
            self.log(f" --> Dropping {len(mask) - mask.sum()} outliers.", 2)

            # Add the estimator as attribute to Balancer
            setattr(self, self.strategy.lower(), estimator)

            X, objective = X.loc[mask, :], objective.loc[mask, :]
            if y is not None:
                y = y[mask]

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


class Balancer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Balance the number of samples per class in the target column.

    Use only for classification tasks.

    Parameters
    ----------
    strategy: str, optional (default="ADASYN")
        Type of algorithm to use for oversampling or undersampling.
        Choose from one of the estimators in the imbalanced-learn
        package.

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

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. Use "auto" for default name.
        - If class: Python `Logger` object.

        The default name consists of the class' name followed by the
        timestamp of the logger's creation.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `numpy.random`.

    **kwargs
        Additional keyword arguments passed to the `strategy` estimator.

    Attributes
    ----------
    <estimator_name>: class
        Estimator instance (attribute name in all lowercase) used to
        oversample/undersample the data, e.g. `balancer.adasyn` for the
        default option.

    mapping: dict
        Dictionary of the target values mapped to their respective
        encoded integer.

    """

    @typechecked
    def __init__(
        self,
        strategy: str = "ADASYN",
        n_jobs: int = 1,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
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
        """Oversample or undersample the data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence, optional (default=-1)
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

        if self.strategy.lower() not in BALANCING_STRATS:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                f"Choose from: {', '.join(BALANCING_STRATS)}."
            )
        estimator = BALANCING_STRATS[self.strategy.lower()](**self.kwargs)

        # Save index and columns to re-convert to pandas
        index = X.index
        columns = X.columns
        name = y.name

        # Create dict of class counts in y
        counts = {}
        if not self.mapping:
            self.mapping = {str(i): v for i, v in enumerate(y.unique())}
        for key, value in self.mapping.items():
            counts[key] = np.sum(y == value)

        if "over_sampling" in estimator.__module__:
            self.log(f"Oversampling with {estimator.__class__.__name__}...", 1)
        else:
            self.log(f"Undersampling with {estimator.__class__.__name__}...", 1)

        # Add n_jobs or random_state if its one of the balancer's parameters
        for param in ["n_jobs", "random_state"]:
            if param in estimator.get_params():
                estimator.set_params(**{param: getattr(self, param)})

        X, y = estimator.fit_resample(X, y)

        # Add the estimator as attribute to Balancer
        setattr(self, self.strategy.lower(), estimator)

        # Print changes
        for key, value in self.mapping.items():
            diff = counts[key] - np.sum(y == value)
            if diff > 0:
                self.log(f" --> Removing {diff} samples from class: {key}.", 2)
            elif diff < 0:
                # Add new indices to the total index
                index = list(index) + list(np.arange(len(index) + diff))
                self.log(f" --> Adding {-diff} samples to class: {key}.", 2)

        X = to_df(X, index=index, columns=columns)
        y = to_series(y, index=index, name=name)
        return X, y

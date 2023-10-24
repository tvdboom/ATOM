# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the data cleaning transformers.

"""

from __future__ import annotations

import re
from collections import defaultdict
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from beartype.typing import Any, Literal
from category_encoders import (
    BackwardDifferenceEncoder, BaseNEncoder, BinaryEncoder, CatBoostEncoder,
    HelmertEncoder, JamesSteinEncoder, MEstimateEncoder, OneHotEncoder,
    OrdinalEncoder, PolynomialEncoder, SumEncoder, TargetEncoder, WOEEncoder,
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN, SMOTE, SMOTEN, SMOTENC, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    AllKNN, CondensedNearestNeighbour, EditedNearestNeighbours,
    InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule,
    OneSidedSelection, RandomUnderSampler, RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from scipy.stats import zscore
from sklearn.base import BaseEstimator
from sklearn.impute import KNNImputer

from atom.basetransformer import BaseTransformer
from atom.utils.types import (
    Bins, Bool, CategoricalStrats, DataFrame, DataFrameTypes,
    DiscretizerStrats, Engine, Estimator, FloatLargerZero, Int, IntLargerTwo,
    NJobs, NormalizerStrats, NumericalStrats, Pandas, PrunerStrats, Scalar,
    ScalerStrats, Sequence, SequenceTypes, Series, SeriesTypes, Transformer,
    Verbose, XSelector, YSelector,
)
from atom.utils.utils import (
    bk, check_is_fitted, composed, crash, get_cols, it, lst, merge,
    method_to_log, n_cols, replace_missing, sign, to_df, to_series,
    variable_return,
)


class TransformerMixin:
    """Mixin class for all transformers in ATOM.

    Different from sklearn, since it accounts for the transformation
    of y and a possible absence of the fit method.

    """

    def fit(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        **fit_params,
    ):
        """Does nothing.

        Implemented for continuity of the API.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, sequence, dataframe-like or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe-like: Target columns with shape=(n_samples,
              n_targets) for multioutput tasks.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        self
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        self._log(f"Fitting {self.__class__.__name__}...", 1)

        return self

    @composed(crash, method_to_log)
    def fit_transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        **fit_params,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Fit to data, then transform it.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, sequence, dataframe-like or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe-like: Target columns with shape=(n_samples,
              n_targets) for multioutput tasks.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        return self.fit(X, y, **fit_params).transform(X, y)

    @composed(crash, method_to_log)
    def inverse_transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Does nothing.

        Returns the input unchanged. Implemented for continuity of the
        API.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, sequence, dataframe-like or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe-like: Target columns with shape=(n_samples,
              n_targets) for multioutput tasks.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        return variable_return(X, y)


class Balancer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Balance the number of samples per class in the target column.

    When oversampling, the newly created samples have an increasing
    integer index for numerical indices, and an index of the form
    [estimator]_N for non-numerical indices, where N stands for the
    N-th sample in the data set. Use only for classification tasks.

    This class can be accessed from atom through the [balance]
    [atomclassifier-balance] method. Read more in the [user guide]
    [balancing-the-data].

    !!! warning
         * The [clustercentroids][] estimator is unavailable because of
           incompatibilities of the APIs.
         * The Balancer class does not support [multioutput tasks][].

    Parameters
    ----------
    strategy: str or estimator, default="ADASYN"
        Type of algorithm with which to balance the dataset. Choose
        from the name of any estimator in the imbalanced-learn package
        or provide a custom instance of such.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 - value.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]_: imblearn estimator
        Object (lowercase strategy) used to balance the data,
        e.g., `balancer.adasyn_` for the default strategy.

    mapping_: dict
        Target values mapped to their respective encoded integers.

    feature_names_in_: np.ndarray
        Names of features seen during fit.

    target_names_in_: np.ndarray
        Names of the target column seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.data_cleaning:Encoder
    atom.data_cleaning:Imputer
    atom.data_cleaning:Pruner

    Examples
    --------

    === "atom"
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.train)

        atom.balance(strategy="smote", verbose=2)

        # Note that the number of rows has increased
        print(atom.train)
        ```

    === "stand-alone"
        ```pycon
        from atom.data_cleaning import Balancer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        print(X)

        balancer = Balancer(strategy="smote", verbose=2)
        X, y = balancer.fit_transform(X, y)

        # Note that the number of rows has increased
        print(X)
        ```

    """

    _train_only = True

    def __init__(
        self,
        strategy: str | Estimator = "ADASYN",
        *,
        n_jobs: NJobs = 1,
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
        random_state: Int | None = None,
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

    @composed(crash, method_to_log)
    def fit(self, X: XSelector, y: YSelector = -1) -> Balancer:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict or sequence, default=-1
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        Returns
        -------
        Balancer
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        if isinstance(y, DataFrameTypes):
            raise ValueError("The Balancer class does not support multioutput tasks.")
        else:
            self.target_names_in_ = np.array([y.name])

        strategies = dict(
            # clustercentroids=ClusterCentroids,  # Has no sample_indices_
            condensednearestneighbour=CondensedNearestNeighbour,
            editednearestneighborus=EditedNearestNeighbours,
            repeatededitednearestneighbours=RepeatedEditedNearestNeighbours,
            allknn=AllKNN,
            instancehardnessthreshold=InstanceHardnessThreshold,
            nearmiss=NearMiss,
            neighbourhoodcleaningrule=NeighbourhoodCleaningRule,
            onesidedselection=OneSidedSelection,
            randomundersampler=RandomUnderSampler,
            tomeklinks=TomekLinks,
            randomoversampler=RandomOverSampler,
            smote=SMOTE,
            smotenc=SMOTENC,
            smoten=SMOTEN,
            adasyn=ADASYN,
            borderlinesmote=BorderlineSMOTE,
            kmeanssmote=KMeansSMOTE,
            svmsmote=SVMSMOTE,
            smoteenn=SMOTEENN,
            smotetomek=SMOTETomek,
        )

        if isinstance(self.strategy, str):
            if self.strategy.lower() not in strategies:
                raise ValueError(
                    f"Invalid value for the strategy parameter, got {self.strategy}. "
                    f"Choose from: {', '.join(strategies)}."
                )
            estimator = strategies[self.strategy.lower()](**self.kwargs)
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
        if not hasattr(self, "mapping_"):
            self.mapping_ = {str(v): v for v in y.sort_values().unique()}

        self._counts = {}
        for key, value in self.mapping_.items():
            self._counts[key] = np.sum(y == value)

        # Add n_jobs or random_state if its one of the estimator's parameters
        self._estimator = self._inherit(estimator).fit(X, y)

        # Add the estimator as attribute to the instance
        setattr(self, f"{estimator.__class__.__name__.lower()}_", self._estimator)

        return self

    @composed(crash, method_to_log)
    def transform(self, X: XSelector, y: YSelector = -1) -> tuple[DataFrame, Series]:
        """Balance the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence, default=-1
            Target column corresponding to X.

            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        dataframe
            Balanced dataframe.

        series
            Transformed target column.

        """

        def log_changes(y):
            """Print the changes per target class."""
            for key, value in self.mapping_.items():
                diff = self._counts[key] - np.sum(y == value)
                if diff > 0:
                    self._log(f" --> Removing {diff} samples from class {key}.", 2)
                elif diff < 0:
                    self._log(f" --> Adding {-diff} samples to class {key}.", 2)

        check_is_fitted(self)
        X, y = self._prepare_input(
            X=X,
            y=y,
            columns=getattr(self, "feature_names_in_", None),
            name=getattr(self, "target_names_in_", None),
        )

        if "over_sampling" in self._estimator.__module__:
            self._log(f"Oversampling with {self._estimator.__class__.__name__}...", 1)

            index = X.index  # Save indices for later reassignment
            X, y = self._estimator.fit_resample(X, y)

            # Create indices for the new samples
            if index.dtype.kind in "ifu":
                new_index = range(max(index) + 1, max(index) + len(X) - len(index) + 1)
            else:
                new_index = [
                    f"{self._estimator.__class__.__name__.lower()}_{i}"
                    for i in range(1, len(X) - len(index) + 1)
                ]

            # Assign the old + new indices
            X.index = list(index) + list(new_index)
            y.index = list(index) + list(new_index)

            log_changes(y)

        elif "under_sampling" in self._estimator.__module__:
            self._log(f"Undersampling with {self._estimator.__class__.__name__}...", 1)

            self._estimator.fit_resample(X, y)

            # Select chosen rows (imblearn doesn't return them in order)
            samples = sorted(self._estimator.sample_indices_)
            X, y = X.iloc[samples, :], y.iloc[samples]

            log_changes(y)

        elif "combine" in self._estimator.__module__:
            self._log(f"Balancing with {self._estimator.__class__.__name__}...", 1)

            index = X.index
            X_new, y_new = self._estimator.fit_resample(X, y)

            # Select rows that were kept by the undersampler
            if self._estimator.__class__.__name__ == "SMOTEENN":
                samples = sorted(self._estimator.enn_.sample_indices_)
            elif self._estimator.__class__.__name__ == "SMOTETomek":
                samples = sorted(self._estimator.tomek_.sample_indices_)

            # Select the remaining samples from the old dataframe
            old_samples = [s for s in samples if s < len(X)]
            X, y = X.iloc[old_samples, :], y.iloc[old_samples]

            # Create indices for the new samples
            if index.dtype.kind in "ifu":
                new_index = range(max(index) + 1, max(index) + len(X_new) - len(X) + 1)
            else:
                new_index = [
                    f"{self._estimator.__class__.__name__.lower()}_{i}"
                    for i in range(1, len(X_new) - len(X) + 1)
                ]

            # Select the new samples and assign the new indices
            X_new = X_new.iloc[-len(X_new) + len(old_samples):, :]
            X_new.index = new_index
            y_new = y_new.iloc[-len(y_new) + len(old_samples):]
            y_new.index = new_index

            # First, output the samples created
            for key, value in self.mapping_.items():
                diff = np.sum(y_new == value)
                if diff > 0:
                    self._log(f" --> Adding {diff} samples to class: {key}.", 2)

            # Then, output the samples dropped
            for key, value in self.mapping_.items():
                diff = self._counts[key] - np.sum(y == value)
                if diff > 0:
                    self._log(f" --> Removing {diff} samples from class: {key}.", 2)

            # Add the new samples to the old dataframe
            X, y = bk.concat([X, X_new]), bk.concat([y, y_new])

        return X, y


class Cleaner(BaseEstimator, TransformerMixin, BaseTransformer):
    """Applies standard data cleaning steps on a dataset.

    Use the parameters to choose which transformations to perform.
    The available steps are:

    - Convert dtypes to the best possible types.
    - Drop columns with specific data types.
    - Remove characters from column names.
    - Strip categorical features from spaces.
    - Drop duplicate rows.
    - Drop rows with missing values in the target column.
    - Encode the target column.

    This class can be accessed from atom through the [clean]
    [atomclassifier-clean] method. Read more in the [user guide]
    [standard-data-cleaning].

    Parameters
    ----------
    convert_dtypes: bool, default=True
        Convert the column's data types to the best possible types
        that support `pd.NA`.

    drop_dtypes: str, sequence or None, default=None
        Columns with these data types are dropped from the dataset.

    drop_chars: str or None, default=None
        Remove the specified regex pattern from column names, e.g.
        `[^A-Za-z0-9]+` to remove all non-alphanumerical characters.

    strip_categorical: bool, default=True
        Whether to strip spaces from categorical columns.

    drop_duplicates: bool, default=False
        Whether to drop duplicate rows. Only the first occurrence of
        every duplicated row is kept.

    drop_missing_target: bool, default=True
        Whether to drop rows with missing values in the target column.
        This transformation is ignored if `y` is not provided.

    encode_target: bool, default=True
        Whether to encode the target column(s). This includes
        converting categorical columns to numerical, and binarizing
        [multilabel][] columns. This transformation is ignored if `y`
        is not provided.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: dict, default={"data": "numpy", "estimator": "sklearn"}
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be a
        dictionary with keys `data` and/or `estimator`, with their
        corresponding choice as values. Choose from:

        - "data":

            - "numpy"
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn"
            - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    Attributes
    ----------
    missing_: list
        Values that are considered "missing". Default values are: "",
        "?", "NA", "nan", "NaN", "NaT", "none", "None", "inf", "-inf".
        Note that `None`, `NaN`, `+inf` and `-inf` are always considered
        missing since they are incompatible with sklearn estimators.

    mapping_: dict
        Target values mapped to their respective encoded integers. Only
        available if encode_target=True.

    feature_names_in_: np.ndarray
        Names of features seen during fit.

    target_names_in_: np.ndarray
        Names of the target column(s) seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.data_cleaning:Encoder
    atom.data_cleaning:Discretizer
    atom.data_cleaning:Scaler

    Examples
    --------

    === "atom"
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        y = ["a" if i else "b" for i in y]

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.y)

        atom.clean(verbose=2)

        print(atom.y)
        ```

    === "stand-alone"
        ```pycon
        from atom.data_cleaning import Cleaner
        from numpy.random import randint

        y = ["a" if i else "b" for i in range(randint(100))]

        cleaner = Cleaner(verbose=2)
        y = cleaner.fit_transform(y=y)

        print(y)
        ```

    """

    _train_only = False

    def __init__(
        self,
        *,
        convert_dtypes: Bool = True,
        drop_dtypes: str | Sequence[str] | None = None,
        drop_chars: str | None = None,
        strip_categorical: Bool = True,
        drop_duplicates: Bool = False,
        drop_missing_target: Bool = True,
        encode_target: Bool = True,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose, logger=logger)
        self.convert_dtypes = convert_dtypes
        self.drop_dtypes = drop_dtypes
        self.drop_chars = drop_chars
        self.strip_categorical = strip_categorical
        self.drop_duplicates = drop_duplicates
        self.drop_missing_target = drop_missing_target
        self.encode_target = encode_target

    @composed(crash, method_to_log)
    def fit(self, X: XSelector | None = None, y: YSelector | None = None) -> Cleaner:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, dict, sequence, dataframe-like or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        Returns
        -------
        Cleaner
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        self.mapping_ = {}
        self._estimators = {}

        if not hasattr(self, "missing_"):
            self.missing_ = [
                "", "?", "NA", "nan", "NaN", "NaT", "none", "None", "inf", "-inf"
            ]

        self._log("Fitting Cleaner...", 1)

        if y is not None:
            if isinstance(y, SeriesTypes):
                self.target_names_in_ = np.array([y.name])
            else:
                self.target_names_in_ = y.columns.values

            if self.drop_chars:
                if isinstance(y, SeriesTypes):
                    y.name = re.sub(self.drop_chars, "", y.name)
                else:
                    y = y.rename(columns=lambda x: re.sub(self.drop_chars, "", str(x)))

            if self.drop_missing_target:
                y = replace_missing(y, self.missing_).dropna(axis=0)

            if self.encode_target:
                for col in get_cols(y):
                    if isinstance(col.iloc[0], SequenceTypes):  # Multilabel
                        est = self._get_est_class("MultiLabelBinarizer", "preprocessing")
                        self._estimators[col.name] = est().fit(col)
                    elif list(uq := np.unique(col)) != list(range(col.nunique())):
                        est = self._get_est_class("LabelEncoder", "preprocessing")
                        self._estimators[col.name] = est().fit(col)
                        self.mapping_.update(
                            {col.name: {str(it(v)): i for i, v in enumerate(uq)}}
                        )

        return self

    @composed(crash, method_to_log)
    def transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Apply the data cleaning steps to the data.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, dict, sequence, dataframe-like or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series
            Transformed target column. Only returned if provided.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(
            X=X,
            y=y,
            columns=getattr(self, "feature_names_in_", None),
            name=getattr(self, "target_names_in_", None),
        )

        self._log("Cleaning the data...", 1)

        if X is not None:
            # Unify all missing values
            X = replace_missing(X, self.missing_)

            for name, column in X.items():
                dtype = column.dtype.name

                # Drop features with an invalid data type
                if dtype in lst(self.drop_dtypes):
                    self._log(
                        f" --> Dropping feature {name} for "
                        f"having a prohibited type: {dtype}.", 2
                    )
                    X = X.drop(name, axis=1)
                    continue

                elif dtype in ("object", "category", "string"):
                    if self.strip_categorical:
                        # Strip strings from blank spaces
                        X[name] = column.apply(
                            lambda val: val.strip() if isinstance(val, str) else val
                        )

            # Drop prohibited chars from column names
            if self.drop_chars:
                X = X.rename(columns=lambda x: re.sub(self.drop_chars, "", str(x)))

            # Drop duplicate samples
            if self.drop_duplicates:
                X = X.drop_duplicates(ignore_index=True)

            if self.convert_dtypes:
                X = X.convert_dtypes()

        if y is not None:
            if self.drop_chars:
                if isinstance(y, SeriesTypes):
                    y.name = re.sub(self.drop_chars, "", y.name)
                else:
                    y = y.rename(columns=lambda x: re.sub(self.drop_chars, "", str(x)))

            # Delete samples with missing values in target
            if self.drop_missing_target:
                length = len(y)  # Save original length to count deleted rows later
                y = replace_missing(y, self.missing_).dropna()

                if X is not None:
                    X = X[X.index.isin(y.index)]  # Select only indices that remain

                if (d := length - len(y)) > 0:
                    self._log(f" --> Dropping {d} rows with missing values in target.", 2)

            if self.encode_target and self._estimators:
                y_transformed = y.__class__(dtype="object")
                for col in get_cols(y):
                    if est := self._estimators.get(col.name):
                        if n_cols(out := est.transform(col)) == 1:
                            self._log(f" --> Label-encoding column {col.name}.", 2)
                            out = to_series(out, y.index, col.name)

                        else:
                            self._log(f" --> Label-binarizing column {col.name}.", 2)
                            out = to_df(
                                data=out,
                                index=y.index,
                                columns=[f"{col.name}_{c}" for c in est.classes_],
                            )

                        # Replace target with encoded column(s)
                        if isinstance(y, SeriesTypes):
                            y_transformed = out
                        else:
                            y_transformed = merge(y_transformed, out)

                    else:  # Add unchanged column
                        y_transformed = merge(y_transformed, col)

                y = y_transformed

            if self.convert_dtypes:
                y = y.convert_dtypes()

        return variable_return(X, y)

    @composed(crash, method_to_log)
    def inverse_transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Inversely transform the label encoding.

        This method only inversely transforms the target encoding.
        The rest of the transformations can't be inverted. If
        `encode_target=False`, the data is returned as is.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        y: int, str, dict, sequence, dataframe-like or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        Returns
        -------
        dataframe
            Unchanged feature set. Only returned if provided.

        series or dataframe
            Original target column. Only returned if provided.

        """
        X, y = self._prepare_input(X, y, columns=getattr(self, "feature_names_in_", None))

        self._log("Inversely cleaning the data...", 1)

        if y is not None and self._estimators:
            y_transformed = y.__class__(dtype="object")
            for col in self.target_names_in_:
                if est := self._estimators.get(col):
                    if est.__class__.__name__ == "LabelEncoder":
                        self._log(f" --> Inversely label-encoding column {col}.", 2)
                        out = est.inverse_transform(bk.DataFrame(y)[col])

                    else:
                        self._log(f" --> Inversely label-binarizing column {col}.", 2)
                        out = est.inverse_transform(
                            y.iloc[:, y.columns.str.startswith(f"{col}_")].values
                        )

                    # Replace encoded columns with target column
                    if isinstance(y, SeriesTypes):
                        y_transformed = to_series(out, y.index, col)
                    else:
                        y_transformed = merge(y_transformed, to_series(out, y.index, col))

                else:  # Add unchanged column
                    y_transformed = merge(y_transformed, bk.DataFrame(y)[col])

            y = y_transformed

        return variable_return(X, y)


class Discretizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Bin continuous data into intervals.

    For each feature, the bin edges are computed during fit and,
    together with the number of bins, they define the intervals.
    Ignores categorical columns.

    This class can be accessed from atom through the [discretize]
    [atomclassifier-discretize] method. Read more in the [user guide]
    [binning-numerical-features].

    !!! tip
        The transformation returns categorical columns. Use the
        [Encoder][] class to convert them back to numerical types.

    Parameters
    ----------
    strategy: str, default="quantile"
        Strategy used to define the widths of the bins. Choose from:

        - "uniform": All bins have identical widths.
        - "quantile": All bins have the same number of points.
        - "kmeans": Values in each bin have the same nearest center of
          a 1D k-means cluster.
        - "custom": Use custom bin edges provided through `bins`.

    bins: int, sequence or dict, default=5
        Bin number or bin edges in which to split every column.

        - If int: Number of bins to produce for all columns. Only for
          strategy!="custom".
        - If sequence:
            - For strategy!="custom": Number of bins per column,
              allowing for non-uniform width. The n-th value corresponds
              to the n-th column that is transformed. Note that
              categorical columns are automatically ignored.
            - For strategy="custom": Bin edges with length=n_bins - 1.
              The outermost edges are always `-inf` and `+inf`, e.g.,
              bins `[1, 2]` indicate `(-inf, 1], (1, 2], (2, inf]`.
        - If dict: One of the aforementioned options per column, where
          the key is the column's name.

    labels: sequence, dict or None, default=None
        Label names with which to replace the binned intervals.

        - If None: Use default labels of the form `(min_edge, max_edge]`.
        - If sequence: Labels to use for all columns.
        - If dict: Labels per column, where the key is the column's name.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: dict, default={"data": "numpy", "estimator": "sklearn"}
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be a
        dictionary with keys `data` and/or `estimator`, with their
        corresponding choice as values. Choose from:

        - "data":

            - "numpy"
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn"
            - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`. Only
        for strategy="quantile".

    Attributes
    ----------
    feature_names_in_: np.ndarray
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.data_cleaning:Encoder
    atom.data_cleaning:Imputer
    atom.data_cleaning:Normalizer

    Examples
    --------

    === "atom"
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom["mean radius"])

        atom.discretize(
            strategy="custom",
            bins=[13, 18],
            labels=["small", "medium", "large"],
            verbose=2,
            columns="mean radius",
        )

        print(atom["mean radius"])
        ```

    === "stand-alone"
        ```pycon
        from atom.data_cleaning import Discretizer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        print(X["mean radius"])

        disc = Discretizer(
            strategy="custom",
            bins=[13, 18],
            labels=["small", "medium", "large"],
            verbose=2,
        )
        X["mean radius"] = disc.fit_transform(X[["mean radius"]])["mean radius"]

        print(X["mean radius"])
        ```

    """

    _train_only = False

    def __init__(
        self,
        strategy: DiscretizerStrats = "quantile",
        *,
        bins: Bins = 5,
        labels: Sequence[str] | dict[str, Sequence[str]] | None = None,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
        random_state: Int | None = None,
    ):
        super().__init__(
            device=device,
            engine=engine,
            verbose=verbose,
            logger=logger,
            random_state=random_state,
        )
        self.strategy = strategy
        self.bins = bins
        self.labels = labels

    @composed(crash, method_to_log)
    def fit(self, X: XSelector, y: YSelector | None = None) -> Discretizer:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Discretizer
            Estimator instance.

        """

        def get_labels(labels, bins):
            """Get labels for the specified bins."""
            if isinstance(labels, dict):
                default = [
                    f"({np.round(bins[i], 2)}, {np.round(bins[i+1], 1)}]"
                    for i in range(len(bins[:-1]))
                ]
                labels = labels.get(col, default)

            if len(bins) - 1 != len(labels):
                raise ValueError(
                    "Invalid value for the labels parameter. The length of "
                    "the bins does not match the length of the labels, got "
                    f"len(bins)={len(bins) - 1} and len(labels)={len(labels)}."
                )

            return labels

        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        self._labels = {}
        self._discretizers = {}
        self._num_cols = list(X.select_dtypes(include="number"))

        self._log("Fitting Discretizer...", 1)

        labels = {} if self.labels is None else self.labels
        for i, col in enumerate(self._num_cols):
            # Assign the proper bins for this column
            if isinstance(self.bins, dict):
                if col in self.bins:
                    bins = self.bins[col]
                else:
                    raise ValueError(
                        "Invalid value for the bins parameter. Column "
                        f"{col} not found in the dictionary."
                    )
            else:
                bins = self.bins

            if self.strategy != "custom":
                if isinstance(bins, SequenceTypes):
                    try:
                        bins = bins[i]  # Fetch the i-th bin for the i-th column
                    except IndexError:
                        raise ValueError(
                            "Invalid value for the bins parameter. The length of "
                            "the bins does not match the length of the columns, got "
                            f"len(bins)={len(bins) } and len(columns)={len(X.columns)}."
                        )

                estimator = self._get_est_class("KBinsDiscretizer", "preprocessing")

                # cuML implementation has no subsample and random_state
                kwargs = {}
                if "subsample" in sign(estimator):
                    kwargs["subsample"] = 200000
                    kwargs["random_state"] = self.random_state

                self._discretizers[col] = estimator(
                    n_bins=bins,
                    encode="ordinal",
                    strategy=self.strategy,
                    **kwargs,
                ).fit(X[[col]])

                # Save labels for transform method
                self._labels[col] = get_labels(
                    labels=labels,
                    bins=self._discretizers[col].bin_edges_[0],
                )

            else:
                if not isinstance(bins, SequenceTypes):
                    raise TypeError(
                        f"Invalid type for the bins parameter, got {bins}. Only "
                        "a sequence of bin edges is accepted when strategy='custom'."
                    )
                else:
                    bins = [-np.inf] + list(bins) + [np.inf]

                estimator = self._get_est_class("FunctionTransformer", "preprocessing")

                # Make of cut a transformer
                self._discretizers[col] = estimator(
                    func=bk.cut,
                    kw_args={"bins": bins, "labels": get_labels(labels, bins)},
                ).fit(X[[col]])

        return self

    @composed(crash, method_to_log)
    def transform(self, X: XSelector, y: YSelector | None = None) -> DataFrame:
        """Bin the data into intervals.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Transformed feature set.

        """
        X, y = self._prepare_input(X, y, columns=self.feature_names_in_)

        self._log("Binning the features...", 1)

        for col in self._num_cols:
            if self.strategy.lower() == "custom":
                X[col] = self._discretizers[col].transform(X[col])
            else:
                X[col] = self._discretizers[col].transform(X[[col]])[:, 0]

                # Replace cluster values with labels
                for i, label in enumerate(self._labels[col]):
                    X[col] = X[col].replace(i, label)

            self._log(f" --> Discretizing feature {col} in {X[col].nunique()} bins.", 2)

        return X


class Encoder(BaseEstimator, TransformerMixin, BaseTransformer):
    """Perform encoding of categorical features.

    The encoding type depends on the number of classes in the column:

    - If n_classes=2 or ordinal feature, use Ordinal-encoding.
    - If 2 < n_classes <= `max_onehot`, use OneHot-encoding.
    - If n_classes > `max_onehot`, use `strategy`-encoding.

    Missing values are propagated to the output column. Unknown
    classes encountered during transforming are imputed according
    to the selected strategy. Infrequent classes can be replaced with
    a value in order to prevent too high cardinality.

    This class can be accessed from atom through the [encode]
    [atomclassifier-encode] method. Read more in the [user guide]
    [encoding-categorical-features].

    !!! warning
        Three category-encoders estimators are unavailable:

        * [OneHotEncoder][]: Use the max_onehot parameter.
        * [HashingEncoder][]: Incompatibility of APIs.
        * [LeaveOneOutEncoder][]: Incompatibility of APIs.

    Parameters
    ----------
    strategy: str or estimator, default="Target"
        Type of encoding to use for high cardinality features. Choose
        from any of the estimators in the category-encoders package
        or provide a custom one.

    max_onehot: int or None, default=10
        Maximum number of unique values in a feature to perform
        one-hot encoding. If None, `strategy`-encoding is always
        used for columns with more than two classes.

    ordinal: dict or None, default=None
        Order of ordinal features, where the dict key is the feature's
        name and the value is the class order, e.g., `{"salary": ["low",
        "medium", "high"]}`.

    infrequent_to_value: int, float or None, default=None
        Replaces infrequent class occurrences in categorical columns
        with the string in parameter `value`. This transformation is
        done before the encoding of the column.

        - If None: Skip this step.
        - If int: Minimum number of occurrences in a class.
        - If float: Minimum fraction of occurrences in a class.

    value: str, default="infrequent"
        Value with which to replace rare classes. This parameter is
        ignored if `infrequent_to_value=None`.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    mapping_: dict of dicts
        Encoded values and their respective mapping. The column name is
        the key to its mapping dictionary. Only for columns mapped to a
        single column (e.g., Ordinal, Leave-one-out, etc...).

    feature_names_in_: np.ndarray
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.data_cleaning:Cleaner
    atom.data_cleaning:Imputer
    atom.data_cleaning:Pruner

    Examples
    --------

    === "atom"
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer
        from numpy.random import randint

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        X["cat_feature_1"] = [f"x{i}" for i in randint(0, 2, len(X))]
        X["cat_feature_2"] = [f"x{i}" for i in randint(0, 3, len(X))]
        X["cat_feature_3"] = [f"x{i}" for i in randint(0, 20, len(X))]

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.X)

        atom.encode(strategy="target", max_onehot=10, verbose=2)

        # Note the one-hot encoded column with name [feature]_[class]
        print(atom.X)
        ```

    === "stand-alone"
        ```pycon
        from atom.data_cleaning import Encoder
        from sklearn.datasets import load_breast_cancer
        from numpy.random import randint

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        X["cat_feature_1"] = [f"x{i}" for i in randint(0, 2, len(X))]
        X["cat_feature_2"] = [f"x{i}" for i in randint(0, 3, len(X))]
        X["cat_feature_3"] = [f"x{i}" for i in randint(0, 20, len(X))]
        print(X)

        encoder = Encoder(strategy="target", max_onehot=10, verbose=2)
        X = encoder.fit_transform(X, y)

        # Note the one-hot encoded column with name [feature]_[class]
        print(X)
        ```

    """

    _train_only = False

    def __init__(
        self,
        strategy: str | Transformer = "YSelector",
        *,
        max_onehot: IntLargerTwo | None = 10,
        ordinal: dict[str, Sequence[Any]] | None = None,
        infrequent_to_value: FloatLargerZero | None = None,
        value: str = "infrequent",
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
        **kwargs,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strategy = strategy
        self.max_onehot = max_onehot
        self.ordinal = ordinal
        self.infrequent_to_value = infrequent_to_value
        self.value = value
        self.kwargs = kwargs

    @composed(crash, method_to_log)
    def fit(self, X: XSelector, y: YSelector | None = None) -> Encoder:
        """Fit to data.

        Note that leaving y=None can lead to errors if the `strategy`
        encoder requires target values. For multioutput tasks, only
        the first target column is used to fit the encoder.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or dataframe-like
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        Returns
        -------
        Encoder
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        self.mapping_ = defaultdict(dict)
        self._to_value = defaultdict(list)
        self._categories = {}
        self._encoders = {}
        self._cat_cols = list(X.select_dtypes(exclude="number").columns)

        strategies = dict(
            backwarddifference=BackwardDifferenceEncoder,
            basen=BaseNEncoder,
            binary=BinaryEncoder,
            catboost=CatBoostEncoder,
            helmert=HelmertEncoder,
            jamesstein=JamesSteinEncoder,
            mestimate=MEstimateEncoder,
            ordinal=OrdinalEncoder,
            polynomial=PolynomialEncoder,
            sum=SumEncoder,
            target=TargetEncoder,
            woe=WOEEncoder,
        )

        if isinstance(self.strategy, str):
            if self.strategy.lower().endswith("encoder"):
                self.strategy = self.strategy[:-7]  # Remove the Encoder at the end
            if self.strategy.lower() not in strategies:
                raise ValueError(
                    f"Invalid value for the strategy parameter, got {self.strategy}. "
                    f"Choose from: {', '.join(strategies)}."
                )
            estimator = strategies[self.strategy.lower()]
        elif callable(self.strategy):
            estimator = self.strategy
        else:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                "For customs estimators, a class is expected, but got an instance."
            )

        if self.max_onehot is None:
            max_onehot = 0
        else:
            max_onehot = self.max_onehot

        if self.infrequent_to_value:
            if self.infrequent_to_value < 1:
                infrequent_to_value = int(self.infrequent_to_value * len(X))
            else:
                infrequent_to_value = self.infrequent_to_value

        self._log("Fitting Encoder...", 1)

        for name, column in X[self._cat_cols].items():
            # Replace infrequent classes with the string in `value`
            if self.infrequent_to_value:
                for category, count in column.value_counts().items():
                    if count <= infrequent_to_value:
                        self._to_value[name].append(category)
                        X[name] = column.replace(category, self.value)

            # Get the unique categories before fitting
            self._categories[name] = column.dropna().sort_values().unique().tolist()

            # Perform encoding type dependent on number of unique values
            ordinal = self.ordinal or {}
            if name in ordinal or len(self._categories[name]) == 2:

                # Check that provided classes match those of column
                ordinal = ordinal.get(name, self._categories[name])
                if column.nunique(dropna=True) != len(ordinal):
                    self._log(
                        f" --> The number of classes passed to feature {name} in the "
                        f"ordinal parameter ({len(ordinal)}) don't match the number "
                        f"of classes in the data ({column.nunique(dropna=True)}).", 1,
                        severity="warning"
                    )

                # Create custom mapping from 0 to N - 1
                mapping = {v: i for i, v in enumerate(ordinal)}
                mapping.setdefault(np.NaN, -1)  # Encoder always needs mapping of NaN
                self.mapping_[name] = mapping

                self._encoders[name] = OrdinalEncoder(
                    mapping=[{"col": name, "mapping": mapping}],
                    cols=[name],  # Specify to not skip bool columns
                    handle_missing="return_nan",
                    handle_unknown="value",
                ).fit(X[[name]])

            elif 2 < len(self._categories[name]) <= max_onehot:
                self._encoders[name] = OneHotEncoder(
                    cols=[name],  # Specify to not skip numerical columns
                    use_cat_names=True,
                    handle_missing="return_nan",
                    handle_unknown="value",
                ).fit(X[[name]])

            else:
                args = [X[[name]]]
                if "y" in sign(estimator.fit):
                    args.append(bk.DataFrame(y).iloc[:, 0])

                self._encoders[name] = estimator(
                    cols=[name],
                    handle_missing="return_nan",
                    handle_unknown="value",
                    **self.kwargs,
                ).fit(*args)

                # Create encoding of unique values for mapping
                data = self._encoders[name].transform(
                    bk.Series(
                        data=self._categories[name],
                        index=self._categories[name],
                        name=name,
                        dtype="object",
                    )
                )

                # Only mapping 1 - 1 column
                if data.shape[1] == 1:
                    for idx, value in data[name].items():
                        self.mapping_[name][idx] = value

        return self

    @composed(crash, method_to_log)
    def transform(self, X: XSelector, y: YSelector | None = None) -> DataFrame:
        """Encode the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Encoded dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y, columns=self.feature_names_in_)

        self._log("Encoding categorical columns...", 1)

        for name, column in X[self._cat_cols].items():
            # Convert infrequent classes to value
            if self._to_value[name]:
                X[name] = column.replace(self._to_value[name], self.value)

            self._log(
                f" --> {self._encoders[name].__class__.__name__[:-7]}-encoding "
                f"feature {name}. Contains {column.nunique()} classes.", 2
            )

            # Count the propagated missingX[[name]] values
            if n_nans := column.isna().sum():
                self._log(f"   --> Propagating {n_nans} missing values.", 2)

            # Get the new encoded columns
            new_cols = self._encoders[name].transform(X[[name]])

            # Drop _nan columns (since missing values are propagated)
            new_cols = new_cols.loc[:, ~new_cols.columns.str.endswith("_nan")]

            # Check for unknown classes
            if uc := len(column.dropna()[~column.isin(self._categories[name])]):
                self._log(f"   --> Handling {uc} unknown classes.", 2)

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


class Imputer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Handle missing values in the data.

    Impute or remove missing values according to the selected strategy.
    Also removes rows and columns with too many missing values. Use
    the `missing` attribute to customize what are considered "missing
    values".

    This class can be accessed from atom through the [impute]
    [atomclassifier-impute] method. Read more in the [user guide]
    [imputing-missing-values].

    Parameters
    ----------
    strat_num: str, int or float, default="drop"
        Imputing strategy for numerical columns. Choose from:

        - "drop": Drop rows containing missing values.
        - "mean": Impute with mean of column.
        - "median": Impute with median of column.
        - "knn": Impute using a K-Nearest Neighbors approach.
        - "most_frequent": Impute with the most frequent value.
        - int or float: Impute with provided numerical value.

    strat_cat: str, default="drop"
        Imputing strategy for categorical columns. Choose from:

        - "drop": Drop rows containing missing values.
        - "most_frequent": Impute with the most frequent value.
        - str: Impute with provided string.

    max_nan_rows: int, float or None, default=None
        Maximum number or fraction of missing values in a row
        (if more, the row is removed). If None, ignore this step.

    max_nan_cols: int, float or None, default=None
        Maximum number or fraction of missing values in a column
        (if more, the column is removed). If None, ignore this step.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: dict, default={"data": "numpy", "estimator": "sklearn"}
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be a
        dictionary with keys `data` and/or `estimator`, with their
        corresponding choice as values. Choose from:

        - "data":

            - "numpy"
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn"
            - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    Attributes
    ----------
    missing_: list
        Values that are considered "missing". Default values are: "",
        "?", "NA", "nan", "NaN", "NaT", "none", "None", "inf", "-inf".
        Note that `None`, `NaN`, `+inf` and `-inf` are always considered
        missing since they are incompatible with sklearn estimators.

    feature_names_in_: np.ndarray
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.data_cleaning:Balancer
    atom.data_cleaning:Discretizer
    atom.data_cleaning:Encoder

    Examples
    --------

    === "atom"
        ```pycon
        import numpy as np
        from atom import ATOMClassifier
        from numpy.random import randint
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Add some random missing values to the data
        for i, j in zip(randint(0, X.shape[0], 600), randint(0, 4, 600)):
            X.iat[i, j] = np.NaN

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.nans)

        atom.impute(strat_num="median", max_nan_rows=0.1, verbose=2)

        print(atom.n_nans)
        ```

    === "stand-alone"
        ```pycon
        import numpy as np
        from atom.data_cleaning import Imputer
        from numpy.random import randint
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Add some random missing values to the data
        for i, j in zip(randint(0, X.shape[0], 600), randint(0, 4, 600)):
            X.iloc[i, j] = np.nan

        imputer = Imputer(strat_num="median", max_nan_rows=0.1, verbose=2)
        X, y = imputer.fit_transform(X, y)

        print(X)
        ```

    """

    _train_only = False

    def __init__(
        self,
        strat_num: NumericalStrats = "drop",
        strat_cat: CategoricalStrats = "drop",
        *,
        max_nan_rows: FloatLargerZero | None = None,
        max_nan_cols: FloatLargerZero | None = None,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose, logger=logger)
        self.strat_num = strat_num
        self.strat_cat = strat_cat
        self.max_nan_rows = max_nan_rows
        self.max_nan_cols = max_nan_cols

    @composed(crash, method_to_log)
    def fit(self, X: XSelector, y: YSelector | None = None) -> Imputer:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Imputer
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        if not hasattr(self, "missing_"):
            self.missing_ = [
                "", "?", "NA", "nan", "NaN", "NaT", "none", "None", "inf", "-inf"
            ]

        self._max_nan_rows = None
        self._drop_cols = []
        self._imputers = {}
        self._num_cols = list(X.select_dtypes(include="number"))

        # Check input Parameters
        if self.max_nan_rows:
            if self.max_nan_rows <= 1:
                self._max_nan_rows = int(X.shape[1] * self.max_nan_rows)
            else:
                self._max_nan_rows = self.max_nan_rows

        if self.max_nan_cols:
            if self.max_nan_cols <= 1:
                max_nan_cols = int(X.shape[0] * self.max_nan_cols)
            else:
                max_nan_cols = self.max_nan_cols

        self._log("Fitting Imputer...", 1)

        # Unify all values to impute
        X = replace_missing(X, self.missing_)

        # Drop rows with too many NaN values
        if self._max_nan_rows is not None:
            X = X.dropna(axis=0, thresh=X.shape[1] - self._max_nan_rows)
            if X.empty:
                raise ValueError(
                    "Invalid value for the max_nan_rows parameter, got "
                    f"{self.max_nan_rows}. All rows contain more than "
                    f"{self._max_nan_rows} missing values. Choose a "
                    f"larger value or set the parameter to None."
                )

        # Reset internal attrs in case of repeated fit
        self._drop_cols = []
        self._imputers = {}

        # Load the imputer class from sklearn or cuml (different modules)
        module = "preprocessing" if self.engine.get("estimator") == "cuml" else "impute"
        estimator = self._get_est_class("SimpleImputer", module)

        # Assign an imputer to each column
        for name, column in X.items():
            # Remember columns with too many missing values
            if self.max_nan_cols and column.isna().sum() > max_nan_cols:
                self._drop_cols.append(name)
                continue

            # Column is numerical
            # Note missing_values=pd.NA also imputes np.NaN in SimpleImputer
            if name in self._num_cols:
                if isinstance(self.strat_num, str):
                    if self.strat_num.lower() == "knn":
                        self._imputers[name] = KNNImputer().fit(X[[name]])

                    elif self.strat_num.lower() == "most_frequent":
                        self._imputers[name] = estimator(
                            missing_values=pd.NA,
                            strategy="most_frequent",
                        ).fit(X[[name]])

                    # Strategies mean or median
                    elif self.strat_num.lower() != "drop":
                        self._imputers[name] = estimator(
                            missing_values=pd.NA,
                            strategy=self.strat_num.lower()
                        ).fit(X[[name]])

            # Column is categorical
            elif self.strat_cat.lower() == "most_frequent":
                self._imputers[name] = estimator(
                    missing_values=pd.NA,
                    strategy="most_frequent",
                ).fit(X[[name]])

        return self

    @composed(crash, method_to_log)
    def transform(
        self,
        X: XSelector,
        y: YSelector | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Impute the missing values.

        Note that leaving y=None can lead to inconsistencies in
        data length between X and y if rows are dropped during
        the transformation.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence, dataframe-like or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        Returns
        -------
        dataframe
            Imputed dataframe.

        series
            Transformed target column. Only returned if provided.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y, columns=self.feature_names_in_)

        self._log("Imputing missing values...", 1)

        # Unify all values to impute
        X = replace_missing(X, self.missing_)

        # Drop rows with too many missing values
        if self._max_nan_rows is not None:
            length = len(X)
            X = X.dropna(axis=0, thresh=X.shape[1] - self._max_nan_rows)
            if diff := length - len(X):
                if y is not None:
                    y = y[y.index.isin(X.index)]  # Select only indices that remain
                self._log(
                    f" --> Dropping {diff} samples for containing "
                    f"more than {self._max_nan_rows} missing values.", 2
                )

        for name, column in X.items():
            nans = column.isna().sum()

            # Drop columns with too many missing values
            if name in self._drop_cols:
                self._log(
                    f" --> Dropping feature {name}. Contains {nans} "
                    f"({nans * 100 // len(X)}%) missing values.", 2
                )
                X = X.drop(name, axis=1)
                continue

            # Apply only if column is numerical and contains missing values
            if name in self._num_cols and nans > 0:
                if not isinstance(self.strat_num, str):
                    self._log(
                        f" --> Imputing {nans} missing values with number "
                        f"{str(self.strat_num)} in feature {name}.", 2
                    )
                    X[name] = column.replace(np.NaN, self.strat_num)

                elif self.strat_num.lower() == "drop":
                    X = X.dropna(subset=[name], axis=0)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self._log(
                        f" --> Dropping {nans} samples due to missing "
                        f"values in feature {name}.", 2
                    )

                elif self.strat_num.lower() == "knn":
                    self._log(
                        f" --> Imputing {nans} missing values using "
                        f"the KNN imputer in feature {name}.", 2
                    )
                    X[name] = self._imputers[name].transform(X[[name]]).flatten()

                else:  # Strategies mean, median or most_frequent
                    n = np.round(self._imputers[name].statistics_[0], 2)
                    self._log(
                        f" --> Imputing {nans} missing values with "
                        f"{self.strat_num.lower()} ({n}) in feature {name}.", 2
                    )
                    X[name] = self._imputers[name].transform(X[[name]]).flatten()

            # The column is categorical and contains missing values
            elif nans > 0:
                if self.strat_cat.lower() not in ("drop", "most_frequent"):
                    self._log(
                        f" --> Imputing {nans} missing values with "
                        f"{self.strat_cat} in feature {name}.", 2
                    )
                    X[name] = column.replace(np.NaN, self.strat_cat)

                elif self.strat_cat.lower() == "drop":
                    X = X.dropna(subset=[name], axis=0)
                    if y is not None:
                        y = y[y.index.isin(X.index)]
                    self._log(
                        f" --> Dropping {nans} samples due to "
                        f"missing values in feature {name}.", 2
                    )

                elif self.strat_cat.lower() == "most_frequent":
                    mode = self._imputers[name].statistics_[0]
                    self._log(
                        f" --> Imputing {nans} missing values with "
                        f"most_frequent ({mode}) in feature {name}.", 2
                    )
                    X[name] = self._imputers[name].transform(X[[name]]).flatten()

        return variable_return(X, y)


class Normalizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Transform the data to follow a Normal/Gaussian distribution.

    This transformation is useful for modeling issues related to
    heteroscedasticity (non-constant variance), or other situations
    where normality is desired. Missing values are disregarded in
    fit and maintained in transform. Categorical columns are ignored.

    This class can be accessed from atom through the [normalize]
    [atomclassifier-normalize] method. Read more in the [user guide]
    [normalizing-the-feature-set].

    !!! warning
        The quantile strategy performs a non-linear transformation.
        This may distort linear correlations between variables measured
        at the same scale but renders variables measured at different
        scales more directly comparable.

    !!! note
        The yeojohnson and boxcox strategies scale the data after
        transforming. Use the `kwargs` to change this behavior.

    Parameters
    ----------
    strategy: str, default="yeojohnson"
        The transforming strategy. Choose from:

        - "[yeojohnson][]"
        - "[boxcox][]" (only works with strictly positive values)
        - "[quantile][]": Transform features using quantiles information.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: dict, default={"data": "numpy", "estimator": "sklearn"}
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be a
        dictionary with keys `data` and/or `estimator`, with their
        corresponding choice as values. Choose from:

        - "data":

            - "numpy"
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn"
            - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    random_state: int or None, default=None
        Seed used by the quantile strategy. If None, the random
        number generator is the `RandomState` used by `np.random`.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]_: sklearn transformer
        Object with which the data is transformed, e.g.,
        `normalizer.yeojohnson` for the default strategy.

    feature_names_in_: np.ndarray
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.data_cleaning:Cleaner
    atom.data_cleaning:Pruner
    atom.data_cleaning:Scaler

    Examples
    --------

    === "atom"
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.dataset)

        atom.plot_distribution(columns=0)

        atom.normalize(verbose=2)

        print(atom.dataset)

        atom.plot_distribution(columns=0)
        ```

    === "stand-alone"
        ```pycon
        from atom.data_cleaning import Normalizer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        normalizer = Normalizer(verbose=2)
        X = normalizer.fit_transform(X)

        print(X)
        ```

    """

    _train_only = False

    def __init__(
        self,
        strategy: NormalizerStrats = "yeojohnson",
        *,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
        random_state: Int | None = None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            engine=engine,
            verbose=verbose,
            logger=logger,
            random_state=random_state,
        )
        self.strategy = strategy
        self.kwargs = kwargs

    @composed(crash, method_to_log)
    def fit(self, X: XSelector, y: YSelector | None = None) -> Normalizer:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Normalizer
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        self._num_cols = list(X.select_dtypes(include="number"))

        strategies = dict(
            yeojohnson="PowerTransformer",
            boxcox="PowerTransformer",
            quantile="QuantileTransformer",
        )

        if self.strategy.lower() in ("yeojohnson", "boxcox"):
            estimator = self._get_est_class(strategies[self.strategy], "preprocessing")
            self._estimator = estimator(
                method=self.strategy.lower()[:3] + "-" + self.strategy.lower()[3:],
                **self.kwargs,
            )
        elif self.strategy.lower() == "quantile":
            kwargs = self.kwargs.copy()
            estimator = self._get_est_class(strategies[self.strategy], "preprocessing")
            self._estimator = estimator(
                output_distribution=kwargs.pop("output_distribution", "normal"),
                random_state=kwargs.pop("random_state", self.random_state),
                **kwargs,
            )
        else:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                f"Choose from: {', '.join(strategies)}."
            )

        self._log("Fitting Normalizer...", 1)
        self._estimator.fit(X[self._num_cols])

        # Add the estimator as attribute to the instance
        setattr(self, f"{self.strategy.lower()}_", self._estimator)

        return self

    @composed(crash, method_to_log)
    def transform(self, X: XSelector, y: YSelector | None = None) -> DataFrame:
        """Apply the transformations to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Normalized dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y, columns=self.feature_names_in_)

        self._log("Normalizing features...", 1)
        X_transformed = self._estimator.transform(X[self._num_cols])

        # If all columns were transformed, just swap sets
        if len(self._num_cols) != X.shape[1]:
            # Replace the numerical columns with the transformed values
            for i, col in enumerate(self._num_cols):
                X[col] = X_transformed[:, i]
        else:
            X = to_df(X_transformed, X.index, X.columns)

        return X

    @composed(crash, method_to_log)
    def inverse_transform(self, X: XSelector, y: YSelector | None = None) -> DataFrame:
        """Apply the inverse transformation to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Original dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self._log("Inversely normalizing features...", 1)
        X_transformed = self._estimator.inverse_transform(X[self._num_cols])

        # If all columns were transformed, just swap sets
        if len(self._num_cols) != X.shape[1]:
            # Replace the numerical columns with the transformed values
            for i, col in enumerate(self._num_cols):
                X[col] = X_transformed[:, i]
        else:
            X = to_df(X_transformed, X.index, X.columns)

        return X


class Pruner(BaseEstimator, TransformerMixin, BaseTransformer):
    """Prune outliers from the data.

    Replace or remove outliers. The definition of outlier depends
    on the selected strategy and can greatly differ from one another.
    Ignores categorical columns.

    This class can be accessed from atom through the [prune]
    [atomclassifier-prune] method. Read more in the [user guide]
    [handling-outliers].

    !!! info
        The "sklearnex" and "cuml" engines are only supported for
        strategy="dbscan".

    Parameters
    ----------
    strategy: str or sequence, default="zscore"
        Strategy with which to select the outliers. If sequence of
        strategies, only samples marked as outliers by all chosen
        strategies are dropped. Choose from:

        - "zscore": Z-score of each data value.
        - "[iforest][]": Isolation Forest.
        - "[ee][]": Elliptic Envelope.
        - "[lof][]": Local Outlier Factor.
        - "[svm][]": One-class SVM.
        - "[dbscan][]": Density-Based Spatial Clustering.
        - "[hdbscan][]": Hierarchical Density-Based Spatial Clustering.
        - "[optics][]": DBSCAN-like clustering approach.

    method: int, float or str, default="drop"
        Method to apply on the outliers. Only the zscore strategy
        accepts another method than "drop". Choose from:

        - "drop": Drop any sample with outlier values.
        - "minmax": Replace outlier with the min/max of the column.
        - Any numerical value with which to replace the outliers.

    max_sigma: int or float, default=3
        Maximum allowed standard deviations from the mean of the
        column. If more, it is considered an outlier. Only if
        strategy="zscore".

    include_target: bool, default=False
        Whether to include the target column in the search for
        outliers. This can be useful for regression tasks. Only
        if strategy="zscore".

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: dict, default={"data": "numpy", "estimator": "sklearn"}
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be a
        dictionary with keys `data` and/or `estimator`, with their
        corresponding choice as values. Choose from:

        - "data":

            - "numpy"
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn"
            - "sklearnex"
            - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator. If
        sequence of strategies, the params should be provided in a dict
        with the strategy's name as key.

    Attributes
    ----------
    [strategy]_: sklearn estimator
        Object used to prune the data, e.g., `pruner.iforest` for the
        isolation forest strategy. Not available for strategy="zscore".

    feature_names_in_: np.ndarray
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.data_cleaning:Balancer
    atom.data_cleaning:Normalizer
    atom.data_cleaning:Scaler

    Examples
    --------

    === "atom"
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.dataset)

        atom.prune(stratgey="iforest", verbose=2)

        # Note the reduced number of rows
        print(atom.dataset)

        atom.plot_distribution(columns=0)
        ```

    === "stand-alone"
        ```pycon
        from atom.data_cleaning import Normalizer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        normalizer = Normalizer(verbose=2)
        X = normalizer.fit_transform(X)

        # Note the reduced number of rows
        print(X)
        ```

    """

    _train_only = True

    def __init__(
        self,
        strategy: PrunerStrats | Sequence[PrunerStrats] = "zscore",
        *,
        method: Scalar | Literal["drop", "minmax"] = "drop",
        max_sigma: FloatLargerZero = 3,
        include_target: Bool = False,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose, logger=logger)
        self.strategy = strategy
        self.method = method
        self.max_sigma = max_sigma
        self.include_target = include_target
        self.kwargs = kwargs

    @composed(crash, method_to_log)
    def transform(
        self,
        X: XSelector,
        y: YSelector | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Apply the outlier strategy on the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence, dataframe-like or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        Returns
        -------
        dataframe
            Transformed feature set.

        series
            Transformed target column. Only returned if provided.

        """
        X, y = self._prepare_input(X, y, columns=getattr(self, "feature_names_in_", None))

        # Estimators with their modules
        strategies = dict(
            iforest=["IsolationForest", "ensemble"],
            ee=["EllipticEnvelope", "covariance"],
            lof=["LocalOutlierFactor", "neighbors"],
            svm=["OneClassSVM", "svm"],
            dbscan=["DBSCAN", "cluster"],
            hdbscan=["HDBSCAN", "cluster"],
            optics=["OPTICS", "cluster"],
        )

        for strat in lst(self.strategy):
            if strat not in ["zscore"] + list(strategies):
                raise ValueError(
                    "Invalid value for the strategy parameter. "
                    f"Choose from: zscore, {', '.join(strategies)}."
                )
            if strat != "zscore" and str(self.method) != "drop":
                raise ValueError(
                    "Invalid value for the method parameter. Only the zscore "
                    f"strategy accepts another method than 'drop', got {self.method}."
                )

        # Allocate kwargs to every estimator
        kwargs = {}
        for strat in lst(self.strategy):
            kwargs[strat] = {}
            for key, value in self.kwargs.items():
                # Parameters for this estimator only
                if key.lower() == strat.lower():
                    kwargs[strat].update(value)
                # Parameters for all estimators
                elif key.lower() not in map(str.lower, lst(self.strategy)):
                    kwargs[strat].update({key: value})

        self._log("Pruning outliers...", 1)

        # Prepare dataset (merge with y and exclude categorical columns)
        objective = merge(X, y) if self.include_target and y is not None else X
        objective = objective.select_dtypes(include=["number"])

        outliers = []
        for strat in lst(self.strategy):
            if strat.lower() == "zscore":
                # stats.zscore only works with np types, therefore convert
                z_scores = zscore(objective.values.astype(float), nan_policy="propagate")

                if not isinstance(self.method, str):
                    cond = np.abs(z_scores) > self.max_sigma
                    objective = objective.mask(cond, self.method)
                    self._log(
                        f" --> Replacing {cond.sum()} outlier "
                        f"values with {self.method}.", 2
                    )

                elif self.method.lower() == "minmax":
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

                    self._log(
                        f" --> Replacing {counts} outlier values "
                        "with the min or max of the column.", 2
                    )

                elif self.method.lower() == "drop":
                    mask = (np.abs(zscore(z_scores)) <= self.max_sigma).all(axis=1)
                    outliers.append(mask)
                    if len(lst(self.strategy)) > 1:
                        self._log(
                            f" --> The zscore strategy detected "
                            f"{len(mask) - sum(mask)} outliers.", 2
                        )

            else:
                estimator = self._get_est_class(*strategies[strat])(**kwargs[strat])
                mask = estimator.fit_predict(objective) >= 0
                outliers.append(mask)
                if len(lst(self.strategy)) > 1:
                    self._log(
                        f" --> The {estimator.__class__.__name__} "
                        f"detected {len(mask) - sum(mask)} outliers.", 2
                    )

                # Add the estimator as attribute to the instance
                setattr(self, f"{strat.lower()}_", estimator)

        if outliers:
            # Select outliers from intersection of strategies
            mask = [any([i for i in strats]) for strats in zip(*outliers)]
            self._log(f" --> Dropping {len(mask) - sum(mask)} outliers.", 2)

            # Keep only the non-outliers from the data
            X = X[mask]
            if y is not None:
                y = y[mask]

        else:  # Replace the columns in X and y with the new values from objective
            X = merge(*[objective.get(col.name, col) for col in get_cols(X)])
            if y is not None:
                y = merge(*[objective.get(col.name, col) for col in get_cols(y)])

        if y is None:
            return X
        else:
            return X, y


class Scaler(BaseEstimator, TransformerMixin, BaseTransformer):
    """Scale the data.

    Apply one of sklearn's scalers. Categorical columns are ignored.

    This class can be accessed from atom through the [scale]
    [atomclassifier-scale] method. Read more in the [user guide]
    [scaling-the-feature-set].

    Parameters
    ----------
    strategy: str, default="standard"
        Strategy with which to scale the data. Choose from:

        - "[standard][]": Remove mean and scale to unit variance.
        - "[minmax][]": Scale features to a given range.
        - "[maxabs][]": Scale features by their maximum absolute value.
        - "[robust][]": Scale using statistics that are robust to outliers.

    include_binary: bool, default=False
        Whether to scale binary columns (only 0s and 1s).

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: dict, default={"data": "numpy", "estimator": "sklearn"}
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be a
        dictionary with keys `data` and/or `estimator`, with their
        corresponding choice as values. Choose from:

        - "data":

            - "numpy"
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn"
            - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]_: sklearn transformer
        Object with which the data is scaled, e.g.,
        `scaler.standard` for the default strategy.

    feature_names_in_: np.ndarray
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.data_cleaning:Balancer
    atom.data_cleaning:Normalizer
    atom.data_cleaning:Scaler

    Examples
    --------

    === "atom"
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.dataset)

        atom.scale(verbose=2)

        # Note the reduced number of rows
        print(atom.dataset)
        ```

    === "stand-alone"
        ```pycon
        from atom.data_cleaning import Scaler
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        scaler = Scaler(verbose=2)
        X = scaler.fit_transform(X)

        # Note the reduced number of rows
        print(X)
        ```

    """

    _train_only = False

    def __init__(
        self,
        strategy: ScalerStrats = "standard",
        include_binary: Bool = False,
        *,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose, logger=logger)
        self.strategy = strategy
        self.include_binary = include_binary
        self.kwargs = kwargs

    @composed(crash, method_to_log)
    def fit(self, X: XSelector, y: YSelector | None = None) -> Scaler:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Scaler
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        self._num_cols = list(X.select_dtypes(include="number"))

        if not self.include_binary:
            self._num_cols = [
                col for col in self._num_cols if ~np.isin(X[col].unique(), [0, 1]).all()
            ]

        strategies = dict(
            standard="StandardScaler",
            minmax="MinMaxScaler",
            maxabs="MaxAbsScaler",
            robust="RobustScaler",
        )

        estimator = self._get_est_class(strategies[self.strategy], "preprocessing")
        self._estimator = estimator(**self.kwargs)

        self._log("Fitting Scaler...", 1)
        self._estimator.fit(X[self._num_cols])

        # Add the estimator as attribute to the instance
        setattr(self, f"{self.strategy.lower()}_", self._estimator)

        return self

    @composed(crash, method_to_log)
    def transform(self, X: XSelector, y: YSelector | None = None) -> DataFrame:
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Scaled dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y, columns=self.feature_names_in_)

        self._log("Scaling features...", 1)
        X_transformed = self._estimator.transform(X[self._num_cols])

        # If all columns were transformed, just swap sets
        if len(self._num_cols) != X.shape[1]:
            # Replace the numerical columns with the transformed values
            for i, col in enumerate(self._num_cols):
                X[col] = X_transformed[:, i]
        else:
            X = to_df(X_transformed, X.index, X.columns)

        return X

    @composed(crash, method_to_log)
    def inverse_transform(self, X: XSelector, y: YSelector | None = None) -> DataFrame:
        """Apply the inverse transformation to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Scaled dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self._log("Inversely scaling features...", 1)
        X_transformed = self._estimator.inverse_transform(X[self._num_cols])

        # If all columns were transformed, just swap sets
        if len(self._num_cols) != X.shape[1]:
            # Replace the numerical columns with the transformed values
            for i, col in enumerate(self._num_cols):
                X[col] = X_transformed[:, i]
        else:
            X = to_df(X_transformed, X.index, X.columns)

        return X

# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the data cleaning transformers.

"""

from __future__ import annotations

from collections import defaultdict
from inspect import signature
from logging import Logger
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
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
from sklearn.base import BaseEstimator, clone
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    FunctionTransformer, PowerTransformer, QuantileTransformer,
)
from typeguard import typechecked

from atom.basetransformer import BaseTransformer
from atom.utils import (
    FLOAT, INT, SCALAR, SEQUENCE, SEQUENCE_TYPES, X_TYPES, Y_TYPES, CustomDict,
    check_is_fitted, composed, crash, it, lst, merge, method_to_log, to_df,
    to_series, variable_return,
)


class TransformerMixin:
    """Mixin class for all transformers in ATOM.

    Different from sklearn, since it accounts for the transformation
    of y and a possible absence of the fit method.

    """

    def fit(
        self,
        X: Optional[X_TYPES] = None,
        /,
        y: Optional[Y_TYPES] = None,
        **fit_params,
    ):
        """Does nothing.

        Implemented for continuity of the API.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

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

        self.log(f"Fitting {self.__class__.__name__}...", 1)

        return self

    @composed(crash, method_to_log, typechecked)
    def fit_transform(
        self,
        X: Optional[X_TYPES] = None,
        /,
        y: Optional[Y_TYPES] = None,
        **fit_params,
    ) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.Series]]:
        """Fit to data, then transform it.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        **fit_params
            Additional keyword arguments for the fit method.

        Returns
        -------
        pd.DataFrame
            Transformed feature set. Only returned if provided.

        pd.Series
            Transformed target column. Only returned if provided.

        """
        return self.fit(X, y, **fit_params).transform(X, y)

    @composed(crash, method_to_log, typechecked)
    def inverse_transform(
        self,
        X: Optional[X_TYPES] = None,
        /,
        y: Optional[Y_TYPES] = None,
    ) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.Series]]:
        """Does nothing.

        Returns the input unchanged. Implemented for continuity of the
        API.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        pd.DataFrame
            Transformed feature set. Only returned if provided.

        pd.Series
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
        The [clustercentroids][] estimator is unavailable because of
        incompatibilities of the APIs.

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
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]: imblearn estimator
        Object (lowercase strategy) used to balance the data,
        e.g. `balancer.adasyn` for the default strategy.

    mapping: dict
        Target values mapped to their respective encoded integer.

    See Also
    --------
    atom.data_cleaning:Encoder
    atom.data_cleaning:Imputer
    atom.data_cleaning:Pruner

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.train)

             mean radius  mean texture  ...  worst fractal dimension  target
        0         18.030         16.85  ...                  0.08225       0
        1         10.950         21.35  ...                  0.09606       0
        2         14.250         22.15  ...                  0.11320       0
        3         17.570         15.05  ...                  0.07919       0
        4         10.600         18.95  ...                  0.07587       1
        ..           ...           ...  ...                      ...     ...
        451        8.888         14.64  ...                  0.10840       1
        452       21.090         26.57  ...                  0.12840       0
        453       16.160         21.54  ...                  0.07619       0
        454       11.260         19.83  ...                  0.07613       1
        455       12.000         15.65  ...                  0.07924       1

        [456 rows x 31 columns]

        >>> atom.balance(strategy="smote", verbose=2)

        Oversampling with SMOTE...
            --> Adding 116 samples to class 0.

        >>> # Note that the number of rows has increased
        >>> print(atom.train)

             mean radius  mean texture  ...  worst fractal dimension  target
        0      11.420000     20.380000  ...                 0.173000       0
        1       9.876000     17.270000  ...                 0.073800       1
        2      13.470000     14.060000  ...                 0.093260       1
        3      16.300000     15.700000  ...                 0.072300       1
        4      12.250000     17.940000  ...                 0.081320       1
        ..           ...           ...  ...                      ...     ...
        567    12.975558     20.580996  ...                 0.118509       0
        568    11.786135     17.120749  ...                 0.091266       0
        569    16.194544     19.737215  ...                 0.106434       0
        570    16.780524     21.261883  ...                 0.086889       0
        571    20.705316     22.635645  ...                 0.085362       0

        [572 rows x 31 columns]

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.data_cleaning import Balancer
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        >>> print(X)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0          17.99         10.38  ...          0.4601                  0.11890
        1          20.57         17.77  ...          0.2750                  0.08902
        2          19.69         21.25  ...          0.3613                  0.08758
        3          11.42         20.38  ...          0.6638                  0.17300
        4          20.29         14.34  ...          0.2364                  0.07678
        ..           ...           ...  ...             ...                      ...
        564        21.56         22.39  ...          0.2060                  0.07115
        565        20.13         28.25  ...          0.2572                  0.06637
        566        16.60         28.08  ...          0.2218                  0.07820
        567        20.60         29.33  ...          0.4087                  0.12400
        568         7.76         24.54  ...          0.2871                  0.07039

        [569 rows x 30 columns]

        >>> balancer = Balancer(strategy="smote", verbose=2)
        >>> X, y = balancer.transform(X, y)

        Oversampling with SMOTE...
            --> Adding 145 samples to class 0.

        >>> # Note that the number of rows has increased
        >>> print(X)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0      17.990000     10.380000  ...        0.460100                 0.118900
        1      20.570000     17.770000  ...        0.275000                 0.089020
        2      19.690000     21.250000  ...        0.361300                 0.087580
        3      11.420000     20.380000  ...        0.663800                 0.173000
        4      20.290000     14.340000  ...        0.236400                 0.076780
        ..           ...           ...  ...             ...                      ...
        709    14.824550     17.497674  ...        0.345200                 0.100678
        710    20.170649     23.997572  ...        0.538881                 0.099281
        711    21.006050     22.305044  ...        0.277181                 0.076740
        712    20.791828     25.103989  ...        0.388202                 0.122836
        713    17.081185     23.560768  ...        0.342508                 0.082558

        [714 rows x 30 columns]

        ```

    """

    _train_only = True

    @typechecked
    def __init__(
        self,
        strategy: Union[str, Any] = "ADASYN",
        *,
        n_jobs: INT = 1,
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
        random_state: Optional[INT] = None,
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
    def transform(self, X: X_TYPES, y: Y_TYPES = -1) -> Tuple[pd.DataFrame, pd.Series]:
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
        pd.DataFrame
            Balanced dataframe.

        pd.Series
            Transformed target column.

        """

        def log_changes(y):
            """Print the changes per target class."""
            for key, value in self.mapping.items():
                diff = counts[key] - np.sum(y == value)
                if diff > 0:
                    self.log(f" --> Removing {diff} samples from class {key}.", 2)
                elif diff < 0:
                    self.log(f" --> Adding {-diff} samples to class {key}.", 2)

        X, y = self._prepare_input(X, y)

        strategies = CustomDict(
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
            if self.strategy not in strategies:
                raise ValueError(
                    f"Invalid value for the strategy parameter, got {self.strategy}. "
                    f"Choose from: {', '.join(strategies)}."
                )
            estimator = strategies[self.strategy](**self.kwargs)
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
            self.mapping = {str(v): v for v in y.sort_values().unique()}
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


class Cleaner(BaseEstimator, TransformerMixin, BaseTransformer):
    """Applies standard data cleaning steps on a dataset.

    Use the parameters to choose which transformations to perform.
    The available steps are:

    - Drop columns with specific data types.
    - Strip categorical features from white spaces.
    - Drop duplicate rows.
    - Drop rows with missing values in the target column.
    - Encode the target column.

    This class can be accessed from atom through the [clean]
    [atomclassifier-clean] method. Read more in the [user guide]
    [standard-data-cleaning].

    Parameters
    ----------
    drop_types: str, sequence or None, default=None
        Columns with these data types are dropped from the dataset.

    strip_categorical: bool, default=True
        Whether to strip spaces from the categorical columns.

    drop_duplicates: bool, default=False
        Whether to drop duplicate rows. Only the first occurrence of
        every duplicated row is kept.

    drop_missing_target: bool, default=True
        Whether to drop rows with missing values in the target column.
        This transformation is ignored if `y` is not provided.

    encode_target: bool, default=True
        Whether to Label-encode the target column. This transformation
        is ignored if `y` is not provided.

    device: str, default="cpu"
        Device on which to train the estimators. Use any string
        that follows the [SYCL_DEVICE_FILTER][] filter selector,
        e.g. `device="gpu"` to use the GPU. Read more in the
        [user guide][accelerating-pipelines].

    engine: str, default="sklearn"
        Execution engine to use for the estimators. Refer to the
        [user guide][accelerating-pipelines] for an explanation
        regarding every choice. Choose from:

        - "sklearn" (only if device="cpu")
        - "cuml" (only if device="gpu")

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
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

    See Also
    --------
    atom.data_cleaning:Encoder
    atom.data_cleaning:Discretizer
    atom.data_cleaning:Scaler

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        >>> y = ["a" if i else "b" for i in y]

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.y)

        0      b
        1      b
        2      b
        3      b
        4      a
              ..
        995    b
        996    a
        997    a
        998    b
        999    b

        Name: target, Length: 1000, dtype: object

        >>> atom.clean(verbose=2)

        Fitting Cleaner...
        Cleaning the data...
         --> Label-encoding the target column.

        >>> print(atom.y)

        0      1
        1      1
        2      1
        3      1
        4      0
              ..
        995    1
        996    0
        997    0
        998    1
        999    1

        Name: target, Length: 1000, dtype: int32

        ```

    === "stand-alone"
        ```pycon
        >>> import numpy as np
        >>> from atom.data_cleaning import Cleaner

        >>> y = ["a" if i else "b" for i in np.randint(100)]

        >>> cleaner = Cleaner(verbose=2)
        >>> y = cleaner.fit_transform(y=y)

        Fitting Cleaner...
        Cleaning the data...
         --> Label-encoding the target column.

        >>> print(y)

        0     0
        1     0
        2     1
        3     0
        4     0
             ..
        95    1
        96    1
        97    0
        98    0
        99    0

        Name: target, Length: 100, dtype: int32

        ```

    """

    _train_only = False

    @typechecked
    def __init__(
        self,
        *,
        drop_types: Optional[Union[str, SEQUENCE_TYPES]] = None,
        strip_categorical: bool = True,
        drop_duplicates: bool = False,
        drop_missing_target: bool = True,
        encode_target: bool = True,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose, logger=logger)
        self.drop_types = drop_types
        self.strip_categorical = strip_categorical
        self.drop_duplicates = drop_duplicates
        self.drop_missing_target = drop_missing_target
        self.encode_target = encode_target

        self.mapping = {}
        self.missing = ["", "?", "NA", "nan", "NaN", "None", "inf"]
        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: Optional[X_TYPES] = None, y: Optional[Y_TYPES] = None) -> Cleaner:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        Cleaner
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        self.log("Fitting Cleaner...", 1)

        if y is not None and self.encode_target:
            if self.drop_missing_target:
                y = y.replace(self.missing + [np.inf, -np.inf], np.NaN).dropna()

            estimator = self._get_est_class("LabelEncoder", "preprocessing")
            self._estimator = estimator().fit(y)
            self.mapping = {
                str(it(v)): i for i, v in enumerate(self._estimator.classes_)
            }

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(
        self,
        X: Optional[X_TYPES] = None,
        y: Optional[Y_TYPES] = None,
    ) -> Union[pd.Series, pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Apply the data cleaning steps to the data.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        pd.DataFrame
            Transformed feature set. Only returned if provided.

        pd.Series
            Transformed target column. Only returned if provided.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Cleaning the data...", 1)

        if X is not None:
            # Replace all missing values with NaN
            X = X.replace(self.missing + [np.inf, -np.inf], np.NaN)

            for name, column in X.items():
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

            # Drop duplicate samples
            if self.drop_duplicates:
                X = X.drop_duplicates(ignore_index=True)

        if y is not None:
            # Delete samples with NaN in target
            if self.drop_missing_target:
                length = len(y)  # Save original length to count deleted rows later
                y = y.replace(self.missing + [np.inf, -np.inf], np.NaN).dropna()

                if X is not None:
                    X = X[X.index.isin(y.index)]  # Select only indices that remain

                diff = length - len(y)  # Difference in size
                if diff > 0:
                    self.log(
                        f" --> Dropping {diff} samples with "
                        "missing values in target column.", 2
                    )

            if self.encode_target and self._estimator:
                self.log(" --> Label-encoding the target column.", 2)
                y = to_series(self._estimator.transform(y), y.index, y.name)

        return variable_return(X, y)

    @composed(crash, method_to_log, typechecked)
    def inverse_transform(
        self,
        X: Optional[X_TYPES] = None,
        y: Optional[Y_TYPES] = None,
    ):
        """Inversely transform the label encoding.

        This method only inversely transforms the label encoding.
        The rest of the transformations can't be inverted. If
        `encode_target=False`, the data is returned as is.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        pd.DataFrame
            Unchanged feature set. Only returned if provided.

        pd.Series
            Original target column. Only returned if provided.

        """
        X, y = self._prepare_input(X, y)

        self.log("Inversely cleaning the data...", 1)

        if y is not None and self.encode_target:
            self.log(" --> Inversely label-encoding the target column.", 2)
            y = to_series(self._estimator.inverse_transform(y), y.index, y.name)

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
              The outermost edges are always `-inf` and `+inf`, e.g.
              bins `[1, 2]` indicate `(-inf, 1], (1, 2], (2, inf]`.
        - If dict: One of the aforementioned options per column, where
          the key is the column's name.

    labels: sequence, dict or None, default=None
        Label names with which to replace the binned intervals.

        - If None: Use default labels of the form `(min_edge, max_edge]`.
        - If sequence: Labels to use for all columns.
        - If dict: Labels per column, where the key is the column's name.

    device: str, default="cpu"
        Device on which to train the estimators. Use any string
        that follows the [SYCL_DEVICE_FILTER][] filter selector,
        e.g. `device="gpu"` to use the GPU. Read more in the
        [user guide][accelerating-pipelines].

    engine: str, default="sklearn"
        Execution engine to use for the estimators. Refer to the
        [user guide][accelerating-pipelines] for an explanation
        regarding every choice. Choose from:

        - "sklearn" (only if device="cpu")
        - "cuml" (only if device="gpu")

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`. Only
        for strategy="quantile".

    Attributes
    ----------
    feature_names_in_: np.array
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
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom["mean radius"])

        0      17.99
        1      20.57
        2      19.69
        3      11.42
        4      20.29
               ...
        564    21.56
        565    20.13
        566    16.60
        567    20.60
        568     7.76

        Name: mean radius, Length: 569, dtype: float64

        >>> atom.discretize(
        ...     strategy="custom",
        ...     bins=[13, 18],
        ...     labels=["small", "medium", "large"],
        ...     verbose=2,
        ...     columns="mean radius",
        ... )

        Fitting Discretizer...
        Binning the features...
         --> Discretizing feature mean radius in 3 bins.

        >>> print(atom["mean radius"])

        0       small
        1      medium
        2      medium
        3      medium
        4       small
                ...
        564     large
        565     small
        566     large
        567     small
        568     small

        Name: mean radius, Length: 569, dtype: category
        Categories (3, object): ['small' < 'medium' < 'large']

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.data_cleaning import Discretizer
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        >>> print(X["mean radius"])

        0      17.99
        1      20.57
        2      19.69
        3      11.42
        4      20.29
               ...
        564    21.56
        565    20.13
        566    16.60
        567    20.60
        568     7.76

        Name: mean radius, Length: 569, dtype: float64

        >>> disc = Discretizer(
        ...     strategy="custom",
        ...     bins=[13, 18],
        ...     labels=["small", "medium", "large"],
        ...     verbose=2,
        ... )
        >>> X["mean radius"] = disc.fit_transform(X[["mean radius"]])["mean radius"]

        Fitting Discretizer...
        Binning the features...
         --> Discretizing feature mean radius in 3 bins.

        >>> print(X["mean radius"])

        0       small
        1      medium
        2      medium
        3      medium
        4       small
                ...
        564     large
        565     small
        566     large
        567     small
        568     small

        Name: mean radius, Length: 569, dtype: category
        Categories (3, object): ['small' < 'medium' < 'large']

        ```

    """

    _train_only = False

    @typechecked
    def __init__(
        self,
        strategy: str = "quantile",
        *,
        bins: Union[INT, SEQUENCE_TYPES, dict] = 5,
        labels: Optional[Union[SEQUENCE_TYPES, dict]] = None,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
        random_state: Optional[INT] = None,
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

        self._num_cols = None
        self._discretizers = {}
        self._labels = {}
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> Discretizer:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
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
        self._num_cols = list(X.select_dtypes(include="number").columns)

        if self.strategy.lower() not in ("uniform", "quantile", "kmeans", "custom"):
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                "Choose from: uniform, quantile, kmeans, custom."
            )

        self.log("Fitting Discretizer...", 1)

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

            if self.strategy.lower() != "custom":
                if isinstance(bins, SEQUENCE):
                    try:
                        bins = bins[i]  # Fetch the i-th bin for the i-th column
                    except IndexError:
                        raise ValueError(
                            "Invalid value for the bins parameter. The length of "
                            "the bins does not match the length of the columns, got "
                            f"len(bins)={len(bins) } and len(columns)={len(X.columns)}."
                        )

                estimator = self._get_est_class("KBinsDiscretizer", "preprocessing")
                self._discretizers[col] = estimator(
                    n_bins=bins,
                    encode="ordinal",
                    strategy=self.strategy.lower(),
                    random_state=self.random_state,
                ).fit(X[[col]])

                # Save labels for transform method
                self._labels[col] = get_labels(
                    labels=labels,
                    bins=self._discretizers[col].bin_edges_[0],
                )

            else:
                if not isinstance(bins, SEQUENCE):
                    raise TypeError(
                        f"Invalid type for the bins parameter, got {bins}. Only "
                        "a sequence of bin edges is accepted when strategy='custom'."
                    )
                else:
                    bins = [-np.inf] + list(bins) + [np.inf]

                # Make of pd.cut a transformer
                self._discretizers[col] = FunctionTransformer(
                    func=pd.cut,
                    kw_args={"bins": bins, "labels": get_labels(labels, bins)},
                ).fit(X[[col]])

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> pd.DataFrame:
        """Bin the data into intervals.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Transformed feature set.

        """
        X, y = self._prepare_input(X, y)

        self.log("Binning the features...", 1)

        for col in self._num_cols:
            if self.strategy.lower() == "custom":
                X[col] = self._discretizers[col].transform(X[col])
            else:
                X[col] = self._discretizers[col].transform(X[[col]])[:, 0]

                # Replace cluster values with labels
                for i, label in enumerate(self._labels[col]):
                    X[col] = X[col].replace(i, label)

            self.log(f" --> Discretizing feature {col} in {X[col].nunique()} bins.", 2)

        return X


class Encoder(BaseEstimator, TransformerMixin, BaseTransformer):
    """Perform encoding of categorical features.

    The encoding type depends on the number of classes in the column:

    - If n_classes=2 or ordinal feature, use Ordinal-encoding.
    - If 2 < n_classes <= `max_onehot`, use OneHot-encoding.
    - If n_classes > `max_onehot`, use `strategy`-encoding.

    Missing values are propagated to the output column. Unknown
    classes encountered during transforming are imputed according
    to the selected strategy. Rare classes can be replaced with a
    value in order to prevent too high cardinality.

    This class can be accessed from atom through the [encode]
    [atomclassifier-encode] method. Read more in the [user guide]
    [encoding-categorical-features].

    !!! warning
        Two category-encoders estimators are unavailable:

        * [OneHotEncoder][]: Use the max_onehot parameter.
        * [HashingEncoder][]: Incompatibility of APIs.

    Parameters
    ----------
    strategy: str or estimator, default="LeaveOneOut"
        Type of encoding to use for high cardinality features. Choose
        from any of the estimators in the category-encoders package
        or provide a custom one.

    max_onehot: int or None, default=10
        Maximum number of unique values in a feature to perform
        one-hot encoding. If None, `strategy`-encoding is always
        used for columns with more than two classes.

    ordinal: dict or None, default=None
        Order of ordinal features, where the dict key is the feature's
        name and the value is the class order, e.g. `{"salary": ["low",
        "medium", "high"]}`.

    rare_to_value: int, float or None, default=None
        Replaces rare class occurrences in categorical columns with the
        string in parameter `value`. This transformation is done before
        the encoding of the column.

        - If None: Skip this step.
        - If int: Minimum number of occurrences in a class.
        - If float: Minimum fraction of occurrences in a class.

    value: str, default="rare"
        Value with which to replace rare classes. This parameter is
        ignored if `rare_to_value=None`.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    mapping: dict of dicts
        Encoded values and their respective mapping. The column name is
        the key to its mapping dictionary. Only for columns mapped to a
        single column (e.g. Ordinal, Leave-one-out, etc...).

    feature_names_in_: np.array
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
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer
        >>> from numpy.random import randint

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        >>> X["cat_feature_1"] = [f"x{i}" for i in randint(0, 2, len(X))]
        >>> X["cat_feature_2"] = [f"x{i}" for i in randint(0, 3, len(X))]
        >>> X["cat_feature_3"] = [f"x{i}" for i in randint(0, 20, len(X))]

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.X)

             mean radius  mean texture  ...  cat_feature_2  cat_feature_3
        0          13.62         23.23  ...             x0             x0
        1          14.86         16.94  ...             x0             x5
        2          16.74         21.59  ...             x2            x15
        3          13.37         16.39  ...             x1            x18
        4          11.37         18.89  ...             x0            x13
        ..           ...           ...  ...            ...            ...
        564        14.06         17.18  ...             x2             x1
        565        11.29         13.04  ...             x0            x10
        566        14.26         19.65  ...             x0             x5
        567        12.05         14.63  ...             x2            x14
        568        18.81         19.98  ...             x1            x13

        [569 rows x 33 columns]

        >>> atom.encode(strategy="leaveoneout", max_onehot=10, verbose=2)

        Fitting Encoder...
        Encoding categorical columns...
         --> Ordinal-encoding feature cat_feature_1. Contains 2 classes.
         --> OneHot-encoding feature cat_feature_2. Contains 3 classes.
         --> LeaveOneOut-encoding feature cat_feature_3. Contains 20 classes.

        >>> # Note the one-hot encoded column with name [feature]_[class]
        >>> print(atom.X)

             mean radius  mean texture  ...  cat_feature_2_x2  cat_feature_3
        0          13.62         23.23  ...               0.0       0.714286
        1          14.86         16.94  ...               0.0       0.555556
        2          16.74         21.59  ...               1.0       0.681818
        3          13.37         16.39  ...               0.0       0.739130
        4          11.37         18.89  ...               0.0       0.521739
        ..           ...           ...  ...               ...            ...
        564        14.06         17.18  ...               1.0       0.772727
        565        11.29         13.04  ...               0.0       0.766667
        566        14.26         19.65  ...               0.0       0.555556
        567        12.05         14.63  ...               1.0       0.411765
        568        18.81         19.98  ...               0.0       0.521739

        [569 rows x 35 columns]

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.data_cleaning import Encoder
        >>> from sklearn.datasets import load_breast_cancer
        >>> from numpy.random import randint

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        >>> X["cat_feature_1"] = [f"x{i}" for i in randint(0, 2, len(X))]
        >>> X["cat_feature_2"] = [f"x{i}" for i in randint(0, 3, len(X))]
        >>> X["cat_feature_3"] = [f"x{i}" for i in randint(0, 20, len(X))]
        >>> print(X)

             mean radius  mean texture  ...  cat_feature_2  cat_feature_3
        0          13.62         23.23  ...             x0             x0
        1          14.86         16.94  ...             x0             x5
        2          16.74         21.59  ...             x2            x15
        3          13.37         16.39  ...             x1            x18
        4          11.37         18.89  ...             x0            x13
        ..           ...           ...  ...            ...            ...
        564        14.06         17.18  ...             x2             x1
        565        11.29         13.04  ...             x0            x10
        566        14.26         19.65  ...             x0             x5
        567        12.05         14.63  ...             x2            x14
        568        18.81         19.98  ...             x1            x13

        [569 rows x 33 columns]

        >>> encoder = Encoder(strategy="leaveoneout", max_onehot=10, verbose=2)
        >>> X = encoder.fit_transform(X, y)

        Fitting Encoder...
        Encoding categorical columns...
         --> Ordinal-encoding feature cat_feature_1. Contains 2 classes.
         --> OneHot-encoding feature cat_feature_2. Contains 3 classes.
         --> LeaveOneOut-encoding feature cat_feature_3. Contains 20 classes.

        >>> # Note the one-hot encoded column with name [feature]_[class]
        >>> print(X)

             mean radius  mean texture  ...  cat_feature_2_x2  cat_feature_3
        0          17.99         10.38  ...               1.0       0.379310
        1          20.57         17.77  ...               1.0       0.714286
        2          19.69         21.25  ...               0.0       0.586207
        3          11.42         20.38  ...               0.0       0.678571
        4          20.29         14.34  ...               0.0       0.714286
        ..           ...           ...  ...               ...            ...
        564        21.56         22.39  ...               0.0       0.580645
        565        20.13         28.25  ...               0.0       0.518519
        566        16.60         28.08  ...               1.0       0.600000
        567        20.60         29.33  ...               1.0       0.586207
        568         7.76         24.54  ...               1.0       0.678571

        [569 rows x 35 columns]

        ```

    """

    _train_only = False

    @typechecked
    def __init__(
        self,
        strategy: Union[str, Any] = "LeaveOneOut",
        *,
        max_onehot: Optional[INT] = 10,
        ordinal: Optional[Dict[str, SEQUENCE_TYPES]] = None,
        rare_to_value: Optional[SCALAR] = None,
        value: str = "rare",
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
        **kwargs,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strategy = strategy
        self.max_onehot = max_onehot
        self.ordinal = ordinal
        self.rare_to_value = rare_to_value
        self.value = value
        self.kwargs = kwargs

        self.mapping = {}
        self._cat_cols = None
        self._max_onehot = None
        self._rare_to_value = None
        self._to_other = defaultdict(list)
        self._categories = {}
        self._encoders = {}
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Y_TYPES = None) -> Encoder:
        """Fit to data.

        Note that leaving y=None can lead to errors if the `strategy`
        encoder requires target values.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        Encoder
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        self._cat_cols = list(X.select_dtypes(exclude="number").columns)

        strategies = CustomDict(
            backwarddifference=BackwardDifferenceEncoder,
            basen=BaseNEncoder,
            binary=BinaryEncoder,
            catboost=CatBoostEncoder,
            helmert=HelmertEncoder,
            jamesstein=JamesSteinEncoder,
            leaveoneout=LeaveOneOutEncoder,
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
            if self.strategy not in strategies:
                raise ValueError(
                    f"Invalid value for the strategy parameter, got {self.strategy}. "
                    f"Choose from: {', '.join(strategies)}."
                )
            estimator = strategies[self.strategy](
                handle_missing="return_nan",
                handle_unknown="value",
                **self.kwargs,
            )
        elif not all(hasattr(self.strategy, attr) for attr in ("fit", "transform")):
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

        if self.rare_to_value:
            if self.rare_to_value < 0:
                raise ValueError(
                    "Invalid value for the rare_to_value parameter. "
                    f"Value should be >0, got {self.rare_to_value}."
                )
            elif self.rare_to_value < 1:
                self._rare_to_value = int(self.rare_to_value * len(X))
            else:
                self._rare_to_value = self.rare_to_value

        self.log("Fitting Encoder...", 1)

        # Reset internal attrs in case of repeated fit
        self.mapping = {}
        self._to_other = defaultdict(list)
        self._categories, self._encoders = {}, {}

        for name, column in X[self._cat_cols].items():
            # Replace rare classes with the string "other"
            if self._rare_to_value:
                for category, count in column.value_counts().items():
                    if count <= self._rare_to_value:
                        self._to_other[name].append(category)
                        X[name] = column.replace(category, self.value)

            # Get the unique categories before fitting
            self._categories[name] = column.sort_values().unique().tolist()

            # Perform encoding type dependent on number of unique values
            ordinal = self.ordinal or {}
            if name in ordinal or len(self._categories[name]) == 2:

                # Check that provided classes match those of column
                ordinal = ordinal.get(name, self._categories[name])
                if column.nunique(dropna=True) != len(ordinal):
                    self.log(
                        f" --> The number of classes passed to feature {name} in the "
                        f"ordinal parameter ({len(ordinal)}) don't match the number "
                        f"of classes in the data ({column.nunique(dropna=True)}).", 1,
                        severity="warning"
                    )

                # Create custom mapping from 0 to N - 1
                mapping = {v: i for i, v in enumerate(ordinal)}
                mapping.setdefault(np.NaN, -1)  # Encoder always needs mapping of NaN
                self.mapping[name] = mapping

                self._encoders[name] = OrdinalEncoder(
                    mapping=[{"col": name, "mapping": mapping}],
                    cols=[name],  # Specify to not skip bool columns
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

                # Create encoding of unique values for mapping
                data = self._encoders[name].transform(
                    pd.Series(
                        data=self._categories[name],
                        index=self._categories[name],
                        name=name,
                        dtype="object",
                    )
                )

                # Only mapping 1 - 1 column
                if data.shape[1] == 1:
                    self.mapping[name] = {}
                    for idx, value in data[name].items():
                        self.mapping[name][idx] = value

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> pd.DataFrame:
        """Encode the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
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
            # Convert uncommon classes to "other"
            if self._to_other[name]:
                X[name] = column.replace(self._to_other[name], self.value)

            n_classes = len(column.unique())
            self.log(
                f" --> {self._encoders[name].__class__.__name__[:-7]}-encoding "
                f"feature {name}. Contains {n_classes} classes.", 2
            )

            # Count the propagated missing values
            n_nans = column.isna().sum()
            if n_nans:
                self.log(f"   --> Propagating {n_nans} missing values.", 2)

            # Get the new encoded columns
            new_cols = self._encoders[name].transform(X[[name]])

            # Drop _nan columns (since missing values are propagated)
            new_cols = new_cols[[col for col in new_cols if not col.endswith("_nan")]]

            # Check for unknown classes
            uc = len([i for i in column.unique() if i not in self._categories[name]])
            if uc:
                self.log(f"   --> Handling {uc} unknown classes.", 2)

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
        - "most_frequent": Impute with most frequent value.
        - int or float: Impute with provided numerical value.

    strat_cat: str, default="drop"
        Imputing strategy for categorical columns. Choose from:

        - "drop": Drop rows containing missing values.
        - "most_frequent": Impute with most frequent value.
        - str: Impute with provided string.

    max_nan_rows: int, float or None, default=None
        Maximum number or fraction of missing values in a row
        (if more, the row is removed). If None, ignore this step.

    max_nan_cols: int, float or None, default=None
        Maximum number or fraction of missing values in a column
        (if more, the column is removed). If None, ignore this step.

    device: str, default="cpu"
        Device on which to train the estimators. Use any string
        that follows the [SYCL_DEVICE_FILTER][] filter selector,
        e.g. `device="gpu"` to use the GPU. Read more in the
        [user guide][accelerating-pipelines].

    engine: str, default="sklearn"
        Execution engine to use for the estimators. Refer to the
        [user guide][accelerating-pipelines] for an explanation
        regarding every choice. Choose from:

        - "sklearn" (only if device="cpu")
        - "cuml" (only if device="gpu")

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
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

    feature_names_in_: np.array
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
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer
        >>> from numpy.random import randint

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> # Add some random missing values to the data
        >>> for i, j in zip(randint(0, X.shape[0], 600), randint(0, 4, 600])
        >>> X.iat[i, j] = np.nan

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.nans)

        mean radius       118
        mean texture      134
        mean perimeter    135
        mean area         140

        dtype: int64

        >>> atom.impute(strat_num="median", max_nan_rows=0.1, verbose=2)

        Fitting Imputer...
        Imputing missing values...
         --> Dropping 3 samples for containing more than 3 missing values.
         --> Imputing 115 missing values with median (13.3) in feature mean radius.
         --> Imputing 131 missing values with median (18.8) in feature mean texture.
         --> Imputing 132 missing values with median (85.86) in feature mean perimeter.
         --> Imputing 137 missing values with median (561.3) in feature mean area.

        >>> print(atom.n_nans)

        0

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.data_cleaning import Imputer
        >>> from sklearn.datasets import load_breast_cancer
        >>> from numpy.random import randint

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> # Add some random missing values to the data
        >>> for i, j in zip(randint(0, X.shape[0], 600), randint(0, 4, 600])
        >>> X.iloc[i, j] = np.nan

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0          17.99           NaN  ...          0.4601                  0.11890
        1          20.57         17.77  ...          0.2750                  0.08902
        2          19.69         21.25  ...          0.3613                  0.08758
        3            NaN         20.38  ...          0.6638                  0.17300
        4            NaN         14.34  ...          0.2364                  0.07678
        ..           ...           ...  ...             ...                      ...
        564          NaN         22.39  ...          0.2060                  0.07115
        565        20.13         28.25  ...          0.2572                  0.06637
        566          NaN           NaN  ...          0.2218                  0.07820
        567          NaN         29.33  ...          0.4087                  0.12400
        568          NaN         24.54  ...          0.2871                  0.07039

        [569 rows x 30 columns]

        >>> imputer = Imputer(strat_num="median", max_nan_rows=0.1, verbose=2)
        >>> X, y = imputer.fit_transform(X, y)

        Fitting Imputer...
        Imputing missing values...
         --> Imputing 135 missing values with median (13.42) in feature mean radius.
         --> Imputing 133 missing values with median (18.81) in feature mean texture.
         --> Imputing 129 missing values with median (86.14) in feature mean perimeter.
         --> Imputing 120 missing values with median (537.9) in feature mean area.

        >>> print(X)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0         17.990         10.38  ...          0.4601                  0.11890
        1         13.415         17.77  ...          0.2750                  0.08902
        2         19.690         21.25  ...          0.3613                  0.08758
        3         11.420         20.38  ...          0.6638                  0.17300
        4         20.290         14.34  ...          0.2364                  0.07678
        ..           ...           ...  ...             ...                      ...
        564       21.560         22.39  ...          0.2060                  0.07115
        565       20.130         28.25  ...          0.2572                  0.06637
        566       13.415         28.08  ...          0.2218                  0.07820
        567       13.415         18.81  ...          0.4087                  0.12400
        568        7.760         24.54  ...          0.2871                  0.07039

        [569 rows x 30 columns]

        ```

    """

    _train_only = False

    @typechecked
    def __init__(
        self,
        strat_num: Union[SCALAR, str] = "drop",
        strat_cat: str = "drop",
        *,
        max_nan_rows: Optional[SCALAR] = None,
        max_nan_cols: Optional[Union[FLOAT]] = None,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose, logger=logger)
        self.strat_num = strat_num
        self.strat_cat = strat_cat
        self.max_nan_rows = max_nan_rows
        self.max_nan_cols = max_nan_cols

        self.missing = ["", "?", "None", "NA", "nan", "NaN", "inf"]
        self._max_nan_rows = None
        self._drop_cols = []
        self._imputers = {}
        self._num_cols = []
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> Imputer:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Imputer
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        self._num_cols = list(X.select_dtypes(include="number").columns)

        # Check input Parameters
        strategies = ["drop", "mean", "median", "knn", "most_frequent"]
        if isinstance(self.strat_num, str) and self.strat_num.lower() not in strategies:
            raise ValueError(
                "Unknown strategy for the strat_num parameter, got "
                f"{self.strat_num}. Choose from: {', '.join(strategies)}."
            )
        if self.max_nan_rows:
            if self.max_nan_rows < 0:
                raise ValueError(
                    "Invalid value for the max_nan_rows parameter. "
                    f"Value should be >0, got {self.max_nan_rows}."
                )
            elif self.max_nan_rows <= 1:
                self._max_nan_rows = int(X.shape[1] * self.max_nan_rows)
            else:
                self._max_nan_rows = self.max_nan_rows

        if self.max_nan_cols:
            if self.max_nan_cols < 0:
                raise ValueError(
                    "Invalid value for the max_nan_cols parameter. "
                    f"Value should be >0, got {self.max_nan_cols}."
                )
            elif self.max_nan_cols <= 1:
                max_nan_cols = int(X.shape[0] * self.max_nan_cols)
            else:
                max_nan_cols = self.max_nan_cols

        self.log("Fitting Imputer...", 1)

        # Replace all missing values with NaN
        X = X.replace(self.missing + [np.inf, -np.inf], np.NaN)

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
        estimator = self._get_est_class(
            name="SimpleImputer",
            module="preprocessing" if self.engine == "cuml" else "impute",
        )

        # Assign an imputer to each column
        for name, column in X.items():
            # Remember columns with too many missing values
            if self.max_nan_cols and column.isna().sum() > max_nan_cols:
                self._drop_cols.append(name)
                continue

            # Column is numerical
            if name in self._num_cols:
                if isinstance(self.strat_num, str):
                    if self.strat_num.lower() == "knn":
                        self._imputers[name] = KNNImputer().fit(X[[name]])

                    elif self.strat_num.lower() == "most_frequent":
                        self._imputers[name] = estimator(
                            strategy="most_frequent",
                        ).fit(X[[name]])

                    # Strategies mean or median
                    elif self.strat_num.lower() != "drop":
                        self._imputers[name] = estimator(
                            strategy=self.strat_num.lower()
                        ).fit(X[[name]])

            # Column is categorical
            elif self.strat_cat.lower() == "most_frequent":
                self._imputers[name] = estimator(
                    strategy="most_frequent",
                ).fit(X[[name]])

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(
        self,
        X: X_TYPES,
        y: Optional[Y_TYPES] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Impute the missing values.

        Note that leaving y=None can lead to inconsistencies in
        data length between X and y if rows are dropped during
        the transformation.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        pd.Series
            Transformed target column. Only returned if provided.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Imputing missing values...", 1)

        # Replace all missing values with NaN
        X = X.replace(self.missing + [np.inf, -np.inf], np.NaN)

        # Drop rows with too many missing values
        if self._max_nan_rows is not None:
            length = len(X)
            X = X.dropna(axis=0, thresh=X.shape[1] - self._max_nan_rows)
            if diff := length - len(X):
                if y is not None:
                    y = y[y.index.isin(X.index)]  # Select only indices that remain
                self.log(
                    f" --> Dropping {diff} samples for containing "
                    f"more than {self._max_nan_rows} missing values.", 2
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
                    n = np.round(self._imputers[name].statistics_[0], 2)
                    self.log(
                        f" --> Imputing {nans} missing values with "
                        f"{self.strat_num.lower()} ({n}) in feature {name}.", 2
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
        transforming. Use the `kwargs` to change this behaviour.

    Parameters
    ----------
    strategy: str, default="yeojohnson"
        The transforming strategy. Choose from:

        - "[yeojohnson][]"
        - "[boxcox][]" (only works with strictly positive values)
        - "[quantile][]": Transform features using quantiles information.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    random_state: int or None, default=None
        Seed used by the quantile strategy. If None, the random
        number generator is the `RandomState` used by `np.random`.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]: sklearn transformer
        Object with which the data is transformed.

    feature_names_in_: np.array
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.
        `NaN`, `+inf` and `-inf` are always considered missing since
        they are incompatible with sklearn estimators.

    See Also
    --------
    atom.data_cleaning:Cleaner
    atom.data_cleaning:Pruner
    atom.data_cleaning:Scaler

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.dataset)

             mean radius  mean texture  ...  worst fractal dimension  target
        0          16.78         18.80  ...                  0.07228       0
        1          15.34         14.26  ...                  0.09946       0
        2          14.22         27.85  ...                  0.07796       1
        3          18.31         18.58  ...                  0.06938       0
        4          18.49         17.52  ...                  0.09445       0
        ..           ...           ...  ...                      ...     ...
        564        13.44         21.58  ...                  0.07146       0
        565        20.47         20.67  ...                  0.06386       0
        566        12.98         19.35  ...                  0.09166       1
        567        14.61         15.69  ...                  0.05695       1
        568        23.27         22.04  ...                  0.09187       0

        [569 rows x 31 columns]

        >>> atom.plot_distribution(columns=0)

        ```
        ![plot_distribution_1](../../img/plots/plot_distribution_1.png)

        ```pycon

        >>> atom.normalize(verbose=2)

        Fitting Normalizer...
        Normalizing features...

        >>> print(atom.dataset)

             mean radius  mean texture  ...  worst fractal dimension  target
        0       0.868700      0.010820  ...                -0.684572       0
        1       0.513904     -1.257343  ...                 1.019875       0
        2       0.200435      1.773390  ...                -0.226619       1
        3       1.197448     -0.042755  ...                -0.945047       0
        4       1.233326     -0.310726  ...                 0.786014       0
        ..           ...           ...  ...                      ...     ...
        564    -0.041166      0.635293  ...                -0.756291       0
        565     1.595052      0.440855  ...                -1.497202       0
        566    -0.193933      0.141884  ...                 0.642613       1
        567     0.313768     -0.816597  ...                -2.307746       1
        568     2.022355      0.730259  ...                 0.653756       0

        [569 rows x 31 columns]


        >>> atom.plot_distribution(columns=0)

        ```

        ![plot_distribution_1](../../img/plots/plot_distribution_2.png)

    === "stand-alone"
        ```pycon
        >>> from atom.data_cleaning import Normalizer
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0          17.99         10.38  ...          0.4601                  0.11890
        1          20.57         17.77  ...          0.2750                  0.08902
        2          19.69         21.25  ...          0.3613                  0.08758
        3          11.42         20.38  ...          0.6638                  0.17300
        4          20.29         14.34  ...          0.2364                  0.07678
        ..           ...           ...  ...             ...                      ...
        564        21.56         22.39  ...          0.2060                  0.07115
        565        20.13         28.25  ...          0.2572                  0.06637
        566        16.60         28.08  ...          0.2218                  0.07820
        567        20.60         29.33  ...          0.4087                  0.12400
        568         7.76         24.54  ...          0.2871                  0.07039

        [569 rows x 30 columns]

        >>> normalizer = Normalizer(verbose=2)
        >>> X = normalizer.fit_transform(X)

        Fitting Normalizer...
        Normalizing features...

        >>> print(X)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0       1.134881     -2.678666  ...        2.197206                 1.723624
        1       1.619346     -0.264377  ...       -0.121997                 0.537179
        2       1.464796      0.547806  ...        1.218181                 0.453955
        3      -0.759262      0.357721  ...        3.250202                 2.517606
        4       1.571260     -1.233520  ...       -0.943554                -0.279402
        ..           ...           ...  ...             ...                      ...
        564     1.781795      0.785604  ...       -1.721528                -0.751459
        565     1.543335      1.845150  ...       -0.480093                -1.210527
        566     0.828589      1.817618  ...       -1.301164                -0.170872
        567     1.624440      2.016299  ...        1.744693                 1.850944
        568    -2.699432      1.203224  ...        0.103122                -0.820663

        [569 rows x 30 columns]

        ```

    """

    _train_only = False

    @typechecked
    def __init__(
        self,
        strategy: str = "yeojohnson",
        *,
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
        random_state: Optional[INT] = None,
        **kwargs,
    ):
        super().__init__(verbose=verbose, logger=logger, random_state=random_state)
        self.strategy = strategy
        self.kwargs = kwargs

        self._num_cols = None
        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> Normalizer:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Normalizer
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        self._num_cols = list(X.select_dtypes(include="number").columns)

        kwargs = self.kwargs.copy()
        if self.strategy.lower() in ("yeojohnson", "boxcox"):
            self._estimator = PowerTransformer(
                method=self.strategy.lower()[:3] + "-" + self.strategy.lower()[3:],
                **kwargs,
            )
        elif self.strategy.lower() == "quantile":
            self._estimator = QuantileTransformer(
                output_distribution=kwargs.pop("output_distribution", "normal"),
                random_state=kwargs.pop("random_state", self.random_state),
                **kwargs,
            )
        else:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                "Choose from: yeojohnson, boxcox, quantile."
            )

        self.log("Fitting Normalizer...", 1)
        self._estimator.fit(X[self._num_cols])

        # Add the estimator as attribute to the instance
        setattr(self, self.strategy.lower(), self._estimator)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> pd.DataFrame:
        """Apply the transformations to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Normalized dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Normalizing features...", 1)
        X_transformed = self._estimator.transform(X[self._num_cols])

        # If all columns were transformed, just swap sets
        if len(self._num_cols) != X.shape[1]:
            # Replace the numerical columns with the transformed values
            for i, col in enumerate(self._num_cols):
                X[col] = X_transformed[:, i]
        else:
            X = to_df(X_transformed, X.index, X.columns)

        return X

    @composed(crash, method_to_log, typechecked)
    def inverse_transform(
        self, X: X_TYPES, y: Optional[Y_TYPES] = None
    ) -> pd.DataFrame:
        """Apply the inverse transformation to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Original dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Inversely normalizing features...", 1)
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
        - "[optics][]": DBSCAN-like clustering approach.

    method: int, float or str, default="drop"
        Method to apply on the outliers. Only the zscore strategy
        accepts another method than "drop". Choose from:

        - "drop": Drop any sample with outlier values.
        - "min_max": Replace outlier with the min/max of the column.
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
        Device on which to train the estimators. Use any string
        that follows the [SYCL_DEVICE_FILTER][] filter selector,
        e.g. `device="gpu"` to use the GPU. Read more in the
        [user guide][accelerating-pipelines].

    engine: str, default="sklearn"
        Execution engine to use for the estimators. Refer to the
        [user guide][accelerating-pipelines] for an explanation
        regarding every choice. Choose from:

        - "sklearn" (only if device="cpu")
        - "sklearnex"
        - "cuml" (only if device="gpu")

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator. If
        sequence of strategies, the params should be provided in a dict
        with the strategy's name as key.

    Attributes
    ----------
    [strategy]: sklearn estimator
        Object used to prune the data, e.g. `pruner.iforest` for the
        isolation forest strategy.

    See Also
    --------
    atom.data_cleaning:Balancer
    atom.data_cleaning:Normalizer
    atom.data_cleaning:Scaler

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.dataset)

             mean radius  mean texture  ...  worst fractal dimension  target
        0          11.04         14.93  ...                  0.07287       1
        1          12.46         24.04  ...                  0.20750       0
        2          13.47         14.06  ...                  0.09326       1
        3          13.44         21.58  ...                  0.07146       0
        4          11.93         21.53  ...                  0.08541       1
        ..           ...           ...  ...                      ...     ...
        564        14.54         27.54  ...                  0.13410       0
        565        18.66         17.12  ...                  0.08456       0
        566        10.95         21.35  ...                  0.09606       0
        567        17.01         20.26  ...                  0.06469       0
        568        12.40         17.68  ...                  0.09359       1

        [569 rows x 31 columns]

        >>> atom.prune(stratgey="iforest", verbose=2)

        Pruning outliers...
         --> Dropping 46 outliers.

        >>> # Note the reduced number of rows
        >>> print(atom.dataset)

             mean radius  mean texture  ...  worst fractal dimension  target
        0          11.04         14.93  ...                  0.07287       1
        1          13.47         14.06  ...                  0.09326       1
        2          13.44         21.58  ...                  0.07146       0
        3          11.93         21.53  ...                  0.08541       1
        4          13.21         25.25  ...                  0.06788       1
        ..           ...           ...  ...                      ...     ...
        518        14.54         27.54  ...                  0.13410       0
        519        18.66         17.12  ...                  0.08456       0
        520        10.95         21.35  ...                  0.09606       0
        521        17.01         20.26  ...                  0.06469       0
        522        12.40         17.68  ...                  0.09359       1

        [523 rows x 31 columns]


        >>> atom.plot_distribution(columns=0)

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.data_cleaning import Normalizer
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0          17.99         10.38  ...          0.4601                  0.11890
        1          20.57         17.77  ...          0.2750                  0.08902
        2          19.69         21.25  ...          0.3613                  0.08758
        3          11.42         20.38  ...          0.6638                  0.17300
        4          20.29         14.34  ...          0.2364                  0.07678
        ..           ...           ...  ...             ...                      ...
        564        21.56         22.39  ...          0.2060                  0.07115
        565        20.13         28.25  ...          0.2572                  0.06637
        566        16.60         28.08  ...          0.2218                  0.07820
        567        20.60         29.33  ...          0.4087                  0.12400
        568         7.76         24.54  ...          0.2871                  0.07039

        [569 rows x 30 columns]

        >>> normalizer = Normalizer(verbose=2)
        >>> X = normalizer.fit_transform(X)

        Fitting Pruner...
        Pruning outliers...
         --> Dropping 74 outliers.

        >>> # Note the reduced number of rows
        >>> print(X)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        1          20.57         17.77  ...          0.2750                  0.08902
        2          19.69         21.25  ...          0.3613                  0.08758
        4          20.29         14.34  ...          0.2364                  0.07678
        5          12.45         15.70  ...          0.3985                  0.12440
        6          18.25         19.98  ...          0.3063                  0.08368
        ..           ...           ...  ...             ...                      ...
        560        14.05         27.15  ...          0.2250                  0.08321
        563        20.92         25.09  ...          0.2929                  0.09873
        564        21.56         22.39  ...          0.2060                  0.07115
        565        20.13         28.25  ...          0.2572                  0.06637
        566        16.60         28.08  ...          0.2218                  0.07820

        [495 rows x 30 columns]


        ```

    """

    _train_only = True

    @typechecked
    def __init__(
        self,
        strategy: Union[str, SEQUENCE_TYPES] = "zscore",
        *,
        method: Union[SCALAR, str] = "drop",
        max_sigma: SCALAR = 3,
        include_target: bool = False,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose, logger=logger)
        self.strategy = strategy
        self.method = method
        self.max_sigma = max_sigma
        self.include_target = include_target
        self.kwargs = kwargs

        self._is_fitted = True

    @composed(crash, method_to_log, typechecked)
    def transform(
        self, X: X_TYPES, y: Optional[Y_TYPES] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Apply the outlier strategy on the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        pd.DataFrame
            Transformed feature set.

        pd.Series
            Transformed target column. Only returned if provided.

        """
        X, y = self._prepare_input(X, y)

        strategies = CustomDict(
            iforest=["IsolationForest", "ensemble"],
            ee=["EllipticEnvelope", "covariance"],
            lof=["LocalOutlierFactor", "neighbors"],
            svm=["OneClassSVM", "svm"],
            dbscan=["DBSCAN", "cluster"],
            optics=["OPTICS", "cluster"],
        )

        for strat in lst(self.strategy):
            if strat.lower() not in ["zscore"] + list(strategies):
                raise ValueError(
                    "Invalid value for the strategy parameter. "
                    f"Choose from: zscore, {', '.join(strategies)}."
                )
            if str(self.method).lower() != "drop" and strat.lower() != "zscore":
                raise ValueError(
                    "Invalid value for the method parameter. Only the zscore "
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
        kwargs = CustomDict()
        for strat in lst(self.strategy):
            kwargs[strat] = {}
            for key, value in self.kwargs.items():
                # Parameters for this estimator only
                if key.lower() == strat.lower():
                    kwargs[strat].update(value)
                # Parameters for all estimators
                elif key.lower() not in map(str.lower, lst(self.strategy)):
                    kwargs[strat].update({key: value})

        self.log("Pruning outliers...", 1)

        # Prepare dataset (merge with y and exclude categorical columns)
        objective = merge(X, y) if self.include_target and y is not None else X
        objective = objective.select_dtypes(include=["number"])

        outliers = []
        for strat in lst(self.strategy):
            if strat.lower() == "zscore":
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
                            f" --> The zscore strategy detected "
                            f"{len(mask) - sum(mask)} outliers.", 2
                        )

            else:
                estimator = self._get_est_class(*strategies[strat])(**kwargs[strat])
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

    device: str, default="cpu"
        Device on which to train the estimators. Use any string
        that follows the [SYCL_DEVICE_FILTER][] filter selector,
        e.g. `device="gpu"` to use the GPU. Read more in the
        [user guide][accelerating-pipelines].

    engine: str, default="sklearn"
        Execution engine to use for the estimators. Refer to the
        [user guide][accelerating-pipelines] for an explanation
        regarding every choice. Choose from:

        - "sklearn" (only if device="cpu")
        - "cuml" (only if device="gpu")

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]: sklearn transformer
        Object with which the data is scaled.

    feature_names_in_: np.array
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
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.dataset)

             mean radius  mean texture  ...  worst fractal dimension  target
        0          17.99         10.38  ...                  0.11890       0
        1          12.25         17.94  ...                  0.08132       1
        2          13.87         20.70  ...                  0.08492       1
        3          12.06         12.74  ...                  0.07898       1
        4          12.62         17.15  ...                  0.07330       1
        ..           ...           ...  ...                      ...     ...
        564        11.34         18.61  ...                  0.06783       1
        565        11.43         17.31  ...                  0.08096       1
        566        11.06         14.96  ...                  0.09080       1
        567        13.20         15.82  ...                  0.08385       1
        568        20.55         20.86  ...                  0.07569       0

        [569 rows x 31 columns]

        >>> atom.scale(verbose=2)

        Fitting Scaler...
        Scaling features...

        >>> # Note the reduced number of rows
        >>> print(atom.dataset)

             mean radius  mean texture  ...  worst fractal dimension  target
        0       1.052603     -2.089926  ...                 1.952598       0
        1      -0.529046     -0.336627  ...                -0.114004       1
        2      -0.082657      0.303467  ...                 0.083968       1
        3      -0.581401     -1.542600  ...                -0.242685       1
        4      -0.427093     -0.519842  ...                -0.555040       1
        ..           ...           ...  ...                      ...     ...
        564    -0.779796     -0.181242  ...                -0.855847       1
        565    -0.754996     -0.482735  ...                -0.133801       1
        566    -0.856949     -1.027742  ...                 0.407321       1
        567    -0.267275     -0.828293  ...                 0.025126       1
        568     1.758008      0.340573  ...                -0.423609       0

        [569 rows x 31 columns]

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.data_cleaning import Scaler
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0          17.99         10.38  ...          0.4601                  0.11890
        1          20.57         17.77  ...          0.2750                  0.08902
        2          19.69         21.25  ...          0.3613                  0.08758
        3          11.42         20.38  ...          0.6638                  0.17300
        4          20.29         14.34  ...          0.2364                  0.07678
        ..           ...           ...  ...             ...                      ...
        564        21.56         22.39  ...          0.2060                  0.07115
        565        20.13         28.25  ...          0.2572                  0.06637
        566        16.60         28.08  ...          0.2218                  0.07820
        567        20.60         29.33  ...          0.4087                  0.12400
        568         7.76         24.54  ...          0.2871                  0.07039

        [569 rows x 30 columns]

        >>> scaler = Scaler(verbose=2)
        >>> X = scaler.fit_transform(X)

        Fitting Scaler...
        Scaling features...

        >>> # Note the reduced number of rows
        >>> print(X)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0       1.097064     -2.073335  ...        2.750622                 1.937015
        1       1.829821     -0.353632  ...       -0.243890                 0.281190
        2       1.579888      0.456187  ...        1.152255                 0.201391
        3      -0.768909      0.253732  ...        6.046041                 4.935010
        4       1.750297     -1.151816  ...       -0.868353                -0.397100
        ..           ...           ...  ...             ...                      ...
        564     2.110995      0.721473  ...       -1.360158                -0.709091
        565     1.704854      2.085134  ...       -0.531855                -0.973978
        566     0.702284      2.045574  ...       -1.104549                -0.318409
        567     1.838341      2.336457  ...        1.919083                 2.219635
        568    -1.808401      1.221792  ...       -0.048138                -0.751207

        [569 rows x 30 columns]


        ```

    """

    _train_only = False

    @typechecked
    def __init__(
        self,
        strategy: str = "standard",
        *,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose, logger=logger)
        self.strategy = strategy
        self.kwargs = kwargs

        self._num_cols = None
        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> Scaler:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Scaler
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        self._num_cols = list(X.select_dtypes(include="number").columns)

        strategies = CustomDict(
            standard="StandardScaler",
            minmax="MinMaxScaler",
            maxabs="MaxAbsScaler",
            robust="RobustScaler",
        )

        if self.strategy in strategies:
            estimator = self._get_est_class(strategies[self.strategy], "preprocessing")
            self._estimator = estimator(**self.kwargs)
        else:
            raise ValueError(
                f"Invalid value for the strategy parameter, got {self.strategy}. "
                f"Choose from: {', '.join(strategies)}."
            )

        self.log("Fitting Scaler...", 1)
        self._estimator.fit(X[self._num_cols])

        # Add the estimator as attribute to the instance
        setattr(self, self.strategy.lower(), self._estimator)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> pd.DataFrame:
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Scaled dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Scaling features...", 1)
        X_transformed = self._estimator.transform(X[self._num_cols])

        # If all columns were transformed, just swap sets
        if len(self._num_cols) != X.shape[1]:
            # Replace the numerical columns with the transformed values
            for i, col in enumerate(self._num_cols):
                X[col] = X_transformed[:, i]
        else:
            X = to_df(X_transformed, X.index, X.columns)

        return X

    @composed(crash, method_to_log, typechecked)
    def inverse_transform(
        self, X: X_TYPES, y: Optional[Y_TYPES] = None
    ) -> pd.DataFrame:
        """Apply the inverse transformation to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Scaled dataframe.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Inversely scaling features...", 1)
        X_transformed = self._estimator.inverse_transform(X[self._num_cols])

        # If all columns were transformed, just swap sets
        if len(self._num_cols) != X.shape[1]:
            # Replace the numerical columns with the transformed values
            for i, col in enumerate(self._num_cols):
                X[col] = X_transformed[:, i]
        else:
            X = to_df(X_transformed, X.index, X.columns)

        return X

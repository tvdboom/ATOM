# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the feature engineering transformers.

"""

from __future__ import annotations

import re
from collections import defaultdict
from inspect import signature
from logging import Logger
from random import sample
from typing import Optional, Union

import featuretools as ft
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from scipy import stats
from sklearn.base import BaseEstimator, is_regressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import (
    RFE, RFECV, SelectFromModel, SelectKBest, SequentialFeatureSelector, chi2,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
)
from sklearn.model_selection import cross_val_score
from typeguard import typechecked
from zoofs import (
    DragonFlyOptimization, GeneticOptimization, GreyWolfOptimization,
    HarrisHawkOptimization, ParticleSwarmOptimization,
)

from atom.basetransformer import BaseTransformer
from atom.data_cleaning import Scaler, TransformerMixin
from atom.models import MODELS
from atom.plots import FeatureSelectorPlot
from atom.utils import (
    FLOAT, INT, SCALAR, SEQUENCE, SEQUENCE_TYPES, X_TYPES, Y_TYPES, CustomDict,
    check_is_fitted, check_scaling, composed, crash, get_custom_scorer,
    get_feature_importance, infer_task, is_sparse, lst, merge, method_to_log,
    to_df,
)


class FeatureExtractor(BaseEstimator, TransformerMixin, BaseTransformer):
    """Extract features from datetime columns.

    Create new features extracting datetime elements (day, month,
    year, etc...) from the provided columns. Columns of dtype
    `datetime64` are used as is. Categorical columns that can be
    successfully converted to a datetime format (less than 30% NaT
    values after conversion) are also used.

    This class can be accessed from atom through the [feature_extraction]
    [atomclassifier-feature_extraction] method. Read more in the
    [user guide][extracting-datetime-features].

    !!! warning
        Decision trees based algorithms build their split rules
        according to one feature at a time. This means that they will
        fail to correctly process cyclic features since the sin/cos
        features should be considered one single coordinate system.

    Parameters
    ----------
    features: str or sequence, default=["day", "month", "year"]
        Features to create from the datetime columns. Note that
        created features with zero variance (e.g. the feature hour
        in a column that only contains dates) are ignored. Allowed
        values are datetime attributes from `pandas.Series.dt`.

    fmt: str, sequence or None, default=None
        Format (`strptime`) of the categorical columns that need
        to be converted to datetime. If sequence, the n-th format
        corresponds to the n-th categorical column that can be
        successfully converted. If None, the format is inferred
        automatically from the first non NaN value. Values that can
        not be converted are returned as `NaT`.

    encoding_type: str, default="ordinal"
        Type of encoding to use. Choose from:

        - "ordinal": Encode features in increasing order.
        - "cyclic": Encode features using sine and cosine to capture
          their cyclic nature. This approach creates two columns for
          every feature. Non-cyclic features still use ordinal encoding.

    drop_columns: bool, default=True
        Whether to drop the original columns after transformation.

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
    feature_names_in_: np.array
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.feature_engineering:FeatureGenerator
    atom.feature_engineering:FeatureGrouper
    atom.feature_engineering:FeatureSelector

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        >>> X["date"] = pd.date_range(start="1/1/2018", periods=len(X))

        >>> atom = ATOMClassifier(X, y)
        >>> atom.feature_extraction(features=["day"], fmt="%d/%m/%Y", verbose=2)

        Extracting datetime features...
         --> Extracting features from column date.
           --> Creating feature date_day.

        >>> # Note the date_day column
        >>> print(atom.dataset)

             mean radius  mean texture  ...  date_day  target
        0         11.300         18.19  ...        31       1
        1         16.460         20.11  ...        27       0
        2         11.370         18.89  ...        17       1
        3          8.598         20.98  ...         3       1
        4         12.800         17.46  ...         2       1
        ..           ...           ...  ...       ...     ...
        564       17.060         21.00  ...         2       0
        565       11.940         20.76  ...        14       1
        566       19.590         25.00  ...        28       0
        567       12.360         18.54  ...        18       1
        568       18.450         21.91  ...        15       0

        [569 rows x 32 columns]

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.feature_engineering import FeatureExtractor
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, _ = load_breast_cancer(return_X_y=True, as_frame=True)
        >>> X["date"] = pd.date_range(start="1/1/2018", periods=len(X))

        >>> fe = FeatureExtractor(features=["day"], fmt="%Y-%m-%d", verbose=2)
        >>> X = fe.transform(X)

        Extracting datetime features...
         --> Extracting features from column date.
           --> Creating feature date_day.

        >>> # Note the date_day column
        >>> print(X)

             mean radius  mean texture  ...  worst fractal dimension  date_day
        0          17.99         10.38  ...                  0.11890         1
        1          20.57         17.77  ...                  0.08902         2
        2          19.69         21.25  ...                  0.08758         3
        3          11.42         20.38  ...                  0.17300         4
        4          20.29         14.34  ...                  0.07678         5
        ..           ...           ...  ...                      ...       ...
        564        21.56         22.39  ...                  0.07115        19
        565        20.13         28.25  ...                  0.06637        20
        566        16.60         28.08  ...                  0.07820        21
        567        20.60         29.33  ...                  0.12400        22
        568         7.76         24.54  ...                  0.07039        23

        [569 rows x 31 columns]

        ```

    """

    def __init__(
        self,
        features: Union[str, SEQUENCE_TYPES] = ["day", "month", "year"],
        fmt: Optional[Union[str, SEQUENCE_TYPES]] = None,
        *,
        encoding_type: str = "ordinal",
        drop_columns: bool = True,
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.fmt = fmt
        self.features = features
        self.encoding_type = encoding_type
        self.drop_columns = drop_columns

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> pd.DataFrame:
        """Extract the new features.

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

        def encode_variable(idx, name, values, min_val=0, max_val=None):
            """Encode a feature in an ordinal or cyclic fashion."""
            if self.encoding_type.lower() == "ordinal" or max_val is None:
                self.log(f"   --> Creating feature {name}.", 2)
                X.insert(idx, name, values)
            elif self.encoding_type.lower() == "cyclic":
                self.log(f"   --> Creating cyclic feature {name}.", 2)
                pos = 2 * np.pi * (values - min_val) / np.array(max_val)
                X.insert(idx, f"{name}_sin", np.sin(pos))
                X.insert(idx + 1, f"{name}_cos", np.cos(pos))

            return X.copy()  # Avoid dataframe fragmentation

        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        # Check parameters
        if self.encoding_type.lower() not in ("ordinal", "cyclic"):
            raise ValueError(
                "Invalid value for the encoding_type parameter, got "
                f"{self.encoding_type}. Choose from: ordinal, cyclic."
            )

        self.log("Extracting datetime features...", 1)

        i = 0
        for name, column in X.select_dtypes(exclude="number").items():
            if column.dtype.name == "datetime64[ns]":
                col_dt = column
                self.log(f" --> Extracting features from column {name}.", 1)
            else:
                col_dt = pd.to_datetime(
                    arg=column,
                    errors="coerce",  # Converts to NaT if he can't format
                    format=self.fmt[i] if isinstance(self.fmt, SEQUENCE) else self.fmt,
                    infer_datetime_format=True,
                )

                # If >30% values are NaT, the conversion was unsuccessful
                nans = 100. * col_dt.isna().sum() / len(X)
                if nans >= 30:
                    continue  # Skip this column
                else:
                    i += 1
                    self.log(
                        f" --> Extracting features from categorical column {name}.", 1
                    )

            # Extract features from the datetime column
            for fx in map(str.lower, lst(self.features)):
                if hasattr(col_dt.dt, fx.lower()):
                    values = getattr(col_dt.dt, fx)
                else:
                    raise ValueError(
                        "Invalid value for the feature parameter. Value "
                        f"{fx.lower()} is not an attribute of pd.Series.dt."
                    )

                # Skip if the information is not present in the format
                if not isinstance(values, pd.Series):
                    self.log(
                        f"   --> Extracting feature {fx} failed. "
                        "Result is not a pd.Series.dt.", 2
                    )
                    continue

                min_val, max_val = 0, None  # max_val=None -> feature is not cyclic
                if self.encoding_type.lower() == "cyclic":
                    if fx == "microsecond":
                        min_val, max_val = 0, 1e6 - 1
                    elif fx in ("second", "minute"):
                        min_val, max_val = 0, 59
                    elif fx == "hour":
                        min_val, max_val = 0, 23
                    elif fx in ("weekday", "dayofweek", "day_of_week"):
                        min_val, max_val = 0, 6
                    elif fx in ("day", "dayofmonth", "day_of_month"):
                        min_val, max_val = 1, col_dt.dt.daysinmonth
                    elif fx in ("dayofyear", "day_of_year"):
                        min_val = 1
                        max_val = [365 if i else 366 for i in col_dt.dt.is_leap_year]
                    elif fx == "month":
                        min_val, max_val = 1, 12
                    elif fx == "quarter":
                        min_val, max_val = 1, 4

                # Add every new feature after the previous one
                X = encode_variable(
                    idx=X.columns.get_loc(name),
                    name=f"{name}_{fx}",
                    values=values,
                    min_val=min_val,
                    max_val=max_val,
                )

            # Drop the original datetime column
            if self.drop_columns:
                X = X.drop(name, axis=1)

        return X


class FeatureGenerator(BaseEstimator, TransformerMixin, BaseTransformer):
    """Generate new features.

    Create new combinations of existing features to capture the
    non-linear relations between the original features.

    This class can be accessed from atom through the [feature_generation]
    [atomclassifier-feature_generation] method. Read more in the
    [user guide][generating-new-features].

    !!! warning
        * Using the `div`, `log` or `sqrt` operators can return new
          features with `inf` or `NaN` values. Check the warnings that
          may pop up or use atom's [nans][atomclassifier-nans] attribute.
        * When using dfs with `n_jobs>1`, make sure to protect your code
          with `if __name__ == "__main__"`. Featuretools uses
          [dask](https://dask.org/), which uses python multiprocessing
          for parallelization. The spawn method on multiprocessing
          starts a new python process, which requires it to import the
          \__main__ module before it can do its task.
        * gfg can be slow for very large populations.

    !!! tip
        dfs can create many new features and not all of them will be
        useful. Use the [FeatureSelector][] class to reduce the number
        of features.

    Parameters
    ----------
    strategy: str, default="dfs"
        Strategy to crate new features. Choose from:

        - "[dfs][]": Deep Feature Synthesis.
        - "[gfg][]": Genetic Feature Generation.

    n_features: int or None, default=None
        Maximum number of newly generated features to add to the
        dataset. If None, select all created features.

    operators: str, sequence or None, default=None
        Mathematical operators to apply on the features. None to use
        all. Choose from: `add`, `sub`, `mul`, `div`, `abs`, `sqrt`,
        `log`, `inv`, `sin`, `cos`, `tan`.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

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
        Additional keyword arguments for the SymbolicTransformer
        instance. Only for the gfg strategy.

    Attributes
    ----------
    gfg: [SymbolicTransformer][]
        Object used to calculate the genetic features. Only for the
        gfg strategy.

    genetic_features: pd.DataFrame
        Information on the newly created non-linear features. Only for
        the gfg strategy. Columns include:

        - **name:** Name of the feature (generated automatically).
        - **description:** Operators used to create this feature.
        - **fitness:** Fitness score.

    feature_names_in_: np.array
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.feature_engineering:FeatureExtractor
    atom.feature_engineering:FeatureGrouper
    atom.feature_engineering:FeatureSelector

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.feature_generation(strategy="dfs", n_features=5, verbose=2)

        Fitting FeatureGenerator...
        Generating new features...
         --> 5 new features were added.

        >>> # Note the texture error / worst symmetry column
        >>> print(atom.dataset)

             mean radius  mean texture  ...  texture error / worst symmetry  target
        0          15.75         19.22  ...                        3.118963       0
        1          12.10         17.72  ...                        5.418170       1
        2          20.16         19.66  ...                        2.246481       0
        3          12.88         18.22  ...                        4.527498       1
        4          13.03         18.42  ...                       11.786613       1
        ..           ...           ...  ...                             ...     ...
        564        21.75         20.99  ...                        4.772326       0
        565        13.64         16.34  ...                        3.936061       1
        566        10.08         15.11  ...                        4.323219       1
        567        12.91         16.33  ...                        3.004630       1
        568        11.60         18.36  ...                        2.385047       1

        [569 rows x 36 columns]

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.feature_engineering import FeatureGenerator
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> fg = FeatureGenerator(strategy="dfs", n_features=5, verbose=2)
        >>> X = fg.fit_transform(X, y)

        Fitting FeatureGenerator...
        Generating new features...
         --> 5 new features were added.

        >>> # Note the radius error * worst smoothness column
        >>> print(X)

             mean radius  ...  radius error * worst smoothness
        0          17.99  ...                         0.177609
        1          20.57  ...                         0.067285
        2          19.69  ...                         0.107665
        3          11.42  ...                         0.103977
        4          20.29  ...                         0.104039
        ..           ...  ...                              ...
        564        21.56  ...                         0.165816
        565        20.13  ...                         0.089257
        566        16.60  ...                         0.051984
        567        20.60  ...                         0.119790
        568         7.76  ...                         0.034698

        [569 rows x 35 columns]

        ```

    """

    def __init__(
        self,
        strategy: str = "dfs",
        *,
        n_features: Optional[INT] = None,
        operators: Optional[Union[str, SEQUENCE_TYPES]] = None,
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
        self.n_features = n_features
        self.operators = operators
        self.kwargs = kwargs

        self.genetic_features = None
        self._operators = None
        self._dfs = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Y_TYPES) -> FeatureGenerator:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence
            Target column corresponding to X.

            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

        Returns
        -------
        self
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        operators = CustomDict(
            add="add_numeric",
            sub="subtract_numeric",
            mul="multiply_numeric",
            div="divide_numeric",
            abs="absolute",
            sqrt="square_root",
            log="natural_logarithm",
            sin="sine",
            cos="cosine",
            tan="tangent",
        )

        if self.strategy.lower() not in ("dfs", "gfg"):
            raise ValueError(
                "Invalid value for the strategy parameter, got "
                f"{self.strategy}. Choose from: dfs or gfg."
            )

        if self.n_features is not None and self.n_features <= 0:
            raise ValueError(
                "Invalid value for the n_features parameter."
                f"Value should be >0, got {self.n_features}."
            )

        if not self.operators:  # None or empty list
            self._operators = list(operators.keys())
        else:
            self._operators = lst(self.operators)
            for operator in self._operators:
                if operator not in operators:
                    raise ValueError(
                        "Invalid value in the operators parameter, got "
                        f"{operator}. Choose from: {', '.join(operators)}."
                    )

        self.log("Fitting FeatureGenerator...", 1)

        if self.strategy.lower() == "dfs":
            # Run deep feature synthesis with transformation primitives
            es = ft.EntitySet(dataframes={"X": (X, "_index", None, None, None, True)})
            self._dfs = ft.dfs(
                target_dataframe_name="X",
                entityset=es,
                trans_primitives=[operators[x] for x in self._operators],
                max_depth=1,
                features_only=True,
                ignore_columns={"X": ["_index"]},
            )

            # Select the new features (dfs also returns originals)
            self._dfs = self._dfs[X.shape[1] - 1:]

            # Get random selection of features
            if self.n_features and self.n_features < len(self._dfs):
                self._dfs = sample(self._dfs, self.n_features)

            # Order the features alphabetically
            self._dfs = sorted(self._dfs, key=lambda x: x._name)

        else:
            kwargs = self.kwargs.copy()  # Copy in case of repeated fit
            hall_of_fame = kwargs.pop("hall_of_fame", max(400, self.n_features or 400))
            self.gfg = SymbolicTransformer(
                population_size=kwargs.pop("population_size", 2000),
                hall_of_fame=hall_of_fame,
                n_components=hall_of_fame,
                init_depth=kwargs.pop("init_depth", (1, 2)),
                const_range=kwargs.pop("const_range", None),
                function_set=self._operators,
                feature_names=X.columns,
                verbose=kwargs.pop("verbose", 0 if self.verbose < 2 else 1),
                n_jobs=kwargs.pop("n_jobs", self.n_jobs),
                random_state=kwargs.pop("random_state", self.random_state),
                **kwargs,
            ).fit(X, y)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> pd.DataFrame:
        """Generate new features.

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
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Generating new features...", 1)

        if self.strategy.lower() == "dfs":
            index = X.index
            es = ft.EntitySet(dataframes={"X": (X, "index", None, None, None, True)})
            dfs = ft.calculate_feature_matrix(
                features=self._dfs,
                entityset=es,
                n_jobs=self.n_jobs,
            )

            # Add the new features to the feature set
            X = pd.concat([X.drop("index", axis=1), dfs], axis=1)
            X.index = index

            self.log(f" --> {len(self._dfs)} new features were added.", 2)

        else:
            # Get the names and fitness of the new features
            df = pd.DataFrame(columns=["name", "description", "fitness"])
            for i, fx in enumerate(self.gfg):
                if str(fx) not in X.columns:  # Drop unchanged features
                    df.loc[i] = ["", str(fx), fx.fitness_]

            # Check if any new features remain
            if len(df) == 0:
                self.log(
                    " --> The genetic algorithm didn't find any improving features.", 2
                )
                return X

            # Select the n_features with the highest fitness
            df = df.drop_duplicates()
            if self.n_features and len(df) > self.n_features:
                df = df.nlargest(self.n_features, columns="fitness")

            # If there are not enough features remaining, notify the user
            if len(df) != self.n_features:
                self.log(
                    f" --> Dropping {(self.n_features or len(self.gfg)) - len(df)} "
                    "features due to repetition.", 2)

            for i, array in enumerate(self.gfg.transform(X)[:, df.index].T):
                # If the column is new, use a default name
                counter = 0
                while True:
                    name = f"x{X.shape[1] + counter}"
                    if name not in X:
                        X[name] = array  # Add new feature to X
                        df.iloc[i, 0] = name
                        break
                    else:
                        counter += 1

            self.log(f" --> {len(df)} new features were added.", 2)
            self.genetic_features = df.reset_index(drop=True)

        return X


class FeatureGrouper(BaseEstimator, TransformerMixin, BaseTransformer):
    """Extract statistics from similar features.

    Replace groups of features with related characteristics with new
    features that summarize statistical properties of te group. The
    statistical operators are calculated over every row of the group.
    The group names and features can be accessed through the `groups`
    method.

    This class can be accessed from atom through the [feature_grouping]
    [atomclassifier-feature_grouping] method. Read more in the
    [user guide][grouping-similar-features].

    !!! tip
        Use a regex pattern with the `groups` parameter to select
        groups easier, e.g. `atom.feature_generation(features="var_.+")`
        to select all features that start with `var_`.

    Parameters
    ----------
    group: str, slice or sequence
        Features that belong to a group. Select them by name, position
        or regex pattern. A feature can belong to multiple groups. Use
        a sequence of sequences to define multiple groups.

    name: str, sequence or None, default=None
        Name of the group. The new features are named combining the
        operator used and the group's name, e.g. `mean(group_1)`. If
        specfified, the length should match with the number of groups
        defined in `features`. If None, default group names of the form
        `group1`, `group2`, etc... are used.

    operators: str, sequence or None, default=None
        Statistical operators to apply on the groups. Any operator from
        `numpy` or `scipy.stats` (checked in that order) that is applied
        on an array can be used. If None, it uses: `min`, `max`, `mean`,
        `median`, `mode` and `std`.

    drop_columns: bool, default=True
        Whether to drop the columns in `groups` after transformation.

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
    groups: dict
        Names and features of every created group.

    feature_names_in_: np.array
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.feature_engineering:FeatureExtractor
    atom.feature_engineering:FeatureGenerator
    atom.feature_engineering:FeatureSelector

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.feature_grouping(group=["mean.+"], name="means", verbose=2)

        Fitting FeatureGrouper...
        Grouping features...
         --> Group means successfully created.

        >>> # Note the mean features are gone and the new std(means) feature
        >>> print(atom.dataset)

             radius error  texture error  ...  std(means)  target
        0          0.2949         1.6560  ...  137.553584       1
        1          0.2351         2.0110  ...   79.830195       1
        2          0.4302         2.8780  ...   80.330330       1
        3          0.2345         1.2190  ...  151.858455       1
        4          0.3511         0.9527  ...  145.769474       1
        ..            ...            ...  ...         ...     ...
        564        0.4866         1.9050  ...  116.749243       1
        565        0.5925         0.6863  ...  378.431333       0
        566        0.2577         1.0950  ...  141.220243       1
        567        0.4615         0.9197  ...  257.903846       0
        568        0.5462         1.5110  ...  194.704033       1

        [569 rows x 27 columns]

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.feature_engineering import FeatureGrouper
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> # Group all features that start with mean
        >>> fg = FeatureGrouper(group="mean.+", name="means", verbose=2)
        >>> X = fg.transform(X)

        Fitting FeatureGrouper...
        Grouping features...
         --> Group means successfully created.

        >>> # Note the mean features are gone and the new std(means) feature
        >>> print(X)

             radius error  texture error  ...  mode(means)  std(means)
        0          1.0950         0.9053  ...      0.07871  297.404540
        1          0.5435         0.7339  ...      0.05667  393.997131
        2          0.7456         0.7869  ...      0.05999  357.203084
        3          0.4956         1.1560  ...      0.09744  114.444620
        4          0.7572         0.7813  ...      0.05883  385.450556
        ..            ...            ...  ...          ...         ...
        564        1.1760         1.2560  ...      0.05623  439.441252
        565        0.7655         2.4630  ...      0.05533  374.274845
        566        0.4564         1.0750  ...      0.05302  254.320568
        567        0.7260         1.5950  ...      0.07016  375.376476
        568        0.3857         1.4280  ...      0.00000   53.739926

        [569 rows x 26 columns]

        ```

    """

    def __init__(
        self,
        group: Union[str, SEQUENCE_TYPES],
        name: Optional[Union[str, SEQUENCE_TYPES]] = None,
        *,
        operators: Optional[Union[str, SEQUENCE_TYPES]] = None,
        drop_columns: bool = True,
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.group = group
        self.name = name
        self.operators = operators
        self.drop_columns = drop_columns
        self.groups = defaultdict(list)

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> pd.DataFrame:
        """Group features.

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
        X, _ = self._prepare_input(X, y)

        self.log("Grouping features...", 1)

        if np.array(self.group).ndim < 2:
            self.group = [self.group]

        # Make the groups
        groups = []
        for group in lst(self.group):
            groups.append([])
            for col in lst(group):
                if isinstance(col, int):
                    try:
                        groups[-1].append(X.columns[col])
                    except IndexError:
                        raise ValueError(
                            f"Invalid value for the groups parameter. Value {col} "
                            f"is out of range for a dataset with {X.shape[1]} columns."
                        )
                else:
                    # Find columns using regex matches
                    matches = [c for c in X.columns if re.fullmatch(col, c)]
                    if matches:
                        groups[-1].extend(matches)
                    else:
                        try:
                            groups[-1].extend(list(X.select_dtypes(col).columns))
                        except TypeError:
                            raise ValueError(
                                "Invalid value for the groups parameter. "
                                f"Could not find any column that matches {col}."
                            )

        if self.name is None:
            names = [f"group_{i}" for i in range(1, len(groups) + 1)]
        else:
            names = lst(self.name)

        if self.operators is None:
            operators = ["min", "max", "mean", "median", "mode", "std"]
        else:
            operators = lst(self.operators)

        if len(groups) != len(names):
            raise ValueError(
                f"Invalid value for the names parameter, got {self.name}. The "
                f"number of groups ({len(groups)}) does not match with the number "
                f"of names ({len(names)})."
            )

        to_drop = set()
        self.groups = defaultdict(list)  # Reset attr for repeated transforms
        for name, group in zip(names, groups):
            group_df = X[group]

            for operator in operators:
                try:
                    result = group_df.apply(getattr(np, operator), axis=1)
                except AttributeError:
                    try:
                        result = getattr(stats, operator)(group_df, axis=1)[0]
                    except AttributeError:
                        raise ValueError(
                            "Invalid value for the operators parameter. Value "
                            f"{operator} is not an attribute of numpy nor scipy.stats."
                        )

                try:
                    X[f"{operator}({name})"] = result
                except ValueError:
                    raise ValueError(
                        "Invalid value for the operators parameter. Value "
                        f"{operator} doesn't return a one-dimensional array."
                    )

            to_drop.update(group)

            self.groups[name] = group
            self.log(f" --> Group {name} successfully created.")

        if self.drop_columns:
            X = X.drop(to_drop, axis=1)

        return X


class FeatureSelector(
    BaseEstimator,
    TransformerMixin,
    BaseTransformer,
    FeatureSelectorPlot,
):
    """Reduce the number of features in the data.

    Apply feature selection or dimensionality reduction, either to
    improve the estimators' accuracy or to boost their performance on
    very high-dimensional datasets. Additionally, remove multicollinear
    and low variance features.

    This class can be accessed from atom through the [feature_selection]
    [atomclassifier-feature_selection] method. Read more in the
    [user guide][selecting-useful-features].

    !!! warning
         - Ties between features with equal scores are broken in an
           unspecified way.
         - For strategy="rfecv", the `n_features` parameter is the
           **minimum** number of features to select, not the actual
           number of features that the transformer returns. It may very
           well be that it returns more!

    !!! info
        If strategy="pca" and the provided data is dense, it's scaled to
        mean=0 and std=1 before fitting the transformer (if it wasn't
        already).

    !!! tip
        Use the [plot_feature_importance][] method to examine how much
        a specific feature contributes to the final predictions. If the
        model doesn't have a `feature_importances_` attribute, use
        [plot_permutation_importance][] instead.

    Parameters
    ----------
    strategy: str or None, default=None
        Feature selection strategy to use. Choose from:

        - None: Do not perform any feature selection strategy.
        - "[univariate][selectkbest]": Univariate statistical F-test.
        - "[pca][]": Principal Component Analysis.
        - "[sfm][]": Select best features according to a model.
        - "[sfs][]": Sequential Feature Selection.
        - "[rfe][]": Recursive Feature Elimination.
        - "[rfecv][]": RFE with cross-validated selection.
        - "[pso][]": Particle Swarm Optimization.
        - "[hho][]": Harris Hawks Optimization.
        - "[gwo][]": Grey Wolf Optimization.
        - "[dfo][]": Dragonfly Optimization.
        - "[go][]": Genetic Optimization.

    solver: str, estimator or None, default=None
        Solver/estimator to use for the feature selection strategy. See
        the corresponding documentation for an extended description of
        the choices. If None, the default value is used (only for pca).

        - for "univariate", choose from:
            - "[f_classif][]"
            - "[f_regression][]"
            - "[mutual_info_classif][]"
            - "[mutual_info_regression][]"
            - "[chi2][]"
            - Any function with signature `func(X, y) -> (scores, p-values)`.
        - for "pca", choose from:
            - If dense data:
                - "auto" (default)
                - "full"
                - "arpack"
                - "randomized"
            - If sparse data:
                - "randomized" (default)
                - "arpack"
            - If gpu implementation:
                - "full" (default)
                - "jacobi"
                - "auto"
        - for the remaining strategies:<br>
          The base estimator. For sfm, rfe and rfecv, it should have
          either a `feature_importances_` or `coef_` attribute after
          fitting. You can use one of the [predefined models][]. Add
          `_class` or `_reg` after the model's  name to specify a
          classification or regression task, e.g. `solver="LGB_reg"`
          (not necessary if called from atom). No default option.

    n_features: int, float or None, default=None
        Number of features to select.

        - If None: Select all features.
        - If <1: Fraction of the total features to select.
        - If >=1: Number of features to select.

        If strategy="sfm" and the threshold parameter is not specified,
        the threshold is automatically set to `-inf` to select
        `n_features` number of features.

        If strategy="rfecv", `n_features` is the minimum number of
        features to select.

        This parameter is ignored if any of the following strategies
        is selected: pso, hho, gwo, dfo, go.

    min_repeated: int, float or None, default=2
        Remove categorical features if there isn't any repeated value
        in at least `min_repeated` rows. The default is to keep all
        features with non-maximum variance, i.e. remove the features
        which number of unique values is equal to the number of rows
        (usually the case for names, IDs, etc...).

        - If None: No check for minimum repetition.
        - If >1: Minimum repetition number.
        - If <=1: Minimum repetition fraction.

    max_repeated: int, float or None, default=1.
        Remove categorical features with the same value in at least
        `max_repeated` rows. The default is to keep all features with
        non-zero variance, i.e. remove the features that have the same
        value in all samples.

        - If None: No check for maximum repetition.
        - If >1: Maximum number of repeated occurences.
        - If <=1: Maximum fraction of repeated occurences.

    max_correlation: float or None, default=1.
        Minimum absolute [Pearson correlation][pearson] to identify
        correlated features. For each group, it removes all except the
        feature with the highest correlation to `y` (if provided, else
        it removes all but the first). The default value removes equal
        columns. If None, skip this step.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

    gpu: bool or str, default=False
        Train strategy on GPU. Only for strategy="pca".

        - If False: Always use CPU implementation.
        - If True: Use GPU implementation if possible.
        - If "force": Force GPU implementation.

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
        Any extra keyword argument for the strategy estimator. See the
        corresponding documentation for the available options.

    Attributes
    ----------
    collinear: pd.DataFrame
        Information on the removed collinear features. Columns include:

        - **drop:** Name of the dropped feature.
        - **corr_feature:** Names of the correlated features.
        - **corr_value:** Corresponding correlation coefficients.

    feature_importance: pd.Series
        Normalized importance scores calculated by the solver for the
        features kept by the transformer. The scores are extracted from
        the coef_ or feature_importances_ attribute, checked in that
        order. Only if strategy is one of univariate, sfm, rfe or rfecv.

    [strategy]: sklearn transformer
        Object used to transform the data, e.g. `fs.pca` for the pca
        strategy.

    feature_names_in_: np.array
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    See Also
    --------
    atom.feature_engineering:FeatureExtractor
    atom.feature_engineering:FeatureGenerator
    atom.feature_engineering:FeatureGrouper

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> atom.feature_selection(strategy="pca", n_features=12, verbose=2)

        Fitting FeatureSelector...
        Performing feature selection ...
         --> Applying Principal Component Analysis...
           --> Scaling features...
           --> Keeping 12 components.
           --> Explained variance ratio: 0.97

        >>> # Note that the column names changed
        >>> print(atom.dataset)

                 pca0      pca1      pca2  ...     pca10     pca11  target
        0   -2.493723  3.082653  1.318595  ... -0.182142 -0.591784       1
        1    4.596102 -0.876940 -0.380685  ...  0.224170  1.155544       0
        2    0.955979 -2.141057 -1.677736  ...  0.306153  0.099138       0
        3    3.221488  4.209911 -2.818757  ...  0.808883 -0.531868       0
        4    1.038000  2.451758 -1.753683  ... -0.312883  0.862319       1
        ..        ...       ...       ...  ...       ...       ...     ...
        564  3.414827 -3.757253 -1.012369  ...  0.387175  0.283633       0
        565 -1.191561 -1.276069 -0.871712  ...  0.106362 -0.449361       1
        566 -2.757000  0.411997 -1.321697  ...  0.185550 -0.025368       1
        567 -3.252533  0.074827  0.549622  ...  0.693073 -0.058251       1
        568  1.607258 -2.076465 -1.025986  ... -0.385542  0.103603       0
        [569 rows x 13 columns]

        >>> atom.plot_pca()
        ```

        ![plot_pca](../../img/plots/plot_pca.png)

    === "stand-alone"
        ```pycon
        >>> from atom.feature_engineering import FeatureSelector
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, _ = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> fs = FeatureSelector(strategy="pca", n_features=12, verbose=2)
        >>> X = fs.fit_transform(X)

        Fitting FeatureSelector...
        Performing feature selection ...
         --> Applying Principal Component Analysis...
           --> Scaling features...
           --> Keeping 12 components.
           --> Explained variance ratio: 0.97

        >>> # Note that the column names changed
        >>> print(X)

                  pca0       pca1      pca2  ...      pca9     pca10     pca11
        0     9.192837   1.948583 -1.123166  ... -0.877402  0.262955 -0.859014
        1     2.387802  -3.768172 -0.529293  ...  1.106995  0.813120  0.157923
        2     5.733896  -1.075174 -0.551748  ...  0.454275 -0.605604  0.124387
        3     7.122953  10.275589 -3.232790  ... -1.116975 -1.151514  1.011316
        4     3.935302  -1.948072  1.389767  ...  0.377704  0.651360 -0.110515
        ..         ...        ...       ...  ...       ...       ...       ...
        564   6.439315  -3.576817  2.459487  ...  0.256989 -0.062651  0.123342
        565   3.793382  -3.584048  2.088476  ... -0.108632  0.244804  0.222753
        566   1.256179  -1.902297  0.562731  ...  0.520877 -0.840512  0.096473
        567  10.374794   1.672010 -1.877029  ... -0.089296 -0.178628 -0.697461
        568  -5.475243  -0.670637  1.490443  ... -0.047726 -0.144094 -0.179496
        [569 rows x 12 columns]

        >>> fs.plot_pca()
        ```

        ![plot_pca](../../img/plots/plot_pca.png)

    """

    def __init__(
        self,
        strategy: Optional[str] = None,
        *,
        solver: Optional[Union[str, callable]] = None,
        n_features: Optional[SCALAR] = None,
        min_repeated: Optional[SCALAR] = 2,
        max_repeated: Optional[SCALAR] = 1.0,
        max_correlation: Optional[FLOAT] = 1.0,
        n_jobs: INT = 1,
        gpu: Union[bool, str] = False,
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
        random_state: Optional[INT] = None,
        **kwargs,
    ):
        super().__init__(
            n_jobs=n_jobs,
            gpu=gpu,
            verbose=verbose,
            logger=logger,
            random_state=random_state,
        )
        self.strategy = strategy
        self.solver = solver
        self.n_features = n_features
        self.min_repeated = min_repeated
        self.max_repeated = max_repeated
        self.max_correlation = max_correlation
        self.kwargs = kwargs

        self.collinear = pd.DataFrame(columns=["drop", "corr_feature", "corr_value"])
        self.feature_importance = []
        self.scaler = None

        self._n_features = None
        self._kwargs = kwargs.copy()
        self._high_variance = {}
        self._low_variance = {}
        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> FeatureSelector:
        """Fit the feature selector to the data.

        The univariate, sfm (when model is not fitted), sfs, rfe and
        rfecv strategies need a target column. Leaving it None raises
        an exception.

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
        self
            Estimator instance.

        """

        def check_y():
            """For some strategies, y needs to be provided."""
            if y is None:
                raise ValueError(
                    "Invalid value for the y parameter. Value cannot "
                    f"be None for strategy='{self.strategy}'."
                )

        def objective_function(model, X_train, y_train, X_valid, y_valid, scoring):
            """Objective function for the advanced optimization strategies."""
            if X_train.equals(X_valid):
                cv = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring)
                return np.mean(cv)
            else:
                model.fit(X_train, y_train)
                return scoring(model, X_valid, y_valid)

        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        strats = CustomDict(
            univariate=SelectKBest,
            pca=PCA,
            sfm=SelectFromModel,
            sfs=SequentialFeatureSelector,
            rfe=RFE,
            rfecv=RFECV,
            pso=ParticleSwarmOptimization,
            hho=HarrisHawkOptimization,
            gwo=GreyWolfOptimization,
            dfo=DragonFlyOptimization,
            go=GeneticOptimization,
        )

        if isinstance(self.strategy, str):
            if self.strategy not in strats:
                raise ValueError(
                    "Invalid value for the strategy parameter, got "
                    f"{self.strategy}. Choose from: {', '.join(strats)}"
                )
            elif self.strategy.lower() not in ("univariate", "pca"):
                if self.solver is None:
                    raise ValueError(
                        "Invalid value for the solver parameter. The "
                        f"value can't be None for strategy={self.strategy}"
                    )
                elif isinstance(self.solver, str):
                    # Assign goal to initialize the predefined model
                    if self.solver[-6:] == "_class":
                        self.goal = "class"
                        solver = self.solver[:-6]
                    elif self.solver[-4:] == "_reg":
                        self.goal = "reg"
                        solver = self.solver[:-4]
                    else:
                        solver = self.solver

                    # Get estimator from predefined models
                    if solver not in MODELS:
                        raise ValueError(
                            "Invalid value for the solver parameter. Unknown "
                            f"model: {solver}. Choose from: {', '.join(MODELS)}."
                        )
                    else:
                        model = MODELS[solver](self, fast_init=True)
                        solver = model.get_estimator()
                else:
                    solver = self.solver

                    # Assign goal to get default scorer for advanced strategies
                    self.goal = "reg" if is_regressor(solver) else "class"

        elif self.kwargs:
            kwargs = ", ".join([f"{str(k)}={str(v)}" for k, v in self.kwargs.items()])
            raise ValueError(
                f"Keyword arguments ({kwargs}) are specified for "
                "the strategy estimator but no strategy is selected."
            )

        if self.n_features is None:
            self._n_features = X.shape[1]
        elif self.n_features <= 0:
            raise ValueError(
                "Invalid value for the n_features parameter. "
                f"Value should be >0, got {self.n_features}."
            )
        elif self.n_features < 1:
            self._n_features = int(self.n_features * X.shape[1])
        else:
            self._n_features = self.n_features

        if self.min_repeated is None:
            min_repeated = 1
        elif self.min_repeated < 0:
            raise ValueError(
                "Invalid value for the min_repeated parameter. Value "
                f"should be >0, got {self.min_repeated}."
            )
        elif self.min_repeated < 1:
            min_repeated = self.min_repeated * len(X)
        else:
            min_repeated = int(self.min_repeated)

        if self.max_repeated is None:
            max_repeated = len(X)
        elif self.max_repeated < 0:
            raise ValueError(
                "Invalid value for the max_repeated parameter. Value "
                f"should be >0, got {self.max_repeated}."
            )
        elif self.max_repeated <= 1:
            max_repeated = self.max_repeated * len(X)
        else:
            max_repeated = int(self.max_repeated)

        if min_repeated > max_repeated:
            raise ValueError(
                "The min_repeated parameter can't be higher than "
                f"max_repeated, got {min_repeated} > {max_repeated}. "
            )

        if self.max_correlation is not None and not 0 <= self.max_correlation <= 1:
            raise ValueError(
                "Invalid value for the max_correlation parameter. Value "
                f"shouldbe between 0 and 1, got {self.max_correlation}."
            )

        self.log("Fitting FeatureSelector...", 1)

        # Remove features with too high variance
        if self.min_repeated is not None:
            for name, column in X.select_dtypes(exclude="number").items():
                max_counts = column.value_counts()
                if min_repeated > max_counts.max():
                    self._high_variance[name] = [max_counts.idxmax(), max_counts.max()]
                    X = X.drop(name, axis=1)
                    break

        # Remove features with too low variance
        if self.max_repeated is not None:
            for name, column in X.select_dtypes(exclude="number").items():
                for category, count in column.value_counts().items():
                    if count >= max_repeated:
                        self._low_variance[name] = [category, 100 * count / len(X)]
                        X = X.drop(name, axis=1)
                        break

        # Remove features with too high correlation
        self.collinear = pd.DataFrame(columns=["drop", "corr_feature", "corr_value"])
        if self.max_correlation:
            # Get the Pearson correlation coefficient matrix
            if y is None:
                corr_X = X.corr()
            else:
                corr_matrix = merge(X, y).corr()
                corr_X, corr_y = corr_matrix.iloc[:-1, :-1], corr_matrix.iloc[:-1, -1]

            corr = {}
            to_drop = []
            for col in corr_X:
                # Select columns that are corr
                corr[col] = corr_X[col][corr_X[col] >= self.max_correlation]

                # Always finds himself with correlation 1
                if len(corr[col]) > 1:
                    if y is None:
                        # Drop all but the first one
                        to_drop.extend(list(corr[col][1:].index))
                    else:
                        # Keep feature with the highest correlation with y
                        keep = corr_y[corr[col].index].idxmax()
                        to_drop.extend(list(corr[col].index.drop(keep)))

            for col in list(dict.fromkeys(to_drop)):
                corr_feature = corr[col].drop(col).index
                corr_value = corr[col].drop(col).round(4).astype(str)
                self.collinear = self.collinear.append(
                    {
                        "drop": col,
                        "corr_feature": ", ".join(corr_feature),
                        "corr_value": ", ".join(corr_value),
                    },
                    ignore_index=True,
                )

            X = X.drop(self.collinear["drop"], axis=1)

        if self.strategy is None:
            self._is_fitted = True
            return self  # Exit feature_engineering

        elif self.strategy.lower() == "univariate":
            solvers_dct = CustomDict(
                f_classif=f_classif,
                f_regression=f_regression,
                mutual_info_classif=mutual_info_classif,
                mutual_info_regression=mutual_info_regression,
                chi2=chi2,
            )

            if not self.solver:
                raise ValueError(
                    "Invalid value for the solver parameter. The "
                    f"value can't be None for strategy={self.strategy}"
                )
            elif self.solver in solvers_dct:
                solver = solvers_dct[self.solver]
            elif isinstance(self.solver, str):
                raise ValueError(
                    "Invalid value for the solver parameter, got "
                    f"{self.solver}. Choose from: {', '.join(solvers_dct)}."
                )
            else:
                solver = self.solver

            check_y()
            self._estimator = SelectKBest(
                score_func=solver,
                k=self._n_features,
            ).fit(X, y)

        elif self.strategy.lower() == "pca":
            # The PCA and TruncatedSVD both get all possible components to use
            # for the plots (n_components must be < n_features and <= n_rows)
            if is_sparse(X):
                self._estimator = TruncatedSVD(
                    n_components=min(len(X), X.shape[1] - 1),
                    algorithm="randomized" if self.solver is None else self.solver,
                    random_state=self.random_state,
                    **self.kwargs,
                )
            else:
                if not check_scaling(X):
                    self.scaler = Scaler().fit(X)
                    X = self.scaler.transform(X)

                s = lambda p: signature(estimator).parameters[p].default

                estimator = self._get_engine(PCA)
                self._estimator = estimator(
                    n_components=min(len(X), X.shape[1] - 1),
                    svd_solver=s("svd_solver") if self.solver is None else self.solver,
                    random_state=self.random_state,
                    **self.kwargs,
                )

            self._estimator.fit(X)
            self._estimator._comps = min(
                self._estimator.components_.shape[0], self._n_features
            )

        elif self.strategy.lower() == "sfm":
            # If any of these attr exists, model is already fitted
            if any(hasattr(solver, a) for a in ("coef_", "feature_importances_")):
                prefit = self._kwargs.pop("prefit", True)
            else:
                prefit = False

            # If threshold is not specified, select only based on _n_features
            if not self.kwargs.get("threshold"):
                self._kwargs["threshold"] = -np.inf

            self._estimator = strats[self.strategy](
                estimator=solver,
                max_features=self._n_features,
                prefit=prefit,
                **self._kwargs,
            )
            if prefit:
                if list(solver.feature_names_in_) != list(X.columns):
                    raise ValueError(
                        "Invalid value for the solver parameter. The "
                        f"{solver.__class__.__name__} estimator "
                        "is fitted using different columns than X!"
                    )
                self._estimator.estimator_ = solver
            else:
                check_y()
                self._estimator.fit(X, y)

        elif self.strategy.lower() == "sfs":
            check_y()

            if self.kwargs.get("scoring"):
                self._kwargs["scoring"] = get_custom_scorer(self.kwargs["scoring"])

            self._estimator = strats[self.strategy](
                estimator=solver,
                n_features_to_select=self._n_features,
                n_jobs=self.n_jobs,
                **self._kwargs,
            ).fit(X, y)

        elif self.strategy.lower() == "rfe":
            check_y()

            self._estimator = strats[self.strategy](
                estimator=solver,
                n_features_to_select=self._n_features,
                **self._kwargs,
            ).fit(X, y)

        elif self.strategy.lower() == "rfecv":
            check_y()

            if self.kwargs.get("scoring"):
                self._kwargs["scoring"] = get_custom_scorer(self.kwargs["scoring"])

            # Invert n_features to select them all (default option)
            if self._n_features == X.shape[1]:
                self._n_features = 1

            self._estimator = strats[self.strategy](
                estimator=solver,
                min_features_to_select=self._n_features,
                n_jobs=self.n_jobs,
                **self._kwargs,
            ).fit(X, y)

        else:
            check_y()

            # Either use a provided validation set or cross-validation over X
            kwargs = self.kwargs.copy()
            if "X_valid" in kwargs:
                if "y_valid" in kwargs:
                    X_valid, y_valid = self._prepare_input(
                        kwargs.pop("X_valid"), kwargs.pop("y_valid")
                    )
                else:
                    raise ValueError(
                        "Invalid value for the y_valid parameter. The value "
                        "cannot be absent when X_valid is provided."
                    )
            else:
                X_valid, y_valid = X, y

            # Get scoring for default objective_function
            if "objective_function" not in kwargs:
                if kwargs.get("scoring"):
                    kwargs["scoring"] = get_custom_scorer(kwargs["scoring"])
                else:
                    task = infer_task(y, goal=self.goal)
                    if task.startswith("bin"):
                        kwargs["scoring"] = get_custom_scorer("f1")
                    elif task.startswith("multi"):
                        kwargs["scoring"] = get_custom_scorer("f1_weighted")
                    else:
                        kwargs["scoring"] = get_custom_scorer("r2")

            self._estimator = strats[self.strategy](
                objective_function=kwargs.pop("objective_function", objective_function),
                minimize=kwargs.pop("minimize", False),
                logger=self.logger,
                **kwargs,
            )

            self._estimator.fit(
                model=solver,
                X_train=X,
                y_train=y,
                X_valid=X_valid,
                y_valid=y_valid,
                verbose=True if self.verbose >= 2 else False,
            )

        # Add the strategy estimator as attribute to the class
        setattr(self, self.strategy.lower(), self._estimator)

        # Assign feature importance (only for some strategies)
        if self.strategy.lower() in ("univariate", "sfm", "rfe", "rfecv"):
            estimator = getattr(self._estimator, "estimator_", self._estimator)
            scores = get_feature_importance(estimator)

            # Some strategies return scores for all features
            if len(scores) == X.shape[1]:
                self.feature_importance = pd.Series(
                    data=scores / max(scores),
                    index=X.columns,
                    name="feature_importance",
                    dtype="float",
                ).sort_values(ascending=False)[:self._n_features]
            else:
                self.feature_importance = pd.Series(
                    data=scores / max(scores),
                    index=X.columns[self._estimator.get_support(indices=True)],
                    name="feature_importance",
                    dtype="float",
                ).sort_values(ascending=False)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Does nothing. Only for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Transformed feature set.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Performing feature selection ...", 1)

        # Remove features with too high variance
        for key, value in self._high_variance.items():
            self.log(
                f" --> Feature {key} was removed due to high variance. "
                f"Value {value[0]} was the most repeated value with "
                f"{value[1]} ({value[1] / len(X):.1f}%) occurrences.", 2
            )
            X = X.drop(key, axis=1)

        # Remove features with too low variance
        for key, value in self._low_variance.items():
            self.log(
                f" --> Feature {key} was removed due to low variance. Value "
                f"{value[0]} repeated in {value[1]}% of the rows.", 2
            )
            X = X.drop(key, axis=1)

        # Remove features with too high correlation
        for col in self.collinear["drop"]:
            self.log(
                f" --> Feature {col} was removed due to "
                "collinearity with another feature.", 2
            )
            X = X.drop(col, axis=1)

        # Perform selection based on strategy
        if self.strategy is None:
            return X

        elif self.strategy.lower() == "univariate":
            self.log(
                f" --> The univariate test selected "
                f"{self._n_features} features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.univariate.get_support()[n]:
                    self.log(
                        f"   --> Dropping feature {column} "
                        f"(score: {self.univariate.scores_[n]:.2f}  "
                        f"p-value: {self.univariate.pvalues_[n]:.2f}).", 2
                    )
                    X = X.drop(column, axis=1)

        elif self.strategy.lower() == "pca":
            self.log(" --> Applying Principal Component Analysis...", 2)

            if self.scaler:
                self.log("   --> Scaling features...", 2)
                X = self.scaler.transform(X)

            X = to_df(
                data=self.pca.transform(X)[:, :self.pca._comps],
                index=X.index,
                columns=[f"pca{str(i)}" for i in range(self.pca._comps)],
            )

            var = np.array(self.pca.explained_variance_ratio_[:self._n_features])
            self.log(f"   --> Keeping {self.pca._comps} components.", 2)
            self.log(f"   --> Explained variance ratio: {round(var.sum(), 3)}", 2)

        elif self.strategy.lower() in ("sfm", "sfs", "rfe", "rfecv"):
            mask = self._estimator.get_support()
            self.log(
                f" --> {self.strategy.lower()} selected "
                f"{sum(mask)} features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not mask[n]:
                    self.log(f"   --> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

        else:  # Advanced strategies
            n_features = len(self._estimator.best_feature_list)
            self.log(
                f" --> {self.strategy.lower()} selected "
                f"{n_features} features from the dataset.", 2
            )

            for column in X:
                if column not in self._estimator.best_feature_list:
                    self.log(f"   --> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

        return X

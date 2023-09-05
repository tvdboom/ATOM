# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the feature engineering transformers.

"""

from __future__ import annotations

import re
from collections import defaultdict
from logging import Logger
from random import sample
from typing import Callable, Literal

import featuretools as ft
import joblib
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from scipy import stats
from sklearn.base import BaseEstimator, is_regressor
from sklearn.feature_selection import (
    RFE, RFECV, SelectFromModel, SelectKBest, SequentialFeatureSelector, chi2,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
)
from sklearn.model_selection import cross_val_score

from zoofs import (
    DragonFlyOptimization, GeneticOptimization, GreyWolfOptimization,
    HarrisHawkOptimization, ParticleSwarmOptimization,
)

from atom.basetransformer import BaseTransformer
from atom.data_cleaning import Scaler, TransformerMixin
from atom.models import MODELS
from atom.plots import FeatureSelectionPlot
from atom.utils.types import (
    BACKEND, BOOL, DATAFRAME, ENGINE, ESTIMATOR, FEATURES, FLOAT, FS_STRATS,
    INT, INT_TYPES, OPERATORS, SCALAR, SEQUENCE, SEQUENCE_TYPES, SERIES_TYPES,
    TARGET,
)
from atom.utils.utils import (
    CustomDict, check_is_fitted, check_scaling, composed, crash,
    get_custom_scorer, infer_task, is_sparse, lst, merge, method_to_log, sign,
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
    features: str or sequence, default=("day", "month", "year")
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
        - If None: Logging isn't used.
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
        import pandas as pd
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Add a datetime column
        X["date"] = pd.date_range(start="1/1/2018", periods=len(X))

        atom = ATOMClassifier(X, y)
        atom.feature_extraction(features=["day"], fmt="%d/%m/%Y", verbose=2)

        # Note the date_day column
        print(atom.dataset)
        ```

    === "stand-alone"
        ```pycon
        import pandas as pd
        from atom.feature_engineering import FeatureExtractor
        from sklearn.datasets import load_breast_cancer

        X, _ = load_breast_cancer(return_X_y=True, as_frame=True)

        # Add a datetime column
        X["date"] = pd.date_range(start="1/1/2018", periods=len(X))

        fe = FeatureExtractor(features=["day"], fmt="%Y-%m-%d", verbose=2)
        X = fe.transform(X)

        # Note the date_day column
        print(X)
        ```

    """

    _train_only = False

    def __init__(
        self,
        features: str | SEQUENCE = ("day", "month", "year"),
        fmt: str | SEQUENCE | None = None,
        *,
        encoding_type: Literal["ordinal", "cyclic"] = "ordinal",
        drop_columns: BOOL = True,
        verbose: Literal[0, 1, 2] = 0,
        logger: str | Logger | None = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.fmt = fmt
        self.features = features
        self.encoding_type = encoding_type
        self.drop_columns = drop_columns

    @composed(crash, method_to_log)
    def transform(self, X: FEATURES, y: TARGET | None = None) -> DATAFRAME:
        """Extract the new features.

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
        X, y = self._prepare_input(X, y, columns=getattr(self, "feature_names_in_", None))
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        self._log("Extracting datetime features...", 1)

        i = 0
        for name, column in X.select_dtypes(exclude="number").items():
            if column.dtype.name == "datetime64[ns]":
                col_dt = column
                self._log(f" --> Extracting features from column {name}.", 1)
            else:
                fmt = self.fmt[i] if isinstance(self.fmt, SEQUENCE_TYPES) else self.fmt
                col_dt = pd.to_datetime(
                    arg=column,
                    errors="coerce",  # Converts to NaT if he can't format
                    format=fmt,
                    infer_datetime_format=True,
                )

                # If >30% values are NaT, the conversion was unsuccessful
                nans = 100. * col_dt.isna().sum() / len(X)
                if nans >= 30:
                    continue  # Skip this column
                else:
                    i += 1
                    self._log(
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
                if not isinstance(values, SERIES_TYPES):
                    self._log(
                        f"   --> Extracting feature {fx} failed. "
                        "Result is not a Series.dt.", 2
                    )
                    continue

                min_val, max_val = 0, None  # max_val=None -> feature is not cyclic
                if self.encoding_type == "cyclic":
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
                new_name = f"{name}_{fx}"
                idx = X.columns.get_loc(name)
                if self.encoding_type == "ordinal" or max_val is None:
                    self._log(f"   --> Creating feature {new_name}.", 2)
                    X.insert(idx, new_name, values)
                elif self.encoding_type == "cyclic":
                    self._log(f"   --> Creating cyclic feature {new_name}.", 2)
                    pos = 2 * np.pi * (values - min_val) / np.array(max_val)
                    X.insert(idx, f"{new_name}_sin", np.sin(pos))
                    X.insert(idx + 1, f"{new_name}_cos", np.cos(pos))

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
        - If None: Logging isn't used.
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
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y)
        atom.feature_generation(strategy="dfs", n_features=5, verbose=2)

        # Note the texture error / worst symmetry column
        print(atom.dataset)
        ```

    === "stand-alone"
        ```pycon
        from atom.feature_engineering import FeatureGenerator
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        fg = FeatureGenerator(strategy="dfs", n_features=5, verbose=2)
        X = fg.fit_transform(X, y)

        # Note the radius error * worst smoothness column
        print(X)
        ```

    """

    _train_only = False

    def __init__(
        self,
        strategy: Literal["dfs", "gfg"] = "dfs",
        *,
        n_features: INT | None = None,
        operators: OPERATORS | SEQUENCE | None = None,
        n_jobs: INT = 1,
        verbose: Literal[0, 1, 2] = 0,
        logger: str | Logger | None = None,
        random_state: INT | None = None,
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
        self._dfs = None
        self._is_fitted = False

    @composed(crash, method_to_log)
    def fit(self, X: FEATURES, y: TARGET | None = None) -> FeatureGenerator:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe-like: Target columns with shape=(n_samples,
              n_targets) for multioutput tasks.

        Returns
        -------
        self
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        all_operators = dict(
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

        if self.n_features is not None and self.n_features <= 0:
            raise ValueError(
                "Invalid value for the n_features parameter."
                f"Value should be >0, got {self.n_features}."
            )

        if not self.operators:  # None or empty list
            operators = list(all_operators)
        else:
            operators = lst(self.operators)
            for op in operators:
                if op not in all_operators:
                    raise ValueError(
                        "Invalid value in the operators parameter, got "
                        f"{op}. Choose from: {', '.join(all_operators)}."
                    )

        self._log("Fitting FeatureGenerator...", 1)

        if self.strategy == "dfs":
            # Run deep feature synthesis with transformation primitives
            es = ft.EntitySet(dataframes={"X": (X, "_index", None, None, None, True)})
            self._dfs = ft.dfs(
                target_dataframe_name="X",
                entityset=es,
                trans_primitives=[all_operators[x] for x in operators],
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
                function_set=operators,
                feature_names=X.columns,
                verbose=kwargs.pop("verbose", 0 if self.verbose < 2 else 1),
                n_jobs=kwargs.pop("n_jobs", self.n_jobs),
                random_state=kwargs.pop("random_state", self.random_state),
                **kwargs,
            ).fit(X, y)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log)
    def transform(self, X: FEATURES, y: TARGET | None = None) -> DATAFRAME:
        """Generate new features.

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
        check_is_fitted(self)
        X, y = self._prepare_input(X, y, columns=self.feature_names_in_)

        self._log("Generating new features...", 1)

        if self.strategy.lower() == "dfs":
            es = ft.EntitySet(dataframes={"X": (X, "index", None, None, None, True)})
            dfs = ft.calculate_feature_matrix(
                features=self._dfs,
                entityset=es,
                n_jobs=self.n_jobs,
            )

            # Add the new features to the feature set
            X = pd.concat([X, dfs], axis=1).set_index("index")

            self._log(f" --> {len(self._dfs)} new features were added.", 2)

        else:
            # Get the names and fitness of the new features
            df = pd.DataFrame(columns=["name", "description", "fitness"])
            for i, fx in enumerate(self.gfg):
                if str(fx) not in X.columns:  # Drop unchanged features
                    df.loc[i] = ["", str(fx), fx.fitness_]

            # Check if any new features remain
            if len(df) == 0:
                self._log(
                    " --> The genetic algorithm didn't find any improving features.", 2
                )
                return X

            # Select the n_features with the highest fitness
            df = df.drop_duplicates()
            df = df.nlargest(self.n_features or len(df), columns="fitness")

            # If there are not enough features remaining, notify the user
            if len(df) != self.n_features:
                self._log(
                    f" --> Dropping {(self.n_features or len(self.gfg)) - len(df)} "
                    "features due to repetition.", 2)

            for i, array in enumerate(self.gfg.transform(X)[:, df.index].T):
                # If the column is new, use a default name
                counter = 0
                while True:
                    name = f"x{X.shape[1] + counter}"
                    if name not in X:
                        X[name] = array  # Add new feature to X
                        df.iat[i, 0] = name
                        break
                    else:
                        counter += 1

            self._log(f" --> {len(df)} new features were added.", 2)
            self.genetic_features = df.reset_index(drop=True)

        return X


class FeatureGrouper(BaseEstimator, TransformerMixin, BaseTransformer):
    """Extract statistics from similar features.

    Replace groups of features with related characteristics with new
    features that summarize statistical properties of the group. The
    statistical operators are calculated over every row of the group.
    The group names and features can be accessed through the `groups`
    method.

    This class can be accessed from atom through the [feature_grouping]
    [atomclassifier-feature_grouping] method. Read more in the
    [user guide][grouping-similar-features].

    !!! tip
        Use a regex pattern with the `groups` parameter to select
        groups easier, e.g. `atom.feature_grouping({"group1": "var_.+")`
        to select all features that start with `var_`.

    Parameters
    ----------
    group: dict
        Group names and features. Select the features by name, position
        or regex pattern. A feature can belong to multiple groups.

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
        - If None: Logging isn't used.
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
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y)
        atom.feature_grouping({"means": ["mean.+"]}, verbose=2)

        # Note the mean features are gone and the new std(means) feature
        print(atom.dataset)
        ```

    === "stand-alone"
        ```pycon
        from atom.feature_engineering import FeatureGrouper
        from sklearn.datasets import load_breast_cancer

        X, _ = load_breast_cancer(return_X_y=True, as_frame=True)

        # Group all features that start with mean
        fg = FeatureGrouper({"means": ["mean.+"]}, verbose=2)
        X = fg.transform(X)

        # Note the mean features are gone and the new std(means) feature
        print(X)
        ```

    """

    _train_only = False

    def __init__(
        self,
        group: dict[str, INT | str | SEQUENCE],
        *,
        operators: str | SEQUENCE | None = None,
        drop_columns: BOOL = True,
        verbose: Literal[0, 1, 2] = 0,
        logger: str | Logger | None = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.group = group
        self.operators = operators
        self.drop_columns = drop_columns
        self.groups = defaultdict(list)

    @composed(crash, method_to_log)
    def transform(self, X: FEATURES, y: TARGET | None = None) -> DATAFRAME:
        """Group features.

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
        X, _ = self._prepare_input(X, y, columns=getattr(self, "feature_names_in_", None))

        self._log("Grouping features...", 1)

        # Make the groups
        self.groups = defaultdict(list)
        for name, group in self.group.items():
            for col in lst(group):
                if isinstance(col, INT_TYPES):
                    try:
                        self.groups[name].append(X.columns[col])
                    except IndexError:
                        raise ValueError(
                            f"Invalid value for the groups parameter. Value {col} "
                            f"is out of range for a dataset with {X.shape[1]} columns."
                        )
                else:
                    # Find columns using regex matches
                    if matches := [c for c in X.columns if re.fullmatch(col, c)]:
                        self.groups[name].extend(matches)
                    else:
                        try:
                            self.groups[name].extend(list(X.select_dtypes(col).columns))
                        except TypeError:
                            raise ValueError(
                                "Invalid value for the groups parameter. "
                                f"Could not find any column that matches {col}."
                            )

        if self.operators is None:
            operators = ["min", "max", "mean", "median", "mode", "std"]
        else:
            operators = lst(self.operators)

        to_drop = set()
        for name, group in self.groups.items():
            for operator in operators:
                try:
                    result = X[group].apply(getattr(np, operator), axis=1)
                except AttributeError:
                    try:
                        result = getattr(stats, operator)(X[group], axis=1)[0]
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
            self._log(f" --> Group {name} successfully created.", 2)

        if self.drop_columns:
            X = X.drop(to_drop, axis=1)

        return X


class FeatureSelector(
    BaseEstimator,
    TransformerMixin,
    BaseTransformer,
    FeatureSelectionPlot,
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
        - The "sklearnex" and "cuml" engines are only supported for
          strategy="pca" with dense datasets.
        - If strategy="pca" and the data is dense and unscaled, it's
          scaled to mean=0 and std=1 before fitting the PCA transformer.
        - If strategy="pca" and the provided data is sparse, the used
          estimator is [TruncatedSVD][], which works more efficiently
          with sparse matrices.

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

    solver: str, func, estimator or None, default=None
        Solver/estimator to use for the feature selection strategy. See
        the corresponding documentation for an extended description of
        the choices. If None, the default value is used (only if
        strategy="pca"). Choose from:

        - If strategy="univariate":

            - "[f_classif][]"
            - "[f_regression][]"
            - "[mutual_info_classif][]"
            - "[mutual_info_regression][]"
            - "[chi2][]"
            - Any function with signature `func(X, y) -> tuple[scores, p-values]`.

        - If strategy="pca":

            - If data is dense:

                - If engine="sklearn":

                    - "auto" (default)
                    - "full"
                    - "arpack"
                    - "randomized"

                - If engine="sklearnex":

                    - "full" (default)

                - If engine="cuml":

                    - "full" (default)
                    - "jacobi"

            - If data is sparse:

                - "randomized" (default)
                - "arpack"

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

    max_repeated: int, float or None, default=1.0
        Remove categorical features with the same value in at least
        `max_repeated` rows. The default is to keep all features with
        non-zero variance, i.e. remove the features that have the same
        value in all samples.

        - If None: No check for maximum repetition.
        - If >1: Maximum number of repeated occurences.
        - If <=1: Maximum fraction of repeated occurences.

    max_correlation: float or None, default=1.0
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

    backend: str, default="loky"
        Parallelization backend. Read more in the
        [user guide][parallel-execution]. Choose from:

        - "loky": Single-node, process-based parallelism.
        - "multiprocessing": Legacy single-node, process-based
          parallelism. Less robust than `loky`.
        - "threading": Single-node, thread-based parallelism.
        - "ray": Multi-node, process-based parallelism.

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
        Any extra keyword argument for the strategy estimator. See the
        corresponding documentation for the available options.

    Attributes
    ----------
    collinear: pd.DataFrame
        Information on the removed collinear features. Columns include:

        - **drop:** Name of the dropped feature.
        - **corr_feature:** Names of the correlated features.
        - **corr_value:** Corresponding correlation coefficients.

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
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y)
        atom.feature_selection(strategy="pca", n_features=12, verbose=2)

        # Note that the column names changed
        print(atom.dataset)

        atom.plot_pca()
        ```

    === "stand-alone"
        ```pycon
        from atom.feature_engineering import FeatureSelector
        from sklearn.datasets import load_breast_cancer

        X, _ = load_breast_cancer(return_X_y=True, as_frame=True)

        fs = FeatureSelector(strategy="pca", n_features=12, verbose=2)
        X = fs.fit_transform(X)

        # Note that the column names changed
        print(X)
        ```

    """

    _train_only = False

    def __init__(
        self,
        strategy: FS_STRATS | None = None,
        *,
        solver: str | Callable[..., SEQUENCE | SEQUENCE] | ESTIMATOR | None = None,
        n_features: SCALAR | None = None,
        min_repeated: SCALAR | None = 2,
        max_repeated: SCALAR | None = 1.0,
        max_correlation: FLOAT | None = 1.0,
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: ENGINE = {"data": "numpy", "estimator": "sklearn"},
        backend: BACKEND = "loky",
        verbose: Literal[0, 1, 2] = 0,
        logger: str | Logger | None = None,
        random_state: INT | None = None,
        **kwargs,
    ):
        super().__init__(
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            backend=backend,
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
        self.scaler = None

        self._multioutput = None
        self._n_features = None
        self._kwargs = kwargs.copy()
        self._high_variance = {}
        self._low_variance = {}
        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log)
    def fit(self, X: FEATURES, y: TARGET | None = None) -> FeatureSelector:
        """Fit the feature selector to the data.

        The univariate, sfm (when model is not fitted), sfs, rfe and
        rfecv strategies need a target column. Leaving it None raises
        an exception.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence, dataframe-like or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe-like: Target columns with shape=(n_samples,
              n_targets) for multioutput tasks.

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
                return np.mean(cv, axis=0)
            else:
                model.fit(X_train, y_train)
                return scoring(model, X_valid, y_valid)

        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        strategies = CustomDict(
            univariate="SelectKBest",
            pca="PCA",
            sfm="SelectFromModel",
            sfs="SequentialFeatureSelector",
            rfe="RFE",
            rfecv="RFECV",
            pso=ParticleSwarmOptimization,
            hho=HarrisHawkOptimization,
            gwo=GreyWolfOptimization,
            dfo=DragonFlyOptimization,
            go=GeneticOptimization,
        )

        if isinstance(self.strategy, str):
            if self.strategy not in ("univariate", "pca"):
                if self.solver is None:
                    raise ValueError(
                        "Invalid value for the solver parameter. The "
                        f"value can't be None for strategy={self.strategy}"
                    )
                elif isinstance(self.solver, str):
                    # Assign goal to initialize the predefined model
                    if self.solver[-6:] == "_class":
                        goal = "class"
                        solver = self.solver[:-6]
                    elif self.solver[-4:] == "_reg":
                        goal = "reg"
                        solver = self.solver[:-4]
                    else:
                        solver = self.solver

                    # Get estimator from predefined models
                    if solver in MODELS:
                        model = MODELS[solver](
                            goal=goal,
                            n_jobs=self.n_jobs,
                            device=self.device,
                            engine=self.engine,
                            backend=self.backend,
                            verbose=self.verbose,
                            logger=self.logger,
                            random_state=self.random_state,
                        )
                        model.task = infer_task(y, goal)
                        solver = model._get_est()
                    else:
                        raise ValueError(
                            "Invalid value for the solver parameter. Unknown "
                            f"model: {solver}. Choose from: {', '.join(MODELS.keys())}."
                        )
                else:
                    solver = self.solver

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
        elif self.min_repeated <= 1:
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

        self._log("Fitting FeatureSelector...", 1)

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
                self.collinear = pd.concat(
                    [
                        self.collinear,
                        pd.DataFrame(
                            {
                                "drop": [col],
                                "corr_feature": [", ".join(corr_feature)],
                                "corr_value": [", ".join(corr_value)],
                            }
                        )
                    ],
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
            self._estimator = SelectKBest(solver, k=self._n_features).fit(X, y)

        elif self.strategy.lower() == "pca":
            if not is_sparse(X):
                # PCA requires the features to be scaled
                if not check_scaling(X):
                    self.scaler = Scaler()
                    X = self.scaler.fit_transform(X)

                estimator = self._get_est_class("PCA", "decomposition")
                solver_param = "svd_solver"
            else:
                estimator = self._get_est_class("TruncatedSVD", "decomposition")
                solver_param = "algorithm"

            if self.solver is None:
                solver = sign(estimator)[solver_param].default
            else:
                solver = self.solver

            # The PCA and TruncatedSVD both get all possible components to use
            # for the plots (n_components must be < n_features and <= n_rows)
            self._estimator = estimator(
                n_components=min(len(X), X.shape[1] - 1),
                **{solver_param: solver},
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

            self._estimator = SelectFromModel(
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

        elif self.strategy.lower() in ("sfs", "rfe", "rfecv"):
            if self.strategy.lower() == "sfs":
                check_y()

                if self.kwargs.get("scoring"):
                    self._kwargs["scoring"] = get_custom_scorer(self.kwargs["scoring"])

                self._estimator = SequentialFeatureSelector(
                    estimator=solver,
                    n_features_to_select=self._n_features,
                    n_jobs=self.n_jobs,
                    **self._kwargs,
                )

            elif self.strategy.lower() == "rfe":
                check_y()

                self._estimator = RFE(
                    estimator=solver,
                    n_features_to_select=self._n_features,
                    **self._kwargs,
                )

            elif self.strategy.lower() == "rfecv":
                check_y()

                if self.kwargs.get("scoring"):
                    self._kwargs["scoring"] = get_custom_scorer(self.kwargs["scoring"])

                # Invert n_features to select them all (default option)
                if self._n_features == X.shape[1]:
                    self._n_features = 1

                self._estimator = RFECV(
                    estimator=solver,
                    min_features_to_select=self._n_features,
                    n_jobs=self.n_jobs,
                    **self._kwargs,
                )

            # Use parallelization backend
            with joblib.parallel_backend(backend=self.backend):
                self._estimator.fit(X, y)

        else:
            check_y()

            # Either use a provided validation set or cross-validation over X
            if "X_valid" in (kwargs := self.kwargs.copy()):
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
                    goal = "reg" if is_regressor(solver) else "class"
                    task = infer_task(y, goal=goal)
                    if task.startswith("bin"):
                        kwargs["scoring"] = get_custom_scorer("f1")
                    elif task.startswith("multi") and goal.startswith("class"):
                        kwargs["scoring"] = get_custom_scorer("f1_weighted")
                    else:
                        kwargs["scoring"] = get_custom_scorer("r2")

            self._estimator = strategies[self.strategy](
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
                verbose=self.verbose >= 2,
            )

        # Add the strategy estimator as attribute to the class
        setattr(self, self.strategy.lower(), self._estimator)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log)
    def transform(self, X: FEATURES, y: TARGET | None = None) -> DATAFRAME:
        """Transform the data.

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
        check_is_fitted(self)
        X, y = self._prepare_input(X, y, columns=self.feature_names_in_)

        self._log("Performing feature selection ...", 1)

        # Remove features with too high variance
        for key, value in self._high_variance.items():
            self._log(
                f" --> Feature {key} was removed due to high variance. "
                f"Value {value[0]} was the most repeated value with "
                f"{value[1]} ({value[1] / len(X):.1f}%) occurrences.", 2
            )
            X = X.drop(key, axis=1)

        # Remove features with too low variance
        for key, value in self._low_variance.items():
            self._log(
                f" --> Feature {key} was removed due to low variance. Value "
                f"{value[0]} repeated in {value[1]}% of the rows.", 2
            )
            X = X.drop(key, axis=1)

        # Remove features with too high correlation
        for col in self.collinear["drop"]:
            self._log(
                f" --> Feature {col} was removed due to "
                "collinearity with another feature.", 2
            )
            X = X.drop(col, axis=1)

        # Perform selection based on strategy
        if self.strategy is None:
            return X

        elif self.strategy.lower() == "univariate":
            self._log(
                f" --> The univariate test selected "
                f"{self._n_features} features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.univariate.get_support()[n]:
                    self._log(
                        f"   --> Dropping feature {column} "
                        f"(score: {self.univariate.scores_[n]:.2f}  "
                        f"p-value: {self.univariate.pvalues_[n]:.2f}).", 2
                    )
                    X = X.drop(column, axis=1)

        elif self.strategy.lower() == "pca":
            self._log(" --> Applying Principal Component Analysis...", 2)

            if self.scaler:
                self._log("   --> Scaling features...", 2)
                X = self.scaler.transform(X)

            X = to_df(
                data=self.pca.transform(X)[:, :self.pca._comps],
                index=X.index,
                columns=[f"pca{str(i)}" for i in range(self.pca._comps)],
            )

            var = np.array(self.pca.explained_variance_ratio_[:self._n_features])
            self._log(f"   --> Keeping {self.pca._comps} components.", 2)
            self._log(f"   --> Explained variance ratio: {round(var.sum(), 3)}", 2)

        elif self.strategy.lower() in ("sfm", "sfs", "rfe", "rfecv"):
            mask = self._estimator.get_support()
            self._log(
                f" --> {self.strategy.lower()} selected "
                f"{sum(mask)} features from the dataset.", 2
            )

            for n, column in enumerate(X):
                if not mask[n]:
                    if hasattr(self._estimator, "ranking_"):
                        self._log(
                            f"   --> Dropping feature {column} "
                            f"(rank {self._estimator.ranking_[n]}).", 2
                        )
                    else:
                        self._log(f"   --> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

        else:  # Advanced strategies
            self._log(
                f" --> {self.strategy.lower()} selected "
                f"{len(self._estimator.best_feature_list)} features from the dataset.", 2
            )

            for column in X:
                if column not in self._estimator.best_feature_list:
                    self._log(f"   --> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

        return X

# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the feature engineering transformers.

"""

from inspect import signature
from random import sample
from typing import Optional, Union

import featuretools as ft
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
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
from atom.plots import FSPlotter
from atom.utils import (
    FLOAT, INT, SCALAR, SEQUENCE, SEQUENCE_TYPES, X_TYPES, Y_TYPES, CustomDict,
    check_is_fitted, check_scaling, composed, crash, get_custom_scorer,
    get_feature_importance, infer_task, is_sparse, lst, method_to_log, to_df, merge
)


class FeatureExtractor(BaseEstimator, TransformerMixin, BaseTransformer):
    """Extract features from datetime columns.

    Create new features extracting datetime elements (day, month,
    year, etc...) from the provided columns. Columns of dtype
    `datetime64` are used as is. Categorical columns that can be
    successfully converted to a datetime format (less than 30% NaT
    values after conversion) are also used.

    Parameters
    ----------
    features: str or sequence, optional (default=["day", "month", "year"])
        Features to create from the datetime columns. Note that
        created features with zero variance (e.g. the feature hour
        in a column that only contains dates) are ignored. Allowed
        values are datetime attributes from `pandas.Series.dt`.

    fmt: str, sequence or None, optional (default=None)
        Format (`strptime`) of the categorical columns that need
        to be converted to datetime. If sequence, the n-th format
        corresponds to the n-th categorical column that can be
        successfully converted. If None, the format is inferred
        automatically from the first non NaN value. Values that can
        not be converted are returned as NaT.

    encoding_type: str, optional (default="ordinal")
        Type of encoding to use. Choose from:
            - "ordinal": Encode features in increasing order.
            - "cyclic": Encode features using sine and cosine to capture
                        their cyclic nature. Note that this creates two
                        columns for every feature. Non-cyclic features
                        still use ordinal encoding.

    drop_columns: bool, optional (default=True)
        Whether to drop the original columns after extracting the
        features from it.

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
    feature_names_in_: np.array
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    """

    def __init__(
        self,
        features: Union[str, SEQUENCE_TYPES] = ["day", "month", "year"],
        fmt: Optional[Union[str, SEQUENCE_TYPES]] = None,
        encoding_type: str = "ordinal",
        drop_columns: bool = True,
        verbose: INT = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.fmt = fmt
        self.features = features
        self.encoding_type = encoding_type
        self.drop_columns = drop_columns

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Extract the new features.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Transformed feature set.

        """

        def encode_variable(idx, name, values, min_val=0, max_val=None):
            """Encode a feature in an ordinal or cyclic fashion."""
            if self.encoding_type.lower() == "ordinal" or max_val is None:
                self.log(f"   >>> Creating feature {name}.", 2)
                X.insert(idx, name, values)
            elif self.encoding_type.lower() == "cyclic":
                self.log(f"   >>> Creating cyclic feature {name}.", 2)
                pos = 2 * np.pi * (values - min_val) / np.array(max_val)
                X.insert(idx, f"{name}_sin", np.sin(pos))
                X.insert(idx, f"{name}_cos", np.cos(pos))

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
                self.log(f" --> Extracting features from datetime column {name}.", 1)
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
                        f"   >>> Extracting feature {fx} failed. "
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
                encode_variable(
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
    """Apply automated feature engineering.

    Create new combinations of existing features to capture the
    non-linear relations between the original features.

    Parameters
    ----------
    strategy: str, optional (default="dfs")
        Strategy to crate new features. Choose from:
            - "dfs": Deep Feature Synthesis.
            - "gfg": Genetic Feature Generation.

    n_features: int or None, optional (default=None)
        Maximum number of newly generated features to add to the
        dataset. If None, select all created features.

    operators: str, sequence or None, optional (default=None)
        Mathematical operators to apply on the features. None for all.
        Choose from: add, sub, mul, div, abs, sqrt, log, inv, sin, cos,
        tan.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 + `n_jobs`.

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
        Additional keyword arguments for the SymbolicTransformer
        instance. Only for the gfg strategy.

    Attributes
    ----------
    gfg: SymbolicTransformer
        Object used to calculate the genetic features. Only for the
        gfg strategy.

    genetic_features: pd.DataFrame
        Information on the newly created non-linear features. Only for
        the gfg strategy. Columns include:
            - name: Name of the feature (automatically created).
            - description: Operators used to create this feature.
            - fitness: Fitness score.

    feature_names_in_: np.array
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    """

    def __init__(
        self,
        strategy: str = "dfs",
        n_features: Optional[INT] = None,
        operators: Optional[Union[str, SEQUENCE_TYPES]] = None,
        n_jobs: INT = 1,
        verbose: INT = 0,
        logger: Optional[Union[str, callable]] = None,
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
    def fit(self, X: X_TYPES, y: Y_TYPES):
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str or sequence
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        Returns
        -------
        FeatureGenerator

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
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Generate new features.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Transformed feature set.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Creating new features...", 1)

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


class FeatureSelector(BaseEstimator, TransformerMixin, BaseTransformer, FSPlotter):
    """Apply feature selection techniques.

    Remove features according to the selected strategy. Ties between
    features with equal scores are broken in an unspecified way.
    Additionally, remove multicollinear and low variance features.

    Parameters
    ----------
    strategy: str or None, optional (default=None)
        Feature selection strategy to use. Choose from:
            - None: Do not perform any feature selection strategy.
            - "univariate": Univariate statistical F-test.
            - "pca": Principal Component Analysis.
            - "sfm": Select best features according to a model.
            - "sfs": Sequential Feature Selection.
            - "rfe": Recursive Feature Elimination.
            - "rfecv": RFE with cross-validated selection.
            - "pso": Particle Swarm Optimization.
            - "hho": Harris Hawks Optimization.
            - "gwo": Grey Wolf Optimization.
            - "dfo": Dragonfly Optimization.
            - "genetic": Genetic Optimization.

    solver: str, estimator or None, optional (default=None)
        Solver/model to use for the feature selection strategy. See the
        corresponding documentation for an extended description of the
        choices. If None, use the estimator's default value (only pca).
            - for "univariate", choose from:
                + "f_classif"
                + "f_regression"
                + "mutual_info_classif"
                + "mutual_info_regression"
                + "chi2"
                + Any function taking two arrays (X, y), and returning
                  arrays (scores, p-values).
            - for "pca", choose from:
                + if dense data:
                    * "auto" (default)
                    * "full"
                    * "arpack"
                    * "randomized"
                + if sparse data:
                    * "randomized" (default)
                    * "arpack"
                + if gpu implementation:
                    * "full" (default)
                    * "jacobi"
                    * "auto"
            - for the remaining strategies:
                The base estimator. For sfm, rfe and rfecv, it should
                have either a `feature_importances_` or `coef_`
                attribute after fitting. You can use one of ATOM's
                predefined models. Add `_class` or `_reg` after the
                model's name to specify a classification or regression
                task, e.g. `solver="LGB_reg"` (not necessary if called
                from atom). No default option.

    n_features: int, float or None, optional (default=None)
        Number of features to select. Choose from:
            - if None: Select all features.
            - if < 1: Fraction of the total features to select.
            - if >= 1: Number of features to select.

        If strategy="sfm" and the threshold parameter is not specified,
        the threshold is automatically set to `-inf` to select
        `n_features` number of features.

        If strategy="rfecv", `n_features` is the minimum number of
        features to select.

        This parameter is ignored if any of the following strategies
        is selected: pso, hho, gwo, dfo, genetic.

    max_frac_repeated: float or None, optional (default=1.)
        Remove features with the same value in at least this fraction
        of the total rows. The default is to keep all features with
        non-zero variance, i.e. remove the features that have the same
        value in all samples. If None, skip this step.

    max_correlation: float or None, optional (default=1.)
        Minimum absolute Pearson correlation to identify correlated
        features. For each group, it removes all except the feature
        with the highest correlation to `y` (if provided, else it
        removes all but the first). The default value removes equal
        columns. If None, skip this step.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 + `n_jobs`.

    gpu: bool or str, optional (default=False)
        Train strategy on GPU (instead of CPU). Only for strategy="pca".
            - If False: Always use CPU implementation.
            - If True: Use GPU implementation if possible.
            - If "force": Force GPU implementation.

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
        Any extra keyword argument for the strategy estimator. See the
        corresponding documentation for the available options.

    Attributes
    ----------
    collinear: pd.DataFrame
        Information on the removed collinear features. Columns include:
            - drop: Name of the dropped feature.
            - corr_feature: Name of the correlated feature(s).
            - corr_value: Corresponding correlation coefficient(s).

    feature_importance: list
        Remaining features ordered by importance. Only if strategy in
        ["univariate", "sfm", "rfe", "rfecv"]. For rfe and rfecv, the
        importance is extracted from the external estimator fitted on
        the reduced set.

    <strategy>: sklearn transformer
        Object used to transform the data, e.g. `balancer.pca` for the pca
        strategy.

    feature_names_in_: np.array
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    """

    def __init__(
        self,
        strategy: Optional[str] = None,
        solver: Optional[Union[str, callable]] = None,
        n_features: Optional[SCALAR] = None,
        max_frac_repeated: Optional[SCALAR] = 1.0,
        max_correlation: Optional[FLOAT] = 1.0,
        n_jobs: INT = 1,
        gpu: Union[bool, str] = False,
        verbose: INT = 0,
        logger: Optional[Union[str, callable]] = None,
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
        self.max_frac_repeated = max_frac_repeated
        self.max_correlation = max_correlation
        self.kwargs = kwargs

        self.collinear = pd.DataFrame(columns=["drop", "corr_feature", "corr_value"])
        self.feature_importance = None
        self.scaler = None

        self._n_features = None
        self._kwargs = kwargs.copy()
        self._low_variance = {}
        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit the feature selector to the data.

        Note that the univariate, sfm (when model is not fitted), sfs,
        rfe and rfecv strategies need a target column. Leaving it None
        will raise an exception.

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
        FeatureSelector
            Fitted instance of self.

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
            genetic=GeneticOptimization,
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

        if self.max_frac_repeated is not None and not 0 <= self.max_frac_repeated <= 1:
            raise ValueError(
                "Invalid value for the max_frac_repeated parameter. Value "
                f"should be between 0 and 1, got {self.max_frac_repeated}."
            )

        if self.max_correlation is not None and not 0 <= self.max_correlation <= 1:
            raise ValueError(
                "Invalid value for the max_correlation parameter. Value "
                f"shouldbe between 0 and 1, got {self.max_correlation}."
            )

        self.log("Fitting FeatureSelector...", 1)

        # Remove features with too low variance
        if self.max_frac_repeated:
            for name, column in X.items():
                for category, count in column.value_counts().items():
                    # If count < fraction of total, drop column
                    if count >= self.max_frac_repeated * len(X):
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

                estimator = self._get_gpu(PCA)
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
                if len(self._estimator.get_support()) != X.shape[1]:
                    raise ValueError(
                        "Invalid value for the solver parameter. The "
                        f"{solver.__class__.__name__} estimator "
                        "is fitted with different columns than X!"
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

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Transform the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Only for continuity of API.

        Returns
        -------
        pd.DataFrame
            Transformed feature set.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)
        columns = X.columns  # Save columns for sfm

        self.log("Performing feature selection ...", 1)

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
            indices = np.argsort(get_feature_importance(self.univariate))
            best_fxs = [X.columns[idx] for idx in indices][::-1]
            self.log(
                f" --> The univariate test selected "
                f"{self._n_features} features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.univariate.get_support()[n]:
                    self.log(
                        f"   >>> Dropping feature {column} "
                        f"(score: {self.univariate.scores_[n]:.2f}  "
                        f"p-value: {self.univariate.pvalues_[n]:.2f}).", 2
                    )
                    X = X.drop(column, axis=1)

            self.feature_importance = [fx for fx in best_fxs if fx in X.columns]

        elif self.strategy.lower() == "pca":
            self.log(" --> Applying Principal Component Analysis...", 2)

            if self.scaler:
                self.log("   >>> Scaling features...", 2)
                X = self.scaler.transform(X)

            X = to_df(
                data=self.pca.transform(X)[:, :self.pca._comps],
                index=X.index,
                columns=[f"pca{str(i)}" for i in range(self.pca._comps)],
            )

            var = np.array(self.pca.explained_variance_ratio_[:self._n_features])
            self.log(f"   >>> Keeping {self.pca._comps} components.", 2)
            self.log(f"   >>> Explained variance ratio: {round(var.sum(), 3)}", 2)

        elif self.strategy.lower() == "sfm":
            # Here we use columns since some cols could be removed before by
            # variance or correlation checks and there would be cols mismatch
            indices = np.argsort(get_feature_importance(self.sfm.estimator_))
            best_fxs = [columns[idx] for idx in indices][::-1]
            self.log(
                f" --> sfm selected {sum(self.sfm.get_support())} "
                "features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.sfm.get_support()[n]:
                    self.log(f"   >>> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

            self.feature_importance = [fx for fx in best_fxs if fx in X.columns]

        elif self.strategy.lower() == "sfs":
            self.log(
                f" --> sfs selected {self.sfs.n_features_to_select_}"
                " features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.sfs.support_[n]:
                    self.log(f"   >>> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

        elif self.strategy.lower() in ("rfe", "rfecv"):
            self.log(
                f" --> {self.strategy.upper()} selected "
                f"{self._estimator.n_features_} features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self._estimator.support_[n]:
                    self.log(
                        f"   >>> Dropping feature {column} "
                        f"(rank {self._estimator.ranking_[n]}).", 2
                    )
                    X = X.drop(column, axis=1)

            idx = np.argsort(get_feature_importance(self._estimator.estimator_))
            self.feature_importance = list(X.columns[idx][::-1])

        else:  # Advanced strategies
            n_features = len(self._estimator.best_feature_list)
            self.log(
                f" --> {self.strategy.lower()} selected "
                f"{n_features} features from the dataset.", 2
            )

            for column in X:
                if column not in self._estimator.best_feature_list:
                    self.log(f"   >>> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

        return X

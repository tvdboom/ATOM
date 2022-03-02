# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the feature engineering transformers.

"""

# Standard packages
import random
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union

# Other packages
import featuretools as ft
from zoofs import ParticleSwarmOptimization
from woodwork.column_schema import ColumnSchema
from gplearn.genetic import SymbolicTransformer
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    chi2,
    SelectKBest,
    SelectFromModel,
    RFE,
    RFECV,
    SequentialFeatureSelector,
)

# Own modules
from .models import MODELS
from .basetransformer import BaseTransformer
from .data_cleaning import TransformerMixin, Scaler
from .plots import FSPlotter
from .utils import (
    SCALAR, SEQUENCE, SEQUENCE_TYPES, X_TYPES, Y_TYPES, lst, to_df,
    is_sparse, get_custom_scorer, check_scaling, check_is_fitted,
    get_feature_importance, composed, crash, method_to_log,
)


def custom_function_for_scorer(model, X_train, y_train, X_valid, y_valid, scorer):  
    model.fit(X_train,y_train)  
    return scorer(model, X_valid, y_valid)


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

    """

    def __init__(
        self,
        features: Union[str, SEQUENCE_TYPES] = ["day", "month", "year"],
        fmt: Optional[Union[str, SEQUENCE_TYPES]] = None,
        encoding_type: str = "ordinal",
        drop_columns: bool = True,
        verbose: int = 0,
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
            elif name in X.select_dtypes(exclude="number").columns:
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

    Use Deep feature Synthesis or a genetic algorithm to create new
    combinations of existing features to capture the non-linear
    relations between the original features.

    Parameters
    ----------
    strategy: str, optional (default="DFS")
        Strategy to crate new features. Choose from:
            - "DFS" to use Deep Feature Synthesis.
            - "GFG" or "genetic" to use Genetic Feature Generation.

    n_features: int or None, optional (default=None)
        Number of newly generated features to add to the dataset (no
        more than 1% of the population for the genetic strategy). If
        None, select all created features.

    generations: int, optional (default=20)
        Number of generations to evolve. Only for the genetic strategy.

    population: int, optional (default=500)
        Number of programs in each generation. Only for the genetic
        strategy.

    operators: str, sequence or None, optional (default=None)
        Mathematical operators to apply on the features. None for all.
        Choose from: add, sub, mul, div, sqrt, log, inv, sin, cos, tan.

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

    Attributes
    ----------
    symbolic_transformer: SymbolicTransformer
        Object used to calculate the genetic features. Only for the
        genetic strategy.

    genetic_features: pd.DataFrame
        Information on the newly created non-linear features. Only for
        the genetic strategy. Columns include:
            - name: Name of the feature (automatically created).
            - description: Operators used to create this feature.
            - fitness: Fitness score.

    """

    def __init__(
        self,
        strategy: str = "DFS",
        n_features: Optional[int] = None,
        generations: int = 20,
        population: int = 500,
        operators: Optional[Union[str, SEQUENCE_TYPES]] = None,
        n_jobs: int = 1,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            logger=logger,
            random_state=random_state,
        )
        self.strategy = strategy
        self.n_features = n_features
        self.generations = generations
        self.population = population
        self.operators = operators

        self.symbolic_transformer = None
        self.genetic_features = None
        self._operators = None
        self._dfs_features = None
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

        if self.n_features is not None and self.n_features <= 0:
            raise ValueError(
                "Invalid value for the n_features parameter."
                f"Value should be >0, got {self.n_features}."
            )

        if self.strategy.lower() in ("gfg", "genetic"):
            if self.population < 100:
                raise ValueError(
                    "Invalid value for the population parameter."
                    f"Value should be >100, got {self.population}."
                )
            if self.generations < 1:
                raise ValueError(
                    "Invalid value for the generations parameter."
                    f"Value should be >100, got {self.generations}."
                )
            if self.n_features and self.n_features > int(0.01 * self.population):
                raise ValueError(
                    "Invalid value for the n_features parameter. Value "
                    f"should be <1% of the population, got {self.n_features}."
                )
        elif self.strategy.lower() != "dfs":
            raise ValueError(
                "Invalid value for the strategy parameter. Value "
                f"should be either 'dfs' or 'genetic', got {self.strategy}."
            )

        default = ["add", "sub", "mul", "div", "sqrt", "log", "sin", "cos", "tan"]
        if not self.operators:  # None or empty list
            self._operators = default
        else:
            self._operators = lst(self.operators)
            for operator in self._operators:
                if operator.lower() not in default:
                    raise ValueError(
                        "Invalid value in the operators parameter, got "
                        f"{operator}. Choose from: {', '.join(default)}."
                    )

        self.log("Fitting FeatureGenerator...", 1)

        if self.strategy.lower() == "dfs":
            trans_primitives = []
            for operator in self._operators:
                if operator.lower() == "add":
                    trans_primitives.append("add_numeric")
                elif operator.lower() == "sub":
                    trans_primitives.append("subtract_numeric")
                elif operator.lower() == "mul":
                    trans_primitives.append("multiply_numeric")
                elif operator.lower() == "div":
                    trans_primitives.append("divide_numeric")
                elif operator.lower() in ("sqrt", "log", "sin", "cos", "tan"):
                    trans_primitives.append(
                        ft.primitives.make_trans_primitive(
                            function=lambda x: getattr(np, operator.lower())(x),
                            input_types=[ColumnSchema()],
                            return_type=ColumnSchema(semantic_tags=["numeric"]),
                            name=operator.lower(),
                        )
                    )

            # Run deep feature synthesis with transformation primitives
            es = ft.EntitySet(dataframes={"X": (X, "_index", None, None, None, True)})
            self._dfs_features = ft.dfs(
                target_dataframe_name="X",
                entityset=es,
                trans_primitives=trans_primitives,
                max_depth=1,
                features_only=True,
                ignore_columns={"X": ["_index"]},
            )

            # Since dfs doesn't return a specific feature order, we
            # enforce order by name to be deterministic
            new_dfs = []
            for feature in sorted(map(str, self._dfs_features[X.shape[1] - 1:])):
                for fx in self._dfs_features:
                    if feature == str(fx):
                        new_dfs.append(fx)
                        break
            self._dfs_features = self._dfs_features[: X.shape[1] - 1] + new_dfs

            # Make sure there are enough features (-1 because of index)
            max_features = len(self._dfs_features) - (X.shape[1] - 1)
            if not self.n_features or self.n_features > max_features:
                n_final_features = max_features
            else:
                n_final_features = self.n_features

            # Get random indices from the feature list
            idx_old = range(X.shape[1] - 1)
            idx_new = random.sample(
                range(X.shape[1] - 1, len(self._dfs_features)), n_final_features
            )
            idx = list(idx_old) + list(idx_new)

            # Get random selection of features
            self._dfs_features = [
                value for i, value in enumerate(self._dfs_features) if i in idx
            ]

        else:
            self.symbolic_transformer = SymbolicTransformer(
                generations=self.generations,
                population_size=self.population,
                hall_of_fame=int(0.1 * self.population),
                n_components=int(0.01 * self.population),
                init_depth=(1, 2),
                function_set=self._operators,
                feature_names=X.columns,
                verbose=0 if self.verbose < 2 else 1,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
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
            Dataset containing the newly generated features.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Creating new features...", 1)

        if self.strategy.lower() == "dfs":
            index = X.index
            es = ft.EntitySet(dataframes={"X": (X, "index", None, None, None, True)})
            X = ft.calculate_feature_matrix(
                    features=self._dfs_features,
                    entityset=es,
                    n_jobs=self.n_jobs,
                )
            X.index = index

            self.log(
                f" --> {len(self._dfs_features)} new "
                "features were added to the dataset.", 2
            )

        else:
            new_features = self.symbolic_transformer.transform(X)

            # ix = indices of new fxs that are not in the original set
            # descript = operators applied to create the new features
            # fitness = list of fitness scores of the new features
            ix, descript, fitness = [], [], []
            for i, program in enumerate(self.symbolic_transformer):
                if str(program) not in X.columns:
                    ix.append(i)
                descript.append(str(program))
                fitness.append(program.fitness_)

            # Remove all identical features to those in the dataset
            new_features = new_features[:, ix]
            descript = [item for i, item in enumerate(descript) if i in ix]
            fitness = [item for i, item in enumerate(fitness) if i in ix]

            # Indices of all non duplicate elements in list
            ix = [ix for ix, v in enumerate(descript) if v not in descript[:ix]]

            # Remove all duplicate elements
            new_features = new_features[:, ix]
            descript = [item for i, item in enumerate(descript) if i in ix]
            fitness = [item for i, item in enumerate(fitness) if i in ix]

            # Check if any new features remain in the loop
            if len(descript) == 0:
                self.log(
                    " --> WARNING! The genetic algorithm couldn't "
                    "find any improving non-linear features.", 1
                )
                return X

            # Get indices of the best features
            if self.n_features and len(descript) > self.n_features:
                index = np.argpartition(fitness, -self.n_features)[-self.n_features:]
            else:
                index = range(len(descript))

            # Select best features only
            new_features = new_features[:, index]

            # Create the genetic_features attribute
            features_df = pd.DataFrame(columns=["name", "description", "fitness"])
            for i, idx in enumerate(index):
                features_df = features_df.append(
                    {
                        "name": "feature " + str(1 + i + len(X.columns)),
                        "description": descript[idx],
                        "fitness": fitness[idx],
                    },
                    ignore_index=True,
                )
            self.genetic_features = features_df

            self.log(
                f" --> {len(self.genetic_features)} new "
                "features were added to the dataset.", 2
            )

            cols = list(X.columns) + list(self.genetic_features["name"])
            X = pd.DataFrame(np.hstack((X, new_features)), index=X.index, columns=cols)

        return X


class FeatureSelector(BaseEstimator, TransformerMixin, BaseTransformer, FSPlotter):
    """Apply feature selection techniques.

    Remove features according to the selected strategy. Ties between
    features with equal scores are broken in an unspecified way.
    Additionally, removes features with too low variance and finds
    pairs of collinear features based on the Pearson correlation
    coefficient. For each pair above the specified limit (in terms of
    absolute value), it removes one of the two.

    Parameters
    ----------
    strategy: str or None, optional (default=None)
        Feature selection strategy to use. Choose from:
            - None: Do not perform any feature selection algorithm.
            - "univariate": Univariate F-test.
            - "PCA": Principal Component Analysis.
            - "SFM": Select best features according to a model.
            - "SFS"" Sequential Feature Selection.
            - "RFE": Recursive Feature Elimination.
            - "RFECV": RFE with cross-validated selection.

        Note that the SFS, RFE and RFECV strategies don't work when the
        solver is a CatBoost model due to incompatibility of the APIs.

    solver: str, estimator or None, optional (default=None)
        Solver or model to use for the feature selection strategy. See
        sklearn's documentation for an extended description of the
        choices. Select None for the default option per strategy (only
        for univariate or PCA).
            - for "univariate", choose from:
                + "f_classif"
                + "f_regression"
                + "mutual_info_classif"
                + "mutual_info_regression"
                + "chi2"
                + Any function taking two arrays (X, y), and returning
                  arrays (scores, p-values).
            - for "PCA", choose from:
                + "auto" (not available for sparse data, default for dense data)
                + "full" (not available for sparse data)
                + "arpack"
                + "randomized" (default for sparse data)
            - for "SFM", "SFS", "RFE" and "RFECV":
                The base estimator. For SFM, RFE and RFECV, it should
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

        If strategy="SFM" and the threshold parameter is not specified,
        the threshold are automatically set to `-inf` to select the
        `n_features` features.

        If strategy="RFECV", `n_features` is the minimum number of
        features to select.

    max_frac_repeated: float or None, optional (default=1.)
        Remove features with the same value in at least this fraction
        of the total rows. The default is to keep all features with
        non-zero variance, i.e. remove the features that have the same
        value in all samples. If None, skip this step.

    max_correlation: float or None, optional (default=1.)
        Minimum Pearson correlation coefficient to identify correlated
        features. A value of 1 removes one of 2 equal columns. A
        dataframe of the removed features and their correlation values
        can be accessed through the collinear attribute. If None, skip
        this step.

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
        Any extra keyword argument for the PCA, SFM, SFS, RFE and
        RFECV estimators. See the corresponding documentation for
        the available options.

    Attributes
    ----------
    collinear: pd.DataFrame
        Information on the removed collinear features. Columns include:
            - drop_feature: Name of the feature dropped by the method.
            - correlated feature: Name of the correlated features.
            - correlation_value: Pearson correlation coefficients of
                                 the feature pairs.

    feature_importance: list
        Remaining features ordered by importance. Only if strategy in
        ["univariate", "SFM, "RFE", "RFECV"]. For RFE and RFECV, the
        importance is extracted from the external estimator fitted on
        the reduced set.

    <strategy>: sklearn transformer
        Object (lowercase strategy) used to transform the data,
        e.g. `balancer.pca` for the PCA strategy.

    """

    def __init__(
        self,
        strategy: Optional[str] = None,
        solver: Optional[Union[str, callable]] = None,
        n_features: Optional[SCALAR] = None,
        max_frac_repeated: Optional[SCALAR] = 1.0,
        max_correlation: Optional[float] = 1.0,
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
        self.solver = solver
        self.n_features = n_features
        self.max_frac_repeated = max_frac_repeated
        self.max_correlation = max_correlation
        self.kwargs = kwargs

        self.collinear = pd.DataFrame(
            columns=["drop_feature", "correlated_feature", "correlation_value"]
        )
        self.feature_importance = None
        self.scaler = None
        self._solver = None
        self._kwargs = kwargs.copy()
        self._n_features = None
        self._low_variance = {}
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

        X, y = self._prepare_input(X, y)

        # Check parameters
        if isinstance(self.strategy, str):
            strats = ["univariate", "pca", "sfm", "sfs", "rfe", "rfecv", "pso"]

            if self.strategy.lower() not in strats:
                raise ValueError(
                    "Invalid value for the strategy parameter. Choose "
                    "from: univariate, PCA, SFM, RFE or RFECV. "
                    "Or from zoofs algos: pso."
                )

            elif self.strategy.lower() == "univariate":
                solvers_dct = dict(
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
                    self._solver = solvers_dct[self.solver]
                elif isinstance(self.solver, str):
                    raise ValueError(
                        "Invalid value for the solver parameter, got "
                        f"{self.solver}. Choose from: {', '.join(solvers_dct)}."
                    )
                else:
                    self._solver = self.solver

            elif self.strategy.lower() == "pca":
                if self.solver is None:
                    if is_sparse(X):
                        self._solver = "randomized"
                    else:
                        self._solver = "auto"
                else:
                    self._solver = self.solver

            else:
                if self.solver is None:
                    raise ValueError(
                        "Invalid value for the solver parameter. The "
                        f"value can't be None for strategy={self.strategy}"
                    )
                elif isinstance(self.solver, str):
                    # Assign goal depending on solver's ending
                    if self.solver[-6:] == "_class":
                        self.goal = "class"
                        self._solver = self.solver[:-6]
                    elif self.solver[-4:] == "_reg":
                        self.goal = "reg"
                        self._solver = self.solver[:-4]
                    else:
                        self._solver = self.solver

                    # Get estimator from predefined models
                    if self._solver not in MODELS:
                        raise ValueError(
                            "Invalid value for the solver parameter. Unknown "
                            f"model: {self._solver}. Choose from: {', '.join(MODELS)}."
                        )
                    else:
                        model = MODELS[self._solver](self, fast_init=True)
                        self._solver = model.get_estimator()
                else:
                    self._solver = self.solver

        elif self.kwargs:
            kwargs = ", ".join([f"{str(k)}={str(v)}" for k, v in self.kwargs.items()])
            raise ValueError(
                f"Keyword arguments ({kwargs}) are specified for "
                f"the strategy estimator but no strategy is selected."
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
            for n, col in enumerate(X):
                unique, count = np.unique(X[col], return_counts=True)
                for u, c in zip(unique, count):
                    # If count is larger than fraction of total...
                    if c >= self.max_frac_repeated * len(X):
                        self._low_variance[col] = [u, c // 100. * len(X)]
                        X = X.drop(col, axis=1)
                        break

        # Remove features with too high correlation
        if self.max_correlation:
            max_ = self.max_correlation
            mtx = X.corr()  # Pearson correlation coefficient matrix

            # Extract the upper triangle of the correlation matrix
            upper = mtx.where(np.triu(np.ones(mtx.shape).astype(bool), k=1))

            # Select the features with correlations above or equal to the threshold
            to_drop = [i for i in upper.columns if any(abs(upper[i] >= max_))]

            # Record the correlated features and corresponding values
            corr_features, corr_values = [], []
            for col in to_drop:
                corr_features.append(list(upper.index[abs(upper[col]) >= max_]))
                corr_values.append(list(round(upper[col][abs(upper[col]) >= max_], 5)))

            self.collinear = pd.DataFrame(
                data={
                    "drop_feature": to_drop,
                    "correlated_feature": [", ".join(fxs) for fxs in corr_features],
                    "correlation_value": [", ".join(map(str, v)) for v in corr_values],
                }
            )

            X = X.drop(to_drop, axis=1)

        if self.strategy is None:
            self._is_fitted = True
            return self  # Exit feature_engineering

        elif self.strategy.lower() == "univariate":
            check_y()
            self.univariate = SelectKBest(
                score_func=self._solver,
                k=self._n_features,
            ).fit(X, y)

        elif self.strategy.lower() == "pca":
            if is_sparse(X):
                self.pca = TruncatedSVD(
                    n_components=self._n_features,
                    algorithm=self._solver,
                    random_state=self.random_state,
                    **self.kwargs,
                )
            else:
                if not check_scaling(X):
                    self.scaler = Scaler().fit(X)
                    X = self.scaler.transform(X)

                self.pca = PCA(
                    n_components=X.shape[1] - 1,  # All -1 because of the pca plot
                    svd_solver=self._solver,
                    random_state=self.random_state,
                    **self.kwargs,
                )

            # Fit and add desired number of components as internal attr
            self.pca.fit(X)
            self.pca._n_components = self._n_features

        elif self.strategy.lower() == "sfm":
            # If any of these attr exists, model is already fitted
            if any(hasattr(self._solver, a) for a in ("coef_", "feature_importances_")):
                prefit = self._kwargs.pop("prefit", True)
            else:
                prefit = False

            # If threshold is not specified, select only based on _n_features
            if not self.kwargs.get("threshold"):
                self._kwargs["threshold"] = -np.inf

            self.sfm = SelectFromModel(
                estimator=self._solver,
                max_features=self._n_features,
                prefit=prefit,
                **self._kwargs,
            )
            if prefit:
                if len(self.sfm.get_support()) != X.shape[1]:
                    raise ValueError(
                        "Invalid value for the solver parameter. The "
                        f"{self._solver.__class__.__name__} estimator "
                        "is fitted with different columns than X!"
                    )
                self.sfm.estimator_ = self._solver
            else:
                check_y()
                self.sfm.fit(X, y)

        elif self.strategy.lower() == "sfs":
            self.sfs = SequentialFeatureSelector(
                estimator=self._solver,
                n_features_to_select=self._n_features,
                n_jobs=self.n_jobs,
                **self._kwargs,
            ).fit(X, y)

        elif self.strategy.lower() == "rfe":
            check_y()
            self.rfe = RFE(
                estimator=self._solver,
                n_features_to_select=self._n_features,
                **self._kwargs,
            ).fit(X, y)

        elif self.strategy.lower() == "pso":
            check_y()
            pso_initialization_params = ['n_iteration','timeout','population_size','minimize','c1','c2','w' ]
            initialization_params_from_kwargs = {k: v for k, v in self.kwargs.items() if k in pso_initialization_params}
            params_for_objective_function = {k: v for k, v in self.kwargs.items() if k not in pso_initialization_params}

            if self.kwargs.get("objective_function"):
                objective_function = self.kwargs['objective_function']

            elif self.kwargs.get("scoring"):
                params_for_objective_function = {"scorer" : get_custom_scorer(self.kwargs["scoring"])}
                objective_function = custom_function_for_scorer

            else:
                params_for_objective_function = {"scorer" : get_custom_scorer("neg_log_loss") if hasattr(self._solver, "predict_proba") else get_custom_scorer("neg_mean_squared_error") }
                objective_function = custom_function_for_scorer 
 
            if 'minimize' not in initialization_params_from_kwargs.keys():
                initialization_params_from_kwargs['minimize'] = False

            self.pso = ParticleSwarmOptimization(objective_function=objective_function,
                                                 **initialization_params_from_kwargs,
                                                 **params_for_objective_function)

            if ("X_valid" in self.kwargs.keys()) and ("y_valid" in self.kwargs.keys()):
                X_valid, y_valid = self._prepare_input(self.kwargs["X_valid"], self.kwargs["y_valid"] )
            else:
                 X_valid, y_valid = X, y

            self.pso.fit(model = self._solver,
                         X_train = X,
                         y_train = y,
                         X_valid = X_valid,
                         y_valid = y_valid,
                         verbose = False if self.verbose==0 else True)
            

        else:
            check_y()

            # Both RFECV and SFS use the scoring parameter
            if self.kwargs.get("scoring"):
                self._kwargs["scoring"] = get_custom_scorer(self.kwargs["scoring"])

            if self.strategy.lower() == "rfecv":
                # Invert n_features to select them all (default option)
                if self._n_features == X.shape[1]:
                    self._n_features = 1

                self.rfecv = RFECV(
                    estimator=self._solver,
                    min_features_to_select=self._n_features,
                    n_jobs=self.n_jobs,
                    **self._kwargs,
                ).fit(X, y)

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
        columns = X.columns  # Save columns for SFM

        self.log("Performing feature selection ...", 1)

        # Remove features with too low variance
        for key, value in self._low_variance.items():
            self.log(
                f" --> Feature {key} was removed due to low variance. Value "
                f"{value[0]} repeated in {value[1]}% of the rows.", 2
            )
            X = X.drop(key, axis=1)

        # Remove features with too high correlation
        for col in self.collinear["drop_feature"]:
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
                data=self.pca.transform(X)[:, :self._n_features],
                index=X.index,
                columns=[f"component {str(i)}" for i in range(1, self._n_features + 1)],
            )

            var = np.array(self.pca.explained_variance_ratio_[:self._n_features])
            self.log(f"   >>> Explained variance ratio: {round(var.sum(), 3)}", 2)

        elif self.strategy.lower() == "sfm":
            # Here we use columns since some cols could be removed before by
            # variance or correlation checks and there would be cols mismatch
            indices = np.argsort(get_feature_importance(self.sfm.estimator_))
            best_fxs = [columns[idx] for idx in indices][::-1]
            self.log(
                f" --> The {self._solver.__class__.__name__} estimator selected "
                f"{sum(self.sfm.get_support())} features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.sfm.get_support()[n]:
                    self.log(f"   >>> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

            self.feature_importance = [fx for fx in best_fxs if fx in X.columns]

        elif self.strategy.lower() == "sfs":
            self.log(
                f" --> SFS selected {self.sfs.n_features_to_select_}"
                " features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.sfs.support_[n]:
                    self.log(f"   >>> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

        elif self.strategy.lower() == "rfe":
            self.log(
                f" --> RFE selected {self.rfe.n_features_}"
                " features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.rfe.support_[n]:
                    self.log(
                        f"   >>> Dropping feature {column} "
                        f"(rank {self.rfe.ranking_[n]}).", 2
                    )
                    X = X.drop(column, axis=1)

            idx = np.argsort(get_feature_importance(self.rfe.estimator_))
            self.feature_importance = list(X.columns[idx][::-1])

        elif self.strategy.lower() == "rfecv":
            self.log(
                f" --> RFECV selected {self.rfecv.n_features_}"
                " features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.rfecv.support_[n]:
                    self.log(
                        f"   >>> Dropping feature {column} "
                        f"(rank {self.rfecv.ranking_[n]}).", 2
                    )
                    X = X.drop(column, axis=1)

            idx = np.argsort(get_feature_importance(self.rfecv.estimator_))
            self.feature_importance = list(X.columns[idx][::-1])

        return X

# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the feature engineering estimators.

"""

# Standard packages
import random
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union

# Other packages
import featuretools as ft
from featuretools.primitives import make_trans_primitive
from featuretools.variable_types import Numeric
from gplearn.genetic import SymbolicTransformer
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
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
from .models import MODEL_LIST
from .basetransformer import BaseTransformer
from .data_cleaning import TransformerMixin, Scaler
from .plots import FSPlotter
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, METRIC_ACRONYMS, lst,
    to_df, check_scaling, check_is_fitted, get_acronym, composed,
    crash, method_to_log
)


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
        Choose from: "add", "sub", "mul", "div", "sqrt", "log", "inv",
        "sin", "cos", "tan".

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - n_jobs.

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

    Attributes
    ----------
    symbolic_transformer: SymbolicTransformer
        Instance used to calculate the genetic features. Only for the
        genetic strategy.

    genetic_features: pd.DataFrame
        Dataframe of the newly created non-linear features. Only for
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
            n_jobs=n_jobs, verbose=verbose, logger=logger, random_state=random_state
        )
        self.strategy = strategy
        self.n_features = n_features
        self.generations = generations
        self.population = population
        self.operators = operators

        self.symbolic_transformer = None
        self.genetic_features = None
        self._dfs_features = None
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
        self: FeatureGenerator

        """

        def sqrt(column):
            return np.sqrt(column)

        def log(column):
            return np.log(column)

        def sin(column):
            return np.sin(column)

        def cos(column):
            return np.cos(column)

        def tan(column):
            return np.tan(column)

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
            self.operators = default
        else:
            self.operators = lst(self.operators)
            for operator in self.operators:
                if operator.lower() not in default:
                    raise ValueError(
                        "Invalid value in the operators parameter, got "
                        f"{operator}. Choose from: {', '.join(default)}."
                    )

        self.log("Fitting FeatureGenerator...", 1)

        if self.strategy.lower() == "dfs":
            # Make an entity set and add the entity
            entity_set = ft.EntitySet(id="atom")
            entity_set.entity_from_dataframe(
                entity_id="data", dataframe=X, make_index=True, index="index"
            )

            # Get list of transformation primitives
            trans_primitives = []
            custom_operators = dict(
                sqrt=make_trans_primitive(sqrt, [Numeric], Numeric),
                log=make_trans_primitive(log, [Numeric], Numeric),
                sin=make_trans_primitive(sin, [Numeric], Numeric),
                cos=make_trans_primitive(cos, [Numeric], Numeric),
                tan=make_trans_primitive(tan, [Numeric], Numeric),
            )
            for operator in self.operators:
                if operator.lower() == "add":
                    trans_primitives.append("add_numeric")
                elif operator.lower() == "sub":
                    trans_primitives.append("subtract_numeric")
                elif operator.lower() == "mul":
                    trans_primitives.append("multiply_numeric")
                elif operator.lower() == "div":
                    trans_primitives.append("divide_numeric")
                elif operator.lower() in ("sqrt", "log", "sin", "cos", "tan"):
                    trans_primitives.append(custom_operators[operator.lower()])

            # Run deep feature synthesis with transformation primitives
            self._dfs_features = ft.dfs(
                entityset=entity_set,
                target_entity="data",
                max_depth=1,
                features_only=True,
                trans_primitives=trans_primitives,
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
                self.n_features = max_features

            # Get random indices from the feature list
            idx_old = range(X.shape[1] - 1)
            idx_new = random.sample(
                range(X.shape[1] - 1, len(self._dfs_features)), self.n_features
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
                function_set=self.operators,
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
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Dataframe containing the newly generated features.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)

        self.log("Creating new features...", 1)

        if self.strategy.lower() == "dfs":
            # Make an entity set and add the entity
            entity_set = ft.EntitySet(id="atom")
            entity_set.entity_from_dataframe(
                entity_id="data", dataframe=X, make_index=True, index="index"
            )

            X = ft.calculate_feature_matrix(
                features=self._dfs_features, entityset=entity_set, n_jobs=self.n_jobs
            )

            X.index.name = ""  # DFS gives index a name automatically
            self.log(
                f" --> {self.n_features} new features were added to the dataset.", 2
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
                    "find any improving non-linear features!", 1
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
                features_df = features_df.append({
                    "name": "Feature " + str(1 + i + len(X.columns)),
                    "description": descript[idx],
                    "fitness": fitness[idx],
                }, ignore_index=True)
            self.genetic_features = features_df

            self.log(
                f" --> {len(self.genetic_features)} new "
                "features were added to the dataset.", 2
            )

            cols = list(X.columns) + list(self.genetic_features["name"])
            X = pd.DataFrame(np.hstack((X, new_features)), columns=cols)

        return X


class FeatureSelector(BaseEstimator, TransformerMixin, BaseTransformer, FSPlotter):
    """Apply feature selection techniques.

    Remove features according to the selected strategy. Ties between
    features with equal scores will be broken in an unspecified way.
    Additionally, removes features with too low variance and finds
    pairs of collinear features based on the Pearson correlation
    coefficient. For each pair above the specified limit (in terms of
    absolute value), it removes one of the two.

    Parameters
    ----------
    strategy: string or None, optional (default=None)
        Feature selection strategy to use. Choose from:
            - None: Do not perform any feature selection algorithm.
            - "univariate": Perform a univariate F-test.
            - "PCA": Perform principal component analysis.
            - "SFM": Select best features from model.
            - "RFE": Recursive feature eliminator.
            - "RFECV": Perform RFE with cross-validated selection.
            - "SFS"" Perform Sequential Feature Selector.

        Note that the RFE, RFECV and SFS strategies don't work when the
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
                + "auto" (default)
                + "full"
                + "arpack"
                + "randomized"
            - for "SFM", "RFE", "RFECV" and "SFS":
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
        the threshold will be automatically set to `-inf` to select the
        `n_features` features.

        If strategy="RFECV", `n_features` is the minimum number of
        features to select.

    max_frac_repeated: float or None, optional (default=1.)
        Remove features with the same value in at least this fraction
        of the total rows. The default is to keep all features with
        non-zero variance, i.e. remove the features that have the same
        value in all samples. None to skip this step.

    max_correlation: float or None, optional (default=1.)
        Minimum Pearson correlation coefficient to identify correlated
        features. A value of 1 will remove one of 2 equal columns. A
        dataframe of the removed features and their correlation values
        can be accessed through the collinear attribute. None to skip
        this step.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - n_jobs.

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
        Any extra keyword argument for the PCA, SFM, RFE, RFECV
        and SFS estimators. See the corresponding documentation
        for the available options.

    Attributes
    ----------
    collinear: pd.DataFrame
        Dataframe of the removed collinear features. Columns include:
            - drop_feature: Name of the feature dropped by the method.
            - correlated feature: Name of the correlated features.
            - correlation_value: Pearson correlation coefficients of
                                 the feature pairs.

    feature_importance: list
        Remaining features ordered by importance. Only if strategy in
        ["univariate", "SFM, "RFE", "RFECV"]. For RFE and RFECV, the
        importance is extracted from the external estimator fitted on
        the reduced set.

    <strategy>: sklearn estimator
        Estimator instance (lowercase strategy) used to transform
        the data, e.g. `balancer.pca` for the PCA strategy.

    """

    def __init__(
        self,
        strategy: Optional[str] = None,
        solver: Optional[Union[str, callable]] = None,
        n_features: Optional[Union[int, float]] = None,
        max_frac_repeated: Optional[Union[int, float]] = 1.0,
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
        self._low_variance = {}
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit the feature selector to the data.

        Note that the univariate, sfm (when model is not fitted), rfe
        and rfecv strategies need a target column. Leaving it None will
        raise an exception.

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
        self: FeatureSelector

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
            strats = ["univariate", "pca", "sfm", "rfe", "rfecv", "sfs"]

            if self.strategy.lower() not in strats:
                raise ValueError(
                    "Invalid value for the strategy parameter. Choose "
                    "from: univariate, PCA, SFM, RFE or RFECV."
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
                    raise ValueError("Choose a solver for the strategy!")
                elif self.solver in solvers_dct:
                    self.solver = solvers_dct[self.solver]
                elif isinstance(self.solver, str):
                    raise ValueError(
                        f"Unknown solver. Choose from: {', '.join(solvers_dct)}."
                    )

            elif self.strategy.lower() == "pca":
                self.solver = "auto" if self.solver is None else self.solver

            else:
                if self.solver is None:
                    raise ValueError("Select a solver for the strategy!")
                elif isinstance(self.solver, str):
                    # Assign goal depending on solver's ending
                    if self.solver[-6:] == "_class":
                        self.goal = "classification"
                        self.solver = self.solver[:-6]
                    elif self.solver[-4:] == "_reg":
                        self.goal = "regression"
                        self.solver = self.solver[:-4]

                    # Set to right acronym and get the model's estimator
                    model = MODEL_LIST[get_acronym(self.solver)](self)
                    self.solver = model.get_estimator()

        if self.n_features is not None and self.n_features <= 0:
            raise ValueError(
                "Invalid value for the n_features parameter. "
                f"Value should be >0, got {self.n_features}."
            )
        if self.max_frac_repeated is not None and not 0 <= self.max_frac_repeated <= 1:
            raise ValueError(
                "Invalid value for the max_frac_repeated parameter. Value should "
                f"be between 0 and 1, got {self.max_frac_repeated}."
            )
        if self.max_correlation is not None and not 0 <= self.max_correlation <= 1:
            raise ValueError(
                "Invalid value for the max_correlation parameter. Value should "
                f"be between 0 and 1, got {self.max_correlation}."
            )

        self.log("Fitting FeatureSelector...", 1)

        # Remove features with too low variance
        if self.max_frac_repeated is not None:
            for n, col in enumerate(X):
                unique, count = np.unique(X[col], return_counts=True)
                for u, c in zip(unique, count):
                    # If count is larger than fraction of total...
                    if c >= self.max_frac_repeated * len(X):
                        self._low_variance[col] = [u, c // len(X) * 100]
                        X = X.drop(col, axis=1)
                        break

        # Remove features with too high correlation
        if self.max_correlation:
            max_ = self.max_correlation
            mtx = X.corr()  # Pearson correlation coefficient matrix

            # Extract the upper triangle of the correlation matrix
            upper = mtx.where(np.triu(np.ones(mtx.shape).astype(bool), k=1))

            # Select the features with correlations above the threshold
            to_drop = [i for i in upper.columns if any(abs(upper[i] > max_))]

            # Iterate to record pairs of correlated features
            for col in to_drop:
                # Find the correlated features and corresponding values
                corr_features = list(upper.index[abs(upper[col]) > max_])
                corr_values = list(round(upper[col][abs(upper[col]) > max_], 5))

                # Update dataframe
                self.collinear = self.collinear.append({
                    "drop_feature": col,
                    "correlated_feature": ", ".join(corr_features),
                    "correlation_value": ", ".join(map(str, corr_values)),
                }, ignore_index=True)

            X = X.drop(to_drop, axis=1)

        # Set n_features as all or fraction of total
        if self.n_features is None:
            self.n_features = X.shape[1]
        elif self.n_features < 1:
            self.n_features = int(self.n_features * X.shape[1])

        # Perform selection based on strategy
        if self.strategy is None:
            self._is_fitted = True
            return self  # Exit feature_engineering

        elif self.strategy.lower() == "univariate":
            check_y()
            self.univariate = SelectKBest(
                score_func=self.solver, k=self.n_features
            ).fit(X, y)

        elif self.strategy.lower() == "pca":
            if not check_scaling(X):
                self.scaler = Scaler().fit(X)
                X = self.scaler.transform(X)

            # Define PCA
            self.pca = PCA(
                n_components=None,
                svd_solver=self.solver,
                random_state=self.random_state,
                **self.kwargs,
            ).fit(X)

            self.pca.n_components_ = self.n_features  # Number of components

        elif self.strategy.lower() == "sfm":
            # If any of these attr exists, model is already fitted
            condition1 = hasattr(self.solver, "coef_")
            condition2 = hasattr(self.solver, "feature_importances_")
            self.kwargs["prefit"] = True if condition1 or condition2 else False

            # If threshold is not specified, select only based on n_features
            if not self.kwargs.get("threshold"):
                self.kwargs["threshold"] = -np.inf

            self.sfm = SelectFromModel(
                estimator=self.solver,
                max_features=self.n_features,
                **self.kwargs,
            )
            if self.kwargs["prefit"]:
                if len(self.sfm.get_support()) != X.shape[1]:
                    raise ValueError(
                        "Invalid value for the solver parameter. The "
                        f"{self.solver.__class__.__name__} estimator "
                        "is fitted with different columns than X!"
                    )
                self.sfm.estimator_ = self.solver
            else:
                check_y()
                self.sfm.fit(X, y)

        elif self.strategy.lower() == "rfe":
            check_y()
            self.rfe = RFE(
                estimator=self.solver,
                n_features_to_select=self.n_features,
                **self.kwargs,
            ).fit(X, y)

        else:
            check_y()

            # Both RFECV and SFS use the scoring parameter
            if isinstance(self.kwargs.get("scoring"), str):
                if self.kwargs.get("scoring", "").lower() in METRIC_ACRONYMS:
                    self.kwargs["scoring"] = METRIC_ACRONYMS[
                        self.kwargs["scoring"].lower()
                    ]

            if self.strategy.lower() == "rfecv":
                # Invert n_features to select them all (default option)
                if self.n_features == X.shape[1]:
                    self.n_features = 1

                self.rfecv = RFECV(
                    estimator=self.solver,
                    min_features_to_select=self.n_features,
                    n_jobs=self.n_jobs,
                    **self.kwargs,
                ).fit(X, y)

            elif self.strategy.lower() == "sfs":
                self.sfs = SequentialFeatureSelector(
                    estimator=self.solver,
                    n_features_to_select=self.n_features,
                    n_jobs=self.n_jobs,
                    **self.kwargs,
                ).fit(X, y)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Transform the data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Only for continuity of API.

        Returns
        -------
        X: pd.DataFrame
            Transformed feature set.

        """

        def get_scores(est):
            """Return the feature scores for a given estimator.

            Return the values of the scores_, feature_importances_ or
            coef_ attributes if available (in that order). For
            multiclass classification tasks, the coef_ attribute has
            shape (n_targets, n_features). In this case, the mean of
            the coef_ value over the targets is returned.

            Parameters
            ----------
            est: class
                Estimator for which to get the scores.

            Returns
            -------
            scores: np.ndarray
                Scores of the selected attribute for the estimator.

            """
            attributes = ["scores_", "feature_importances_", "coef_"]
            scores = getattr(est, next(i for i in attributes if hasattr(est, i)))
            if scores.ndim > 1:
                scores = np.mean(scores, axis=0)

            return scores

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
            indices = np.argsort(get_scores(self.univariate))
            best_fxs = [X.columns[idx] for idx in indices][::-1]
            self.log(
                f" --> The univariate test selected "
                f"{self.n_features} features from the dataset.", 2
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
            self.log(f" --> Applying Principal Component Analysis...", 2)

            if self.scaler:
                self.log("   >>> Scaling features...", 2)
                X = self.scaler.transform(X)

            # Define PCA, keep in mind that it has all components still
            n = self.pca.n_components_
            var = np.array(self.pca.explained_variance_ratio_[:n])
            X = to_df(self.pca.transform(X)[:, :n], index=X.index, pca=True)
            self.log(f"   >>> Total explained variance: {round(var.sum(), 3)}", 2)

        elif self.strategy.lower() == "sfm":
            # Here we use columns since some cols could be removed before by
            # variance or correlation checks and there would be cols mismatch
            indices = np.argsort(get_scores(self.sfm.estimator_))
            best_fxs = [columns[idx] for idx in indices][::-1]
            self.log(
                f" --> The {self.solver.__class__.__name__} estimator selected "
                f"{sum(self.sfm.get_support())} features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.sfm.get_support()[n]:
                    self.log(f"   >>> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

            self.feature_importance = [fx for fx in best_fxs if fx in X.columns]

        elif self.strategy.lower() == "rfe":
            self.log(
                f" --> The RFE selected {self.rfe.n_features_}"
                " features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.rfe.support_[n]:
                    self.log(
                        f"   >>> Dropping feature {column} "
                        f"(rank {self.rfe.ranking_[n]}).", 2
                    )
                    X = X.drop(column, axis=1)

            idx = np.argsort(get_scores(self.rfe.estimator_))
            self.feature_importance = list(X.columns[idx][::-1])

        elif self.strategy.lower() == "rfecv":
            self.log(
                f" --> The RFECV selected {self.rfecv.n_features_}"
                " features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.rfecv.support_[n]:
                    self.log(
                        f"   >>> Dropping feature {column} "
                        f"(rank {self.rfecv.ranking_[n]}).", 2
                    )
                    X = X.drop(column, axis=1)

            idx = np.argsort(get_scores(self.rfecv.estimator_))
            self.feature_importance = list(X.columns[idx][::-1])

        elif self.strategy.lower() == "sfs":
            self.log(
                f" --> The SFS selected {self.sfs.n_features_to_select_}"
                " features from the dataset.", 2
            )
            for n, column in enumerate(X):
                if not self.sfs.support_[n]:
                    self.log(f"   >>> Dropping feature {column}.", 2)
                    X = X.drop(column, axis=1)

        return X

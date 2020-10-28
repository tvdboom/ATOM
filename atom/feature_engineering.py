# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the FeatureGenerator and FeatureSelector estimators.

"""

# Standard packages
import random
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union, Sequence

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
)

# Own modules
from .models import MODEL_LIST
from .basetransformer import BaseTransformer
from .data_cleaning import BaseCleaner, Scaler
from .plots import FeatureSelectorPlotter
from .utils import (
    METRIC_ACRONYMS, X_TYPES, Y_TYPES, to_df, check_scaling, check_is_fitted,
    get_model_name, composed, crash, method_to_log
)


class FeatureGenerator(BaseEstimator, BaseTransformer, BaseCleaner):
    """Apply automated feature engineering.

    Use Deep feature Synthesis or a genetic algorithm to create new combinations
    of existing features to capture the non-linear relations between the original
    features.

    Parameters
    ----------
    strategy: str, optional (default="DFS")
        Strategy to crate new features. Choose from:
            - "DFS" to use Deep Feature Synthesis.
            - "GFG" or "genetic" to use Genetic Feature Generation.

    n_features: int, optional (default=None)
        Number of newly generated features to add to the dataset (if
        strategy="genetic", no more than 1% of the population). If None,
        select all created.

    generations: int, optional (default=20)
        Number of generations to evolve. Only if strategy="genetic".

    population: int, optional (default=500)
        Number of programs in each generation. Only if strategy="genetic".

    operators: str, list, tuple or None, optional (default=None)
        Mathematical operators to apply on the features. None for all. Choose from:
        "add", "sub", "mul", "div", "sqrt", "log", "inv", "sin", "cos", "tan".

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

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If str: name of the logging file. "auto" for default name.
        - If class: python `Logger` object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` instance used by `numpy.random`.

    """

    def __init__(
        self,
        strategy: str = "DFS",
        n_features: Optional[int] = None,
        generations: int = 20,
        population: int = 500,
        operators: Optional[Union[str, Sequence[str]]] = None,
        n_jobs: int = 1,
        verbose: int = 0,
        logger: Optional[Union[bool, str, callable]] = None,
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

        self._dfs_features = None
        self.symbolic_transformer = None
        self.genetic_features = None

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Y_TYPES):
        """Fit the data according to the selected strategy.

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

        # Check Parameters
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

        # Check operators
        default = ["add", "sub", "mul", "div", "sqrt", "log", "sin", "cos", "tan"]
        if not self.operators:  # None or empty list
            self.operators = default
        else:
            if not isinstance(self.operators, (list, tuple)):
                self.operators = [self.operators]
            for self.operator in self.operators:
                if self.operator.lower() not in default:
                    raise ValueError(
                        "Invalid value in the operators parameter, got "
                        f"{self.operator}. Choose from: {', '.join(default)}."
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

            # Since dfs doesn't return a specific order in the features and we need
            # an order for the selection to be deterministic, order by name
            new_dfs = []
            for feature in sorted(map(str, self._dfs_features[X.shape[1] - 1:])):
                for fx in self._dfs_features:
                    if feature == str(fx):
                        new_dfs.append(fx)
                        break
            self._dfs_features = self._dfs_features[: X.shape[1] - 1] + new_dfs

            # Make sure there are enough features (-1 because X has an index column)
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

        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Generate the new features.

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, list, tuple,  np.array or pd.Series, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Dataframe containing the newly generated features.

        """
        check_is_fitted(self, ("_dfs_features", "symbolic_transformer"))
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

            # ix = indices of all new features that are not in the original set
            # descript = list of the operators applied to create the new features
            # fitness = list of fitness scores of the new features
            ix, descript, fitness = [], [], []
            for i, program in enumerate(self.symbolic_transformer):
                if str(program) not in X.columns:
                    ix.append(i)
                descript.append(str(program))
                fitness.append(program.fitness_)

            # Remove all features that are identical to those in the dataset
            new_features = new_features[:, ix]
            descript = [descript[i] for i in range(len(descript)) if i in ix]
            fitness = [fitness[i] for i in range(len(fitness)) if i in ix]

            # Indices of all non duplicate elements in list
            ix = [ix for ix, v in enumerate(descript) if v not in descript[:ix]]

            # Remove all duplicate elements
            new_features = new_features[:, ix]
            descript = [descript[i] for i in range(len(descript)) if i in ix]
            fitness = [fitness[i] for i in range(len(fitness)) if i in ix]

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


class FeatureSelector(
    BaseEstimator, BaseTransformer, BaseCleaner, FeatureSelectorPlotter
):
    """Apply feature selection techniques.

    Remove features according to the selected strategy. Ties between
    features with equal scores will be broken in an unspecified way.
    Additionally, removes features with too low variance and finds pairs of
    collinear features based on the Pearson correlation coefficient. For
    each pair above the specified limit (in terms of absolute value), it
    removes one of the two.

    Parameters
    ----------
    strategy: string or None, optional (default=None)
        Feature selection strategy to use. Choose from:
            - None: Do not perform any feature selection algorithm.
            - "univariate": Perform a univariate F-test.
            - "PCA": Perform principal component analysis.
            - "SFM": Select best features from model.
            - "RFE": Recursive feature eliminator.
            - "RFECV": RFE with cross-validated selection.

        Note that the RFE and RFECV strategies don't work when the solver is a
        CatBoost model due to incompatibility of the APIs.

    solver: string, callable or None, optional (default=None)
        Solver or model to use for the feature selection strategy. See the
        sklearn documentation for an extended description of the choices.
        Select None for the default option per strategy (not applicable
        for SFM, RFE and RFECV).
            - for "univariate", choose from:
                + "f_classif"
                + "f_regression"
                + "mutual_info_classif"
                + "mutual_info_regression"
                + "chi2"
                + Any function taking two arrays (X, y), and returning
                  arrays (scores, p-values). See the sklearn documentation.
            - for "PCA", choose from:
                + "auto" (default)
                + "full"
                + "arpack"
                + "randomized"
            - for "SFM", "RFE" and "RFECV:
                Estimator with either a `feature_importances_` or `coef_` attribute
                after fitting. You can use one of ATOM's pre-defined models. Add
                `_class` or `_reg` after the model's name to specify a classification
                or regression task, e.g. `solver="LGB_reg"` (not necessary if called
                from an `atom` instance. No default option.

    n_features: int, float or None, optional (default=None)
        Number of features to select. Choose from:
            - if None: Select all features.
            - if < 1: Fraction of the total features to select.
            - if >= 1: Number of features to select.

        If `strategy="SFM"` and the threshold parameter is not specified, the
        threshold will be set to `-np.inf` in order to make this parameter the
        number of features to select.
        If `strategy="RFECV"`, it's the minimum number of features to select.

    max_frac_repeated: float or None, optional (default=1.)
        Remove features with the same value in at least this fraction of
        the total rows. The default is to keep all features with non-zero
        variance, i.e. remove the features that have the same value in all
        samples. None to skip this step.

    max_correlation: float or None, optional (default=1.)
        Minimum value of the Pearson correlation coefficient to identify
        correlated features. A value of 1 removes one of 2 equal columns.
        A dataframe of the removed features and their correlation values
        can be accessed through the collinear attribute. None to skip this step.

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

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If str: name of the logging file. "auto" for default name.
        - If class: python `Logger` object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` instance used by `numpy.random`.

    **kwargs
        Any extra keyword argument for the PCA, SFM, RFE or RFECV estimators.
        See the corresponding sklearn documentation for the available options.

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
        logger: Optional[Union[bool, str, callable]] = None,
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

        self._low_variance = {}
        self.collinear = pd.DataFrame(
            columns=["drop_feature", "correlated_feature", "correlation_value"]
        )
        self.univariate = None
        self.scaler = None
        self.pca = None
        self.sfm = None
        self.rfe = None
        self.rfecv = None
        self.feature_importance = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit the data according to the selected strategy.

        Note that the univariate, sfm (when model is not fitted), rfe and rfecv
        strategies need a target column. Leaving it None will raise an exception.

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
            strats = ["univariate", "pca", "sfm", "rfe", "rfecv"]

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

            elif self.strategy.lower() in ["sfm", "rfe", "rfecv"]:
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

                    if self.solver.lower() not in map(str.lower, MODEL_LIST):
                        raise ValueError(
                            "Unknown value for the solver parameter, got "
                            f"{self.solver}. Try one of {list(MODEL_LIST)}."
                        )
                    else:  # Set to right model name and call model's method
                        model_class = MODEL_LIST[get_model_name(self.solver)]
                        self.solver = model_class(self).get_estimator()

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
                        self._low_variance[col] = [u, int(c / len(X) * 100)]
                        X.drop(col, axis=1, inplace=True)
                        break

        # Remove features with too high correlation
        if self.max_correlation:
            max_ = self.max_correlation
            mtx = X.corr()  # Pearson correlation coefficient matrix

            # Extract the upper triangle of the correlation matrix
            upper = mtx.where(np.triu(np.ones(mtx.shape).astype(np.bool), k=1))

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

            X.drop(to_drop, axis=1, inplace=True)

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
            # Always fit in case the data to transform is not scaled
            self.scaler = Scaler().fit(X)
            if not check_scaling(X):
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
                estimator=self.solver, max_features=self.n_features, **self.kwargs
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

        elif self.strategy.lower() == "rfecv":
            check_y()
            if self.n_features == X.shape[1]:
                self.n_features = 1

            if isinstance(self.kwargs.get("scoring"), str):
                if self.kwargs.get("scoring", "").lower() in METRIC_ACRONYMS:
                    self.kwargs["scoring"] = METRIC_ACRONYMS[
                        self.kwargs["scoring"].lower()
                    ]

            self.rfecv = RFECV(
                estimator=self.solver,
                min_features_to_select=self.n_features,
                n_jobs=self.n_jobs,
                **self.kwargs,
            ).fit(X, y)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Transform the data according to the selected strategy.

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, list, tuple,  np.array or pd.Series, optional (default=None)
            Does nothing. Only for continuity of API.

        Returns
        -------
        X: pd.DataFrame
            Copy of the feature dataset.

        """

        def get_scores(est):
            """Return the feature scores for a given estimator.

            Return the values of the attributes scores_, feature_importances_ or
            coef_ if available (in that order). For multiclass classification tasks,
            the coef_ attribute has shape (n_targets, n_features). In this case, the
            mean of the coef_ value over the targets is returned.

            Parameters
            ----------
            est: class
                Estimator for which to get the scores.

            Returns
            -------
            scores: np.ndarray
                Scores of the selected attribute for the provided estimator.

            """
            attributes = ["scores_", "feature_importances_", "coef_"]
            scores = getattr(est, next(i for i in attributes if hasattr(est, i)))
            if scores.ndim > 1:
                scores = np.mean(scores, axis=0)

            return scores

        check_is_fitted(self, "_is_fitted")
        X, y = self._prepare_input(X, y)
        columns = X.columns  # Save columns for SFM

        self.log("Performing feature selection ...", 1)

        # Remove features with too low variance
        for key, value in self._low_variance.items():
            self.log(
                f" --> Feature {key} was removed due to low variance. Value "
                f"{value[0]} repeated in {value[1]}% of the rows.", 2
            )
            X.drop(key, axis=1, inplace=True)

        # Remove features with too high correlation
        for col in self.collinear["drop_feature"]:
            self.log(
                f" --> Feature {col} was removed due to "
                "collinearity with another feature.", 2
            )
            X.drop(col, axis=1, inplace=True)

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
                    X.drop(column, axis=1, inplace=True)

            self.feature_importance = [fx for fx in best_fxs if fx in X.columns]

        elif self.strategy.lower() == "pca":
            self.log(f" --> Applying Principal Component Analysis...", 2)

            if not check_scaling(X):
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
                    X.drop(column, axis=1, inplace=True)

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
                    X.drop(column, axis=1, inplace=True)

            self.feature_importance = X.columns[
                [np.argsort(get_scores(self.rfe.estimator_))]
            ][::-1]

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
                    X.drop(column, axis=1, inplace=True)

            self.feature_importance = X.columns[
                [np.argsort(get_scores(self.rfecv.estimator_))]
            ][::-1]

        return X

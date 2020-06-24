# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the feature selection and generator estimators.

"""

# Standard packages
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union, Tuple

# Other packages
from gplearn.genetic import SymbolicTransformer
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    chi2, SelectKBest, SelectFromModel, RFE, RFECV
    )

# Own modules
from .models import MODEL_LIST, get_model_name
from .basetransformer import BaseTransformer
from .data_cleaning import BaseCleaner, Scaler
from .utils import (
    X_TYPES, Y_TYPES, variable_return, to_df, check_scaling,
    check_is_fitted, composed, crash
    )
from .plots import BasePlotter, plot_pca, plot_components, plot_rfecv


# Classes =================================================================== >>

class FeatureGenerator(BaseEstimator, BaseTransformer, BaseCleaner):
    """Create new non-linear features.

    Use a genetic algorithm to create new combinations of existing
    features and add them to the original dataset in order to capture
    the non-linear relations between the original features. A dataframe
    containing the description of the newly generated features and their
    scores can be accessed through the `genetic_features` attribute. It is
    recommended to only use this method when fitting linear models.
    Dependency: gplearn.

    Parameters
    ----------
    n_features: int, optional (default=2)
        Maximum number of newly generated features (no more than 1%
        of the population).

    generations: int, optional (default=20)
        Number of generations to evolve.

    population: int, optional (default=500)
        Number of programs in each generation.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If -1, use all available cores
            - If <-1, use available_cores - 1 + value

        Beware that using multiple processes on the same machine may
        cause memory issues for large datasets.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: bool, str, class or None, optional (default=None)
        - If None: No logger.
        - If bool: True for logging file with default name, False for no logger.
        - If string: name of the logging file. 'auto' for default name.
        - If class: python Logger object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    def __init__(self,
                 n_features: int = 2,
                 generations: int = 20,
                 population: int = 500,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 logger: Optional[Union[bool, str, callable]] = None,
                 random_state: Optional[int] = None):
        super().__init__(n_jobs=n_jobs,
                         verbose=verbose,
                         logger=logger,
                         random_state=random_state)

        # Check Parameters
        if population < 100:
            raise ValueError("Invalid value for the population parameter." +
                             f"Value should be >100, got {population}.")
        if generations < 1:
            raise ValueError("Invalid value for the generations parameter." +
                             f"Value should be >100, got {generations}.")
        if n_features <= 0:
            raise ValueError("Invalid value for the n_features parameter." +
                             f"Value should be >0, got {n_features}.")
        elif n_features > int(0.01 * population):
            raise ValueError("Invalid value for the n_features parameter." +
                             "Value should be <1% of the population, " +
                             f"got {n_features}.")

        # Define attributes
        self.population = population
        self.generations = generations
        self.n_features = n_features

        self.symbolic_transformer = None
        self.genetic_features = pd.DataFrame(
            columns=['name', 'description', 'fitness'])

    @composed(crash, typechecked)
    def fit(self, X: X_TYPES, y: Y_TYPES):
        """Fit the individual encoders on each column.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Data target column with shape=(n_samples,).

        Returns
        -------
        self: FeatureInsertor

        """
        X, y = self._prepare_input(X, y)

        self.log("Fitting genetic algorithm...", 1)

        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs',
                        'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']

        self.symbolic_transformer = \
            SymbolicTransformer(generations=self.generations,
                                population_size=self.population,
                                hall_of_fame=int(0.1 * self.population),
                                n_components=int(0.01 * self.population),
                                init_depth=(1, 2),
                                function_set=function_set,
                                feature_names=X.columns,
                                max_samples=1.0,
                                verbose=0 if self.verbose < 3 else 1,
                                n_jobs=self.n_jobs,
                                random_state=self.random_state)
        self.symbolic_transformer.fit(X, y)

        return self

    @composed(crash, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the encoding transformations.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            Does nothing. Only for continuity of API. Is returned unchanged if
            provided.

        Returns
        -------
        X: pd.DataFrame
            Dataframe containing the newly generated features.

        """
        check_is_fitted(self, 'symbolic_transformer')
        X, y = self._prepare_input(X, y)

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

        self.log("-" * 49, 2)

        # Check if any new features remain in the loop
        if len(descript) == 0:
            self.log("WARNING! The genetic algorithm couldn't find any " +
                     "improving non-linear features!", 1)
            return variable_return(X, y)

        # Get indices of the best features
        if len(descript) > self.n_features:
            index = np.argpartition(fitness, -self.n_features)[-self.n_features:]
        else:
            index = range(len(descript))

        # Select best features only
        new_features = new_features[:, index]
        for i, idx in enumerate(index):
            self.genetic_features = self.genetic_features.append(
                {'name': 'Feature ' + str(1 + i + len(X.columns)),
                 'description': descript[idx],
                 'fitness': fitness[idx]},
                ignore_index=True)

            self.log(f" --> New feature {descript[idx]} added to the dataset.", 2)

        cols = list(X.columns) + list(self.genetic_features['name'])
        X = pd.DataFrame(np.hstack((X, new_features)), columns=cols)

        return variable_return(X, y)


class FeatureSelector(BaseEstimator, BaseTransformer, BaseCleaner, BasePlotter):
    """Apply feature selection techniques.

    Remove features according to the selected strategy. Ties between
    features with equal scores will be broken in an unspecified way.
    Also removes features with too low variance and finds pairs of
    collinear features based on the Pearson correlation coefficient. For
    each pair above the specified limit (in terms of absolute value), it
    removes one of the two.

    Parameters
    ----------
    strategy: string or None, optional (default=None)
        Feature selection strategy to use. Choose from:
            - None: Do not perform any feature selection algorithm.
            - 'univariate': Perform a univariate F-test.
            - 'PCA': Perform principal component analysis.
            - 'SFM': Select best features from model.
            - 'RFE': Recursive feature eliminator.
            - 'RFECV': RFE with cross-validated selection.

        The sklearn objects are attached as attributes to the ATOM instance under
        the names: univariate, pca, sfm, rfe and rfecv.

        Note that the RFE and RFECV strategies don't work when the solver is a
        CatBoost model due to incompatibility of the APIs.

    solver: string, callable or None, optional (default=None)
        Solver or model to use for the feature selection strategy. See the
        sklearn documentation for an extended description of the choices.
        Select None for the default option per strategy (not applicable
        for SFM, RFE and RFECV).
            - for 'univariate', choose from:
                + 'f_classif'
                + 'f_regression'
                + 'mutual_info_classif'
                + 'mutual_info_regression'
                + 'chi2'
                + Any function taking two arrays (X, y), and returning
                  arrays (scores, pvalues). See the sklearn documentation.
            - for 'PCA', choose from:
                + 'auto' (default)
                + 'full'
                + 'arpack'
                + 'randomized'
            - for 'SFM': choose a base estimator from which the
                         transformer is built. The estimator must have
                         either a feature_importances_ or coef_ attribute
                         after fitting. No default option. You can use a
                         model from the ATOM package. No default option.
            - for 'RFE': choose a supervised learning estimator. The
                         estimator must have either a feature_importances_
                         or coef_ attribute after fitting. You can use a
                         model from the ATOM package. No default option.
            - for 'RFECV': choose a supervised learning estimator. The
                           estimator must have either feature_importances_
                           or coef_ attribute after fitting. You can use a
                           model from the ATOM package. No default option.

    n_features: int, float or None, optional (default=None)
        Number of features to select (except for RFECV, where it's the
        minimum number of features to select).
            - None: Select all features.
            - if < 1: Fraction of the total features to select.
            - if >= 1: Number of features to select.

    max_frac_repeated: float or None, optional (default=1.)
        Remove features with the same value in at least this fraction of
        the total rows. The default is to keep all features with non-zero
        variance, i.e. remove the features that have the same value in all
        samples. None to skip this step.

    max_correlation: float or None, optional (default=0.98)
        Minimum value of the Pearson correlation coefficient to identify
        correlated features. A dataframe of the removed features and their
        correlation values can be accessed through the collinear attribute.
        None to skip this step.

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
        - If None: No logger.
        - If bool: True for logging file with default name, False for no logger.
        - If string: name of the logging file. 'auto' for default name.
        - If class: python Logger object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the RandomState instance used by `np.random`.

    **kwargs
        Any extra keyword argument for the PCA, SFM, RFE or RFECV estimators.
        See the corresponding sklearn documentation for the available options.

    """

    def __init__(self,
                 strategy: Optional[str] = None,
                 solver: Optional[Union[str, callable]] = None,
                 n_features: Optional[Union[int, float]] = None,
                 max_frac_repeated: Optional[Union[int, float]] = 1.,
                 max_correlation: Optional[float] = 1.,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 logger: Optional[Union[bool, str, callable]] = None,
                 random_state: Optional[int] = None,
                 **kwargs):
        super().__init__(n_jobs=n_jobs,
                         verbose=verbose,
                         logger=logger,
                         random_state=random_state)

        # Check parameters
        if isinstance(strategy, str):
            strats = ['univariate', 'pca', 'sfm', 'rfe', 'rfecv']

            if strategy.lower() not in strats:
                raise ValueError("Invalid value for the strategy parameter. " +
                                 "Choose from: univariate, PCA, SFM, RFE or " +
                                 "RFECV.")

            elif strategy.lower() == 'univariate':
                solvers_dct = dict(f_classif=f_classif,
                                   f_regression=f_regression,
                                   mutual_info_classif=mutual_info_classif,
                                   mutual_info_regression=mutual_info_regression,
                                   chi2=chi2)

                if not solver:
                    raise ValueError("Choose a solver for the strategy!")
                elif solver in solvers_dct:
                    solver = solvers_dct[solver]
                elif isinstance(solver, str):
                    raise ValueError(
                        f"Unknown solver. Choose from: {', '.join(solvers_dct)}.")

            elif strategy.lower() == 'pca':
                solver = 'auto' if solver is None else solver

            elif strategy.lower() in ['sfm', 'rfe', 'rfecv']:
                if solver is None:
                    raise ValueError("Select a solver for the strategy!")
                elif isinstance(solver, str):
                    # Assign goal depending on solver's ending
                    if solver[-6:] == '_class':
                        self.goal = 'classification'
                        solver = solver[:-6]
                    elif solver[-4:] == '_reg':
                        self.goal = 'regression'
                        solver = solver[:-4]

                    if solver.lower() not in map(str.lower, MODEL_LIST.keys()):
                        raise ValueError("Unknown value for the solver " +
                                         f"parameter, got {solver}. Try " +
                                         f"one of {MODEL_LIST.keys()}.")
                    else:  # Set to right model name and call model's method
                        model_class = MODEL_LIST[get_model_name(solver)]
                        solver = model_class(self).get_model()

        if n_features is not None and n_features <= 0:
            raise ValueError("Invalid value for the n_features parameter. " +
                             f"Value should be >0, got {n_features}.")
        if max_frac_repeated is not None and not 0 <= max_frac_repeated <= 1:
            raise ValueError(
                "Invalid value for the max_frac_repeated parameter. Value should " +
                f"be between 0 and 1, got {max_frac_repeated}.")
        if max_correlation is not None and not 0 <= max_correlation <= 1:
            raise ValueError(
                "Invalid value for the max_correlation parameter. Value should " +
                f"be between 0 and 1, got {max_correlation}.")

        # Define attributes
        self.strategy = strategy
        self.solver = solver
        self.n_features = n_features
        self.max_frac_repeated = max_frac_repeated
        self.max_correlation = max_correlation
        self.kwargs = kwargs

        self.collinear = None
        self.univariate = None
        self.scaler = None
        self.pca = None
        self.sfm = None
        self.rfe = None
        self.rfecv = None
        self._is_fitted = False

    def _remove_low_variance(self, X, _vb=2):
        """Remove features with too low variance.

        Parameters
        ----------
        X: pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        _vb: int, optional (default=2)
            Internal parameter to silence the fit method. If default, prints.

        Returns
        -------
        X: pd.DataFrame
            Dataframe with no low variance columns.

        """
        for n, col in enumerate(X):
            unique, count = np.unique(X[col], return_counts=True)
            for u, c in zip(unique, count):
                # If count is larger than fraction of total...
                if c > self.max_frac_repeated * len(X):
                    self.log(f" --> Feature {col} was removed due to " +
                             f"low variance. Value {u} repeated in " +
                             f"{round(c/len(X)*100., 1)}% of rows.", _vb)
                    X.drop(col, axis=1, inplace=True)
                    break
        return X

    def _remove_collinear(self, X, _vb=2):
        """Remove collinear features.

        Finds pairs of collinear features based on the Pearson correlation
        coefficient. For each pair above the specified limit (in terms of
        absolute value), it removes one of the two. Using code adapted from:
        https://chrisalbon.com/machine_learning/feature_selection/
        drop_highly_correlated_features

        Parameters
        ----------
        X: pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        _vb: int, optional (default=2)
            Internal parameter to silence the fit method. If default, prints.

        Returns
        -------
        X: pd.DataFrame
            Dataframe with no highly correlated features.

        """
        max_ = self.max_correlation
        mtx = X.corr()  # Pearson correlation coefficient matrix

        # Extract the upper triangle of the correlation matrix
        upper = mtx.where(np.triu(np.ones(mtx.shape).astype(np.bool), k=1))

        # Select the features with correlations above the threshold
        to_drop = [i for i in upper.columns if any(abs(upper[i] > max_))]

        # Dataframe to hold correlated pairs
        self.collinear = pd.DataFrame(columns=['drop_feature',
                                               'correlated_feature',
                                               'correlation_value'])

        # Iterate to record pairs of correlated features
        for col in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[abs(upper[col]) > max_])

            # Find the correlated values
            corr_values = list(round(upper[col][abs(upper[col]) > max_], 5))
            drop_features = set([col for _ in corr_features])

            # Add to class attribute
            self.collinear = self.collinear.append(
                {'drop_feature': ', '.join(drop_features),
                 'correlated_feature': ', '.join(corr_features),
                 'correlation_value': ', '.join(map(str, corr_values))},
                ignore_index=True)

            self.log(f" --> Feature {col} was removed due to " +
                     "collinearity with another feature.", _vb)

        return X.drop(to_drop, axis=1)

    @composed(crash, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit the data to the selected strategy.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series
            - If None: y is not used in the estimator.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Data target column with shape=(n_samples,).

        Returns
        -------
        self: FeatureSelector

        """
        X, y = self._prepare_input(X, y)

        # Remove features with too low variance
        if self.max_frac_repeated is not None:
            X = self._remove_low_variance(X, _vb=10)

        # Remove features with too high correlation
        if self.max_correlation is not None:
            X = self._remove_collinear(X, _vb=10)

        # Set n_features as all or fraction of total
        if self.n_features is None:
            self.n_features = X.shape[1]
        elif self.n_features < 1:
            self.n_features = int(self.n_features * X.shape[1])

        # Perform selection based on strategy
        if self.strategy is None:
            self._is_fitted = True
            return self  # Exit feature_selection

        elif self.strategy.lower() == 'univariate':
            self.univariate = SelectKBest(self.solver, k=self.n_features)
            self.univariate.fit(X, y)

        elif self.strategy.lower() == 'pca':
            # Always fit in case the data to transform is not scaled
            self.scaler = Scaler().fit(X)
            if not check_scaling(X):
                X = self.scaler.transform(X)

            # Define PCA
            self.pca = PCA(n_components=None,
                           svd_solver=self.solver,
                           **self.kwargs)
            self.pca.fit(X)
            self.pca.n_components_ = self.n_features  # Number of components

        elif self.strategy.lower() == 'sfm':
            # If any of these attr exists, model is already fitted
            condition1 = hasattr(self.solver, 'coef_')
            condition2 = hasattr(self.solver, 'feature_importances_')
            self.kwargs['prefit'] = True if condition1 or condition2 else False

            self.sfm = SelectFromModel(estimator=self.solver,
                                       max_features=self.n_features,
                                       **self.kwargs)
            if not self.kwargs['prefit']:
                self.sfm.fit(X, y)

        elif self.strategy.lower() == 'rfe':
            self.rfe = RFE(estimator=self.solver,
                           n_features_to_select=self.n_features,
                           **self.kwargs)
            self.rfe.fit(X, y)

        elif self.strategy.lower() == 'rfecv':
            if self.n_features == X.shape[1]:
                self.n_features = 1

            self.rfecv = RFECV(estimator=self.solver,
                               min_features_to_select=self.n_features,
                               n_jobs=self.n_jobs,
                               **self.kwargs)
            self.rfecv.fit(X, y)

        self._is_fitted = True
        return self

    @composed(crash, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Transform the data according to the selected strategy.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array or pd.Series, optional (default=None)
            Does nothing. Only for continuity of API. Is returned unchanged if
            provided.

        Returns
        -------
        X: pd.DataFrame
            Copy of the feature dataset.

        """
        check_is_fitted(self, '_is_fitted')
        X, y = self._prepare_input(X, y)

        self.log("Performing feature selection...", 1)

        # Remove features with too low variance
        if self.max_frac_repeated is not None:
            X = self._remove_low_variance(X)

        # Remove features with too high correlation
        if self.max_correlation is not None:
            X = self._remove_collinear(X)

        # Perform selection based on strategy
        if self.strategy is None:
            return variable_return(X, y)

        elif self.strategy.lower() == 'univariate':
            for n, col in enumerate(X):
                if not self.univariate.get_support()[n]:
                    self.log(f" --> Feature {col} was removed after the uni" +
                             "variate test (score: {:.2f}  p-value: {:.2f})."
                             .format(self.univariate.scores_[n],
                                     self.univariate.pvalues_[n]), 2)
                    X.drop(col, axis=1, inplace=True)

        elif self.strategy.lower() == 'pca':
            self.log(f" --> Applying Principal Component Analysis...", 2)

            if not check_scaling(X):
                self.log("   >>> Scaling features...", 2)
                X = self.scaler.transform(X)

            # Define PCA, keep in mind that it has all components still
            n = self.pca.n_components_
            var = np.array(self.pca.explained_variance_ratio_[:n])
            X = to_df(self.pca.transform(X)[:, :n], index=X.index, pca=True)
            self.log("   >>> Total explained variance: {}"
                     .format(round(var.sum(), 3)), 2)

        elif self.strategy.lower() == 'sfm':
            for n, column in enumerate(X):
                if not self.sfm.get_support()[n]:
                    self.log(f" --> Feature {column} was removed by the " +
                             f"{self.solver.__class__.__name__}.", 2)
                    X.drop(column, axis=1, inplace=True)

        elif self.strategy.lower() == 'rfe':
            for n, column in enumerate(X):
                if not self.rfe.support_[n]:
                    self.log(f" --> Feature {column} was removed by the " +
                             "recursive feature eliminator.", 2)
                    X.drop(column, axis=1, inplace=True)

        elif self.strategy.lower() == 'rfecv':
            for n, column in enumerate(X):
                if not self.rfecv.support_[n]:
                    self.log(f" --> Feature {column} was removed by the " +
                             "RFECV.", 2)
                    X.drop(column, axis=1, inplace=True)

        return variable_return(X, y)

    # << ======================= Plot methods ======================= >>

    @composed(crash, typechecked)
    def plot_pca(self,
                 title: Optional[str] = None,
                 figsize: Optional[Tuple[int, int]] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the explained variance ratio vs the number of component.

        Only if PCA was applied on the dataset through the feature_selection
        method.

        """
        plot_pca(self, title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_components(self,
                        show: Optional[int] = None,
                        title: Optional[str] = None,
                        figsize: Optional[Tuple[int, int]] = None,
                        filename: Optional[str] = None,
                        display: bool = True):
        """Plot the explained variance ratio per component.

        Only if PCA was applied on the dataset through the feature_selection
        method.

        """
        plot_components(self, show, title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_rfecv(self,
                   title: Optional[str] = None,
                   figsize: Optional[Tuple[int, int]] = (10, 6),
                   filename: Optional[str] = None,
                   display: bool = True):
        """Plot the RFECV results.

        Plot the scores obtained by the estimator fitted on every subset of
        the data. Only if RFECV was applied on the dataset through the
        feature_selection method.

        """
        plot_rfecv(self, title, figsize, filename, display)

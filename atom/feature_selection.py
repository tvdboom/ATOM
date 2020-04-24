# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the feature selection and insertion estimators.

"""

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
import pandas as pd
from typing import Optional, Union
from sklearn.base import BaseEstimator

# Third party packages
from gplearn.genetic import SymbolicTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    chi2, SelectKBest, SelectFromModel, RFE, RFECV
    )

# Own modules
from .models import model_list
from .data_cleaning import DataCleanerEstimator, Scaler
from .utils import scalar, variable_return, to_df, check_scaling


# << ================ Classes ================= >>

class FeatureGenerator(BaseEstimator, DataCleanerEstimator):
    """Create new features using a genetic algorithm."""

    def __init__(self,
                 n_features: int = 2,
                 generations: int = 20,
                 population: int = 500,
                 **kwargs):
        """Initialize class.

        Use a genetic algorithm to create new combinations of existing
        features and add them to the original dataset in order to capture
        the non-linear relations between the original features. A dataframe
        containing the description of the newly generated features and their
        scores can be accessed through the `genetic_features` attribute. The
        algorithm is implemented using the Symbolic Transformer method, which
        can be accessed through the `genetic_algorithm` attribute. It is
        advised to only use this method when fitting linear models.
        Dependency: gplearn.

        Parameters -------------------------------------

        n_features: int, optional (default=2)
            Maximum number of newly generated features (no more than 1%
            of the population).

        generations: int, optional (default=20)
            Number of generations to evolve.

        population: int, optional (default=500)
            Number of programs in each generation.

        **kwargs
            Additional parameters for the estimator.

        """
        super().__init__(**kwargs)

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

        self.genetic_algorithm = None
        self.genetic_features = None

    def fit(self, X, y):
        """Fit the individual encoders on each column.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: sequence, np.array or pd.Series, optional (default=None)
            Data target column with shape=(n_samples,)

        Returns
        -------
        self: FeatureInsertor

        """
        X, y = self.prepare_input(X, y)

        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs',
                        'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']

        self.genetic_algorithm = \
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
        self.genetic_algorithm.fit(X, y)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Apply the encoding transformations.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: None
            Does nothing. Only for continuity of API. Is returned unchanged if
            provided.

        Returns
        -------
        X: pd.DataFrame
            Dataframe containing the newly generated features.

        """
        self._check_is_fitted()
        X, y = self.prepare_input(X, y)

        self._log("Running genetic algorithm...", 1)

        new_features = self.genetic_algorithm.transform(X)

        # ix = indices of all new features that are not in the original set
        # descript = list of the operators applied to create the new features
        # fitness = list of fitness scores of the new features
        ix, descript, fitness = [], [], []
        for i, program in enumerate(self.genetic_algorithm):
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
            self._log("WARNING! The genetic algorithm couldn't find any " +
                      "improving non-linear features!", 1)
            return X, y

        # Get indices of the best features
        if len(descript) > self.n_features:
            ix = np.argpartition(fitness, -self.n_features)[-self.n_features:]
        else:
            ix = range(len(descript))

        # Select best features only
        new_features = new_features[:, ix]
        descript = [descript[i] for i in range(len(descript)) if i in ix]
        fitness = [fitness[i] for i in range(len(fitness)) if i in ix]
        names = ['Feature ' + str(1 + i + len(X.columns))
                 for i in range(new_features.shape[1])]

        # Create dataframe attribute
        data = {'Name': names, 'Description': descript, 'Fitness': fitness}
        self.genetic_features = pd.DataFrame(data)
        self._log("---------------------------------------------------" +
                  "---------------------------------------", 2)

        for feature in descript:
            self._log(f" --> New feature {feature} added to the dataset.", 1)

        X = pd.DataFrame(np.hstack((X, new_features)),
                         columns=X.columns.to_list() + names)

        return variable_return(X, y)


class FeatureSelector(BaseEstimator, DataCleanerEstimator):
    """Apply feature selection techniques."""

    def __init__(self,
                 strategy: Optional[str] = None,
                 solver: Optional[Union[str, callable]] = None,
                 n_features: Optional[scalar] = None,
                 max_frac_repeated: Optional[scalar] = 1.,
                 max_correlation: Optional[float] = 0.98,
                 **kwargs):
        """Initialize class.

        Remove features according to the selected strategy. Ties between
        features with equal scores will be broken in an unspecified way.
        Also removes features with too low variance and finds pairs of
        collinear features based on the Pearson correlation coefficient. For
        each pair above the specified limit (in terms of absolute value), it
        removes one of the two.

        Note that the RFE and RFECV strategies don't work when the solver is a
        CatBoost model due to incompatibility of the APIs. If the pipeline has
        already ran before running the RFECV, the scoring parameter will be set
        to the selected metric (if scoring=None).

        Parameters
        ----------
        strategy: string or None, optional (default=None)
            Feature selection strategy to use. Choose from:
                - None: do not perform any feature selection algorithm
                - 'univariate': perform a univariate F-test
                - 'PCA': perform principal component analysis
                - 'SFM': select best features from model
                - 'RFE': recursive feature eliminator
                - 'RFECV': RFE with cross-validated selection

            The sklearn objects can be found under the univariate, PCA, SFM,
            RFE or RFECV attributes of the class.

        solver: string, callable or None, optional (default=None)
            Solver or model to use for the feature selection strategy. See the
            sklearn documentation for an extended description of the choices.
            Select None for the default option per strategy (not applicable
            for SFM, RFE and RFECV).
                - for 'univariate', choose from:
                    + 'f_classif' (default for classification tasks)
                    + 'f_regression' (default for regression tasks)
                    + 'mutual_info_classif'
                    + 'mutual_info_regression'
                    + 'chi2'
                    + Any function taking two arrays (X, y), and returning
                      arrays (scores, pvalues). See the documentation.
                - for 'PCA', choose from:
                    + 'auto' (default)
                    + 'full'
                    + 'arpack'
                    + 'randomized'
                - for 'SFM': choose a base estimator from which the
                             transformer is built. The estimator must have
                             either a feature_importances_ or coef_ attribute
                             after fitting. No default option. You can use a
                             model from the ATOM pipeline. No default option.
                - for 'RFE': choose a supervised learning estimator. The
                             estimator must have either a feature_importances_
                             or coef_ attribute after fitting. You can use a
                             model from the ATOM pipeline. No default option.
                - for 'RFECV': choose a supervised learning estimator. The
                               estimator must have either feature_importances_
                               or coef_ attribute after fitting. You can use a
                               model from the ATOM pipeline. No default option.

        n_features: int, float or None, optional (default=None)
            Number of features to select (except for RFECV, where it's the
            minimum number of features to select).
                - if < 1: fraction of features to select
                - if >= 1: number of features to select
                - None to select all

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

        **kwargs
            Additional parameters for the estimator.

        """
        super().__init__(**kwargs)

        # Check Parameters
        if strategy is not None:
            if strategy.lower() == 'univariate':
                solvers_dct = \
                    dict(f_classif=f_classif,
                         f_regression=f_regression,
                         mutual_info_classif=mutual_info_classif,
                         mutual_info_regression=mutual_info_regression,
                         chi2=chi2)

                if solver is None and self.task == 'regression':
                    solver = f_regression
                elif solver is None:
                    solver = f_classif
                elif solver in solvers_dct.keys():
                    solver = solvers_dct[solver]
                elif isinstance(solver, str):
                    raise ValueError("Unknown solver. Choose from: {}."
                                     .format(', '.join(solvers_dct.keys())))

            elif strategy.lower() == 'pca':
                solver = 'auto' if solver is None else solver

            elif strategy.lower() in ['sfm', 'rfe', 'rfecv']:
                if solver is None:
                    if self.winner is None:
                        raise ValueError("Select a model for the solver!")
                    else:
                        solver = self.winner.best_model_fit

                elif isinstance(solver, str):
                    if solver.lower() not in map(str.lower, model_list.keys()):
                        raise ValueError("Unknown value for the solver " +
                                         f"parameter, got {solver}. Try " +
                                         f"one of {model_list.keys()}.")
                    else:  # Set to right model name and call estimator
                        for m in model_list.keys():
                            if m.lower() == solver.lower():
                                solver = model_list[m](self).get_model()
                                break

            else:
                raise ValueError("Invalid value for the strategy parameter. " +
                                 "Choose from: univariate, PCA, SFM, " +
                                 "RFE or RFECV.")

        if n_features is not None and n_features <= 0:
            raise ValueError("Invalid value for the n_features parameter." +
                             f"Value should be >0, got {n_features}.")
        if max_frac_repeated is not None and not 0 <= max_frac_repeated <= 1:
            raise ValueError("Invalid value for the max_frac_repeated param" +
                             "eter. Value should be between 0 and 1, got {}."
                             .format(max_frac_repeated))
        if max_correlation is not None and not 0 <= max_correlation <= 1:
            raise ValueError("Invalid value for the max_correlation param" +
                             "eter. Value should be between 0 and 1, got {}."
                             .format(max_correlation))

        # Define attributes
        self.strategy = strategy
        self.solver = solver
        self.n_features = n_features
        self.max_frac_repeated = max_frac_repeated
        self.max_correlation = max_correlation

        self.collinear = None
        self.univariate = None
        self.scaler = None
        self.PCA = None
        self.PCA2 = None
        self.SFM = None
        self.RFE = None
        self.RFECV = None

    def remove_low_variance(self, X):
        """Remove features with too low variance.

        Parameters
        ----------
        X: pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

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
                    self._log(f" --> Feature {col} was removed due to " +
                              f"low variance. Value {u} repeated in " +
                              f"{round(c/len(X)*100., 1)}% of rows.",
                              2)
                    X.drop(col, axis=1, inplace=True)
                    break
        return X

    def remove_collinear(self, X):
        """Remove collinear features.

        Finds pairs of collinear features based on the Pearson
        correlation coefficient. For each pair above the specified
        limit (in terms of absolute value), it removes one of the two.
        Using code adapted from: https://chrisalbon.com/machine_learning/
        feature_selection/drop_highly_correlated_features

        Parameters
        ----------
        X: pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        Returns
        -------
        X: pd.DataFrame
            Dataframe with no highly correlated features.

        """
        limit = self.max_correlation
        mtx = X.corr()  # Pearson correlation coefficient matrix

        # Extract the upper triangle of the correlation matrix
        upper = mtx.where(np.triu(np.ones(mtx.shape).astype(np.bool), k=1))

        # Select the features with correlations above the threshold
        to_drop = [i for i in upper.columns if any(abs(upper[i]) > limit)]

        # Dataframe to hold correlated pairs
        self.collinear = pd.DataFrame(columns=['drop_feature',
                                               'correlated_feature',
                                               'correlation_value'])

        # Iterate to record pairs of correlated features
        for column in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[abs(upper[column]) > limit])

            # Find the correlated values
            corr_values = list(round(
                upper[column][abs(upper[column]) > limit], 5))
            drop_features = set([column for _ in corr_features])

            # Add to class attribute
            self.collinear = self.collinear.append(
                {'drop_feature': ', '.join(drop_features),
                 'correlated_feature': ', '.join(corr_features),
                 'correlation_value': ', '.join(map(str, corr_values))},
                ignore_index=True)

            self._log(f" --> Feature {column} was removed due to " +
                      "collinearity with another feature.", 2)

        return X.drop(to_drop, axis=1)

    def fit(self, X, y):
        """Fit the data to the selected strategy.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, dict, sequence, np.array, pd.Series or None
            Target column index, name or array-like, with shape=(n_samples,).

        Returns
        -------
        self: FeatureSelector

        """
        X, y = self.prepare_input(X, y)
        self._is_fitted = True

        # Remove features with too low variance
        if self.max_frac_repeated is not None:
            X = self.remove_low_variance(X)

        # Remove features with too high correlation
        if self.max_correlation is not None:
            X = self.remove_collinear(X)

        # Set n_features as all or fraction of total
        if self.n_features is None:
            self.n_features = X.shape[1]
        elif self.n_features < 1:
            self.n_features = int(self.n_features * X.shape[1])

        # Perform selection based on strategy
        if self.strategy is None:
            return self  # Exit feature_selection

        if self.strategy.lower() == 'univariate':
            self.univariate = SelectKBest(self.solver, k=self.n_features)
            self.univariate.fit(X, y)

        elif self.strategy.lower() == 'pca':
            # Always fit in case the data to transform is not scaled
            self.scaler = Scaler(verbose=self.verbose).fit(X)
            if not check_scaling(X):
                X = self.scaler.transform(X)

            # Define PCA
            self.PCA = PCA(n_components=self.n_features,
                           svd_solver=self.solver,
                           **self.kwargs)
            self.PCA.fit(X)

            # Another PCA object to get the explained variances for all the
            # components for the plots
            self.PCA2 = PCA(n_components=None,
                            svd_solver=self.solver,
                            **self.kwargs)
            self.PCA2.fit(X)

        elif self.strategy.lower() == 'sfm':

            # If any of these attr exists, model is already fitted
            condition1 = hasattr(self.solver, 'coef_')
            condition2 = hasattr(self.solver, 'feature_importances_')
            self.kwargs['prefit'] = True if condition1 or condition2 else False

            self.SFM = SelectFromModel(estimator=self.solver,
                                       max_features=self.n_features,
                                       **self.kwargs)
            if not self.kwargs['prefit']:
                self.SFM.fit(X, y)

        elif self.strategy.lower() == 'rfe':
            self.RFE = RFE(estimator=self.solver,
                           n_features_to_select=self.n_features,
                           **self.kwargs)
            self.RFE.fit(X, y)

        elif self.strategy.lower() == 'rfecv':
            if self.n_features == X.shape[1]:
                self.n_features = 1

            # If pipeline ran already, use selected metric
            if hasattr(self, 'metric') and 'scoring' not in self.kwargs.keys():
                self.kwargs['scoring'] = self.metric

            self.RFECV = RFECV(estimator=self.solver,
                               min_features_to_select=self.n_features,
                               n_jobs=self.n_jobs,
                               **self.kwargs)
            self.RFECV.fit(X, y)

        return self

    def transform(self, X, y=None):
        """Transform the data according to the selected strategy.

        Parameters
        ----------
        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: None
            Does nothing. Only for continuity of API. Is returned unchanged if
            provided.

        Returns
        -------
        X: pd.DataFrame
            Copy of the feature dataset.

        """
        self._check_is_fitted()
        X, y = self.prepare_input(X, y)

        self._log("Performing feature selection...", 1)

        # Remove features with too low variance
        if self.max_frac_repeated is not None:
            X = self.remove_low_variance(X)

        # Remove features with too high correlation
        if self.max_correlation is not None:
            X = self.remove_collinear(X)

        # Set n_features as all or fraction of total
        if self.n_features is None:
            self.n_features = X.shape[1]
        elif self.n_features < 1:
            self.n_features = int(self.n_features * X.shape[1])

        # Perform selection based on strategy
        if self.strategy is None:
            return variable_return(X, y)

        elif self.strategy.lower() == 'univariate':
            for n, col in enumerate(X):
                if not self.univariate.get_support()[n]:
                    self._log(f" --> Feature {col} was removed after the uni" +
                              "variate test (score: {:.2f}  p-value: {:.2f})."
                              .format(self.univariate.scores_[n],
                                      self.univariate.pvalues_[n]), 2)
                    X.drop(col, axis=1, inplace=True)

        elif self.strategy.lower() == 'pca':
            self._log(f" --> Applying Principal Component Analysis...", 2)

            if not check_scaling(X):
                X = self.scaler.transform(X)

            # Define PCA
            var = np.array(self.PCA.explained_variance_ratio_)
            X = to_df(self.PCA.transform(X), pca=True)
            self._log("   >>> Total explained variance: {}"
                      .format(round(var.sum(), 3)), 2)

        elif self.strategy.lower() == 'sfm':
            for n, column in enumerate(X):
                if not self.SFM.get_support()[n]:
                    self._log(f" --> Feature {column} was removed by the " +
                              f"{self.solver.__class__.__name__}.", 2)
                    X.drop(column, axis=1, inplace=True)

        elif self.strategy.lower() == 'rfe':
            for n, column in enumerate(X):
                if not self.RFE.support_[n]:
                    self._log(f" --> Feature {column} was removed by the " +
                              "recursive feature eliminator.", 2)
                    X.drop(column, axis=1, inplace=True)

        elif self.strategy.lower() == 'rfecv':
            for n, column in enumerate(X):
                if not self.RFECV.support_[n]:
                    self._log(f" --> Feature {column} was removed by the " +
                              "RFECV.", 2)
                    X.drop(column, axis=1, inplace=True)

        return variable_return(X, y)

# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing all the available models for the fit method
             of the ATOM class. All classes must have the following structure:

        Name
        ----
        Name of the model's class in camel case format.

        Attributes
        ----------
        name: string
            Short acronym of the model's longname for calling.

        longname: string
            Name of the model.

        Methods
        -------
        __init__(self, *args):
            Class initializer (contains super() to parent class).

        get_params(self, x):
            Return a dictionary of the model´s hyperparameters.

        get_model(self, params={}):
            Return the model object with unpacked parameters.

        custom_fit(model, train, validation):
            If the direct fit method of the model is not enough and you desire to
            customize it a bit, make a custom_fit method. It will run instead.

        get_domain(self):
            Return a list of the bounds for the hyperparameters.

        get_init_values(self):
            Return the default values of the model's hyperparameters.


To add a new model:
    1. Add the model's class to models.py
    2. Add the model to the list MODEL_LIST in models.py
    3. Add the name to all the relevant variables in utils.py


List of available models:
    - 'GNB' for Gaussian Naive Bayes (no hyperparameter tuning)
    - 'MNB' for Multinomial Naive Bayes
    - 'BNB' for Bernoulli Naive Bayes
    - 'GP' for Gaussian Process (no hyperparameter tuning)
    - 'OLS' for Ordinary Least Squares (no hyperparameter tuning)
    - 'Ridge' for Ridge Linear
    - 'Lasso' for Lasso Linear Regression
    - 'EN' for ElasticNet Linear Regression
    - 'BR' for Bayesian Regression (with ridge regularization)
    - 'LR' for Logistic Regression
    - 'LDA' for Linear Discriminant Analysis
    - 'QDA' for Quadratic Discriminant Analysis
    - 'KNN' for K-Nearest Neighbors
    - 'Tree' for a single Decision Tree
    - 'Bag' for Bagging (with decision tree as base estimator)
    - 'ET' for Extra-Trees
    - 'RF' for Random Forest
    - 'AdaB' for AdaBoost (with decision tree as base estimator)
    - 'GBM' for Gradient Boosting Machine
    - 'XGB' for XGBoost (if package is available)
    - 'LGB' for LightGBM (if package is available)
    - 'CatB' for CatBoost (if package is available)
    - 'lSVM' for Linear Support Vector Machine
    - 'kSVM' for Kernel (non-linear) Support Vector Machine
    - 'PA' for Passive Aggressive
    - 'SGD' for Stochastic Gradient Descent
    - 'MLP' for Multilayer Perceptron

"""

# Standard packages
import random
import numpy as np

# Others
from skopt.space.space import Real, Integer, Categorical

# Sklearn models
from sklearn.gaussian_process import (
    GaussianProcessClassifier, GaussianProcessRegressor
    )
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import (
    LinearRegression, RidgeClassifier, Ridge as RidgeRegressor,
    Lasso as LassoRegressor, ElasticNet as ElasticNetRegressor,
    BayesianRidge, LogisticRegression as LR
    )
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
    )
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
    )
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.linear_model import (
    PassiveAggressiveClassifier as PAC, PassiveAggressiveRegressor as PAR,
    SGDClassifier, SGDRegressor
    )
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Own modules
from .basemodel import BaseModel


# Classes =================================================================== >>

class GaussianProcess(BaseModel):
    """Gaussian process model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'GP', 'Gaussian Process'

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return GaussianProcessClassifier(random_state=self.T.random_state,
                                             n_jobs=self.T.n_jobs,
                                             **params)
        else:
            return GaussianProcessRegressor(random_state=self.T.random_state,
                                            **params)


class GaussianNaiveBayes(BaseModel):
    """Gaussian Naive Bayes model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'GNB', 'Gaussian Naive Bayes'

    @staticmethod
    def get_model(params={}):
        """Call the model object."""
        return GaussianNB(**params)


class MultinomialNaiveBayes(BaseModel):
    """Multinomial Naive Bayes model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'MNB', 'Multinomial Naive Bayes'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model's hyperparameters."""
        params = {'alpha': round(x[0], 3),
                  'fit_prior': x[1]}
        return params

    @staticmethod
    def get_model(params={}):
        """Call the model object."""
        return MultinomialNB(**params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Real(1e-3, 10, 'log-uniform', name='alpha'),
                Categorical([True, False], name='fit_prior')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [1, True]


class BernoulliNaiveBayes(BaseModel):
    """Bernoulli Naive Bayes model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'BNB', 'Bernoulli Naive Bayes'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'alpha': round(x[0], 3),
                  'fit_prior': x[1]}
        return params

    @staticmethod
    def get_model(params={}):
        """Call the model object."""
        return BernoulliNB(**params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Real(1e-3, 10, 'log-uniform', name='alpha'),
                Categorical([True, False], name='fit_prior')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [1, True]


class OrdinaryLeastSquares(BaseModel):
    """Linear Regression without regularization (OLS)."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'OLS', 'Ordinary Least Squares'

    def get_model(self, params={}):
        """Call the model object."""
        return LinearRegression(n_jobs=self.T.n_jobs, **params)


class Ridge(BaseModel):
    """Linear Regression/Classification with ridge regularization."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name = 'Ridge'
        if self.T.goal.startswith('class'):
            self.longname = 'Ridge Classification'
        else:
            self.longname = 'Ridge Regression'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'max_iter': x[0],
                  'alpha': round(x[1], 3)}
        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return RidgeClassifier(random_state=self.T.random_state, **params)
        else:
            return RidgeRegressor(random_state=self.T.random_state, **params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(100, 1000, name='max_iter'),
                Real(1e-3, 10, 'log-uniform', name='alpha')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [500, 1.0]


class Lasso(BaseModel):
    """Linear Regression with lasso regularization."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'Lasso', 'Lasso Regression'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'max_iter': x[0],
                  'alpha': round(x[1], 3)}
        return params

    def get_model(self, params={}):
        """Call the model object."""
        return LassoRegressor(random_state=self.T.random_state, **params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(100, 1000, name='max_iter'),
                Real(1e-3, 10, 'log-uniform', name='alpha')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [500, 1.0]


class ElasticNet(BaseModel):
    """Linear Regression with both lasso and ridge regularization."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'EN', 'ElasticNet Regression'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'max_iter': x[0],
                  'alpha': round(x[1], 3),
                  'l1_ratio': round(x[2], 1)}
        return params

    def get_model(self, params={}):
        """Call the model object."""
        return ElasticNetRegressor(random_state=self.T.random_state, **params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(100, 1000, name='max_iter'),
                Real(1e-3, 10, 'log-uniform', name='alpha'),
                Categorical(np.linspace(0.1, 0.9, 9), name='l1_ratio')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [1000, 1.0, 0.5]


class BayesianRegression(BaseModel):
    """Linear Bayesian Regression model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'BR', 'Bayesian Regression'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'n_iter': x[0]}
        return params

    @staticmethod
    def get_model(params={}):
        """Call the model object."""
        return BayesianRidge(**params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(100, 1000, name='max_iter')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [300]


class LogisticRegression(BaseModel):
    """Logistic Regression model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'LR', 'Logistic Regression'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'max_iter': x[0],
                  'solver': x[1],
                  'class_weight': x[5]}

        # Limitations on solver and penalty combinations
        condition1 = (x[3] == 'none' and x[1] == 'liblinear')
        condition2 = (x[3] == 'l1' and x[1] not in ['liblinear', 'saga'])
        condition3 = (x[3] == 'elasticnet' and x[1] != 'saga')

        if condition1 or condition2 or condition3:
            x[3] = 'l2'  # Change to default value

        params['penalty'] = x[3]
        if x[3] == 'elasticnet':
            params['l1_ratio'] = round(x[4], 1)
        if x[3] != 'none':
            params['C'] = round(x[2], 3)

        return params

    def get_model(self, params={}):
        """Call the model object."""
        return LR(random_state=self.T.random_state,
                  n_jobs=self.T.n_jobs,
                  **params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        penalties = ['none', 'l1', 'l2', 'elasticnet']
        return [Integer(100, 1000, name='max_iter'),
                Categorical(solvers, name='solver'),
                Real(1e-3, 100, 'log-uniform', name='C'),
                Categorical(penalties, name='penalty'),
                Categorical(np.linspace(0.1, 0.9, 9), name='l1_ratio'),
                Categorical([None, 'balanced'], name='class_weight')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [100, 'lbfgs', 1.0, 'l2', 0.5, None]


class LinearDiscriminantAnalysis(BaseModel):
    """Linear Discriminant Analysis model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'LDA', 'Linear Discriminant Analysis'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'solver': x[0]}

        if params['solver'] != 'svd':  # Add extra parameter: shrinkage
            params['shrinkage'] = round(x[1], 1)

        return params

    @staticmethod
    def get_model(params={}):
        """Call the model object."""
        return LDA(**params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Categorical(['svd', 'lsqr', 'eigen'], name='solver'),
                Categorical(np.linspace(0.0, 1.0, 11), name='shrinkage')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return ['svd', 0]


class QuadraticDiscriminantAnalysis(BaseModel):
    """Quadratic Discriminant Analysis model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'QDA', 'Quadratic Discriminant Analysis'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'reg_param': round(x[0], 1)}
        return params

    @staticmethod
    def get_model(params={}):
        """Call the model object."""
        return QDA(**params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Categorical(np.linspace(0.0, 1.0, 11), name='reg_param')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [0]


class KNearestNeighbors(BaseModel):
    """K-Nearest Neighbors model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'KNN', 'K-Nearest Neighbors'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'n_neighbors': x[0],
                  'leaf_size': x[1],
                  'p': x[2],
                  'weights': x[3]}
        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return KNeighborsClassifier(n_jobs=self.T.n_jobs, **params)
        else:
            return KNeighborsRegressor(n_jobs=self.T.n_jobs, **params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(1, 100, name='n_neighbors'),
                Integer(20, 40, name='leaf_size'),
                Categorical([1, 2], name='p'),
                Categorical(['distance', 'uniform'], name='weights')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [5, 30, 2, 'distance']


class DecisionTree(BaseModel):
    """Single Decision Tree."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'Tree', 'Decision Tree'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'criterion': x[0],
                  'splitter': x[1],
                  'max_depth': x[2],
                  'max_features': round(x[3], 1),
                  'min_samples_split': x[4],
                  'min_samples_leaf': x[5],
                  'ccp_alpha': round(x[6], 3)}
        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return DecisionTreeClassifier(random_state=self.T.random_state,
                                          **params)
        else:
            return DecisionTreeRegressor(random_state=self.T.random_state,
                                         **params)

    def get_domain(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            criterion = ['gini', 'entropy']
        else:
            criterion = ['mse', 'mae', 'friedman_mse']
        return [Categorical(criterion, name='criterion'),
                Categorical(['best', 'random'], name='splitter'),
                Integer(3, 10, name='max_depth'),
                Categorical(np.linspace(0.5, 1.0, 6), name='max_features'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 20, name='min_samples_leaf'),
                Real(0, 0.035, name='ccp_alpha')]

    def get_init_values(self):
        """Return the default values of the model's hyperparameters."""
        criterion = 'gini' if self.T.task != 'regression' else 'mse'
        return [criterion, 'best', 10, 1.0, 2, 1, 0.0]


class Bagging(BaseModel):
    """Bagging model (with decision tree as base estimator)."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name = 'Bag'
        if self.T.goal.startswith('class'):
            self.longname = 'Bagging Classifier'
        else:
            self.longname = 'Bagging Regressor'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'n_estimators': x[0],
                  'max_samples': round(x[1], 1),
                  'max_features': round(x[2], 1),
                  'bootstrap': x[3],
                  'bootstrap_features': x[4]}
        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return BaggingClassifier(random_state=self.T.random_state,
                                     n_jobs=self.T.n_jobs,
                                     **params)
        else:
            return BaggingRegressor(random_state=self.T.random_state,
                                    n_jobs=self.T.n_jobs,
                                    **params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(10, 500, name='n_estimators'),
                Categorical(np.linspace(0.5, 1.0, 6), name='max_samples'),
                Categorical(np.linspace(0.5, 1.0, 6), name='max_features'),
                Categorical([True, False], name='bootstrap'),
                Categorical([True, False], name='bootstrap_features')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [10, 1.0, 1.0, True, False]


class ExtraTrees(BaseModel):
    """Extremely Randomized Trees."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'ET', 'Extra-Trees'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'n_estimators': x[0],
                  'max_depth': x[1],
                  'max_features': round(x[2], 1),
                  'criterion': x[3],
                  'min_samples_split': x[4],
                  'min_samples_leaf': x[5],
                  'ccp_alpha': round(x[6], 3),
                  'bootstrap': x[7]}

        if params['bootstrap']:
            params['max_samples'] = round(x[8], 1)

        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return ExtraTreesClassifier(random_state=self.T.random_state,
                                        n_jobs=self.T.n_jobs,
                                        **params)
        else:
            return ExtraTreesRegressor(random_state=self.T.random_state,
                                       n_jobs=self.T.n_jobs,
                                       **params)

    def get_domain(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            criterion = ['gini', 'entropy']
        else:
            criterion = ['mse', 'mae']
        return [Integer(10, 500, name='n_estimators'),
                Integer(3, 10, name='max_depth'),
                Categorical(np.linspace(0.5, 1.0, 6), name='max_features'),
                Categorical(criterion, name='criterion'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 20, name='min_samples_leaf'),
                Real(0, 0.035, name='ccp_alpha'),
                Categorical([True, False], name='bootstrap'),
                Categorical(np.linspace(0.5, 0.9, 5), name='max_samples')]

    def get_init_values(self):
        """Return the default values of the model's hyperparameters."""
        criterion = 'gini' if self.T.task != 'regression' else 'mse'
        return [100, 10, 1.0, criterion, 2, 1, 0.0, False, 0.9]


class RandomForest(BaseModel):
    """Random Forest model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'RF', 'Random Forest'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'n_estimators': x[0],
                  'max_depth': x[1],
                  'max_features': round(x[2], 1),
                  'criterion': x[3],
                  'min_samples_split': x[4],
                  'min_samples_leaf': x[5],
                  'ccp_alpha': round(x[6], 3),
                  'bootstrap': x[7]}

        if params['bootstrap']:
            params['max_samples'] = round(x[8], 1)

        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return RandomForestClassifier(random_state=self.T.random_state,
                                          n_jobs=self.T.n_jobs,
                                          **params)
        else:
            return RandomForestRegressor(random_state=self.T.random_state,
                                         n_jobs=self.T.n_jobs,
                                         **params)

    def get_domain(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            criterion = ['gini', 'entropy']
        else:
            criterion = ['mse', 'mae']
        return [Integer(10, 500, name='n_estimators'),
                Integer(3, 10, name='max_depth'),
                Categorical(np.linspace(0.5, 1.0, 6), name='max_features'),
                Categorical(criterion, name='criterion'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 20, name='min_samples_leaf'),
                Real(0, 0.035, name='ccp_alpha'),
                Categorical([True, False], name='bootstrap'),
                Categorical(np.linspace(0.5, 0.9, 5), name='max_samples')]

    def get_init_values(self):
        """Return the default values of the model's hyperparameters."""
        criterion = 'gini' if self.T.task != 'regression' else 'mse'
        return [100, 10, 1.0, criterion, 2, 1, 0.0, True, 0.9]


class AdaBoost(BaseModel):
    """Adaptive Boosting (with decision tree as base estimator)."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'AdaB', 'AdaBoost'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'n_estimators': x[0],
                  'learning_rate': round(x[1], 2)}
        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return AdaBoostClassifier(random_state=self.T.random_state,
                                      **params)
        else:
            return AdaBoostRegressor(random_state=self.T.random_state,
                                     **params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(50, 500, name='n_estimators'),
                Real(0.01, 1, 'log-uniform', name='learning_rate')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [50, 1]


class GradientBoostingMachine(BaseModel):
    """Gradient Boosting Machine model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'GBM', 'Gradient Boosting Machine'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'n_estimators': x[0],
                  'learning_rate': round(x[1], 2),
                  'subsample': round(x[2], 1),
                  'max_depth': x[3],
                  'max_features': round(x[4], 1),
                  'criterion': x[5],
                  'min_samples_split': x[6],
                  'min_samples_leaf': x[7],
                  'ccp_alpha': round(x[8], 3)}
        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return GradientBoostingClassifier(random_state=self.T.random_state,
                                              **params)
        else:
            return GradientBoostingRegressor(random_state=self.T.random_state,
                                             **params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(50, 500, name='n_estimators'),
                Real(0.01, 1, 'log-uniform', name='learning_rate'),
                Categorical(np.linspace(0.5, 1.0, 6), name='subsample'),
                Integer(1, 10, name='max_depth'),
                Categorical(np.linspace(0.5, 1.0, 6), name='max_features'),
                Categorical(['friedman_mse', 'mae', 'mse'], name='criterion'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 20, name='min_samples_leaf'),
                Real(0, 0.035, name='ccp_alpha')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [100, 0.1, 1.0, 3, 1.0, 'friedman_mse', 2, 1, 0.0]


class XGBoost(BaseModel):
    """Extreme Gradient Boosting."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'XGB', 'XGBoost'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'n_estimators': x[0],
                  'learning_rate': round(x[1], 2),
                  'max_depth': x[2],
                  'gamma': round(x[3], 2),
                  'min_child_weight': x[4],
                  'subsample': round(x[5], 1),
                  'colsample_bytree': round(x[6], 1),
                  'reg_alpha': x[7],
                  'reg_lambda': x[8]}
        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        from xgboost import XGBClassifier, XGBRegressor
        # XGBoost can't handle random_state to be None
        if self.T.random_state is None:
            random_state = random.randint(0, np.iinfo(np.int16).max)
        else:
            random_state = self.T.random_state
        if self.T.goal.startswith('class'):
            return XGBClassifier(n_jobs=self.T.n_jobs,
                                 random_state=random_state,
                                 verbosity=0,
                                 **params)
        else:
            return XGBRegressor(n_jobs=self.T.n_jobs,
                                random_state=random_state,
                                verbosity=0,
                                **params)

    def custom_fit(self, model, train, validation):
        """Fit the model using early stopping and update evals attr."""
        # Determine early stopping rounds
        if not self._early_stopping or self._early_stopping >= 1:  # None or int
            rounds = self._early_stopping
        elif self._early_stopping < 1:
            rounds = int(model.get_params()['n_estimators'] * self._early_stopping)

        # Fit the model
        model.fit(train[0], train[1],
                  eval_set=[train, validation],
                  early_stopping_rounds=rounds,
                  verbose=False)

        # Create evals attribute with train and validation scores
        # Invert sign since XGBoost minimizes the metric
        metric_name = list(model.evals_result()['validation_0'])[0]
        self.evals = {'metric': metric_name,
                      'train': model.evals_result()['validation_0'][metric_name],
                      'test': model.evals_result()['validation_1'][metric_name]}

        iters = len(self.evals['train'])  # Iterations reached
        tot = int(model.get_params()['n_estimators'])  # Total iterations in params
        self._stopped = (iters, tot) if iters < tot else None

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(50, 500, name='n_estimators'),
                Real(0.01, 1, 'log-uniform', name='learning_rate'),
                Integer(1, 10, name='max_depth'),
                Real(0, 1, name='gamma'),
                Integer(1, 20, name='min_child_weight'),
                Categorical(np.linspace(0.5, 1, 6), name='subsample'),
                Categorical(np.linspace(0.3, 1, 8), name='colsample_by_tree'),
                Categorical([0, 0.01, 0.1, 1, 10, 100], name='reg_alpha'),
                Categorical([0, 0.01, 0.1, 1, 10, 100], name='reg_lambda')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [100, 0.1, 3, 0.0, 1, 1.0, 1.0, 0, 1]


class LightGBM(BaseModel):
    """Light Gradient Boosting Machine."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'LGB', 'LightGBM'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'n_estimators': x[0],
                  'learning_rate': round(x[1], 2),
                  'max_depth': x[2],
                  'num_leaves': x[3],
                  'min_child_weight': x[4],
                  'min_child_samples': x[5],
                  'subsample': round(x[6], 1),
                  'colsample_bytree': round(x[7], 1),
                  'reg_alpha': x[8],
                  'reg_lambda': x[9]}
        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
        if self.T.goal.startswith('class'):
            return LGBMClassifier(n_jobs=self.T.n_jobs,
                                  random_state=self.T.random_state,
                                  **params)
        else:
            return LGBMRegressor(n_jobs=self.T.n_jobs,
                                 random_state=self.T.random_state,
                                 **params)

    def custom_fit(self, model, train, validation):
        """Fit the model using early stopping and update evals attr."""
        # Determine early stopping rounds
        if not self._early_stopping or self._early_stopping >= 1:  # None or int
            rounds = self._early_stopping
        elif self._early_stopping < 1:
            rounds = int(model.get_params()['n_estimators'] * self._early_stopping)

        # Fit the model
        model.fit(train[0], train[1],
                  eval_set=[train, validation],
                  early_stopping_rounds=rounds,
                  verbose=False)

        # Create evals attribute with train and validation scores
        metric_name = list(model.evals_result_['training'])[0]  # Get first key
        self.evals = {'metric': metric_name,
                      'train': model.evals_result_['training'][metric_name],
                      'test': model.evals_result_['valid_1'][metric_name]}

        iters = len(self.evals['train'])  # Iterations reached
        tot = int(model.get_params()['n_estimators'])  # Total iterations in params
        self._stopped = (iters, tot) if iters < tot else None

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(20, 500, name='n_estimators'),
                Real(0.01, 1, 'log-uniform', name='learning_rate'),
                Integer(1, 10, name='max_depth'),
                Integer(20, 40, name='num_leaves'),
                Integer(1, 20, name='min_child_weight'),
                Integer(10, 30, name='min_child_samples'),
                Categorical(np.linspace(0.5, 1, 6), name='subsample'),
                Categorical(np.linspace(0.3, 1, 8), name='colsample_by_tree'),
                Categorical([0, 0.01, 0.1, 1, 10, 100], name='reg_alpha'),
                Categorical([0, 0.01, 0.1, 1, 10, 100], name='reg_lambda')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [100, 0.1, 3, 31, 1, 20, 1.0, 1.0, 0, 0]


class CatBoost(BaseModel):
    """Categorical Boosting Machine."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'CatB', 'CatBoost'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'n_estimators': x[0],
                  'learning_rate': round(x[1], 2),
                  'max_depth': x[2],
                  'subsample': round(x[3], 1),
                  'colsample_bylevel': round(x[4], 1),
                  'reg_lambda': x[5]}
        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        from catboost import CatBoostClassifier, CatBoostRegressor
        if self.T.goal.startswith('class'):
            # subsample only works when bootstrap_type=Bernoulli
            return CatBoostClassifier(bootstrap_type='Bernoulli',
                                      train_dir='',
                                      allow_writing_files=False,
                                      random_state=self.T.random_state,
                                      verbose=False,
                                      **params)
        else:
            return CatBoostRegressor(bootstrap_type='Bernoulli',
                                     train_dir='',
                                     allow_writing_files=False,
                                     random_state=self.T.random_state,
                                     verbose=False,
                                     **params)

    def custom_fit(self, model, train, validation):
        """Fit the model using early stopping and update evals attr."""
        # Determine early stopping rounds
        if not self._early_stopping or self._early_stopping >= 1:  # None or int
            rounds = self._early_stopping
        elif self._early_stopping < 1:
            rounds = int(model.get_params()['n_estimators'] * self._early_stopping)

        # Fit the model
        model.fit(train[0], train[1],
                  eval_set=validation,
                  early_stopping_rounds=rounds)

        # Create evals attribute with train and validation scores
        metric_name = list(model.evals_result_['learn'])[0]  # Get first key
        self.evals = {'metric': metric_name,
                      'train': model.evals_result_['learn'][metric_name],
                      'test': model.evals_result_['validation'][metric_name]}

        iters = len(self.evals['train'])  # Iterations reached
        tot = int(model.get_all_params()['iterations'])  # Total iterations in params
        self._stopped = (iters, tot) if iters < tot else None

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        # num_leaves and min_child_samples not available for CPU implementation
        return [Integer(20, 500, name='n_estimators'),
                Real(0.01, 1, 'log-uniform', name='learning_rate'),
                Integer(1, 10, name='max_depth'),
                Categorical(np.linspace(0.5, 1, 6), name='subsample'),
                Categorical(np.linspace(0.3, 1, 8), name='colsample_by_level'),
                Categorical([0, 0.01, 0.1, 1, 10, 100], name='reg_lambda')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [100, 0.1, 3, 1.0, 1.0, 0]


class LinearSVM(BaseModel):
    """Linear Support Vector Machine."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'lSVM', 'Linear SVM'

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'C': round(x[0], 3),
                  'loss': x[1]}

        # l1 regularization can't be combined with hinge
        # l1 regularization can't be combined with squared_hinge when dual=True
        if self.T.goal.startswith('class'):
            params['penalty'] = x[2] if x[1] == 'squared_hinge' else 'l2'
            params['dual'] = True if params['penalty'] == 'l2' else False

        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return LinearSVC(random_state=self.T.random_state, **params)
        else:
            return LinearSVR(random_state=self.T.random_state, **params)

    def get_domain(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            return [Real(1e-3, 100, 'log-uniform', name='C'),
                    Categorical(['hinge', 'squared_hinge'], name='loss'),
                    Categorical(['l1', 'l2'], name='penalty')]
        else:
            return [Real(1e-3, 100, 'log-uniform', name='C'),
                    Categorical(['epsilon_insensitive',
                                 'squared_epsilon_insensitive'], name='loss')]

    def get_init_values(self):
        """Return the default values of the model's hyperparameters."""
        if self.T.goal.startswith('class'):
            return [1, 'squared_hinge', 'l2']
        else:
            return [1, 'squared_epsilon_insensitive']


class KernelSVM(BaseModel):
    """Kernel (non-linear) Support Vector Machine."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'kSVM', 'Kernel SVM'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'C': round(x[0], 3),
                  'kernel': x[1],
                  'gamma': x[3],
                  'shrinking': x[5]}

        if x[1] == 'poly':
            params['degree'] = x[2]
            params['gamma'] = 'scale'  # Crashes in combination with 'auto'

        if x[1] != 'rbf':
            params['coef0'] = round(x[4], 2)

        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return SVC(random_state=self.T.random_state, **params)
        else:
            return SVR(**params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Real(1e-3, 100, 'log-uniform', name='C'),
                Categorical(['poly', 'rbf', 'sigmoid'], name='kernel'),
                Integer(2, 5, name='degree'),
                Categorical(['auto', 'scale'], name='gamma'),
                Real(-1, 1, name='coef0'),
                Categorical([True, False], name='shrinking')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [1, 'rbf', 3, 'auto', 0, True]


class PassiveAggressive(BaseModel):
    """Passive Aggressive model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'PA', 'Passive Aggressive'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'loss': x[0],
                  'C': round(x[1], 3),
                  'average': x[2]}

        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return PAC(random_state=self.T.random_state,
                       n_jobs=self.T.n_jobs,
                       **params)
        else:
            return PAR(random_state=self.T.random_state, **params)

    def get_domain(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            loss = ['hinge', 'squared_hinge']
        else:
            loss = ['epsilon_insensitive', 'squared_epsilon_insensitive']
        return [Categorical(loss, name='loss'),
                Real(1e-3, 100, 'log-uniform', name='C'),
                Categorical([True, False], name='average')]

    def get_init_values(self):
        """Return the default values of the model's hyperparameters."""
        if self.T.goal.startswith('class'):
            loss = 'hinge'
        else:
            loss = 'epsilon_insensitive'
        return [loss, 1, 1]


class StochasticGradientDescent(BaseModel):
    """Stochastic Gradient Descent model."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'SGD', 'Stochastic Gradient Descent'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {'loss': x[0],
                  'penalty': x[1],
                  'alpha': round(x[2], 4),
                  'average': x[3],
                  'epsilon': round(x[4], 4),
                  'learning_rate': x[6],
                  'power_t': round(x[8], 2)}

        if params['penalty'] == 'elasticnet':
            params['l1_ratio'] = round(x[7], 1)

        if params['learning_rate'] != 'optimal':
            params['eta0'] = round(x[5], 4)

        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return SGDClassifier(random_state=self.T.random_state,
                                 n_jobs=self.T.n_jobs,
                                 **params)
        else:
            return SGDRegressor(random_state=self.T.random_state, **params)

    def get_domain(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            loss = ['hinge', 'log', 'modified_huber', 'squared_hinge',
                    'perceptron', 'squared_loss', 'huber',
                    'epsilon_insensitive', 'squared_epsilon_insensitive']
        else:
            loss = ['squared_loss', 'huber',
                    'epsilon_insensitive', 'squared_epsilon_insensitive']

        return [Categorical(loss, name='loss'),
                Categorical(['none', 'l1', 'l2', 'elasticnet'],
                            name='penalty'),
                Real(1e-4, 1, 'log-uniform', name='alpha'),
                Categorical([True, False], name='average'),
                Real(1e-4, 1, 'log-uniform', name='epsilon'),
                Real(1e-4, 1, 'log-uniform', name='eta0'),
                Categorical(['constant', 'invscaling', 'optimal', 'adaptive'],
                            name='learning_rate'),
                Categorical(np.linspace(0.1, 0.9, 9), name='l1_ratio'),
                Categorical(np.linspace(0.05, 1, 20), name='power_t')]

    def get_init_values(self):
        """Return the default values of the model's hyperparameters."""
        loss = 'hinge' if self.T.task != 'regression' else 'squared_loss'
        return [loss, 'l2', 1e-4, False, 0.1, 0.01, 'optimal', 0.1, 0.25]


class MultilayerPerceptron(BaseModel):
    """Multilayer Perceptron with 1 to 3 hidden layers."""

    def __init__(self, *args):
        """Class initializer."""
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'MLP', 'Multilayer Perceptron'

    @staticmethod
    def get_params(x):
        """Return a dictionary of the model´s hyperparameters."""
        # Set the number of neurons per layer
        n1, n2, n3 = x[0], x[1], x[2]
        if n2 == 0:
            layers = (n1,)
        elif n3 == 0:
            layers = (n1, n2)
        else:
            layers = (n1, n2, n3)

        params = {'hidden_layer_sizes': layers,
                  'alpha': round(x[3], 4),
                  'learning_rate_init': round(x[4], 3),
                  'max_iter': x[5],
                  'batch_size': x[6]}
        return params

    def get_model(self, params={}):
        """Return the model object with unpacked parameters."""
        if self.T.goal.startswith('class'):
            return MLPClassifier(random_state=self.T.random_state, **params)
        else:
            return MLPRegressor(random_state=self.T.random_state, **params)

    @staticmethod
    def get_domain():
        """Return a list of the bounds for the hyperparameters."""
        return [Integer(10, 100, name='hidden_layer_1'),
                Integer(0, 100, name='hidden_layer_2'),
                Integer(0, 100, name='hidden_layer_3'),
                Real(1e-4, 0.1, 'log-uniform', name='alpha'),
                Real(1e-3, 0.1, 'log-uniform', name='learning_rate_init'),
                Integer(50, 500, name='max_iter'),
                Categorical([1, 8, 16, 32, 64], name='batch_size')]

    @staticmethod
    def get_init_values():
        """Return the default values of the model's hyperparameters."""
        return [20, 0, 0, 1e-4, 1e-3, 200, 32]


# Global constants ========================================================== >>

# List of all the available models
MODEL_LIST = dict(GP=GaussianProcess,
                  GNB=GaussianNaiveBayes,
                  MNB=MultinomialNaiveBayes,
                  BNB=BernoulliNaiveBayes,
                  OLS=OrdinaryLeastSquares,
                  Ridge=Ridge,
                  Lasso=Lasso,
                  EN=ElasticNet,
                  BR=BayesianRegression,
                  LR=LogisticRegression,
                  LDA=LinearDiscriminantAnalysis,
                  QDA=QuadraticDiscriminantAnalysis,
                  KNN=KNearestNeighbors,
                  Tree=DecisionTree,
                  Bag=Bagging,
                  ET=ExtraTrees,
                  RF=RandomForest,
                  AdaB=AdaBoost,
                  GBM=GradientBoostingMachine,
                  XGB=XGBoost,
                  LGB=LightGBM,
                  CatB=CatBoost,
                  lSVM=LinearSVM,
                  kSVM=KernelSVM,
                  PA=PassiveAggressive,
                  SGD=StochasticGradientDescent,
                  MLP=MultilayerPerceptron)

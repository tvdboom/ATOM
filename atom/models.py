# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing all available models. All classes must have the
             following structure:

        Name
        ----
        Name of the model's class in camel case format.

        Attributes
        ----------
        name: str
            Short acronym of the model's longname for calling.

        longname: str
            Complete name of the model.

        type: str
            Model type (to use in plots). Choose from: linear, tree, kernel.

        evals: dict
            Evaluation metric and scores. Only for models that allow
            in-train evaluation.

        params: dict
            All the estimator's parameters for the BO. The values should be a
            list with the parameter's default value and the number of decimals.

        Methods
        -------
        __init__(self, *args):
            Class initializer (contains super() to parent class).

        get_init_values(self):
            Return the initial values for the estimator. Don't implement if
            parent method in BaseModel (default behaviour) is sufficient.

        get_params(self, x):
            Return the parameters with rounded decimals and (optional) custom
            changes to the params. Don't implement if parent method in BaseModel
            (default behaviour) is sufficient.

        get_estimator(self, params={}):
            Return the model's estimator with unpacked parameters.

        custom_fit(model, train, validation):
            If the direct fit method of the model is not enough and you desire to
            customize it a bit, make a custom_fit method. It will run instead.

        get_dimensions(self):
            Return a list of the bounds for the hyperparameters.


To add a new model:
    1. Add the model's class to models.py
    2. Add the model to the list MODEL_LIST in models.py
    3. Add the name to all the relevant variables in utils.py


List of available models:
    - 'GNB' for Gaussian Naive Bayes (no hyperparameter tuning)
    - 'MNB' for Multinomial Naive Bayes
    - 'BNB' for Bernoulli Naive Bayes
    - 'CatNB' for Categorical Naive Bayes
    - 'CNB' for Complement Naive Bayes
    - 'GP' for Gaussian Process (no hyperparameter tuning)
    - 'OLS' for Ordinary Least Squares (no hyperparameter tuning)
    - 'Ridge' for Ridge Linear
    - 'Lasso' for Lasso Linear Regression
    - 'EN' for ElasticNet Linear Regression
    - 'BR' for Bayesian Ridge
    - 'ARD' for Automated Relevance Determination
    - 'LR' for Logistic Regression
    - 'LDA' for Linear Discriminant Analysis
    - 'QDA' for Quadratic Discriminant Analysis
    - 'KNN' for K-Nearest Neighbors
    - 'RNN' for Radius Nearest Neighbors
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
    - 'MLP' for Multi-layer Perceptron

"""

# Standard packages
import random
import numpy as np
from scipy.spatial.distance import minkowski

# Others
from skopt.space.space import Real, Integer, Categorical

# Sklearn models
from sklearn.gaussian_process import (
    GaussianProcessClassifier, GaussianProcessRegressor
)
from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB
)
from sklearn.linear_model import (
    LinearRegression, RidgeClassifier, Ridge as RidgeRegressor,
    Lasso as LassoRegressor, ElasticNet as ElasticNetRegressor,
    BayesianRidge as BayesianRidgeRegressor, ARDRegression,
    LogisticRegression as LR
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
)
from sklearn.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor,
    RadiusNeighborsClassifier, RadiusNeighborsRegressor
)
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
    PassiveAggressiveClassifier, PassiveAggressiveRegressor,
    SGDClassifier, SGDRegressor
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Own modules
from .basemodel import BaseModel


# Classes =================================================================== >>

class GaussianProcess(BaseModel):
    """Gaussian process."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'GP', 'Gaussian Process'
        self.type = 'kernel'

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        kwargs = dict(random_state=self.T.random_state, **params)
        if self.T.goal.startswith('class'):
            estimator = GaussianProcessClassifier(n_jobs=self.T.n_jobs, **kwargs)
        else:
            estimator = GaussianProcessRegressor(**kwargs)

        return estimator


class GaussianNaiveBayes(BaseModel):
    """Gaussian Naive Bayes."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'GNB', 'Gaussian Naive Bayes'
        self.type = 'kernel'

    @staticmethod
    def get_estimator(params={}):
        """Call the estimator object."""
        return GaussianNB(**params)


class MultinomialNaiveBayes(BaseModel):
    """Multinomial Naive Bayes."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'MNB', 'Multinomial Naive Bayes'
        self.type = 'kernel'
        self.params = dict(alpha=[1.0, 3], fit_prior=[True, 0])

    @staticmethod
    def get_estimator(params={}):
        """Call the estimator object."""
        return MultinomialNB(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, 'log-uniform', name='alpha'),
            Categorical([True, False], name='fit_prior')
        ]
        return [d for d in dimensions if d.name in self.params]


class BernoulliNaiveBayes(BaseModel):
    """Bernoulli Naive Bayes."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'BNB', 'Bernoulli Naive Bayes'
        self.type = 'kernel'
        self.params = dict(alpha=[1.0, 3], fit_prior=[True, 0])

    @staticmethod
    def get_estimator(params={}):
        """Call the estimator object."""
        return BernoulliNB(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, 'log-uniform', name='alpha'),
            Categorical([True, False], name='fit_prior')
        ]
        return [d for d in dimensions if d.name in self.params]


class CategoricalNaiveBayes(BaseModel):
    """Categorical Naive Bayes."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'CatNB', 'Categorical Naive Bayes'
        self.type = 'kernel'
        self.params = dict(alpha=[1.0, 3], fit_prior=[True, 0])

    @staticmethod
    def get_estimator(params={}):
        """Call the estimator object."""
        return CategoricalNB(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, 'log-uniform', name='alpha'),
            Categorical([True, False], name='fit_prior')
        ]
        return [d for d in dimensions if d.name in self.params]


class ComplementNaiveBayes(BaseModel):
    """Complement Naive Bayes."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'CNB', 'Complement Naive Bayes'
        self.type = 'kernel'
        self.params = dict(alpha=[1.0, 3], fit_prior=[True, 0], norm=[False, 0])

    @staticmethod
    def get_estimator(params={}):
        """Call the estimator object."""
        return ComplementNB(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, 'log-uniform', name='alpha'),
            Categorical([True, False], name='fit_prior'),
            Categorical([True, False], name='norm')
        ]
        return [d for d in dimensions if d.name in self.params]


class OrdinaryLeastSquares(BaseModel):
    """Linear Regression (without regularization)."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'OLS', 'Ordinary Least Squares'
        self.type = 'linear'

    def get_estimator(self, params={}):
        """Call the estimator object."""
        return LinearRegression(n_jobs=self.T.n_jobs, **params)


class Ridge(BaseModel):
    """Linear Regression/Classification with ridge regularization."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name = 'Ridge'
        if self.T.goal.startswith('class'):
            self.longname = 'Ridge Classification'
        else:
            self.longname = 'Ridge Regression'
        self.type = 'linear'
        self.params = dict(alpha=[1.0, 3], solver=['auto', 0])

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal.startswith('class'):
            estimator = RidgeClassifier(random_state=self.T.random_state, **params)
        else:
            estimator = RidgeRegressor(random_state=self.T.random_state, **params)

        return estimator

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        solvers = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        dimensions = [
            Real(1e-3, 10, 'log-uniform', name='alpha'),
            Categorical(solvers, name='solver')
        ]
        return [d for d in dimensions if d.name in self.params]


class Lasso(BaseModel):
    """Linear Regression with lasso regularization."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'Lasso', 'Lasso Regression'
        self.type = 'linear'
        self.params = dict(alpha=[1.0, 3], selection=['cyclic', 0])

    def get_estimator(self, params={}):
        """Call the estimator object."""
        return LassoRegressor(random_state=self.T.random_state, **params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, 'log-uniform', name='alpha'),
            Categorical(['cyclic', 'random'], name='selection')
        ]
        return [d for d in dimensions if d.name in self.params]


class ElasticNet(BaseModel):
    """Linear Regression with elasticnet regularization."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'EN', 'ElasticNet Regression'
        self.type = 'linear'
        self.params = dict(
            alpha=[1.0, 3],
            l1_ratio=[0.5, 1],
            selection=['cyclic', 0]
        )

    def get_estimator(self, params={}):
        """Call the estimator object."""
        return ElasticNetRegressor(random_state=self.T.random_state, **params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, 'log-uniform', name='alpha'),
            Categorical(np.linspace(0.1, 0.9, 9), name='l1_ratio'),
            Categorical(['cyclic', 'random'], name='selection')
        ]
        return [d for d in dimensions if d.name in self.params]


class BayesianRidge(BaseModel):
    """Bayesian ridge regression."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'BR', 'Bayesian Ridge'
        self.type = 'linear'
        self.params = dict(
            n_iter=[300, 0],
            alpha_1=[1e-6, 8],
            alpha_2=[1e-6, 8],
            lambda_1=[1e-6, 8],
            lambda_2=[1e-6, 8]
        )

    @staticmethod
    def get_estimator(params={}):
        """Call the estimator object."""
        return BayesianRidgeRegressor(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(100, 1000, name='n_iter'),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name='alpha_1'),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name='alpha_2'),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name='lambda_1'),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name='lambda_2')
        ]
        return [d for d in dimensions if d.name in self.params]


class AutomaticRelevanceDetermination(BaseModel):
    """Automatic Relevance Determination."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'ARD', 'Automatic Relevance Determination'
        self.type = 'linear'
        self.params = dict(
            n_iter=[300, 0],
            alpha_1=[1e-6, 8],
            alpha_2=[1e-6, 8],
            lambda_1=[1e-6, 8],
            lambda_2=[1e-6, 8]
        )

    @staticmethod
    def get_estimator(params={}):
        """Call the estimator object."""
        return ARDRegression(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(100, 1000, name='n_iter'),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name='alpha_1'),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name='alpha_2'),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name='lambda_1'),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name='lambda_2')
        ]
        return [d for d in dimensions if d.name in self.params]


class LogisticRegression(BaseModel):
    """Logistic Regression."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'LR', 'Logistic Regression'
        self.type = 'linear'
        self.params = dict(
            penalty=['l2', 0],
            C=[1.0, 3],
            solver=['lbfgs', 0],
            max_iter=[100, 0],
            l1_ratio=[0.5, 1]
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        # Limitations on penalty + solver combinations
        penalty, solver = params.get('penalty'), params.get('solver')
        cond_1 = (penalty == 'none' and solver == 'liblinear')
        cond_2 = (penalty == 'l1' and solver not in ['liblinear', 'saga'])
        cond_3 = (penalty == 'elasticnet' and solver != 'saga')

        if cond_1 or cond_2 or cond_3:
            params['penalty'] = 'l2'  # Change to default value

        if params.get('penalty') != 'elasticnet':
            params.pop('l1_ratio')
        if params.get('penalty') == 'none':
            params.pop('C')

        return params

    def get_estimator(self, params={}):
        """Call the estimator object."""
        return LR(random_state=self.T.random_state, n_jobs=self.T.n_jobs, **params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        solvers = ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']
        dimensions = [
            Categorical(['none', 'l1', 'l2', 'elasticnet'], name='penalty'),
            Real(1e-3, 100, 'log-uniform', name='C'),
            Categorical(solvers, name='solver'),
            Integer(100, 1000, name='max_iter'),
            Categorical(np.linspace(0.1, 0.9, 9), name='l1_ratio')
        ]
        return [d for d in dimensions if d.name in self.params]


class LinearDiscriminantAnalysis(BaseModel):
    """Linear Discriminant Analysis."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'LDA', 'Linear Discriminant Analysis'
        self.type = 'kernel'
        self.params = dict(solver=['svd', 0], shrinkage=[0, 1])

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if params.get('solver') == 'svd':
            params.pop('shrinkage')

        return params

    @staticmethod
    def get_estimator(params={}):
        """Call the estimator object."""
        return LDA(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Categorical(['svd', 'lsqr', 'eigen'], name='solver'),
            Categorical(np.linspace(0.0, 1.0, 11), name='shrinkage')
        ]
        return [d for d in dimensions if d.name in self.params]


class QuadraticDiscriminantAnalysis(BaseModel):
    """Quadratic Discriminant Analysis."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'QDA', 'Quadratic Discriminant Analysis'
        self.type = 'kernel'
        self.params = dict(reg_param=[0, 1])

    @staticmethod
    def get_estimator(params={}):
        """Call the estimator object."""
        return QDA(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [Categorical(np.linspace(0.0, 1.0, 11), name='reg_param')]
        return [d for d in dimensions if d.name in self.params]


class KNearestNeighbors(BaseModel):
    """K-Nearest Neighbors."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'KNN', 'K-Nearest Neighbors'
        self.type = 'kernel'
        self.params = dict(
            n_neighbors=[5, 0],
            weights=['uniform', 0],
            algorithm=['auto', 0],
            leaf_size=[30, 0],
            p=[2, 0]
        )

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal.startswith('class'):
            estimator = KNeighborsClassifier(n_jobs=self.T.n_jobs, **params)
        else:
            estimator = KNeighborsRegressor(n_jobs=self.T.n_jobs, **params)

        return estimator

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(1, 100, name='n_neighbors'),
            Categorical(['uniform', 'distance'], name='weights'),
            Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
            Integer(20, 40, name='leaf_size'),
            Integer(1, 2, name='p')
        ]
        return [d for d in dimensions if d.name in self.params]


class RadiusNearestNeighbors(BaseModel):
    """Radius Nearest Neighbors."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'RNN', 'Radius Nearest Neighbors'
        self.type = 'kernel'
        self._distances = []
        self.params = dict(
            radius=[None, 3],  # We need the scaler to calculate the distances
            weights=['uniform', 0],
            algorithm=['auto', 0],
            leaf_size=[30, 0],
            p=[2, 0]
        )

    def get_init_values(self):
        """Custom method to return a valid radius."""
        return [np.mean(self.distances)] + super().get_init_values()[1:]

    @property
    def distances(self):
        """Return distances between a random subsample of rows in the dataset."""
        if not self._distances:
            len_ = len(self.X_train)
            sample = np.random.randint(len_, size=int(0.1 * len_))
            for i in range(0, len(sample), 2):
                self._distances.append(minkowski(
                    self.X_train.loc[i], self.X_train.loc[i+1]))

        return self._distances

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        if params.get('radius'):
            radius = params.pop('radius')
        else:
            radius = np.mean(self.distances)
        kwargs = dict(radius=radius, n_jobs=self.T.n_jobs, **params)
        if self.T.goal.startswith('class'):
            estimator = RadiusNeighborsClassifier(
                outlier_label='most_frequent',
                **kwargs
            )
        else:
            estimator = RadiusNeighborsRegressor(**kwargs)

        return estimator

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']

        dimensions = [
            Real(min(self.distances), max(self.distances), name='radius'),
            Categorical(['uniform', 'distance'], name='weights'),
            Categorical(algorithms, name='algorithm'),
            Integer(20, 40, name='leaf_size'),
            Integer(1, 2, name='p')
        ]
        return [d for d in dimensions if d.name in self.params]


class DecisionTree(BaseModel):
    """Single Decision Tree."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'Tree', 'Decision Tree'
        self.type = 'tree'
        self.params = dict(
            criterion=['gini' if self.T.goal.startswith('class') else 'mse', 0],
            splitter=['best', 0],
            max_depth=[None, 0],
            min_samples_split=[2, 0],
            min_samples_leaf=[1, 0],
            max_features=[None, 0],
            ccp_alpha=[0, 3]
        )

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        kwargs = dict(random_state=self.T.random_state, **params)
        if self.T.goal.startswith('class'):
            estimator = DecisionTreeClassifier
        else:
            estimator = DecisionTreeRegressor

        return estimator(**kwargs)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            criterion = ['gini', 'entropy']
        else:
            criterion = ['mse', 'mae', 'friedman_mse']

        dimensions = [
            Categorical(criterion, name='criterion'),
            Categorical(['best', 'random'], name='splitter'),
            Categorical([None, *list(range(1, 10))], name='max_depth'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 20, name='min_samples_leaf'),
            Categorical([None, *np.linspace(0.5, 0.9, 5)], name='max_features'),
            Real(0, 0.035, name='ccp_alpha')
        ]
        return [d for d in dimensions if d.name in self.params]


class Bagging(BaseModel):
    """Bagging model (with decision tree as base estimator)."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name = 'Bag'
        if self.T.goal.startswith('class'):
            self.longname = 'Bagging Classifier'
        else:
            self.longname = 'Bagging Regressor'
        self.type = 'tree'
        self.params = dict(
            n_estimators=[10, 0],
            max_samples=[1.0, 1],
            max_features=[1.0, 1],
            bootstrap=[True, 0],
            bootstrap_features=[False, 0]
        )

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        kwargs = dict(random_state=self.T.random_state, n_jobs=self.T.n_jobs, **params)
        if self.T.goal.startswith('class'):
            estimator = BaggingClassifier
        else:
            estimator = BaggingRegressor

        return estimator(**kwargs)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(10, 500, name='n_estimators'),
            Categorical(np.linspace(0.5, 1.0, 6), name='max_samples'),
            Categorical(np.linspace(0.5, 1.0, 6), name='max_features'),
            Categorical([True, False], name='bootstrap'),
            Categorical([True, False], name='bootstrap_features')
        ]
        return [d for d in dimensions if d.name in self.params]


class ExtraTrees(BaseModel):
    """Extremely Randomized Trees."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'ET', 'Extra-Trees'
        self.type = 'tree'
        self.params = dict(
            n_estimators=[100, 0],
            criterion=['gini' if self.T.goal.startswith('class') else 'mse', 0],
            max_depth=[None, 0],
            min_samples_split=[2, 0],
            min_samples_leaf=[1, 0],
            max_features=[None, 0],
            bootstrap=[False, 0],
            ccp_alpha=[0, 3],
            max_samples=[0.9, 1]
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if not params.get('bootstrap'):
            params.pop('max_samples')

        return params

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        kwargs = dict(random_state=self.T.random_state, n_jobs=self.T.n_jobs, **params)
        if self.T.goal.startswith('class'):
            estimator = ExtraTreesClassifier
        else:
            estimator = ExtraTreesRegressor

        return estimator(**kwargs)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            criterion = ['gini', 'entropy']
        else:
            criterion = ['mse', 'mae']

        dimensions = [
            Integer(10, 500, name='n_estimators'),
            Categorical(criterion, name='criterion'),
            Categorical([None, *list(range(1, 10))], name='max_depth'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 20, name='min_samples_leaf'),
            Categorical([None, *np.linspace(0.5, 0.9, 5)], name='max_features'),
            Categorical([True, False], name='bootstrap'),
            Real(0, 0.035, name='ccp_alpha'),
            Categorical(np.linspace(0.5, 0.9, 5), name='max_samples')
        ]
        return [d for d in dimensions if d.name in self.params]


class RandomForest(BaseModel):
    """Random Forest."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'RF', 'Random Forest'
        self.type = 'tree'
        self.params = dict(
            n_estimators=[100, 0],
            criterion=['gini' if self.T.goal.startswith('class') else 'mse', 0],
            max_depth=[None, 0],
            min_samples_split=[2, 0],
            min_samples_leaf=[1, 0],
            max_features=[None, 0],
            bootstrap=[False, 0],
            ccp_alpha=[0, 3],
            max_samples=[0.9, 1]
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if not params.get('bootstrap'):
            params.pop('max_samples')

        return params

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        kwargs = dict(random_state=self.T.random_state, n_jobs=self.T.n_jobs, **params)
        if self.T.goal.startswith('class'):
            estimator = RandomForestClassifier
        else:
            estimator = RandomForestRegressor

        return estimator(**kwargs)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            criterion = ['gini', 'entropy']
        else:
            criterion = ['mse', 'mae']

        dimensions = [
            Integer(10, 500, name='n_estimators'),
            Categorical(criterion, name='criterion'),
            Categorical([None, *list(range(1, 10))], name='max_depth'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 20, name='min_samples_leaf'),
            Categorical([None, *np.linspace(0.5, 0.9, 5)], name='max_features'),
            Categorical([True, False], name='bootstrap'),
            Real(0, 0.035, name='ccp_alpha'),
            Categorical(np.linspace(0.5, 0.9, 5), name='max_samples')
        ]
        return [d for d in dimensions if d.name in self.params]


class AdaBoost(BaseModel):
    """Adaptive Boosting (with decision tree as base estimator)."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'AdaB', 'AdaBoost'
        self.type = 'tree'
        self.params = dict(
            n_estimators=[50, 0],
            learning_rate=[1.0, 2],
        )

        if self.T.goal.startswith('class'):
            self.params['algorithm'] = ['SAMME.R', 0]
        else:
            self.params['loss'] = ['linear', 0]

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        kwargs = dict(random_state=self.T.random_state, **params)
        if self.T.goal.startswith('class'):
            estimator = AdaBoostClassifier
        else:
            estimator = AdaBoostRegressor

        return estimator(**kwargs)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(50, 500, name='n_estimators'),
            Real(0.01, 1.0, 'log-uniform', name='learning_rate'),
            Categorical(['SAMME.R', 'SAMME'], name='algorithm'),
            Categorical(['linear', 'square', 'exponential'], name='loss')
        ]
        return [d for d in dimensions if d.name in self.params]


class GradientBoostingMachine(BaseModel):
    """Gradient Boosting Machine."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=False)
        self.name, self.longname = 'GBM', 'Gradient Boosting Machine'
        self.type = 'tree'
        self.params = dict(
            learning_rate=[0.1, 2],
            n_estimators=[100, 0],
            subsample=[1.0, 1],
            criterion=['friedman_mse', 0],
            min_samples_split=[2, 0],
            min_samples_leaf=[1, 0],
            max_depth=[3, 0],
            max_features=[None, 0],
            ccp_alpha=[0, 3],
        )

        if self.T.task.startswith('bin'):
            self.params['loss'] = ['deviance', 0]
        elif self.T.task.startswith('reg'):
            self.params['loss'] = ['ls', 0]
            self.params['alpha'] = [0.9, 1]

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if self.T.task.startswith('reg'):
            if params.get('loss') not in ['huber', 'quantile']:
                params.pop('alpha')

        return params

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        kwargs = dict(random_state=self.T.random_state, **params)
        if self.T.goal.startswith('class'):
            estimator = GradientBoostingClassifier
        else:
            estimator = GradientBoostingRegressor

        return estimator(**kwargs)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            loss = ['deviance', 'exponential']  # Will never be used when multiclass
        else:
            loss = ['ls', 'lad', 'huber', 'quantile']

        dimensions = [
            Real(0.01, 1.0, 'log-uniform', name='learning_rate'),
            Integer(10, 500, name='n_estimators'),
            Categorical(np.linspace(0.5, 1.0, 6), name='subsample'),
            Categorical(['friedman_mse', 'mae', 'mse'], name='criterion'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 20, name='min_samples_leaf'),
            Integer(1, 10, name='max_depth'),
            Categorical([None, *np.linspace(0.5, 0.9, 5)], name='max_features'),
            Real(0, 0.035, name='ccp_alpha'),
            Categorical(loss, name='loss'),
            Categorical(np.linspace(0.5, 0.9, 5), name='alpha')
        ]
        return [d for d in dimensions if d.name in self.params]


class XGBoost(BaseModel):
    """Extreme Gradient Boosting."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'XGB', 'XGBoost'
        self.type = 'tree'
        self.evals = {}
        self.params = dict(
            n_estimators=[100, 0],
            learning_rate=[0.1, 2],
            max_depth=[6, 0],
            gamma=[0.0, 2],
            min_child_weight=[1, 0],
            subsample=[1.0, 1],
            colsample_bytree=[1.0, 1],
            reg_alpha=[0, 0],
            reg_lambda=[1, 0]
        )

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        from xgboost import XGBClassifier, XGBRegressor
        # XGBoost can't handle random_state to be None
        if self.T.random_state is None:
            random_state = random.randint(0, np.iinfo(np.int16).max)
        else:
            random_state = self.T.random_state

        kwargs = dict(
            n_jobs=self.T.n_jobs,
            random_state=random_state,
            verbosity=0,
            **params
        )
        if self.T.goal.startswith('class'):
            estimator = XGBClassifier
        else:
            estimator = XGBRegressor

        return estimator(**kwargs)

    def custom_fit(self, model, train, validation):
        """Fit the model using early stopping and update evals attr."""
        # Determine early stopping rounds
        if not self._early_stopping or self._early_stopping >= 1:  # None or int
            rounds = self._early_stopping
        elif self._early_stopping < 1:
            rounds = int(model.get_params()['n_estimators'] * self._early_stopping)

        # Fit the model
        model.fit(
            X=train[0],
            y=train[1],
            eval_set=[train, validation],
            early_stopping_rounds=rounds,
            verbose=False
        )

        # Create evals attribute with train and validation scores
        # Invert sign since XGBoost minimizes the metric
        metric_name = list(model.evals_result()['validation_0'])[0]
        self.evals = {
            'metric': metric_name,
            'train': model.evals_result()['validation_0'][metric_name],
            'test': model.evals_result()['validation_1'][metric_name]
        }

        iters = len(self.evals['train'])  # Iterations reached
        tot = int(model.get_params()['n_estimators'])  # Total iterations in params
        self._stopped = (iters, tot) if iters < tot else None

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(20, 500, name='n_estimators'),
            Real(0.01, 1.0, 'log-uniform', name='learning_rate'),
            Integer(1, 10, name='max_depth'),
            Real(0, 1.0, name='gamma'),
            Integer(1, 20, name='min_child_weight'),
            Categorical(np.linspace(0.5, 1.0, 6), name='subsample'),
            Categorical(np.linspace(0.3, 1.0, 8), name='colsample_bytree'),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name='reg_alpha'),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name='reg_lambda')
        ]
        return [d for d in dimensions if d.name in self.params]


class LightGBM(BaseModel):
    """Light Gradient Boosting Machine."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'LGB', 'LightGBM'
        self.type = 'tree'
        self.evals = {}
        self.params = dict(
            n_estimators=[100, 0],
            learning_rate=[0.1, 2],
            max_depth=[-1, 0],
            num_leaves=[31, 0],
            min_child_weight=[1, 0],
            min_child_samples=[20, 0],
            subsample=[1.0, 1],
            colsample_bytree=[1.0, 1],
            reg_alpha=[0, 0],
            reg_lambda=[0, 0]
        )

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
        kwargs = dict(
            n_jobs=self.T.n_jobs,
            random_state=self.T.random_state,
            ** params
        )
        if self.T.goal.startswith('class'):
            estimator = LGBMClassifier
        else:
            estimator = LGBMRegressor

        return estimator(**kwargs)

    def custom_fit(self, model, train, validation):
        """Fit the model using early stopping and update evals attr."""
        # Determine early stopping rounds
        if not self._early_stopping or self._early_stopping >= 1:  # None or int
            rounds = self._early_stopping
        elif self._early_stopping < 1:
            rounds = int(model.get_params()['n_estimators'] * self._early_stopping)

        model.fit(
            X=train[0],
            y=train[1],
            eval_set=[train, validation],
            early_stopping_rounds=rounds,
            verbose=False
        )

        # Create evals attribute with train and validation scores
        metric_name = list(model.evals_result_['training'])[0]  # Get first key
        self.evals = {
            'metric': metric_name,
            'train': model.evals_result_['training'][metric_name],
            'test': model.evals_result_['valid_1'][metric_name]
        }

        iters = len(self.evals['train'])  # Iterations reached
        tot = int(model.get_params()['n_estimators'])  # Total iterations in params
        self._stopped = (iters, tot) if iters < tot else None

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(20, 500, name='n_estimators'),
            Real(0.01, 1.0, 'log-uniform', name='learning_rate'),
            Categorical([-1, *list(range(1, 10))], name='max_depth'),
            Integer(20, 40, name='num_leaves'),
            Integer(1, 20, name='min_child_weight'),
            Integer(10, 30, name='min_child_samples'),
            Categorical(np.linspace(0.5, 1.0, 6), name='subsample'),
            Categorical(np.linspace(0.3, 1.0, 8), name='colsample_bytree'),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name='reg_alpha'),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name='reg_lambda')
        ]
        return [d for d in dimensions if d.name in self.params]


class CatBoost(BaseModel):
    """Categorical Boosting Machine."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'CatB', 'CatBoost'
        self.type = 'tree'
        self.evals = {}
        self.params = dict(
            n_estimators=[100, 0],
            learning_rate=[0.1, 2],
            max_depth=[None, 0],
            subsample=[1.0, 1],
            colsample_bylevel=[1.0, 1],
            reg_lambda=[0, 0]
        )

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        from catboost import CatBoostClassifier, CatBoostRegressor
        # The subsample parameter only works when bootstrap_type=Bernoulli
        kwargs = dict(
            bootstrap_type='Bernoulli',
            train_dir='',
            allow_writing_files=False,
            thread_count=self.T.n_jobs,
            random_state=self.T.random_state,
            verbose=False,
            **params
        )
        if self.T.goal.startswith('class'):
            estimator = CatBoostClassifier
        else:
            estimator = CatBoostRegressor

        return estimator(**kwargs)

    def custom_fit(self, model, train, validation):
        """Fit the model using early stopping and update evals attr."""
        # Determine early stopping rounds
        if not self._early_stopping or self._early_stopping >= 1:  # None or int
            rounds = self._early_stopping
        elif self._early_stopping < 1:
            rounds = int(model.get_params()['n_estimators'] * self._early_stopping)

        model.fit(
            X=train[0],
            y=train[1],
            eval_set=validation,
            early_stopping_rounds=rounds
        )

        # Create evals attribute with train and validation scores
        metric_name = list(model.evals_result_['learn'])[0]  # Get first key
        self.evals = {
            'metric': metric_name,
            'train': model.evals_result_['learn'][metric_name],
            'test': model.evals_result_['validation'][metric_name]
        }

        iters = len(self.evals['train'])  # Iterations reached
        tot = int(model.get_all_params()['iterations'])  # Total iterations in params
        self._stopped = (iters, tot) if iters < tot else None

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        # num_leaves and min_child_samples not available for CPU implementation
        dimensions = [
            Integer(20, 500, name='n_estimators'),
            Real(0.01, 1.0, 'log-uniform', name='learning_rate'),
            Categorical([None, *list(range(1, 10))], name='max_depth'),
            Categorical(np.linspace(0.5, 1.0, 6), name='subsample'),
            Categorical(np.linspace(0.3, 1.0, 8), name='colsample_bylevel'),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name='reg_lambda')
        ]
        return [d for d in dimensions if d.name in self.params]


class LinearSVM(BaseModel):
    """Linear Support Vector Machine."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'lSVM', 'Linear-SVM'
        self.type = 'kernel'
        self.params = dict(loss=['epsilon_insensitive', 0], C=[1.0, 3])

        # Different params for classification tasks
        if self.T.goal.startswith('class'):
            self.params['loss'] = ['squared_hinge', 0]
            self.params['penalty'] = ['l2', 0]

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        # l1 regularization can't be combined with hinge
        # l1 regularization can't be combined with squared_hinge when dual=True
        if self.T.goal.startswith('class'):
            if params.get('loss') == 'hinge':
                params['penalty'] = 'l2'
            if params.get('penalty') == 'l1' and params.get('loss') == 'squared_hinge':
                params['dual'] = False

        return params

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal.startswith('class'):
            estimator = LinearSVC(random_state=self.T.random_state, **params)
        else:
            estimator = LinearSVR(random_state=self.T.random_state, **params)

        return estimator

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            dimensions = [
                Categorical(['hinge', 'squared_hinge'], name='loss'),
                Real(1e-3, 100, 'log-uniform', name='C'),
                Categorical(['l1', 'l2'], name='penalty')
            ]
        else:
            dimensions = [
                Categorical(['epsilon_insensitive',
                             'squared_epsilon_insensitive'], name='loss'),
                Real(1e-3, 100, 'log-uniform', name='C')
            ]
        return [d for d in dimensions if d.name in self.params]


class KernelSVM(BaseModel):
    """Kernel (non-linear) Support Vector Machine."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'kSVM', 'Kernel-SVM'
        self.type = 'kernel'
        self.params = dict(
            C=[1.0, 3],
            kernel=['rbf', 0],
            degree=[3, 0],
            gamma=['scale', 0],
            coef0=[0, 2],
            shrinking=[True, 0]
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if params.get('kernel') == 'poly':
            params['gamma'] = 'scale'  # Crashes in combination with 'auto'
        else:
            params.pop('degree')

        if params.get('kernel') != 'rbf':
            params.pop('coef0')

        return params

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal.startswith('class'):
            estimator = SVC(random_state=self.T.random_state, **params)
        else:
            estimator = SVR(**params)

        return estimator

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 100, 'log-uniform', name='C'),
            Categorical(['poly', 'rbf', 'sigmoid'], name='kernel'),
            Integer(2, 5, name='degree'),
            Categorical(['scale', 'auto'], name='gamma'),
            Real(-1.0, 1.0, name='coef0'),
            Categorical([True, False], name='shrinking')
        ]
        return [d for d in dimensions if d.name in self.params]


class PassiveAggressive(BaseModel):
    """Passive Aggressive."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'PA', 'Passive Aggressive'
        self.type = 'linear'

        if self.T.goal.startswith('class'):
            loss = 'hinge'
        else:
            loss = 'epsilon_insensitive'
        self.params = dict(
            C=[1.0, 3],
            loss=[loss, 0],
            average=[False, 0]
        )

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal.startswith('class'):
            estimator = PassiveAggressiveClassifier(
                random_state=self.T.random_state,
                n_jobs=self.T.n_jobs,
                **params
            )
        else:
            estimator = PassiveAggressiveRegressor(
                random_state=self.T.random_state,
                **params
            )

        return estimator

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith('class'):
            loss = ['hinge', 'squared_hinge']
        else:
            loss = ['epsilon_insensitive', 'squared_epsilon_insensitive']

        dimensions = [
            Real(1e-3, 100, 'log-uniform', name='C'),
            Categorical(loss, name='loss'),
            Categorical([True, False], name='average')
        ]
        return [d for d in dimensions if d.name in self.params]


class StochasticGradientDescent(BaseModel):
    """Stochastic Gradient Descent."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'SGD', 'Stochastic Gradient Descent'
        self.type = 'linear'
        self.params = dict(
            loss=['squared_loss' if self.T.task.startswith('reg') else 'hinge', 0],
            penalty=['l2', 0],
            alpha=[1e-4, 4],
            l1_ratio=[0.15, 2],
            epsilon=[0.1, 4],
            learning_rate=['optimal', 0],
            eta0=[0.01, 4],
            power_t=[0.5, 1],
            average=[False, 0]
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if params.get('penalty') != 'elasticnet':
            params.pop('l1_ratio')

        if params.get('learning_rate') == 'optimal':
            params.pop('eta0')

        return params

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        kwargs = dict(random_state=self.T.random_state, **params)
        if self.T.goal.startswith('class'):
            estimator = SGDClassifier(n_jobs=self.T.n_jobs, **kwargs)
        else:
            estimator = SGDRegressor(**kwargs)

        return estimator

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        loss = [
            'hinge',
            'log',
            'modified_huber',
            'squared_hinge',
            'perceptron',
            'squared_loss',
            'huber',
            'epsilon_insensitive',
            'squared_epsilon_insensitive'
        ]
        loss = loss[:-4] if self.T.goal.startswith('class') else loss[-4:]
        learning_rate = ['constant', 'invscaling', 'optimal', 'adaptive']

        dimensions = [
            Categorical(loss, name='loss'),
            Categorical(['none', 'l1', 'l2', 'elasticnet'], name='penalty'),
            Real(1e-4, 1.0, 'log-uniform', name='alpha'),
            Categorical(np.linspace(0.05, 0.95, 19), name='l1_ratio'),
            Real(1e-4, 1.0, 'log-uniform', name='epsilon'),
            Categorical(learning_rate, name='learning_rate'),
            Real(1e-4, 1.0, 'log-uniform', name='eta0'),
            Categorical(np.linspace(0.1, 0.9, 9), name='power_t'),
            Categorical([True, False], name='average')
        ]
        return [d for d in dimensions if d.name in self.params]


class MultilayerPerceptron(BaseModel):
    """Multi-layer Perceptron."""

    def __init__(self, *args):
        super().__init__(T=args[0], need_scaling=True)
        self.name, self.longname = 'MLP', 'Multi-layer Perceptron'
        self.type = 'kernel'
        self.params = dict(
            hidden_layer_sizes=[(100, 0, 0), 0],
            activation=['relu', 0],
            solver=['adam', 0],
            alpha=[1e-4, 4],
            batch_size=[200, 0],
            learning_rate=['constant', 0],
            learning_rate_init=[0.001, 3],
            power_t=[0.5, 1],
            max_iter=[200, 0]
        )

    def get_init_values(self):
        """Custom method to return the correct hidden_layer_sizes."""
        init_values = []
        for key, value in self.params.items():
            if key == 'hidden_layer_sizes':
                init_values.extend(value[0])
            else:
                init_values.append(value[0])

        return init_values

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {}
        for i, key in enumerate(self.params):
            # Add extra counter for the hidden layers
            j = 2 if 'hidden_layer_sizes' in self.params else 0

            if key == 'hidden_layer_sizes':
                # Set the number of neurons per layer
                n1, n2, n3 = x[i], x[i+1], x[i+2]
                if n2 == 0:
                    layers = (n1,)
                elif n3 == 0:
                    layers = (n1, n2)
                else:
                    layers = (n1, n2, n3)

                params['hidden_layer_sizes'] = layers

            elif self.params[key][1]:  # If it has decimals...
                params[key] = round(x[i+j], self.params[key][1])
            else:
                params[key] = x[i+j]

        if params.get('solver') != 'sgd':
            params.pop('learning_rate')
            params.pop('power_t')
        else:
            params.pop('learning_rate_init')

        return params

    def get_estimator(self, params={}):
        """Return the model's estimator with unpacked parameters."""
        kwargs = dict(random_state=self.T.random_state, **params)
        if self.T.goal.startswith('class'):
            estimator = MLPClassifier
        else:
            estimator = MLPRegressor

        return estimator(**kwargs)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(10, 100, name='hidden_layer_sizes'),
            Integer(0, 100, name='hidden_layer_sizes'),
            Integer(0, 100, name='hidden_layer_sizes'),
            Categorical(['identity', 'logistic', 'tanh', 'relu'], name='activation'),
            Categorical(['lbfgs', 'sgd', 'adam'], name='solver'),
            Real(1e-4, 0.1, 'log-uniform', name='alpha'),
            Integer(8, 250, name='batch_size'),
            Categorical(['constant', 'invscaling', 'adaptive'], name='learning_rate'),
            Real(1e-3, 0.1, 'log-uniform', name='learning_rate_init'),
            Categorical(np.linspace(0.1, 0.9, 9), name='power_t'),
            Integer(50, 500, name='max_iter')
        ]
        return [d for d in dimensions if d.name in self.params]


# Global constants ========================================================== >>

# List of all the available models
MODEL_LIST = dict(
    GP=GaussianProcess,
    GNB=GaussianNaiveBayes,
    MNB=MultinomialNaiveBayes,
    BNB=BernoulliNaiveBayes,
    CatNB=CategoricalNaiveBayes,
    CNB=ComplementNaiveBayes,
    OLS=OrdinaryLeastSquares,
    Ridge=Ridge,
    Lasso=Lasso,
    EN=ElasticNet,
    BR=BayesianRidge,
    ARD=AutomaticRelevanceDetermination,
    LR=LogisticRegression,
    LDA=LinearDiscriminantAnalysis,
    QDA=QuadraticDiscriminantAnalysis,
    KNN=KNearestNeighbors,
    RNN=RadiusNearestNeighbors,
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
    MLP=MultilayerPerceptron
)

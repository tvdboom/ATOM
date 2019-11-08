# -*- coding: utf-8 -*-

'''
Automated Tool for Optimized Modelling (ATOM)
Author: tvdboom

'''

# << ============ Import Packages ============ >>

# Standard packages
import numpy as np
from .basemodel import BaseModel

# Sklearn models
from sklearn.gaussian_process import (
    GaussianProcessClassifier, GaussianProcessRegressor
    )
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
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
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import (
    PassiveAggressiveClassifier, PassiveAggressiveRegressor,
    SGDClassifier, SGDRegressor
    )
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor


# << ============ Functions ============ >>

def set_init(data, metric, task, log, verbose, scaled=False):
    ''' Returns BaseModel's (class) parameters as dictionary '''

    if scaled:
        params = {'X': data['X_scaled'],
                  'X_train': data['X_train_scaled'],
                  'X_test': data['X_test_scaled']}
    else:
        params = {'X': data['X'],
                  'X_train': data['X_train'],
                  'X_test': data['X_test']}

    for p in ('Y', 'Y_train', 'Y_test'):
        params[p] = data[p]
        params['metric'] = metric
        params['task'] = task
        params['log'] = log
        params['verbose'] = verbose

    return params


# << ============ Classes ============ >>

class GP(BaseModel):

    def __init__(self, *args):
        ''' Class initializer '''

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Gaussian Process', 'GP'

    def get_params(self, x):
        ''' GP has no hyperparameters to optimize '''

        return False

    def get_model(self):
        ''' Returns the sklearn model '''

        if self.task != 'regression':
            return GaussianProcessClassifier()
        else:
            return GaussianProcessRegressor()

    def get_domain(self):
        return None

    def get_init_values(self):
        return None


class GNB(BaseModel):

    def __init__(self, *args):
        ''' Class initializer '''

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Gaussian Naïve Bayes', 'GNB'

    def get_params(self, x):
        ''' GNB has no hyperparameters to optimize '''

        return False

    def get_model(self):
        ''' Returns the sklearn model '''

        return GaussianNB()

    def get_domain(self):
        return None

    def get_init_values(self):
        return None


class MNB(BaseModel):

    def __init__(self, *args):
        ''' Class initializer '''

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Multinomial Naïve Bayes', 'MNB'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        prior = [True, False]
        params = {'alpha': round(x[0, 0], 2),
                  'fit_prior': prior[int(x[0, 1])]}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return MultinomialNB(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        return [{'name': 'alpha',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)},
                {'name': 'fit_prior',
                 'type': 'discrete',
                 'domain': range(2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[1, 0]])


class BNB(BaseModel):

    def __init__(self, *args):
        ''' Class initializer '''

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Bernoulli Naïve Bayes', 'BNB'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        prior = [True, False]
        params = {'alpha': round(x[0, 0], 2),
                  'fit_prior': prior[int(x[0, 1])]}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return BernoulliNB(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        return [{'name': 'alpha',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)},
                {'name': 'fit_prior',
                 'type': 'discrete',
                 'domain': range(2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[1, 0]])


class LinReg(BaseModel):

    def __init__(self, *args):
        ''' Class initializer '''

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Linear Regression', 'LinReg'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'max_iter': int(x[0, 0]),
                  'alpha': round(x[0, 1], 2),
                  'l1_ratio': round(x[0, 2], 1)}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return ElasticNet(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'max_iter',
                 'type': 'discrete',
                 'domain': range(100, 501)},
                {'name': 'alpha',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 5, 500)},
                {'name': 'l1_ratio',
                 'type': 'discrete',
                 'domain': np.linspace(0, 1, 10)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[250, 1.0, 0.5]])


class LogReg(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Logistic Regression', 'LogReg'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        regularization = ['l1', 'l2', 'elasticnet', 'none']
        penalty = regularization[int(x[0, 2])]
        params = {'max_iter': int(x[0, 0]),
                  'penalty': penalty}

        if penalty != 'none':
            params['C'] = float(round(x[0, 1], 1))
        if penalty == 'elasticnet':  # Add extra parameter: l1_ratio
            params['l1_ratio'] = float(round(x[0, 3], 1))

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return LogisticRegression(solver='saga', multi_class='auto', **params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'max_iter',
                 'type': 'discrete',
                 'domain': range(100, 501)},
                {'name': 'C',
                 'type': 'discrete',
                 'domain': np.linspace(0.1, 5, 50)},
                {'name': 'penalty',
                 'type': 'discrete',
                 'domain': range(4)},
                {'name': 'l1_ratio',
                 'type': 'discrete',
                 'domain': np.linspace(0.1, 0.9, 9)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[250, 1.0, 1, 0.5]])


class LDA(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Linear Discriminant Analysis', 'LDA'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        solver_types = ['svd', 'lsqr', 'eigen']
        solver = solver_types[int(x[0, 0])]
        params = {'solver': solver,
                  'n_components': int(x[0, 2]),
                  'tol': round(x[0, 3], 5)}

        if solver != 'svd':  # Add extra parameter: shrinkage
            params['shrinkage'] = x[0, 1]

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return LinearDiscriminantAnalysis(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'solver',
                 'type': 'discrete',
                 'domain': range(3)},
                {'name': 'shrinkage',
                 'type': 'discrete',
                 'domain': np.linspace(0, 1, 11)},
                {'name': 'n_components',
                 'type': 'discrete',
                 'domain': range(1, 251)},
                {'name': 'tol',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[0, 0, 200, 1e-3]])


class QDA(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Quadratic Discriminant Analysis', 'QDA'

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'reg_param': round(x[0, 0], 1)}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        return QuadraticDiscriminantAnalysis(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'reg_param',
                 'type': 'discrete',
                 'domain': np.linspace(0.1, 1, 10)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[0]])


class KNN(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'K-Nearest Neighbors', 'KNN'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        weights = ['distance', 'uniform']
        params = {'n_neighbors': int(x[0, 0]),
                  'p': int(x[0, 1]),
                  'weights': weights[int(x[0, 2])]}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return KNeighborsClassifier(**params)
        else:
            return KNeighborsRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_neighbors',
                 'type': 'discrete',
                 'domain': range(1, 101)},
                {'name': 'p',
                 'type': 'discrete',
                 'domain': range(1, 3)},
                {'name': 'weights',
                 'type': 'discrete',
                 'domain': range(2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[5, 2, 1]])


class Tree(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Decision Tree', 'Tree'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.task != 'regression':
            criterion = ['entropy', 'gini']
        else:
            criterion = ['mse', 'mae', 'friedman_mse']

        params = {'criterion': criterion[int(x[0, 0])],
                  'max_depth': int(x[0, 1]),
                  'min_samples_split': int(x[0, 2]),
                  'min_samples_leaf': int(x[0, 3])}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return DecisionTreeClassifier(**params)
        else:
            return DecisionTreeRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'criterion',
                 'type': 'discrete',
                 'domain': range(2 if self.task != 'regression' else 3)},
                {'name': 'max_depth',
                 'type': 'discrete',
                 'domain': range(1, 11)},
                {'name': 'min_samples_split',
                 'type': 'discrete',
                 'domain': range(2, 21)},
                {'name': 'min_samples_leaf',
                 'type': 'discrete',
                 'domain': range(1, 21)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[0, 3, 2, 1]])


class Bag(BaseModel):
    'Bagging class'

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Bagging', 'Bag'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        bootstrap = [True, False]
        params = {'n_estimators': int(x[0, 0]),
                  'max_samples': round(x[0, 1], 1),
                  'bootstrap': bootstrap[int(x[0, 2])]}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return BaggingClassifier(**params)
        else:
            return BaggingRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'discrete',
                 'domain': range(20, 501)},
                {'name': 'max_samples',
                 'type': 'discrete',
                 'domain': np.linspace(0.3, 1, 8)},
                {'name': 'bootstrap',
                 'type': 'discrete',
                 'domain': range(2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[50, 1, 1]])


class ET(BaseModel):
    'Extremely Randomized Trees class'

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Extra-Trees', 'ET'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.task != 'regression':
            criterion = ['entropy', 'gini']
        else:
            criterion = ['mse', 'mae']
        bootstrap = [True, False]
        params = {'n_estimators': int(x[0, 0]),
                  'max_features': round(x[0, 1], 1),
                  'criterion': criterion[int(x[0, 2])],
                  'bootstrap': bootstrap[int(x[0, 3])],
                  'min_samples_split': int(x[0, 4]),
                  'min_samples_leaf': int(x[0, 5])}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return ExtraTreesClassifier(**params)
        else:
            return ExtraTreesRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'discrete',
                 'domain': range(20, 501)},
                {'name': 'max_features',
                 'type': 'discrete',
                 'domain': np.linspace(0.3, 1, 8)},
                {'name': 'criterion',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'bootstrap',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'min_samples_split',
                 'type': 'discrete',
                 'domain': range(2, 21)},
                {'name': 'min_samples_leaf',
                 'type': 'discrete',
                 'domain': range(1, 21)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[50, 1, 1, 0, 2, 1]])


class RF(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Random Forest', 'RF'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.task != 'regression':
            criterion = ['entropy', 'gini']
        else:
            criterion = ['mse', 'mae', 'friedman_mse']
        bootstrap = [True, False]
        params = {'n_estimators': int(x[0, 0]),
                  'max_features': round(x[0, 1], 1),
                  'criterion': criterion[int(x[0, 2])],
                  'bootstrap': bootstrap[int(x[0, 3])],
                  'min_samples_split': int(x[0, 4]),
                  'min_samples_leaf': int(x[0, 5])}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'discrete',
                 'domain': range(2, 101)},
                {'name': 'max_features',
                 'type': 'discrete',
                 'domain': np.linspace(0.3, 1, 8)},
                {'name': 'criterion',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'bootstrap',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'min_samples_split',
                 'type': 'discrete',
                 'domain': range(2, 21)},
                {'name': 'min_samples_leaf',
                 'type': 'discrete',
                 'domain': range(1, 21)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[10, 1, 1, 0, 2, 1]])


class AdaB(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'AdaBoost', 'AdaB'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': round(x[0, 1], 2)}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return AdaBoostClassifier(**params)
        else:
            return AdaBoostRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'discrete',
                 'domain': range(50, 501)},
                {'name': 'learning_rate',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[50, 1]])


class GBM(BaseModel):
    'Gradient Boosting Machine'

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=False))

        # Class attributes
        self.name, self.shortname = 'Gradient Boosting Machine', 'GBM'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        criterion = ['friedman_mse', 'mae', 'mse']
        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': round(x[0, 1], 2),
                  'subsample': round(x[0, 2], 1),
                  'max_depth': int(x[0, 3]),
                  'criterion': criterion[int(x[0, 4])],
                  'min_samples_split': int(x[0, 5]),
                  'min_samples_leaf': int(x[0, 6])}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return GradientBoostingClassifier(**params)
        else:
            return GradientBoostingRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'discrete',
                 'domain': range(50, 501)},
                {'name': 'learning_rate',
                 'type': 'discrete',
                 'domain': (0.01, 1)},
                {'name': 'subsample',
                 'type': 'discrete',
                 'domain': np.linspace(0.3, 1.0, 8)},
                {'name': 'max_depth',
                 'type': 'discrete',
                 'domain': range(1, 11)},
                {'name': 'criterion',
                 'type': 'discrete',
                 'domain': range(3)},
                {'name': 'min_samples_split',
                 'type': 'discrete',
                 'domain': range(2, 21)},
                {'name': 'min_samples_leaf',
                 'type': 'discrete',
                 'domain': range(1, 21)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[100, 0.1, 1.0, 3, 0, 2, 1]])


class XGB(BaseModel):
    'Extreme Gradient Boosting'

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'XGBoost', 'XGB'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': round(x[0, 1], 2),
                  'min_child_weight': int(x[0, 2]),
                  'reg_alpha': round(x[0, 3], 1),
                  'reg_lambda': round(x[0, 4], 1),
                  'subsample': round(x[0, 5], 1),
                  'max_depth': int(x[0, 6]),
                  'colsample_bytree': round(x[0, 7], 1)}
        return params

    def get_model(self, params):
        ''' Returns the model with unpacked hyperparameters '''

        from xgboost import XGBClassifier, XGBRegressor
        if self.task != 'regression':
            return XGBClassifier(**params, verbosity=0)
        else:
            return XGBRegressor(**params, verbosity=0)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'discrete',
                 'domain': range(20, 501)},
                {'name': 'learning_rate',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)},
                {'name': 'min_child_weight',
                 'type': 'discrete',
                 'domain': range(1, 21)},
                {'name': 'reg_alpha',
                 'type': 'discrete',
                 'domain': np.linspace(0, 80, 800)},
                {'name': 'reg_lambda',
                 'type': 'discrete',
                 'domain': np.linspace(0, 80, 800)},
                {'name': 'subsample',
                 'type': 'discrete',
                 'domain': np.linspace(0.3, 1.0, 8)},
                {'name': 'max_depth',
                 'type': 'discrete',
                 'domain': range(1, 11)},
                {'name': 'colsample_bytree',
                 'type': 'discrete',
                 'domain': np.linspace(0.3, 1.0, 8)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[100, 0.1, 1, 0, 1, 1.0, 3, 1.0]])


class LGB(BaseModel):
    'Light Gradient Boosting Machine'

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'LightGBM', 'LGB'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': round(x[0, 1], 2),
                  'min_child_samples': int(x[0, 2]),
                  'reg_alpha': round(x[0, 3], 1),
                  'reg_lambda': round(x[0, 4], 1),
                  'subsample': round(x[0, 5], 1),
                  'max_depth': int(x[0, 6]),
                  'colsample_bytree': round(x[0, 7], 1),
                  'num_leaves': int(x[0, 8])}
        return params

    def get_model(self, params):
        ''' Returns the model with unpacked hyperparameters '''

        from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
        if self.task != 'regression':
            return LGBMClassifier(**params)
        else:
            return LGBMRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'discrete',
                 'domain': range(20, 501)},
                {'name': 'learning_rate',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)},
                {'name': 'min_child_samples',
                 'type': 'discrete',
                 'domain': range(10, 41)},
                {'name': 'reg_alpha',
                 'type': 'discrete',
                 'domain': np.linspace(0, 80, 800)},
                {'name': 'reg_lambda',
                 'type': 'discrete',
                 'domain': np.linspace(0, 80, 800)},
                {'name': 'subsample',
                 'type': 'discrete',
                 'domain': np.linspace(0.3, 1.0, 8)},
                {'name': 'max_depth',
                 'type': 'discrete',
                 'domain': range(1, 11)},
                {'name': 'colsample_bytree',
                 'type': 'discrete',
                 'domain': np.linspace(0.3, 1.0, 8)},
                {'name': 'num_leaves',
                 'type': 'discrete',
                 'domain': range(20, 41)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[100, 0.1, 20, 0, 1.0, 1.0, 3, 1, 31]])


class CatB(BaseModel):
    'Categorical Boosting Machine'

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'CatBoost', 'CatB'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        params = {'n_estimators': int(x[0, 0]),
                  'learning_rate': round(x[0, 1], 2),
                  'reg_lambda': round(x[0, 2], 1),
                  'subsample': round(x[0, 3], 1),
                  'max_depth': int(x[0, 4]),
                  'colsample_bylevel': round(x[0, 5], 1)}
        return params

    def get_model(self, params):
        ''' Returns the model with unpacked hyperparameters '''

        from catboost import CatBoostClassifier, CatBoostRegressor
        if self.task != 'regression':
            return CatBoostClassifier(**params,
                                      train_dir='',
                                      allow_writing_files=False,
                                      verbose=False)
        else:
            return CatBoostRegressor(**params,
                                     train_dir='',
                                     allow_writing_files=False,
                                     verbose=False)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'n_estimators',
                 'type': 'discrete',
                 'domain': range(20, 501)},
                {'name': 'learning_rate',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)},
                {'name': 'reg_lambda',
                 'type': 'discrete',
                 'domain': np.linspace(0, 80, 800)},
                {'name': 'subsample',
                 'type': 'discrete',
                 'domain': np.linspace(0.3, 1.0, 8)},
                {'name': 'max_depth',
                 'type': 'discrete',
                 'domain': range(1, 11)},
                {'name': 'colsample_bylevel',
                 'type': 'discrete',
                 'domain': np.linspace(0.3, 1.0, 8)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[100, 0.1, 1.0, 1.0, 3, 1]])


class lSVM(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Linear SVM', 'lSVM'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.task != 'regression':
            losses = ['hinge', 'squared_hinge']
        else:
            losses = ['epsilon_insensitive', 'squared_epsilon_insensitive']
        loss = losses[int(x[0, 1])]
        penalties = ['l1', 'l2']

        # l1 regularization can't be combined with hinge
        penalty = penalties[int(x[0, 2])] if loss == 'squared_hinge' else 'l2'

        # l1 regularization can't be combined with squared_hinge when dual=True
        dual = True if penalty == 'l2' else False

        params = {'C': round(x[0, 0], 2),
                  'loss': loss,
                  'tol': round(x[0, 3], 4),
                  'dual': dual}

        if self.task != 'regression':
            params['penalty'] = penalty

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return LinearSVC(**params)
        else:
            return LinearSVR(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'C',
                 'type': 'discrete',
                 'domain': (0.01, 0.1, 1, 10, 100)},
                {'name': 'loss',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'penalty',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'tol',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[1, 1, 1, 1e-3]])


class kSVM(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Non-linear SVM', 'kSVM'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        kernels = ['poly', 'rbf', 'sigmoid']
        kernel = kernels[int(x[0, 4])]
        gamma = ['auto', 'scale']
        shrinking = [True, False]

        params = {'C': round(x[0, 0], 2),
                  'degree': int(x[0, 1]),
                  'gamma': gamma[int(x[0, 2])],
                  'kernel': kernel,
                  'shrinking': shrinking[int(x[0, 5])],
                  'tol': round(x[0, 6], 4)}

        if kernel != 'rbf':
            params['coef0'] = round(x[0, 3], 2)

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return SVC(**params)
        else:
            return SVR(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'C',
                 'type': 'discrete',
                 'domain': (0.01, 0.1, 1, 10, 100)},
                {'name': 'degree',
                 'type': 'discrete',
                 'domain': range(2, 6)},
                {'name': 'gamma',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'coef0',
                 'type': 'discrete',
                 'domain': np.linspace(-1, 1, 200)},
                {'name': 'kernel',
                 'type': 'discrete',
                 'domain': range(3)},
                {'name': 'shrinking',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'tol',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[1, 3, 0, 0, 1, 0, 1e-3]])


class PA(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Passive Aggressive', 'PA'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.task != 'regression':
            loss = ['hinge', 'squared_hinge']
        else:
            loss = ['epsilon_insensitive', 'squared_epsilon_insensitive']
        average = [True, False]

        params = {'loss': loss[int(x[0, 0])],
                  'C': round(x[0, 1], 4),
                  'tol': round(x[0, 2], 4),
                  'average': average[int(x[0, 3])]}

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return PassiveAggressiveClassifier(**params)
        else:
            return PassiveAggressiveRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'loss',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'C',
                 'type': 'discrete',
                 'domain': (1e-4, 1e-3, 0.01, 0.1, 1, 10)},
                {'name': 'tol',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)},
                {'name': 'average',
                 'type': 'discrete',
                 'domain': range(2)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[0, 1,  1e-3, 1]])


class SGD(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Stochastic Gradient Descent', 'SGD'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        if self.task != 'regression':
            loss = ['hinge', 'log', 'modified_huber', 'squared_hinge',
                    'perceptron', 'squared_loss', 'huber',
                    'epsilon_insensitive', 'squared_epsilon_insensitive']
        else:
            loss = ['squared_loss', 'huber',
                    'epsilon_insensitive', 'squared_epsilon_insensitive']
        penalty = ['none', 'l1', 'l2', 'elasticnet']
        average = [True, False]
        lr = ['constant', 'invscaling', 'optimal', 'adaptive']

        params = {'loss': loss[int(x[0, 0])],
                  'penalty': penalty[int(x[0, 1])],
                  'alpha': round(x[0, 2], 4),
                  'average': average[int(x[0, 3])],
                  'epsilon': round(x[0, 4], 4),
                  'learning_rate': lr[int(x[0, 6])],
                  'power_t': round(x[0, 8], 4),
                  'tol': round(x[0, 9], 4)}

        if penalty == 'elasticnet':
            params['l1_ratio'] = round(x[0, 7], 2)

        if lr != 'optimal':
            params['eta0'] = round(x[0, 5], 5)

        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return SGDClassifier(**params)
        else:
            return SGDRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'loss',
                 'type': 'discrete',
                 'domain': range(9 if self.task != 'regression' else 4)},
                {'name': 'penalty',
                 'type': 'discrete',
                 'domain': range(4)},
                {'name': 'alpha',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)},
                {'name': 'average',
                 'type': 'discrete',
                 'domain': range(2)},
                {'name': 'epsilon',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)},
                {'name': 'eta0',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)},
                {'name': 'learning_rate',
                 'type': 'discrete',
                 'domain': range(4)},
                {'name': 'l1_ratio',
                 'type': 'discrete',
                 'domain': np.linspace(0.01, 1, 100)},
                {'name': 'power_t',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)},
                {'name': 'tol',
                 'type': 'discrete',
                 'domain': np.linspace(1e-4, 0.1, 1e3)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[0, 2, 1e-3, 1, 0.1, 0.01, 2, 0.15, 0.5, 1e-3]])


class MLP(BaseModel):

    def __init__(self, *args):

        # BaseModel class initializer
        super().__init__(**set_init(*args, scaled=True))

        # Class attributes
        self.name, self.shortname = 'Multilayer Perceptron', 'MLP'
        self.task = args[2]

    def get_params(self, x):
        ''' Returns the hyperparameters as a dictionary '''

        # Set the number of neurons per layer
        n1, n2, n3 = int(x[0, 0]), int(x[0, 1]), int(x[0, 2])
        if n2 == 0:
            layers = (n1,)
        elif n3 == 0:
            layers = (n1, n2)
        else:
            layers = (n1, n2, n3)

        params = {'hidden_layer_sizes': layers,
                  'alpha': round(x[0, 3], 4),
                  'learning_rate_init': round(x[0, 4], 3),
                  'max_iter': int(x[0, 5]),
                  'batch_size': int(x[0, 6])}
        return params

    def get_model(self, params):
        ''' Returns the sklearn model with unpacked hyperparameters '''

        if self.task != 'regression':
            return MLPClassifier(**params)
        else:
            return MLPRegressor(**params)

    def get_domain(self):
        ''' Returns the bounds for the hyperparameters '''

        # Dict should be in order of continuous and then discrete types
        return [{'name': 'hidden_layer_1',
                 'type': 'discrete',
                 'domain': range(10, 101)},
                {'name': 'hidden_layer_2',
                 'type': 'discrete',
                 'domain': range(101)},
                {'name': 'hidden_layer_3',
                 'type': 'discrete',
                 'domain': range(101)},
                {'name': 'alpha',
                 'type': 'discrete',
                 'domain': (1e-4, 1e-3, 0.01, 0.1)},
                {'name': 'learning_rate_init',
                 'type': 'discrete',
                 'domain': np.linspace(0.001, 0.1, 1e2)},
                {'name': 'max_iter',
                 'type': 'discrete',
                 'domain': range(100, 501)},
                {'name': 'batch_size',
                 'type': 'discrete',
                 'domain': (1, 8, 32, 64)}]

    def get_init_values(self):
        ''' Returns initial values for the BO trials '''

        return np.array([[20, 0, 0, 1e-4, 1e-3, 200, 32]])

# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing all available models. All classes must
             have the following structure:

        Name
        ----
        Name of the model's class in camel case format.


        Class attributes
        ----------------
        acronym: str
            Acronym of the model's fullname.

        fullname: str
            Complete name of the model. If None, the estimator's
            __name__ is used.

        needs_scaling: bool
            Whether the model needs scaled features.

        accepts_sparse: bool
            Whether the model has native support for sparse matrices.

        goal: str
            If the model is only for classification ("class"),
            regression ("reg") or both ("both").


        Instance attributes
        -------------------
        T: class
            Trainer from which the model is called.

        name: str
            Name of the model. Defaults to the same as the acronym
            but can be different if the same model is called multiple
            times. The name is assigned in the basemodel.py module.

        evals: dict
            Evaluation metric and scores. Only for models that allow
            in-training evaluation.

        Properties
        ----------
        est_class: estimator's base class
            Base class (not instance) of the underlying estimator.


        Methods
        -------
        get_parameters(self, x):
            Return the parameters with rounded decimals and (optional)
            custom changes to the params. Don't implement if the method
            in BaseModel (default behaviour) is sufficient.

        get_estimator(self, **params):
            Return the model's estimator with unpacked parameters.

        custom_fit(model, train, validation, est_params):
            This method is called instead of directly running the
            estimator's fit method. Implement only to customize the fit.

        get_dimensions(self):
            Return a list of the hyperparameter space for optimization.


To add a new model:
    1. Add the model's class to models.py
    2. Add the model to the list MODELS in models.py


List of available models:
    - "Dummy" for Dummy Estimator
    - "GNB" for Gaussian Naive Bayes (no hyperparameter tuning)
    - "MNB" for Multinomial Naive Bayes
    - "BNB" for Bernoulli Naive Bayes
    - "CatNB" for Categorical Naive Bayes
    - "CNB" for Complement Naive Bayes
    - "GP" for Gaussian Process (no hyperparameter tuning)
    - "OLS" for Ordinary Least Squares (no hyperparameter tuning)
    - "Ridge" for Ridge Estimator
    - "Lasso" for Lasso Linear Regression
    - "EN" for ElasticNet Linear Regression
    - "BR" for Bayesian Ridge
    - "ARD" for Automated Relevance Determination
    - "LR" for Logistic Regression
    - "LDA" for Linear Discriminant Analysis
    - "QDA" for Quadratic Discriminant Analysis
    - "KNN" for K-Nearest Neighbors
    - "RNN" for Radius Nearest Neighbors
    - "Tree" for a single Decision Tree
    - "Bag" for Bagging
    - "ET" for Extra-Trees
    - "RF" for Random Forest
    - "AdaB" for AdaBoost
    - "GBM" for Gradient Boosting Machine
    - "hGBM" for Hist Gradient Boosting Machine
    - "XGB" for XGBoost (if package is available)
    - "LGB" for LightGBM (if package is available)
    - "CatB" for CatBoost (if package is available)
    - "lSVM" for Linear Support Vector Machine
    - "kSVM" for Kernel (non-linear) Support Vector Machine
    - "PA" for Passive Aggressive
    - "SGD" for Stochastic Gradient Descent
    - "MLP" for Multi-layer Perceptron

Additionally, ATOM implements two ensemble models:
    - "Stack" for Stacking
    - "Vote" for Voting

"""

# Standard packages
import numpy as np
from random import randint
from inspect import signature
from scipy.spatial.distance import cdist
from skopt.space.space import Real, Integer, Categorical

# Sklearn estimators
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor
)
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
)
from sklearn.linear_model import (
    LinearRegression,
    RidgeClassifier,
    Ridge as RidgeRegressor,
    Lasso as LassoRegressor,
    ElasticNet as ElasticNetRegressor,
    BayesianRidge as BayesianRidgeRegressor,
    ARDRegression,
    LogisticRegression as LR,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Own modules
from .basemodel import BaseModel
from .pipeline import Pipeline
from .ensembles import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from .utils import create_acronym, CustomDict


class CustomModel(BaseModel):
    """Custom model. Estimator provided by user."""

    def __init__(self, *args, **kwargs):
        self.est = kwargs["estimator"]  # Estimator provided by the user

        # If no fullname is provided, use the class' name
        if hasattr(self.est, "fullname"):
            self.fullname = self.est.fullname
        elif callable(self.est):
            self.fullname = self.est.__name__
        else:
            self.fullname = self.est.__class__.__name__

        # If no acronym is provided, use capital letters in the class' name
        if hasattr(self.est, "acronym"):
            self.acronym = self.est.acronym
        else:
            self.acronym = create_acronym(self.fullname)

        self.needs_scaling = getattr(self.est, "needs_scaling", False)
        super().__init__(*args)

    @property
    def est_class(self):
        """Return the estimator's class."""
        if callable(self.est):
            return self.est
        else:
            return self.est.__class__

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        sign = signature(self.est.__init__).parameters

        # The provided estimator can be a class or an instance
        if callable(self.est):
            # Add n_jobs and random_state to the estimator (if available)
            for p in ("n_jobs", "random_state"):
                if p in sign:
                    params[p] = params.pop(p, getattr(self.T, p))

            return self.est(**params)

        else:
            # Update the parameters (only if it's a BaseEstimator)
            if all(hasattr(self.est, attr) for attr in ("get_params", "set_params")):
                for p in ("n_jobs", "random_state"):
                    # If the class has the parameter and it's the default value
                    if p in sign and self.est.get_params()[p] == sign[p]._default:
                        params[p] = params.pop(p, getattr(self.T, p))

                self.est.set_params(**params)

            return self.est


class Dummy(BaseModel):
    """Dummy classifier/regressor."""

    acronym = "Dummy"
    fullname = "Dummy Estimator"
    needs_scaling = False
    accepts_sparse = False
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return DummyClassifier
        else:
            return DummyRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal == "class":
            return self.est_class(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return self.est_class(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal == "class":
            strategies = ["stratified", "most_frequent", "prior", "uniform"]
            return [Categorical(strategies, name="strategy")]
        else:
            return [
                Categorical(["mean", "median", "quantile"], name="strategy"),
                Real(0.0, 1.0, name="quantile"),
            ]


class GaussianProcess(BaseModel):
    """Gaussian process."""

    acronym = "GP"
    fullname = "Gaussian Process"
    needs_scaling = False
    accepts_sparse = False
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return GaussianProcessClassifier
        else:
            return GaussianProcessRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal == "class":
            return self.est_class(
                random_state=params.pop("random_state", self.T.random_state),
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )
        else:
            return self.est_class(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )


class GaussianNaiveBayes(BaseModel):
    """Gaussian Naive Bayes."""

    acronym = "GNB"
    fullname = "Gaussian Naive Bayes"
    needs_scaling = False
    accepts_sparse = False
    goal = "class"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return GaussianNB

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(**params)


class MultinomialNaiveBayes(BaseModel):
    """Multinomial Naive Bayes."""

    acronym = "MNB"
    fullname = "Multinomial Naive Bayes"
    needs_scaling = False
    accepts_sparse = True
    goal = "class"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return MultinomialNB

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(**params)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]


class BernoulliNaiveBayes(BaseModel):
    """Bernoulli Naive Bayes."""

    acronym = "BNB"
    fullname = "Bernoulli Naive Bayes"
    needs_scaling = False
    accepts_sparse = True
    goal = "class"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return BernoulliNB

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(**params)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]


class CategoricalNaiveBayes(BaseModel):
    """Categorical Naive Bayes."""

    acronym = "CatNB"
    fullname = "Categorical Naive Bayes"
    needs_scaling = False
    accepts_sparse = True
    goal = "class"

    def __init__(self, *args):
        super().__init__(*args)
        self.params = {"alpha": [1.0, 3], "fit_prior": [True, 0]}

    @property
    def est_class(self):
        """Return the estimator's class."""
        return CategoricalNB

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(**params)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]


class ComplementNaiveBayes(BaseModel):
    """Complement Naive Bayes."""

    acronym = "CNB"
    fullname = "Complement Naive Bayes"
    needs_scaling = False
    accepts_sparse = True
    goal = "class"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return ComplementNB

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(**params)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
            Categorical([True, False], name="norm"),
        ]


class OrdinaryLeastSquares(BaseModel):
    """Linear Regression (without regularization)."""

    acronym = "OLS"
    fullname = "Ordinary Least Squares"
    needs_scaling = True
    accepts_sparse = True
    goal = "reg"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return LinearRegression

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(n_jobs=params.pop("n_jobs", self.T.n_jobs), **params)


class Ridge(BaseModel):
    """Linear least squares with l2 regularization."""

    acronym = "Ridge"
    fullname = "Ridge Estimator"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return RidgeClassifier
        else:
            return RidgeRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(solvers, name="solver"),
        ]


class Lasso(BaseModel):
    """Linear Regression with lasso regularization."""

    acronym = "Lasso"
    fullname = "Lasso Regression"
    needs_scaling = True
    accepts_sparse = True
    goal = "reg"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return LassoRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(["cyclic", "random"], name="selection"),
        ]


class ElasticNet(BaseModel):
    """Linear Regression with elasticnet regularization."""

    acronym = "EN"
    fullname = "ElasticNet Regression"
    needs_scaling = True
    accepts_sparse = True
    goal = "reg"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return ElasticNetRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(np.linspace(0.1, 0.9, 9), name="l1_ratio"),
            Categorical(["cyclic", "random"], name="selection"),
        ]


class BayesianRidge(BaseModel):
    """Bayesian ridge regression."""

    acronym = "BR"
    fullname = "Bayesian Ridge"
    needs_scaling = True
    accepts_sparse = False
    goal = "reg"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return BayesianRidgeRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(**params)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Integer(100, 1000, name="n_iter"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="alpha_1"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="alpha_2"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="lambda_1"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="lambda_2"),
        ]


class AutomaticRelevanceDetermination(BaseModel):
    """Automatic Relevance Determination."""

    acronym = "ARD"
    fullname = "Automatic Relevant Determination"
    needs_scaling = True
    accepts_sparse = False
    goal = "reg"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return ARDRegression

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(**params)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Integer(100, 1000, name="n_iter"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="alpha_1"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="alpha_2"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="lambda_1"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="lambda_2"),
        ]


class LogisticRegression(BaseModel):
    """Logistic Regression."""

    acronym = "LR"
    fullname = "Logistic Regression"
    needs_scaling = True
    accepts_sparse = True
    goal = "class"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return LR

    def get_parameters(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_parameters(x)

        # Limitations on penalty + solver combinations
        penalty, solver = params.get("penalty"), params.get("solver")
        cond_1 = penalty == "none" and solver == "liblinear"
        cond_2 = penalty == "l1" and solver not in ("liblinear", "saga")
        cond_3 = penalty == "elasticnet" and solver != "saga"

        if cond_1 or cond_2 or cond_3:
            params.replace("penalty", "l2")  # Change to default value

        if params.get("penalty") != "elasticnet":
            params.pop("l1_ratio", None)
        if params.get("penalty") == "none":
            params.pop("C", None)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        solvers = ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
        return [
            Categorical(["none", "l1", "l2", "elasticnet"], name="penalty"),
            Real(1e-3, 100, "log-uniform", name="C"),
            Categorical(solvers, name="solver"),
            Integer(100, 1000, name="max_iter"),
            Categorical(np.linspace(0.1, 0.9, 9), name="l1_ratio"),
        ]


class LinearDiscriminantAnalysis(BaseModel):
    """Linear Discriminant Analysis."""

    acronym = "LDA"
    fullname = "Linear Discriminant Analysis"
    needs_scaling = False
    accepts_sparse = False
    goal = "class"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return LDA

    def get_parameters(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_parameters(x)

        if params.get("solver") == "svd":
            params.pop("shrinkage", None)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(**params)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Categorical(["svd", "lsqr", "eigen"], name="solver"),
            Categorical(np.linspace(0.0, 1.0, 11), name="shrinkage"),
        ]


class QuadraticDiscriminantAnalysis(BaseModel):
    """Quadratic Discriminant Analysis."""

    acronym = "QDA"
    fullname = "Quadratic Discriminant Analysis"
    needs_scaling = False
    accepts_sparse = False
    goal = "class"

    @property
    def est_class(self):
        """Return the estimator's class."""
        return QDA

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(**params)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [Categorical(np.linspace(0.0, 1.0, 11), name="reg_param")]


class KNearestNeighbors(BaseModel):
    """K-Nearest Neighbors."""

    acronym = "KNN"
    fullname = "K-Nearest Neighbors"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return KNeighborsClassifier
        else:
            return KNeighborsRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            **params,
        )

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Integer(1, 100, name="n_neighbors"),
            Categorical(["uniform", "distance"], name="weights"),
            Categorical(["auto", "ball_tree", "kd_tree", "brute"], name="algorithm"),
            Integer(20, 40, name="leaf_size"),
            Integer(1, 2, name="p"),
        ]


class RadiusNearestNeighbors(BaseModel):
    """Radius Nearest Neighbors."""

    acronym = "RNN"
    fullname = "Radius Nearest Neighbors"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    def __init__(self, *args):
        super().__init__(*args)
        self._distances = None

    @property
    def distances(self):
        """Return distances between a random subsample of rows."""
        if self._distances is None:
            self._distances = cdist(
                self.X_train.select_dtypes("number").sample(min(50, len(self.X_train))),
                self.X_train.select_dtypes("number").sample(min(50, len(self.X_train))),
            ).flatten()

        return self._distances

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return RadiusNeighborsClassifier
        else:
            return RadiusNeighborsRegressor

    def _get_default_params(self):
        """Custom method to return a valid radius."""
        x0 = super()._get_default_params()

        # Replace sklearn's default value for the mean of the distances
        x0.replace("radius", np.mean(self.distances))

        return x0

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal == "class":
            return self.est_class(
                outlier_label=params.pop("outlier_label", "most_frequent"),
                radius=params.pop("radius", np.mean(self.distances)),
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )
        else:
            return self.est_class(
                radius=params.pop("radius", np.mean(self.distances)),
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(min(self.distances), max(self.distances), name="radius"),
            Categorical(["uniform", "distance"], name="weights"),
            Categorical(["auto", "ball_tree", "kd_tree", "brute"], name="algorithm"),
            Integer(20, 40, name="leaf_size"),
            Integer(1, 2, name="p"),
        ]


class DecisionTree(BaseModel):
    """Single Decision Tree."""

    acronym = "Tree"
    fullname = "Decision Tree"
    needs_scaling = False
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return DecisionTreeClassifier
        else:
            return DecisionTreeRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error", "friedman_mse", "poisson"]

        return [
            Categorical(criterion, name="criterion"),
            Categorical(["best", "random"], name="splitter"),
            Categorical([None, *list(range(1, 10))], name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 20, name="min_samples_leaf"),
            Categorical([None, *np.linspace(0.5, 0.9, 5)], name="max_features"),
            Real(0, 0.035, name="ccp_alpha"),
        ]


class Bagging(BaseModel):
    """Bagging model (with decision tree as base estimator)."""

    acronym = "Bag"
    fullname = "Bagging"
    needs_scaling = False
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return BaggingClassifier
        else:
            return BaggingRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Integer(10, 500, name="n_estimators"),
            Categorical(np.linspace(0.5, 1.0, 6), name="max_samples"),
            Categorical(np.linspace(0.5, 1.0, 6), name="max_features"),
            Categorical([True, False], name="bootstrap"),
            Categorical([True, False], name="bootstrap_features"),
        ]


class ExtraTrees(BaseModel):
    """Extremely Randomized Trees."""

    acronym = "ET"
    fullname = "Extra-Trees"
    needs_scaling = False
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return ExtraTreesClassifier
        else:
            return ExtraTreesRegressor

    def get_parameters(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_parameters(x)

        if not params.get("bootstrap"):
            params.pop("max_samples", None)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error"]

        return [
            Integer(10, 500, name="n_estimators"),
            Categorical(criterion, name="criterion"),
            Categorical([None, *list(range(1, 10))], name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 20, name="min_samples_leaf"),
            Categorical([None, *np.linspace(0.5, 0.9, 5)], name="max_features"),
            Categorical([True, False], name="bootstrap"),
            Real(0, 0.035, name="ccp_alpha"),
            Categorical(np.linspace(0.5, 0.9, 5), name="max_samples"),
        ]


class RandomForest(BaseModel):
    """Random Forest."""

    acronym = "RF"
    fullname = "Random Forest"
    needs_scaling = False
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return RandomForestClassifier
        else:
            return RandomForestRegressor

    def get_parameters(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_parameters(x)

        if not params.get("bootstrap"):
            params.pop("max_samples", None)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error", "poisson"]

        return [
            Integer(10, 500, name="n_estimators"),
            Categorical(criterion, name="criterion"),
            Categorical([None, *list(range(1, 10))], name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 20, name="min_samples_leaf"),
            Categorical([None, *np.linspace(0.5, 0.9, 5)], name="max_features"),
            Categorical([True, False], name="bootstrap"),
            Real(0, 0.035, name="ccp_alpha"),
            Categorical(np.linspace(0.5, 0.9, 5), name="max_samples"),
        ]


class AdaBoost(BaseModel):
    """Adaptive Boosting (with decision tree as base estimator)."""

    acronym = "AdaB"
    fullname = "AdaBoost"
    needs_scaling = False
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return AdaBoostClassifier
        else:
            return AdaBoostRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(50, 500, name="n_estimators"),
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
        ]

        if self.T.goal == "class":
            dimensions.append(Categorical(["SAMME.R", "SAMME"], name="algorithm"))
        else:
            loss = ["linear", "square", "exponential"]
            dimensions.append(Categorical(loss, name="loss"))

        return dimensions


class GradientBoostingMachine(BaseModel):
    """Gradient Boosting Machine."""

    acronym = "GBM"
    fullname = "Gradient Boosting Machine"
    needs_scaling = False
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return GradientBoostingClassifier
        else:
            return GradientBoostingRegressor

    def get_parameters(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_parameters(x)

        if params.get("loss") not in ("huber", "quantile"):
            params.pop("alpha", None)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = []  # Multiclass classification only works with deviance loss
        if self.T.task.startswith("bin"):
            dimensions.append(Categorical(["deviance", "exponential"], name="loss"))
        elif self.T.task.startswith("reg"):
            loss = ["squared_error", "absolute_error", "huber", "quantile"]
            dimensions.append(Categorical(loss, name="loss"))

        dimensions.extend(
            [
                Real(0.01, 1.0, "log-uniform", name="learning_rate"),
                Integer(10, 500, name="n_estimators"),
                Categorical(np.linspace(0.5, 1.0, 6), name="subsample"),
                Categorical(["friedman_mse", "squared_error"], name="criterion"),
                Integer(2, 20, name="min_samples_split"),
                Integer(1, 20, name="min_samples_leaf"),
                Integer(1, 10, name="max_depth"),
                Categorical([None, *np.linspace(0.5, 0.9, 5)], name="max_features"),
                Real(0, 0.035, name="ccp_alpha"),
            ]
        )

        if self.goal == "reg":
            dimensions.append(Categorical(np.linspace(0.5, 0.9, 5), name="alpha"))

        return dimensions


class HistGBM(BaseModel):
    """Histogram-based Gradient Boosting Machine."""

    acronym = "hGBM"
    fullname = "HistGBM"
    needs_scaling = False
    accepts_sparse = False
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return HistGradientBoostingClassifier
        else:
            return HistGradientBoostingRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = []
        if self.T.task == "reg":
            loss = ["squared_error", "absolute_error", "poisson"]
            dimensions.append(Categorical(loss, name="loss"))

        dimensions.extend(
            [
                Real(0.01, 1.0, "log-uniform", name="learning_rate"),
                Integer(10, 500, name="max_iter"),
                Integer(10, 50, name="max_leaf_nodes"),
                Categorical([None, *np.linspace(1, 10, 10)], name="max_depth"),
                Integer(10, 30, name="min_samples_leaf"),
                Categorical([*np.linspace(0.0, 1.0, 11)], name="l2_regularization"),
            ]
        )

        return dimensions


class XGBoost(BaseModel):
    """Extreme Gradient Boosting."""

    acronym = "XGB"
    fullname = "XGBoost"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        from xgboost import XGBClassifier, XGBRegressor

        if self.T.goal == "class":
            return XGBClassifier
        else:
            return XGBRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self.T.random_state is None:  # XGBoost can't handle random_state to be None
            random_state = params.pop("random_state", randint(0, 1e5))
        else:
            random_state = params.pop("random_state", self.T.random_state)
        return self.est_class(
            use_label_encoder=params.pop("use_label_encoder", False),
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            random_state=random_state,
            verbosity=params.pop("verbosity", 0),
            **params,
        )

    def custom_fit(self, est, train, validation=None, **params):
        """Fit the model using early stopping and update evals attr."""
        from xgboost.callback import EarlyStopping

        n_estimators = est.get_params().get("n_estimators", 100)
        rounds = self._get_early_stopping_rounds(params, n_estimators)
        eval_set = params.pop("eval_set", [train, validation] if validation else None)
        callbacks = params.pop("callbacks", [])
        if rounds:  # Add early stopping callback
            callbacks.append(EarlyStopping(rounds, maximize=True))

        est.fit(
            X=train[0],
            y=train[1],
            eval_set=eval_set,
            verbose=params.get("verbose", False),
            callbacks=callbacks,
            **params,
        )

        if validation:
            # Create evals attribute with train and validation scores
            metric_name = list(est.evals_result()["validation_0"])[0]
            self.evals = {
                "metric": metric_name,
                "train": est.evals_result()["validation_0"][metric_name],
                "test": est.evals_result()["validation_1"][metric_name],
            }
            self._stopped = (len(self.evals["train"]), n_estimators)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Integer(20, 500, name="n_estimators"),
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
            Integer(1, 10, name="max_depth"),
            Real(0, 1.0, name="gamma"),
            Integer(1, 10, name="min_child_weight"),
            Categorical(np.linspace(0.5, 1.0, 6), name="subsample"),
            Categorical(np.linspace(0.3, 1.0, 8), name="colsample_bytree"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_alpha"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_lambda"),
        ]


class LightGBM(BaseModel):
    """Light Gradient Boosting Machine."""

    acronym = "LGB"
    fullname = "LightGBM"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        from lightgbm.sklearn import LGBMClassifier, LGBMRegressor

        if self.T.goal == "class":
            return LGBMClassifier
        else:
            return LGBMRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def custom_fit(self, est, train, validation=None, **params):
        """Fit the model using early stopping and update evals attr."""
        from lightgbm.callback import early_stopping, log_evaluation

        n_estimators = est.get_params().get("n_estimators", 100)
        rounds = self._get_early_stopping_rounds(params, n_estimators)
        eval_set = params.pop("eval_set", [train, validation] if validation else None)
        callbacks = params.pop("callbacks", [log_evaluation(0)])
        if rounds:  # Add early stopping callback
            callbacks.append(early_stopping(rounds, True, False))

        est.fit(
            X=train[0],
            y=train[1],
            eval_set=eval_set,
            callbacks=callbacks,
            **params,
        )

        if validation:
            # Create evals attribute with train and validation scores
            metric_name = list(est.evals_result_["training"])[0]  # Get first key
            self.evals = {
                "metric": metric_name,
                "train": est.evals_result_["training"][metric_name],
                "test": est.evals_result_["valid_1"][metric_name],
            }
            self._stopped = (len(self.evals["train"]), n_estimators)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Integer(20, 500, name="n_estimators"),
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
            Categorical([-1, *list(range(1, 10))], name="max_depth"),
            Integer(20, 40, name="num_leaves"),
            Categorical([1e-5, 1e-3, 0.1, 1, 10, 100], name="min_child_weight"),
            Integer(10, 30, name="min_child_samples"),
            Categorical(np.linspace(0.5, 1.0, 6), name="subsample"),
            Categorical(np.linspace(0.3, 1.0, 8), name="colsample_bytree"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_alpha"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_lambda"),
        ]


class CatBoost(BaseModel):
    """Categorical Boosting Machine."""

    acronym = "CatB"
    fullname = "CatBoost"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        from catboost import CatBoostClassifier, CatBoostRegressor

        if self.T.goal == "class":
            return CatBoostClassifier
        else:
            return CatBoostRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            bootstrap_type=params.pop("bootstrap_type", "Bernoulli"),  # For subsample
            train_dir=params.pop("train_dir", ""),
            allow_writing_files=params.pop("allow_writing_files", False),
            thread_count=params.pop("n_jobs", self.T.n_jobs),
            random_state=params.pop("random_state", self.T.random_state),
            verbose=params.pop("verbose", False),
            **params,
        )

    def custom_fit(self, est, train, validation=None, **params):
        """Fit the model using early stopping and update evals attr."""
        n_estimators = est.get_params().get("n_estimators", 100)
        rounds = self._get_early_stopping_rounds(params, n_estimators)

        est.fit(
            X=train[0],
            y=train[1],
            eval_set=params.pop("eval_set", validation),
            early_stopping_rounds=rounds,
            **params,
        )

        if validation:
            # Create evals attribute with train and validation scores
            metric_name = list(est.evals_result_["learn"])[0]  # Get first key
            self.evals = {
                "metric": metric_name,
                "train": est.evals_result_["learn"][metric_name],
                "test": est.evals_result_["validation"][metric_name],
            }
            self._stopped = (len(self.evals["train"]), n_estimators)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        # num_leaves and min_child_samples not available for CPU implementation
        return [
            Integer(20, 500, name="n_estimators"),
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
            Categorical([None, *list(range(1, 10))], name="max_depth"),
            Categorical(np.linspace(0.5, 1.0, 6), name="subsample"),
            Categorical(np.linspace(0.3, 1.0, 8), name="colsample_bylevel"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_lambda"),
        ]


class LinearSVM(BaseModel):
    """Linear Support Vector Machine."""

    acronym = "lSVM"
    fullname = "Linear-SVM"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return LinearSVC
        else:
            return LinearSVR

    def get_parameters(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_parameters(x)

        if self.T.goal == "class":
            # l1 regularization can't be combined with hinge
            if params.get("loss") == "hinge":
                params.replace("penalty", "l2")
            # l1 regularization can't be combined with squared_hinge when dual=True
            if params.get("penalty") == "l1" and params.get("loss") == "squared_hinge":
                params.replace("dual", False)
            # l2 regularization can't be combined with hinge when dual=False
            if params.get("penalty") == "l2" and params.get("loss") == "hinge":
                params.replace("dual", True)
        elif params.get("loss") == "epsilon_insensitive":
            params.replace("dual", True)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = []
        if self.T.goal == "class":
            loss = ["hinge", "squared_hinge"]
            dimensions.append(Categorical(["l1", "l2"], name="penalty"))
        else:
            loss = ["epsilon_insensitive", "squared_epsilon_insensitive"]

        dimensions.extend(
            [
                Categorical(loss, name="loss"),
                Real(1e-3, 100, "log-uniform", name="C"),
                Categorical([True, False], name="dual"),
            ]
        )

        return dimensions


class KernelSVM(BaseModel):
    """Kernel (non-linear) Support Vector Machine."""

    acronym = "kSVM"
    fullname = "Kernel-SVM"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return SVC
        else:
            return SVR

    def get_parameters(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_parameters(x)

        if params.get("kernel") == "poly":
            params.replace("gamma", "scale")  # Crashes in combination with "auto"
        else:
            params.pop("degree", None)

        if params.get("kernel") != "rbf":
            params.pop("coef0", None)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal == "class":
            return self.est_class(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return self.est_class(**params)

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(1e-3, 100, "log-uniform", name="C"),
            Categorical(["linear", "poly", "rbf", "sigmoid"], name="kernel"),
            Integer(2, 5, name="degree"),
            Categorical(["scale", "auto"], name="gamma"),
            Real(-1.0, 1.0, name="coef0"),
            Categorical([True, False], name="shrinking"),
        ]


class PassiveAggressive(BaseModel):
    """Passive Aggressive."""

    acronym = "PA"
    fullname = "Passive Aggressive"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return PassiveAggressiveClassifier
        else:
            return PassiveAggressiveRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal == "class":
            return self.est_class(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )
        else:
            return self.est_class(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal == "class":
            loss = ["hinge", "squared_hinge"]
        else:
            loss = ["epsilon_insensitive", "squared_epsilon_insensitive"]

        return [
            Real(1e-3, 100, "log-uniform", name="C"),
            Categorical(loss, name="loss"),
            Categorical([True, False], name="average"),
        ]


class StochasticGradientDescent(BaseModel):
    """Stochastic Gradient Descent."""

    acronym = "SGD"
    fullname = "Stochastic Gradient Descent"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return SGDClassifier
        else:
            return SGDRegressor

    def get_parameters(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_parameters(x)

        if params.get("penalty") != "elasticnet":
            params.pop("l1_ratio", None)

        if params.get("learning_rate") == "optimal":
            params.pop("eta0", None)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal == "class":
            return self.est_class(
                random_state=params.pop("random_state", self.T.random_state),
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )
        else:
            return self.est_class(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        loss = [
            "hinge",
            "log",
            "modified_huber",
            "squared_hinge",
            "perceptron",
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ]
        learning_rate = ["constant", "invscaling", "optimal", "adaptive"]

        return [
            Categorical(loss if self.T.goal == "class" else loss[-4:], name="loss"),
            Categorical(["none", "l1", "l2", "elasticnet"], name="penalty"),
            Real(1e-4, 1.0, "log-uniform", name="alpha"),
            Categorical(np.linspace(0.05, 0.95, 19), name="l1_ratio"),
            Real(1e-4, 1.0, "log-uniform", name="epsilon"),
            Categorical(learning_rate, name="learning_rate"),
            Real(1e-4, 1.0, "log-uniform", name="eta0"),
            Categorical(np.linspace(0.1, 0.9, 9), name="power_t"),
            Categorical([True, False], name="average"),
        ]


class MultilayerPerceptron(BaseModel):
    """Multi-layer Perceptron."""

    acronym = "MLP"
    fullname = "Multi-layer Perceptron"
    needs_scaling = True
    accepts_sparse = True
    goal = "both"

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return MLPClassifier
        else:
            return MLPRegressor

    @property
    def _dims(self):
        """Custom method to return hidden_layer_sizes."""
        params = [p for p in super()._dims if not p.startswith("hidden_layer")]
        if len(params) != len(super()._dims):
            params.insert(0, "hidden_layer_sizes")

        return params

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        hidden_layer_sizes = []
        for param in [p for p in sorted(params) if p.startswith("hidden_layer")]:
            if params[param] > 0:
                hidden_layer_sizes.append(params[param])
            else:
                break

        # Drop hidden layers and add hidden_layer_sizes
        if hidden_layer_sizes:
            params = params[[p for p in params if not p.startswith("hidden_layer")]]
            params.insert(0, "hidden_layer_sizes", tuple(hidden_layer_sizes))

        # Solve rest of parameters
        if params.get("solver") != "sgd":
            params.pop("learning_rate", None)
            params.pop("power_t", None)
        else:
            params.pop("learning_rate_init", None)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(10, 100, name="hidden_layer_1"),
            Integer(0, 100, name="hidden_layer_2"),
            Integer(0, 100, name="hidden_layer_3"),
            Categorical(["identity", "logistic", "tanh", "relu"], name="activation"),
            Categorical(["lbfgs", "sgd", "adam"], name="solver"),
            Real(1e-4, 0.1, "log-uniform", name="alpha"),
            Categorical([8, 16, 32, 64, 128, 256], name="batch_size"),
            Categorical(["constant", "invscaling", "adaptive"], name="learning_rate"),
            Real(1e-3, 0.1, "log-uniform", name="learning_rate_init"),
            Categorical(np.linspace(0.1, 0.9, 9), name="power_t"),
            Integer(50, 500, name="max_iter"),
        ]

        # Drop layers if sizes are specified by user
        if "hidden_layer_sizes" in self._est_params:
            return [d for d in dimensions if not d.name.startswith("hidden_layer")]
        else:
            return dimensions


# Ensembles ======================================================== >>

class Stacking(BaseModel):
    """Class for stacking the models in the pipeline."""

    acronym = "Stack"
    fullname = "Stacking"
    needs_scaling = False
    goal = "both"

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._models = kwargs.pop("models")
        self._est_params = kwargs

        if any(m.branch is not self.branch for m in self._models.values()):
            raise ValueError(
                "Invalid value for the models parameter. All "
                "models must have been fitted on the current branch."
            )

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return StackingClassifier
        else:
            return StackingRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        estimators = []
        for m in self._models.values():
            if m.scaler:
                name = f"pipeline_{m.name}"
                est = Pipeline([("scaler", m.scaler), (m.name, m.estimator)])
            else:
                name = m.name
                est = m.estimator

            estimators.append((name, est))

        return self.est_class(
            estimators=estimators,
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            **params,
        )


class Voting(BaseModel):
    """Soft Voting/Majority Rule voting."""

    acronym = "Vote"
    fullname = "Voting"
    needs_scaling = False
    goal = "both"

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._models = kwargs.pop("models")
        self._est_params = kwargs

        if any(m.branch is not self.branch for m in self._models.values()):
            raise ValueError(
                "Invalid value for the models parameter. All "
                "models must have been fitted on the current branch."
            )

    @property
    def est_class(self):
        """Return the estimator's class."""
        if self.T.goal == "class":
            return VotingClassifier
        else:
            return VotingRegressor

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        estimators = []
        for m in self._models.values():
            if m.scaler:
                name = f"pipeline_{m.name}"
                est = Pipeline([("scaler", m.scaler), (m.name, m.estimator)])
            else:
                name = m.name
                est = m.estimator

            estimators.append((name, est))

        return self.est_class(
            estimators=estimators,
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            **params,
        )


# Variables ======================================================== >>

# List of available models
MODELS = CustomDict(
    Dummy=Dummy,
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
    hGBM=HistGBM,
    XGB=XGBoost,
    LGB=LightGBM,
    CatB=CatBoost,
    lSVM=LinearSVM,
    kSVM=KernelSVM,
    PA=PassiveAggressive,
    SGD=StochasticGradientDescent,
    MLP=MultilayerPerceptron,
)

# List of available ensembles
ENSEMBLES = CustomDict(Stack=Stacking, Vote=Voting)

# List of all models + ensembles
MODELS_ENSEMBLES = CustomDict(**MODELS, **ENSEMBLES)

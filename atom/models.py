# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing all available models. All classes must
             have the following structure:

        Name
        ----
        Name of the model's class in camel case format.

        Attributes
        ----------
        T: class
            Trainer from which the model is called.

        name: str
            Name of the model. Defaults to the same as the acronym.
            Is assigned in the basemodel.py module.

        acronym: str
            Acronym of the model's fullname.

        fullname: str
            Complete name of the model.

        needs_scaling: bool
            Whether the model needs scaled features.

        type: str
            Model's type. Used to select SHAP's explainer for plotting.
            Options are: "linear", "tree" or "kernel".

        evals: dict
            Evaluation metric and scores. Only for models that allow
            in-train evaluation.

        params: dict
            All the estimator's parameters for the BO. The values
            should be a list with two elements, the parameter's
            default value and the number of decimals.

        Methods
        -------
        __init__(self, *args):
            Class initializer. Contains super() to the ModelOptimizer class.

        get_init_values(self):
            Return the initial values for the estimator. Don't implement
            if the method in ModelOptimizer (default behaviour) is sufficient.

        get_params(self, x):
            Return the parameters with rounded decimals and (optional)
            custom changes to the params. Don't implement if the method
            in ModelOptimizer (default behaviour) is sufficient.

        get_estimator(self, params=None):
            Return the model's estimator with unpacked parameters.

        custom_fit(model, train, validation, est_params):
            This method will be called instead of directly running the
            estimator's fit method. Implement only to customize the fit.

        get_dimensions(self):
            Return a list of the bounds for the hyperparameters.


To add a new model:
    1. Add the model's class to models.py
    2. Add the model to the list MODEL_LIST in models.py
    3. Add the name to all the relevant variables in utils.py


List of available models:
    - "GNB" for Gaussian Naive Bayes (no hyperparameter tuning)
    - "MNB" for Multinomial Naive Bayes
    - "BNB" for Bernoulli Naive Bayes
    - "CatNB" for Categorical Naive Bayes
    - "CNB" for Complement Naive Bayes
    - "GP" for Gaussian Process (no hyperparameter tuning)
    - "OLS" for Ordinary Least Squares (no hyperparameter tuning)
    - "Ridge" for Ridge Linear
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
    - "Bag" for Bagging (with decision tree as base estimator)
    - "ET" for Extra-Trees
    - "RF" for Random Forest
    - "AdaB" for AdaBoost (with decision tree as base estimator)
    - "GBM" for Gradient Boosting Machine
    - "XGB" for XGBoost (if package is available)
    - "LGB" for LightGBM (if package is available)
    - "CatB" for CatBoost (if package is available)
    - "lSVM" for Linear Support Vector Machine
    - "kSVM" for Kernel (non-linear) Support Vector Machine
    - "PA" for Passive Aggressive
    - "SGD" for Stochastic Gradient Descent
    - "MLP" for Multi-layer Perceptron

"""

# Standard packages
import numpy as np
from random import randint
from inspect import signature
from scipy.spatial.distance import minkowski
from skopt.space.space import Real, Integer, Categorical

# Sklearn estimators
from sklearn.gaussian_process import (
    GaussianProcessClassifier, GaussianProcessRegressor
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
from .modeloptimizer import ModelOptimizer
from .utils import dct, create_acronym


class CustomModel(ModelOptimizer):
    """Custom model. Estimator provided by user."""

    def __init__(self, *args, **kwargs):
        self.est = kwargs["estimator"]  # Estimator provided by the user

        # If no fullname is provided, use the class' name
        if hasattr(self.est, "fullname"):
            fullname = self.est.fullname
        elif callable(self.est):
            fullname = self.est.__name__
        else:
            fullname = self.est.__class__.__name__

        # If no acronym is provided, use capital letters in the class' name
        if hasattr(self.est, "acronym"):
            acronym = self.est.acronym
        else:
            acronym = create_acronym(fullname)

        super().__init__(
            *args,
            acronym=acronym,
            fullname=fullname,
            needs_scaling=getattr(self.est, "needs_scaling", False),
            type=getattr(self.est, "type", "kernel"),
            params={},
        )

    def get_estimator(self, params=None):
        """Call the estimator object."""
        params = dct(params)
        sign = signature(self.est.__init__).parameters

        # The provided estimator can be a class or an instance
        if callable(self.est):
            # Add n_jobs and random_state to the estimator (if available)
            for p in ["n_jobs", "random_state"]:
                if p in sign:
                    params[p] = params.pop(p, getattr(self.T, p))

            return self.est(**params)

        else:
            # Update the parameters (only if it's a BaseEstimator)
            if all(hasattr(self.est, attr) for attr in ["get_params", "set_params"]):
                for p in ["n_jobs", "random_state"]:
                    # If the class has the parameter and it's the default value
                    if p in sign and self.est.get_params()[p] == sign[p]._default:
                        params[p] = params.pop(p, getattr(self.T, p))

                self.est.set_params(**params)

            return self.est


class GaussianProcess(ModelOptimizer):
    """Gaussian process."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="GP",
            fullname="Gaussian Process",
            needs_scaling=False,
            type="kernel",
        )

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return GaussianProcessClassifier(
                random_state=params.pop("random_state", self.T.random_state),
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )
        else:
            return GaussianProcessRegressor(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )


class GaussianNaiveBayes(ModelOptimizer):
    """Gaussian Naive Bayes."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="GNB",
            fullname="Gaussian Naive Bayes",
            needs_scaling=False,
            type="kernel",
        )

    @staticmethod
    def get_estimator(params=None):
        """Call the estimator object."""
        return GaussianNB(**dct(params))


class MultinomialNaiveBayes(ModelOptimizer):
    """Multinomial Naive Bayes."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="MNB",
            fullname="Multinomial Naive Bayes",
            needs_scaling=False,
            type="kernel",
            params={"alpha": [1.0, 3], "fit_prior": [True, 0]},
        )

    @staticmethod
    def get_estimator(params=None):
        """Call the estimator object."""
        return MultinomialNB(**dct(params))

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]
        return [d for d in dimensions if d.name in self.params]


class BernoulliNaiveBayes(ModelOptimizer):
    """Bernoulli Naive Bayes."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="BNB",
            fullname="Bernoulli Naive Bayes",
            needs_scaling=False,
            type="kernel",
            params={"alpha": [1.0, 3], "fit_prior": [True, 0]},
        )

    @staticmethod
    def get_estimator(params=None):
        """Call the estimator object."""
        return BernoulliNB(**dct(params))

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]
        return [d for d in dimensions if d.name in self.params]


class CategoricalNaiveBayes(ModelOptimizer):
    """Categorical Naive Bayes."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="CatNB",
            fullname="Categorical Naive Bayes",
            needs_scaling=False,
            type="kernel",
            params={"alpha": [1.0, 3], "fit_prior": [True, 0]},
        )

    @staticmethod
    def get_estimator(params=None):
        """Call the estimator object."""
        return CategoricalNB(**dct(params))

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]
        return [d for d in dimensions if d.name in self.params]


class ComplementNaiveBayes(ModelOptimizer):
    """Complement Naive Bayes."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="CNB",
            fullname="Complement Naive Bayes",
            needs_scaling=False,
            type="kernel",
            params={"alpha": [1.0, 3], "fit_prior": [True, 0], "norm": [False, 0]},
        )

    @staticmethod
    def get_estimator(params=None):
        """Call the estimator object."""
        return ComplementNB(**dct(params))

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
            Categorical([True, False], name="norm"),
        ]
        return [d for d in dimensions if d.name in self.params]


class OrdinaryLeastSquares(ModelOptimizer):
    """Linear Regression (without regularization)."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="OLS",
            fullname="Ordinary Least Squares",
            needs_scaling=True,
            type="linear",
        )

    def get_estimator(self, params=None):
        """Call the estimator object."""
        params = dct(params)
        return LinearRegression(n_jobs=params.pop("n_jobs", self.T.n_jobs), **params)


class Ridge(ModelOptimizer):
    """Linear Regression/Classification with ridge regularization."""

    def __init__(self, *args):
        if args[0].goal.startswith("class"):
            fullname = "Ridge Classification"
        else:
            fullname = "Ridge Regression"

        super().__init__(
            *args,
            acronym="Ridge",
            fullname=fullname,
            needs_scaling=True,
            type="linear",
            params={"alpha": [1.0, 3], "solver": ["auto", 0]},
        )

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return RidgeClassifier(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return RidgeRegressor(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        dimensions = [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(solvers, name="solver"),
        ]
        return [d for d in dimensions if d.name in self.params]


class Lasso(ModelOptimizer):
    """Linear Regression with lasso regularization."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="Lasso",
            fullname="Lasso Regression",
            needs_scaling=True,
            type="linear",
            params={"alpha": [1.0, 3], "selection": ["cyclic", 0]},
        )

    def get_estimator(self, params=None):
        """Call the estimator object."""
        params = dct(params)
        return LassoRegressor(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(["cyclic", "random"], name="selection"),
        ]
        return [d for d in dimensions if d.name in self.params]


class ElasticNet(ModelOptimizer):
    """Linear Regression with elasticnet regularization."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="EN",
            fullname="ElasticNet Regression",
            needs_scaling=True,
            type="linear",
            params={
                "alpha": [1.0, 3],
                "l1_ratio": [0.5, 1],
                "selection": ["cyclic", 0],
            },
        )

    def get_estimator(self, params=None):
        """Call the estimator object."""
        params = dct(params)
        return ElasticNetRegressor(
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(np.linspace(0.1, 0.9, 9), name="l1_ratio"),
            Categorical(["cyclic", "random"], name="selection"),
        ]
        return [d for d in dimensions if d.name in self.params]


class BayesianRidge(ModelOptimizer):
    """Bayesian ridge regression."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="BR",
            fullname="Bayesian Ridge",
            needs_scaling=True,
            type="linear",
            params={
                "n_iter": [300, 0],
                "alpha_1": [1e-6, 8],
                "alpha_2": [1e-6, 8],
                "lambda_1": [1e-6, 8],
                "lambda_2": [1e-6, 8],
            },
        )

    @staticmethod
    def get_estimator(params=None):
        """Call the estimator object."""
        return BayesianRidgeRegressor(**dct(params))

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(100, 1000, name="n_iter"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="alpha_1"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="alpha_2"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="lambda_1"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="lambda_2"),
        ]
        return [d for d in dimensions if d.name in self.params]


class AutomaticRelevanceDetermination(ModelOptimizer):
    """Automatic Relevance Determination."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="ARD",
            fullname="Automatic Relevant Determination",
            needs_scaling=True,
            type="linear",
            params={
                "n_iter": [300, 0],
                "alpha_1": [1e-6, 8],
                "alpha_2": [1e-6, 8],
                "lambda_1": [1e-6, 8],
                "lambda_2": [1e-6, 8],
            },
        )

    @staticmethod
    def get_estimator(params=None):
        """Call the estimator object."""
        return ARDRegression(**dct(params))

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(100, 1000, name="n_iter"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="alpha_1"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="alpha_2"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="lambda_1"),
            Categorical([1e-8, 1e-6, 1e-4, 1e-2], name="lambda_2"),
        ]
        return [d for d in dimensions if d.name in self.params]


class LogisticRegression(ModelOptimizer):
    """Logistic Regression."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="LR",
            fullname="Logistic Regression",
            needs_scaling=True,
            type="linear",
            params={
                "penalty": ["l2", 0],
                "C": [1.0, 3],
                "solver": ["lbfgs", 0],
                "max_iter": [100, 0],
                "l1_ratio": [0.5, 1],
            },
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        # Limitations on penalty + solver combinations
        penalty, solver = params.get("penalty"), params.get("solver")
        cond_1 = penalty == "none" and solver == "liblinear"
        cond_2 = penalty == "l1" and solver not in ["liblinear", "saga"]
        cond_3 = penalty == "elasticnet" and solver != "saga"

        if cond_1 or cond_2 or cond_3:
            params["penalty"] = "l2"  # Change to default value

        if params.get("penalty") != "elasticnet":
            params.pop("l1_ratio")
        if params.get("penalty") == "none":
            params.pop("C")

        return params

    def get_estimator(self, params=None):
        """Call the estimator object."""
        params = dct(params)
        return LR(
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        solvers = ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
        dimensions = [
            Categorical(["none", "l1", "l2", "elasticnet"], name="penalty"),
            Real(1e-3, 100, "log-uniform", name="C"),
            Categorical(solvers, name="solver"),
            Integer(100, 1000, name="max_iter"),
            Categorical(np.linspace(0.1, 0.9, 9), name="l1_ratio"),
        ]
        return [d for d in dimensions if d.name in self.params]


class LinearDiscriminantAnalysis(ModelOptimizer):
    """Linear Discriminant Analysis."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="LDA",
            fullname="Linear Discriminant Analysis",
            needs_scaling=False,
            type="kernel",
            params={"solver": ["svd", 0], "shrinkage": [0, 1]},
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if params.get("solver") == "svd":
            params.pop("shrinkage")

        return params

    @staticmethod
    def get_estimator(params=None):
        """Call the estimator object."""
        return LDA(**dct(params))

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Categorical(["svd", "lsqr", "eigen"], name="solver"),
            Categorical(np.linspace(0.0, 1.0, 11), name="shrinkage"),
        ]
        return [d for d in dimensions if d.name in self.params]


class QuadraticDiscriminantAnalysis(ModelOptimizer):
    """Quadratic Discriminant Analysis."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="QDA",
            fullname="Quadratic Discriminant Analysis",
            needs_scaling=False,
            type="kernel",
            params={"reg_param": [0, 1]},
        )

    @staticmethod
    def get_estimator(params=None):
        """Call the estimator object."""
        return QDA(**dct(params))

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [Categorical(np.linspace(0.0, 1.0, 11), name="reg_param")]
        return [d for d in dimensions if d.name in self.params]


class KNearestNeighbors(ModelOptimizer):
    """K-Nearest Neighbors."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="KNN",
            fullname="K-Nearest Neighbors",
            needs_scaling=True,
            type="kernel",
            params={
                "n_neighbors": [5, 0],
                "weights": ["uniform", 0],
                "algorithm": ["auto", 0],
                "leaf_size": [30, 0],
                "p": [2, 0],
            },
        )

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return KNeighborsClassifier(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )
        else:
            return KNeighborsRegressor(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(1, 100, name="n_neighbors"),
            Categorical(["uniform", "distance"], name="weights"),
            Categorical(["auto", "ball_tree", "kd_tree", "brute"], name="algorithm"),
            Integer(20, 40, name="leaf_size"),
            Integer(1, 2, name="p"),
        ]
        return [d for d in dimensions if d.name in self.params]


class RadiusNearestNeighbors(ModelOptimizer):
    """Radius Nearest Neighbors."""

    def __init__(self, *args):
        self._distances = []
        super().__init__(
            *args,
            acronym="RNN",
            fullname="Radius Nearest Neighbors",
            needs_scaling=True,
            type="kernel",
            params={
                "radius": [None, 3],  # We need the scaler to calculate the distances
                "weights": ["uniform", 0],
                "algorithm": ["auto", 0],
                "leaf_size": [30, 0],
                "p": [2, 0],
            },
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
                self._distances.append(
                    minkowski(self.X_train.loc[i], self.X_train.loc[i + 1])
                )

        return self._distances

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return RadiusNeighborsClassifier(
                outlier_label="most_frequent",
                radius=params.pop("radius", np.mean(self.distances)),
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )
        else:
            return RadiusNeighborsRegressor(
                radius=params.pop("radius", np.mean(self.distances)),
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        algorithms = ["auto", "ball_tree", "kd_tree", "brute"]

        dimensions = [
            Real(min(self.distances), max(self.distances), name="radius"),
            Categorical(["uniform", "distance"], name="weights"),
            Categorical(algorithms, name="algorithm"),
            Integer(20, 40, name="leaf_size"),
            Integer(1, 2, name="p"),
        ]
        return [d for d in dimensions if d.name in self.params]


class DecisionTree(ModelOptimizer):
    """Single Decision Tree."""

    def __init__(self, *args):
        if args[0].goal.startswith("class"):
            criterion = "gini"
        else:
            criterion = "mse"

        super().__init__(
            *args,
            acronym="Tree",
            fullname="Decision Tree",
            needs_scaling=False,
            type="tree",
            params={
                "criterion": [criterion, 0],
                "splitter": ["best", 0],
                "max_depth": [None, 0],
                "min_samples_split": [2, 0],
                "min_samples_leaf": [1, 0],
                "max_features": [None, 0],
                "ccp_alpha": [0, 3],
            },
        )

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return DecisionTreeClassifier(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return DecisionTreeRegressor(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith("class"):
            criterion = ["gini", "entropy"]
        else:
            criterion = ["mse", "mae", "friedman_mse"]

        dimensions = [
            Categorical(criterion, name="criterion"),
            Categorical(["best", "random"], name="splitter"),
            Categorical([None, *list(range(1, 10))], name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 20, name="min_samples_leaf"),
            Categorical([None, *np.linspace(0.5, 0.9, 5)], name="max_features"),
            Real(0, 0.035, name="ccp_alpha"),
        ]
        return [d for d in dimensions if d.name in self.params]


class Bagging(ModelOptimizer):
    """Bagging model (with decision tree as base estimator)."""

    def __init__(self, *args):
        if args[0].goal.startswith("class"):
            fullname = "Bagging Classifier"
        else:
            fullname = "Bagging Regressor"

        super().__init__(
            *args,
            acronym="Bag",
            fullname=fullname,
            needs_scaling=False,
            type="tree",
            params={
                "n_estimators": [10, 0],
                "max_samples": [1.0, 1],
                "max_features": [1.0, 1],
                "bootstrap": [True, 0],
                "bootstrap_features": [False, 0],
            },
        )

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return BaggingClassifier(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return BaggingRegressor(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(10, 500, name="n_estimators"),
            Categorical(np.linspace(0.5, 1.0, 6), name="max_samples"),
            Categorical(np.linspace(0.5, 1.0, 6), name="max_features"),
            Categorical([True, False], name="bootstrap"),
            Categorical([True, False], name="bootstrap_features"),
        ]
        return [d for d in dimensions if d.name in self.params]


class ExtraTrees(ModelOptimizer):
    """Extremely Randomized Trees."""

    def __init__(self, *args):
        if args[0].goal.startswith("class"):
            criterion = "gini"
        else:
            criterion = "mse"

        super().__init__(
            *args,
            acronym="ET",
            fullname="Extra-Trees",
            needs_scaling=False,
            type="tree",
            params={
                "n_estimators": [100, 0],
                "criterion": [criterion, 0],
                "max_depth": [None, 0],
                "min_samples_split": [2, 0],
                "min_samples_leaf": [1, 0],
                "max_features": [None, 0],
                "bootstrap": [False, 0],
                "ccp_alpha": [0, 3],
                "max_samples": [0.9, 1],
            },
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if not params.get("bootstrap"):
            params.pop("max_samples")

        return params

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return ExtraTreesClassifier(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return ExtraTreesRegressor(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith("class"):
            criterion = ["gini", "entropy"]
        else:
            criterion = ["mse", "mae"]

        dimensions = [
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
        return [d for d in dimensions if d.name in self.params]


class RandomForest(ModelOptimizer):
    """Random Forest."""

    def __init__(self, *args):
        if args[0].goal.startswith("class"):
            criterion = "gini"
        else:
            criterion = "mse"

        super().__init__(
            *args,
            acronym="RF",
            fullname="Random Forest",
            needs_scaling=False,
            type="tree",
            params={
                "n_estimators": [100, 0],
                "criterion": [criterion, 0],
                "max_depth": [None, 0],
                "min_samples_split": [2, 0],
                "min_samples_leaf": [1, 0],
                "max_features": [None, 0],
                "bootstrap": [False, 0],
                "ccp_alpha": [0, 3],
                "max_samples": [0.9, 1],
            },
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if not params.get("bootstrap"):
            params.pop("max_samples")

        return params

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return RandomForestClassifier(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return RandomForestRegressor(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith("class"):
            criterion = ["gini", "entropy"]
        else:
            criterion = ["mse", "mae"]

        dimensions = [
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
        return [d for d in dimensions if d.name in self.params]


class AdaBoost(ModelOptimizer):
    """Adaptive Boosting (with decision tree as base estimator)."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="AdaB",
            fullname="AdaBoost",
            needs_scaling=False,
            type="tree",
            params={"n_estimators": [50, 0], "learning_rate": [1.0, 2]},
        )

        # Add extra parameters depending on the task
        if self.T.goal.startswith("class"):
            self.params["algorithm"] = ["SAMME.R", 0]
        else:
            self.params["loss"] = ["linear", 0]

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return AdaBoostClassifier(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return AdaBoostRegressor(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(50, 500, name="n_estimators"),
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
            Categorical(["SAMME.R", "SAMME"], name="algorithm"),
            Categorical(["linear", "square", "exponential"], name="loss"),
        ]
        return [d for d in dimensions if d.name in self.params]


class GradientBoostingMachine(ModelOptimizer):
    """Gradient Boosting Machine."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="GBM",
            fullname="Gradient Boosting Machine",
            needs_scaling=False,
            type="tree",
            params={
                "learning_rate": [0.1, 2],
                "n_estimators": [100, 0],
                "subsample": [1.0, 1],
                "criterion": ["friedman_mse", 0],
                "min_samples_split": [2, 0],
                "min_samples_leaf": [1, 0],
                "max_depth": [3, 0],
                "max_features": [None, 0],
                "ccp_alpha": [0, 3],
            },
        )

        # Add extra parameters depending on the task
        if self.T.task.startswith("bin"):
            self.params["loss"] = ["deviance", 0]
        elif self.T.task.startswith("reg"):
            self.params["loss"] = ["ls", 0]
            self.params["alpha"] = [0.9, 1]

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if self.T.task.startswith("reg"):
            if params.get("loss") not in ["huber", "quantile"]:
                params.pop("alpha")

        return params

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return GradientBoostingClassifier(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return GradientBoostingRegressor(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith("class"):
            loss = ["deviance", "exponential"]  # Will never be used when multiclass
        else:
            loss = ["ls", "lad", "huber", "quantile"]

        dimensions = [
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
            Integer(10, 500, name="n_estimators"),
            Categorical(np.linspace(0.5, 1.0, 6), name="subsample"),
            Categorical(["friedman_mse", "mae", "mse"], name="criterion"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 20, name="min_samples_leaf"),
            Integer(1, 10, name="max_depth"),
            Categorical([None, *np.linspace(0.5, 0.9, 5)], name="max_features"),
            Real(0, 0.035, name="ccp_alpha"),
            Categorical(loss, name="loss"),
            Categorical(np.linspace(0.5, 0.9, 5), name="alpha"),
        ]
        return [d for d in dimensions if d.name in self.params]


class XGBoost(ModelOptimizer):
    """Extreme Gradient Boosting."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="XGB",
            fullname="XGBoost",
            needs_scaling=True,
            type="tree",
            evals={},
            params={
                "n_estimators": [100, 0],
                "learning_rate": [0.1, 2],
                "max_depth": [6, 0],
                "gamma": [0.0, 2],
                "min_child_weight": [1, 0],
                "subsample": [1.0, 1],
                "colsample_bytree": [1.0, 1],
                "reg_alpha": [0, 0],
                "reg_lambda": [1, 0],
            },
        )

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        from xgboost import XGBClassifier, XGBRegressor

        params = dct(params)
        if self.T.random_state is None:  # XGBoost can't handle random_state to be None
            random_state = params.pop("random_state", randint(0, 1e5))
        else:
            random_state = params.pop("random_state", self.T.random_state)
        if self.T.goal.startswith("class"):
            return XGBClassifier(
                use_label_encoder=False,
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                random_state=random_state,
                verbosity=0,
                **params,
            )
        else:
            return XGBRegressor(
                use_label_encoder=False,
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                random_state=random_state,
                verbosity=0,
                **params,
            )

    def custom_fit(self, est, train, validation=None, params=None):
        """Fit the model using early stopping and update evals attr."""
        params = dct(params)

        # Determine early stopping rounds
        if "early_stopping_rounds" in params:
            rounds = params["early_stopping_rounds"]
        elif not self.T._early_stopping or self.T._early_stopping >= 1:  # None or int
            rounds = self.T._early_stopping
        elif self.T._early_stopping < 1:
            rounds = int(est.get_params()["n_estimators"] * self.T._early_stopping)

        pre_defined_params = ["early_stopping_rounds", "verbose"]
        est.fit(
            X=train[0],
            y=train[1],
            eval_set=[train, validation] if validation else None,
            early_stopping_rounds=rounds,
            verbose=params.get("verbose", False),
            **{k: v for k, v in params.items() if k not in pre_defined_params}
        )

        if validation:
            # Create evals attribute with train and validation scores
            metric_name = list(est.evals_result()["validation_0"])[0]
            self.evals = {
                "metric": metric_name,
                "train": est.evals_result()["validation_0"][metric_name],
                "test": est.evals_result()["validation_1"][metric_name],
            }

            iters = len(self.evals["train"])  # Iterations reached
            tot = int(est.get_params()["n_estimators"])  # Iterations in params
            self._stopped = (iters, tot) if iters < tot else None

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(20, 500, name="n_estimators"),
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
            Integer(1, 10, name="max_depth"),
            Real(0, 1.0, name="gamma"),
            Integer(1, 20, name="min_child_weight"),
            Categorical(np.linspace(0.5, 1.0, 6), name="subsample"),
            Categorical(np.linspace(0.3, 1.0, 8), name="colsample_bytree"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_alpha"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_lambda"),
        ]
        return [d for d in dimensions if d.name in self.params]


class LightGBM(ModelOptimizer):
    """Light Gradient Boosting Machine."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="LGB",
            fullname="LightGBM",
            needs_scaling=True,
            type="tree",
            evals={},
            params={
                "n_estimators": [100, 0],
                "learning_rate": [0.1, 2],
                "max_depth": [-1, 0],
                "num_leaves": [31, 0],
                "min_child_weight": [1, 0],
                "min_child_samples": [20, 0],
                "subsample": [1.0, 1],
                "colsample_bytree": [1.0, 1],
                "reg_alpha": [0, 0],
                "reg_lambda": [0, 0],
            },
        )

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        from lightgbm.sklearn import LGBMClassifier, LGBMRegressor

        params = dct(params)
        if self.T.goal.startswith("class"):
            return LGBMClassifier(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return LGBMRegressor(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def custom_fit(self, est, train, validation=None, params=None):
        """Fit the model using early stopping and update evals attr."""
        params = dct(params)

        # Determine early stopping rounds
        if "early_stopping_rounds" in params:
            rounds = params["early_stopping_rounds"]
        elif not self.T._early_stopping or self.T._early_stopping >= 1:  # None or int
            rounds = self.T._early_stopping
        elif self.T._early_stopping < 1:
            rounds = int(est.get_params()["n_estimators"] * self.T._early_stopping)

        pre_defined_params = ["early_stopping_rounds", "verbose"]
        est.fit(
            X=train[0],
            y=train[1],
            eval_set=[train, validation] if validation else None,
            early_stopping_rounds=rounds,
            verbose=params.get("verbose", False),
            **{k: v for k, v in params.items() if k not in pre_defined_params}
        )

        if validation:
            # Create evals attribute with train and validation scores
            metric_name = list(est.evals_result_["training"])[0]  # Get first key
            self.evals = {
                "metric": metric_name,
                "train": est.evals_result_["training"][metric_name],
                "test": est.evals_result_["valid_1"][metric_name],
            }

            iters = len(self.evals["train"])  # Iterations reached
            tot = int(est.get_params()["n_estimators"])  # Iterations in params
            self._stopped = (iters, tot) if iters < tot else None

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(20, 500, name="n_estimators"),
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
            Categorical([-1, *list(range(1, 10))], name="max_depth"),
            Integer(20, 40, name="num_leaves"),
            Integer(1, 20, name="min_child_weight"),
            Integer(10, 30, name="min_child_samples"),
            Categorical(np.linspace(0.5, 1.0, 6), name="subsample"),
            Categorical(np.linspace(0.3, 1.0, 8), name="colsample_bytree"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_alpha"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_lambda"),
        ]
        return [d for d in dimensions if d.name in self.params]


class CatBoost(ModelOptimizer):
    """Categorical Boosting Machine."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="CatB",
            fullname="CatBoost",
            needs_scaling=True,
            type="tree",
            evals={},
            params={
                "n_estimators": [100, 0],
                "learning_rate": [0.1, 2],
                "max_depth": [None, 0],
                "subsample": [1.0, 1],
                "colsample_bylevel": [1.0, 1],
                "reg_lambda": [0, 0],
            },
        )

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        from catboost import CatBoostClassifier, CatBoostRegressor

        params = dct(params)
        if self.T.goal.startswith("class"):
            return CatBoostClassifier(
                bootstrap_type="Bernoulli",  # subsample only works with Bernoulli
                train_dir="",
                allow_writing_files=False,
                thread_count=params.pop("n_jobs", self.T.n_jobs),
                random_state=params.pop("random_state", self.T.random_state),
                verbose=False,
                **params,
            )
        else:
            return CatBoostRegressor(
                bootstrap_type="Bernoulli",
                train_dir="",
                allow_writing_files=False,
                thread_count=params.pop("n_jobs", self.T.n_jobs),
                random_state=params.pop("random_state", self.T.random_state),
                verbose=False,
                **params,
            )

    def custom_fit(self, est, train, validation=None, params=None):
        """Fit the model using early stopping and update evals attr."""
        params = dct(params)

        # Determine early stopping rounds
        if "early_stopping_rounds" in params:
            rounds = params["early_stopping_rounds"]
        elif not self.T._early_stopping or self.T._early_stopping >= 1:  # None or int
            rounds = self.T._early_stopping
        elif self.T._early_stopping < 1:
            rounds = int(est.get_params()["n_estimators"] * self.T._early_stopping)

        est.fit(
            X=train[0],
            y=train[1],
            eval_set=validation,
            early_stopping_rounds=rounds,
            **{k: v for k, v in params.items() if k != "early_stopping_rounds"}
        )

        if validation:
            # Create evals attribute with train and validation scores
            metric_name = list(est.evals_result_["learn"])[0]  # Get first key
            self.evals = {
                "metric": metric_name,
                "train": est.evals_result_["learn"][metric_name],
                "test": est.evals_result_["validation"][metric_name],
            }

            iters = len(self.evals["train"])  # Iterations reached
            tot = int(est.get_all_params()["iterations"])  # Iterations in params
            self._stopped = (iters, tot) if iters < tot else None

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        # num_leaves and min_child_samples not available for CPU implementation
        dimensions = [
            Integer(20, 500, name="n_estimators"),
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
            Categorical([None, *list(range(1, 10))], name="max_depth"),
            Categorical(np.linspace(0.5, 1.0, 6), name="subsample"),
            Categorical(np.linspace(0.3, 1.0, 8), name="colsample_bylevel"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_lambda"),
        ]
        return [d for d in dimensions if d.name in self.params]


class LinearSVM(ModelOptimizer):
    """Linear Support Vector Machine."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="lSVM",
            fullname="Linear-SVM",
            needs_scaling=True,
            type="kernel",
            params={"loss": ["epsilon_insensitive", 0], "C": [1.0, 3]},
        )

        # Different params for classification tasks
        if self.T.goal.startswith("class"):
            self.params["loss"] = ["squared_hinge", 0]
            self.params["penalty"] = ["l2", 0]

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        # l1 regularization can't be combined with hinge
        # l1 regularization can't be combined with squared_hinge when dual=True
        if self.T.goal.startswith("class"):
            if params.get("loss") == "hinge":
                params["penalty"] = "l2"
            if params.get("penalty") == "l1" and params.get("loss") == "squared_hinge":
                params["dual"] = False

        return params

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return LinearSVC(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return LinearSVR(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith("class"):
            dimensions = [
                Categorical(["hinge", "squared_hinge"], name="loss"),
                Real(1e-3, 100, "log-uniform", name="C"),
                Categorical(["l1", "l2"], name="penalty"),
            ]
        else:
            dimensions = [
                Categorical(
                    ["epsilon_insensitive", "squared_epsilon_insensitive"], name="loss"
                ),
                Real(1e-3, 100, "log-uniform", name="C"),
            ]
        return [d for d in dimensions if d.name in self.params]


class KernelSVM(ModelOptimizer):
    """Kernel (non-linear) Support Vector Machine."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="kSVM",
            fullname="Kernel-SVM",
            needs_scaling=True,
            type="kernel",
            params={
                "C": [1.0, 3],
                "kernel": ["rbf", 0],
                "degree": [3, 0],
                "gamma": ["scale", 0],
                "coef0": [0, 2],
                "shrinking": [True, 0],
            },
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if params.get("kernel") == "poly":
            params["gamma"] = "scale"  # Crashes in combination with "auto"
        else:
            params.pop("degree")

        if params.get("kernel") != "rbf":
            params.pop("coef0")

        return params

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return SVC(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return SVR(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 100, "log-uniform", name="C"),
            Categorical(["poly", "rbf", "sigmoid"], name="kernel"),
            Integer(2, 5, name="degree"),
            Categorical(["scale", "auto"], name="gamma"),
            Real(-1.0, 1.0, name="coef0"),
            Categorical([True, False], name="shrinking"),
        ]
        return [d for d in dimensions if d.name in self.params]


class PassiveAggressive(ModelOptimizer):
    """Passive Aggressive."""

    def __init__(self, *args):
        if args[0].goal.startswith("class"):
            loss = "hinge"
        else:
            loss = "epsilon_insensitive"

        super().__init__(
            *args,
            acronym="PA",
            fullname="Passive Aggressive",
            needs_scaling=True,
            params={"C": [1.0, 3], "loss": [loss, 0], "average": [False, 0]},
        )

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return PassiveAggressiveClassifier(
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )
        else:
            return PassiveAggressiveRegressor(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal.startswith("class"):
            loss = ["hinge", "squared_hinge"]
        else:
            loss = ["epsilon_insensitive", "squared_epsilon_insensitive"]

        dimensions = [
            Real(1e-3, 100, "log-uniform", name="C"),
            Categorical(loss, name="loss"),
            Categorical([True, False], name="average"),
        ]
        return [d for d in dimensions if d.name in self.params]


class StochasticGradientDescent(ModelOptimizer):
    """Stochastic Gradient Descent."""

    def __init__(self, *args):
        if args[0].task.startswith("reg"):
            loss = "squared_loss"
        else:
            loss = "hinge"

        super().__init__(
            *args,
            acronym="SGD",
            fullname="Stochastic Gradient Descent",
            needs_scaling=True,
            type="linear",
            params={
                "loss": [loss, 0],
                "penalty": ["l2", 0],
                "alpha": [1e-4, 4],
                "l1_ratio": [0.15, 2],
                "epsilon": [0.1, 4],
                "learning_rate": ["optimal", 0],
                "eta0": [0.01, 4],
                "power_t": [0.5, 1],
                "average": [False, 0],
            },
        )

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = super().get_params(x)

        if params.get("penalty") != "elasticnet":
            params.pop("l1_ratio")

        if params.get("learning_rate") == "optimal":
            params.pop("eta0")

        return params

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return SGDClassifier(
                random_state=params.pop("random_state", self.T.random_state),
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )
        else:
            return SGDRegressor(
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
            "squared_loss",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ]
        loss = loss[:-4] if self.T.goal.startswith("class") else loss[-4:]
        learning_rate = ["constant", "invscaling", "optimal", "adaptive"]

        dimensions = [
            Categorical(loss, name="loss"),
            Categorical(["none", "l1", "l2", "elasticnet"], name="penalty"),
            Real(1e-4, 1.0, "log-uniform", name="alpha"),
            Categorical(np.linspace(0.05, 0.95, 19), name="l1_ratio"),
            Real(1e-4, 1.0, "log-uniform", name="epsilon"),
            Categorical(learning_rate, name="learning_rate"),
            Real(1e-4, 1.0, "log-uniform", name="eta0"),
            Categorical(np.linspace(0.1, 0.9, 9), name="power_t"),
            Categorical([True, False], name="average"),
        ]
        return [d for d in dimensions if d.name in self.params]


class MultilayerPerceptron(ModelOptimizer):
    """Multi-layer Perceptron."""

    def __init__(self, *args):
        super().__init__(
            *args,
            acronym="MLP",
            fullname="Multi-layer Perceptron",
            needs_scaling=True,
            type="kernel",
            params={
                "hidden_layer_sizes": [(100, 0, 0), 0],
                "activation": ["relu", 0],
                "solver": ["adam", 0],
                "alpha": [1e-4, 4],
                "batch_size": [200, 0],
                "learning_rate": ["constant", 0],
                "learning_rate_init": [0.001, 3],
                "power_t": [0.5, 1],
                "max_iter": [200, 0],
            },
        )

    def get_init_values(self):
        """Custom method to return the correct hidden_layer_sizes."""
        init_values = []
        for key, value in self.params.items():
            if key == "hidden_layer_sizes":
                init_values.extend(value[0])
            else:
                init_values.append(value[0])

        return init_values

    def get_params(self, x):
        """Return a dictionary of the model´s hyperparameters."""
        params = {}
        for i, key in enumerate(self.params):
            # Add extra counter for the hidden layers
            j = 2 if "hidden_layer_sizes" in self.params else 0

            if key == "hidden_layer_sizes":
                # Set the number of neurons per layer
                n1, n2, n3 = x[i], x[i + 1], x[i + 2]
                if n2 == 0:
                    layers = (n1,)
                elif n3 == 0:
                    layers = (n1, n2)
                else:
                    layers = (n1, n2, n3)

                params["hidden_layer_sizes"] = layers

            elif self.params[key][1]:  # If it has decimals...
                params[key] = round(x[i + j], self.params[key][1])
            else:
                params[key] = x[i + j]

        if params.get("solver") != "sgd":
            params.pop("learning_rate")
            params.pop("power_t")
        else:
            params.pop("learning_rate_init")

        return params

    def get_estimator(self, params=None):
        """Return the model's estimator with unpacked parameters."""
        params = dct(params)
        if self.T.goal.startswith("class"):
            return MLPClassifier(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )
        else:
            return MLPRegressor(
                random_state=params.pop("random_state", self.T.random_state),
                **params,
            )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(10, 100, name="hidden_layer_sizes"),
            Integer(0, 100, name="hidden_layer_sizes"),
            Integer(0, 100, name="hidden_layer_sizes"),
            Categorical(["identity", "logistic", "tanh", "relu"], name="activation"),
            Categorical(["lbfgs", "sgd", "adam"], name="solver"),
            Real(1e-4, 0.1, "log-uniform", name="alpha"),
            Integer(8, 250, name="batch_size"),
            Categorical(["constant", "invscaling", "adaptive"], name="learning_rate"),
            Real(1e-3, 0.1, "log-uniform", name="learning_rate_init"),
            Categorical(np.linspace(0.1, 0.9, 9), name="power_t"),
            Integer(50, 500, name="max_iter"),
        ]
        return [d for d in dimensions if d.name in self.params]


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
    MLP=MultilayerPerceptron,
)

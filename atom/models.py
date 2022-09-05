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

        supports_engines: list
            Engines that can be used to run this model.

        _module: str
            Module from which to load the class. If one of engines,
            ignore the engine name, i.e. use "ensemble" instead of
            "sklearn.ensemble".

        _estimators: CustomDict
            Name of the estimators per goal.

        Instance attributes
        -------------------
        T: class
            Parent class from which the model is called.

        name: str
            Name of the model. Defaults to the same as the acronym
            but can be different if the same model is called multiple
            times. The name is assigned in the basemodel.py module.

        evals: dict
            Evaluation metric and scores. Only for models that allow
            in-training evaluation.

        Methods
        -------
        get_parameters(self, x):
            Return the parameters with rounded decimals and (optional)
            custom changes to the params. Don't implement if the method
            in BaseModel (default behaviour) is sufficient.

        get_estimator(self, **params):
            Return the model's estimator with unpacked parameters.
            Implement only if custom parameters (aside n_jobs and
            random_state) are needed.

        custom_fit(model, train, validation, est_params):
            This method is called instead of directly running the
            estimator's fit method. Implement only to customize the fit.

        get_dimensions(self):
            Return a list of the hyperparameter space for optimization.


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
- "Lars" for Least Angle Regression
- "BR" for Bayesian Ridge
- "ARD" for Automated Relevance Determination
- "Huber" for Huber Regression
- "Perc" for Perceptron
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

import random
from typing import Union

import numpy as np
from scipy.spatial.distance import cdist
from skopt.space.space import Categorical, Integer, Real

from atom.basemodel import BaseModel
from atom.pipeline import Pipeline
from atom.utils import CustomDict, create_acronym


# Variables ======================================================== >>

zero_to_one_exc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
zero_to_one_inc = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
half_to_one_exc = [0.5, 0.6, 0.7, 0.8, 0.9]
half_to_one_inc = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# Classes ========================================================== >>

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
        if callable(self.est):
            # Add n_jobs and random_state to the estimator (if available)
            for p in ("n_jobs", "random_state"):
                if p in self._sign():
                    params[p] = params.pop(p, getattr(self.T, p))

            return self.est(**params)
        else:
            # Update the parameters if it's a child class of BaseEstimator
            # If the class has the param and it's the default value, change it
            if all(hasattr(self.est, attr) for attr in ("get_params", "set_params")):
                for p in ("n_jobs", "random_state"):
                    if p in self._sign():
                        if self.est.get_params()[p] == self._sign()[p]._default:
                            params[p] = params.pop(p, getattr(self.T, p))

                self.est.set_params(**params)

            return self.est


class AdaBoost(BaseModel):
    """Adaptive Boosting (with decision tree as base estimator).

    AdaBoost is a meta-estimator that begins by fitting a
    classifier/regressor on the original dataset and then fits
    additional copies of the algorithm on the same dataset but where
    the weights of instances are adjusted according to the error of
    the current prediction.

    Corresponding estimators are:

    - [AdaBoostClassifier][] for classification tasks.
    - [AdaBoostRegressor][] for regression tasks.

    Read more in sklearn's [documentation][adabdocs].

    See Also
    --------
    atom.models:DecisionTree
    atom.models:ExtraTrees
    atom.models:RandomForest

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="AdaB", verbose=2)

    Training ========================= >>
    Models: AdaB
    Metric: f1


    Results for AdaBoost:
    Fit ---------------------------------------------
    Train evaluation --> f1: 1.0
    Test evaluation --> f1: 0.9722
    Time elapsed: 0.108s
    -------------------------------------------------
    Total time: 0.108s


    Final results ==================== >>
    Duration: 0.109s
    -------------------------------------
    AdaBoost --> f1: 0.9722

    ```

    """

    acronym = "AdaB"
    fullname = "AdaBoost"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "AdaBoostClassifier", "reg": "AdaBoostRegressor"}
    )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(50, 500, name="n_estimators"),
            Real(0.01, 10, "log-uniform", name="learning_rate"),
        ]

        if self.T.goal == "class":
            dimensions.append(Categorical(["SAMME.R", "SAMME"], name="algorithm"))
        else:
            loss = ["linear", "square", "exponential"]
            dimensions.append(Categorical(loss, name="loss"))

        return dimensions


class AutomaticRelevanceDetermination(BaseModel):
    """Automatic Relevance Determination."""

    acronym = "ARD"
    fullname = "Automatic Relevant Determination"
    needs_scaling = True
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "ARDRegression"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Integer(100, 1000, name="n_iter"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="alpha_1"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="alpha_2"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="lambda_1"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="lambda_2"),
        ]


class Bagging(BaseModel):
    """Bagging model (with decision tree as base estimator)."""

    acronym = "Bag"
    fullname = "Bagging"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict({"class": "BaggingClassifier", "reg": "BaggingRegressor"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Integer(10, 500, name="n_estimators"),
            Categorical(half_to_one_inc, name="max_samples"),
            Categorical(half_to_one_inc, name="max_features"),
            Categorical([True, False], name="bootstrap"),
            Categorical([True, False], name="bootstrap_features"),
        ]


class BayesianRidge(BaseModel):
    """Bayesian ridge regression."""

    acronym = "BR"
    fullname = "Bayesian Ridge"
    needs_scaling = True
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "BayesianRidge"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Integer(100, 1000, name="n_iter"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="alpha_1"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="alpha_2"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="lambda_1"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="lambda_2"),
        ]


class BernoulliNaiveBayes(BaseModel):
    """Bernoulli Naive Bayes."""

    acronym = "BNB"
    fullname = "Bernoulli Naive Bayes"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "BernoulliNB"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(0.01, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]


class CatBoost(BaseModel):
    """Categorical Boosting Machine."""

    acronym = "CatB"
    fullname = "CatBoost"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = []

    _module = "catboost"
    _estimators = CustomDict(
        {"class": "CatBoostClassifier", "reg": "CatBoostRegressor"}
    )

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            bootstrap_type=params.pop("bootstrap_type", "Bernoulli"),  # For subsample
            train_dir=params.pop("train_dir", ""),
            allow_writing_files=params.pop("allow_writing_files", False),
            thread_count=params.pop("n_jobs", self.T.n_jobs),
            task_type=params.pop("task_type", "GPU" if self._gpu else "CPU"),
            devices=str(self.T._device_id),
            verbose=params.pop("verbose", False),
            random_state=params.pop("random_state", self.T.random_state),
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
        return [
            Integer(20, 500, name="n_estimators"),
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
            Categorical([None, *range(1, 17)], name="max_depth"),
            Integer(1, 30, name="min_child_samples"),
            Categorical(half_to_one_inc, name="subsample"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_lambda"),
        ]


class CategoricalNaiveBayes(BaseModel):
    """Categorical Naive Bayes."""

    acronym = "CatNB"
    fullname = "Categorical Naive Bayes"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "CategoricalNB"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(0.01, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]


class ComplementNaiveBayes(BaseModel):
    """Complement Naive Bayes."""

    acronym = "CNB"
    fullname = "Complement Naive Bayes"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "ComplementNB"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(0.01, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
            Categorical([True, False], name="norm"),
        ]


class DecisionTree(BaseModel):
    """Single Decision Tree."""

    acronym = "Tree"
    fullname = "Decision Tree"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn"]

    _module = "tree"
    _estimators = CustomDict(
        {"class": "DecisionTreeClassifier", "reg": "DecisionTreeRegressor"}
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
            Categorical([None, *range(1, 17)], name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 20, name="min_samples_leaf"),
            Categorical(
                categories=["auto", "sqrt", "log2", *half_to_one_exc, None],
                name="max_features",
            ),
            Real(0, 0.035, name="ccp_alpha"),
        ]


class Dummy(BaseModel):
    """Dummy classifier/regressor."""

    acronym = "Dummy"
    fullname = "Dummy Estimator"
    needs_scaling = False
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "dummy"
    _estimators = CustomDict({"class": "DummyClassifier", "reg": "DummyRegressor"})

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        if self._get_param(params, "strategy") != "quantile":
            params.pop("quantile", None)
        elif params.get("quantile") is None:
            # quantile can't be None with strategy="quantile" (select value randomly)
            params.replace_value("quantile", random.choice(zero_to_one_inc))

        return params

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal == "class":
            categories = ["most_frequent", "prior", "stratified", "uniform"]
            dimensions = [Categorical(categories, name="strategy")]
        else:
            dimensions = [
                Categorical(["mean", "median", "quantile"], name="strategy"),
                Categorical([None, *zero_to_one_inc], name="quantile"),
            ]

        return dimensions


class ElasticNet(BaseModel):
    """Linear Regression with elasticnet regularization."""

    acronym = "EN"
    fullname = "ElasticNet"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "ElasticNet"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(half_to_one_exc, name="l1_ratio"),
            Categorical(["cyclic", "random"], name="selection"),
        ]


class ExtraTrees(BaseModel):
    """Extremely Randomized Trees."""

    acronym = "ET"
    fullname = "Extra-Trees"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "ExtraTreesClassifier", "reg": "ExtraTreesRegressor"}
    )

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        if not self._get_param(params, "bootstrap"):
            params.pop("max_samples", None)

        return params

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error"]

        return [
            Integer(10, 500, name="n_estimators"),
            Categorical(criterion, name="criterion"),
            Categorical([None, *range(1, 17)], name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 20, name="min_samples_leaf"),
            Categorical(
                categories=["sqrt", "log2", *half_to_one_exc, None],
                name="max_features",
            ),
            Categorical([True, False], name="bootstrap"),
            Categorical([None, *half_to_one_exc], name="max_samples"),
            Real(0, 0.035, name="ccp_alpha"),
        ]


class GaussianNaiveBayes(BaseModel):
    """Gaussian Naive Bayes."""

    acronym = "GNB"
    fullname = "Gaussian Naive Bayes"
    needs_scaling = False
    accepts_sparse = False
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "GaussianNB"})


class GaussianProcess(BaseModel):
    """Gaussian process."""

    acronym = "GP"
    fullname = "Gaussian Process"
    needs_scaling = False
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "gaussian_process"
    _estimators = CustomDict(
        {"class": "GaussianProcessClassifier", "reg": "GaussianProcessRegressor"}
    )


class GradientBoostingMachine(BaseModel):
    """Gradient Boosting Machine."""

    acronym = "GBM"
    fullname = "Gradient Boosting Machine"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "GradientBoostingClassifier", "reg": "GradientBoostingRegressor"}
    )

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        if self._get_param(params, "loss") not in ("huber", "quantile"):
            params.pop("alpha", None)

        return params

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
                Categorical(half_to_one_inc, name="subsample"),
                Categorical(["friedman_mse", "squared_error"], name="criterion"),
                Integer(2, 20, name="min_samples_split"),
                Integer(1, 20, name="min_samples_leaf"),
                Integer(1, 21, name="max_depth"),
                Categorical(
                    categories=["auto", "sqrt", "log2", *half_to_one_exc, None],
                    name="max_features",
                ),
                Real(0, 0.035, name="ccp_alpha"),
            ]
        )

        if self.T.goal == "reg":
            dimensions.append(Categorical(half_to_one_exc, name="alpha"))

        return dimensions


class HuberRegression(BaseModel):
    """Huber regression."""

    acronym = "Huber"
    fullname = "Huber Regression"
    needs_scaling = True
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "HuberRegressor"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(1, 10, "log-uniform", name="epsilon"),
            Integer(50, 500, name="max_iter"),
            Categorical([1e-4, 1e-3, 1e-2, 1e-1, 1], name="alpha"),
        ]


class HistGBM(BaseModel):
    """Histogram-based Gradient Boosting Machine."""

    acronym = "hGBM"
    fullname = "HistGBM"
    needs_scaling = False
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict(
        {
            "class": "HistGradientBoostingClassifier",
            "reg": "HistGradientBoostingRegressor",
        }
    )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = []
        if self.T.goal == "reg":
            loss = ["squared_error", "absolute_error", "poisson"]
            dimensions.append(Categorical(loss, name="loss"))

        dimensions.extend(
            [
                Real(0.01, 1.0, "log-uniform", name="learning_rate"),
                Integer(10, 500, name="max_iter"),
                Integer(10, 50, name="max_leaf_nodes"),
                Categorical([None, *range(1, 17)], name="max_depth"),
                Integer(10, 30, name="min_samples_leaf"),
                Categorical(zero_to_one_inc, name="l2_regularization"),
            ]
        )

        return dimensions


class KNearestNeighbors(BaseModel):
    """K-Nearest Neighbors."""

    acronym = "KNN"
    fullname = "K-Nearest Neighbors"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "neighbors"
    _estimators = CustomDict(
        {"class": "KNeighborsClassifier", "reg": "KNeighborsRegressor"}
    )

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [Integer(1, 100, name="n_neighbors")]

        if not self._gpu:
            dimensions.extend(
                [
                    Categorical(["uniform", "distance"], name="weights"),
                    Categorical(
                        categories=["auto", "ball_tree", "kd_tree", "brute"],
                        name="algorithm",
                    ),
                    Integer(20, 40, name="leaf_size"),
                    Integer(1, 2, name="p"),
                ]
            )

        return dimensions


class KernelSVM(BaseModel):
    """Kernel (non-linear) Support Vector Machine."""

    acronym = "kSVM"
    fullname = "Kernel SVM"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "svm"
    _estimators = CustomDict({"class": "SVC", "reg": "SVR"})

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        if self.T.goal == "class":
            params.pop("epsilon", None)

        kernel = self._get_param(params, "kernel")
        if kernel == "poly":
            params.replace_value("gamma", "scale")  # Crashes in combination with "auto"
        else:
            params.pop("degree", None)

        if kernel not in ("rbf", "poly", "sigmoid"):
            params.pop("gamma", None)

        if kernel not in ("poly", "sigmoid"):
            params.pop("coef0", None)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self._gpu and self.T.goal == "class":
            return self.est_class(
                probability=True,
                random_state=params.pop("random_state", self.T.random_state),
                **params)
        else:
            return super().get_estimator(**params)

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Real(1e-3, 100, "log-uniform", name="C"),
            Categorical(["linear", "poly", "rbf", "sigmoid"], name="kernel"),
            Integer(2, 5, name="degree"),
            Categorical(["scale", "auto"], name="gamma"),
            Real(-1.0, 1.0, name="coef0"),
        ]

        if not self._gpu:
            dimensions.extend(
                [
                    Real(1e-3, 100, "log-uniform", name="epsilon"),
                    Categorical([True, False], name="shrinking"),
                ]
            )

        return dimensions


class Lasso(BaseModel):
    """Linear Regression with lasso regularization."""

    acronym = "Lasso"
    fullname = "Lasso Regression"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "sklearnex"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "Lasso"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(["cyclic", "random"], name="selection"),
        ]


class LeastAngleRegression(BaseModel):
    """Least Angle Regression."""

    acronym = "Lars"
    fullname = "Least Angle Regression"
    needs_scaling = True
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "Lars"})


class LightGBM(BaseModel):
    """Light Gradient Boosting Machine."""

    acronym = "LGB"
    fullname = "LightGBM"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = []

    _module = "lightgbm.sklearn"
    _estimators = CustomDict({"class": "LGBMClassifier", "reg": "LGBMRegressor"})

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self.est_class(
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            device=params.pop("device", "gpu" if self._gpu else "cpu"),
            gpu_device_id=params.pop("gpu_device_id", self.T._device_id or -1),
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
            Categorical([-1, *range(1, 17)], name="max_depth"),
            Integer(20, 40, name="num_leaves"),
            Categorical([1e-4, 1e-3, 0.01, 0.1, 1, 10, 100], name="min_child_weight"),
            Integer(1, 30, name="min_child_samples"),
            Categorical(half_to_one_inc, name="subsample"),
            Categorical([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], name="colsample_bytree"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_alpha"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_lambda"),
        ]


class LinearDiscriminantAnalysis(BaseModel):
    """Linear Discriminant Analysis."""

    acronym = "LDA"
    fullname = "Linear Discriminant Analysis"
    needs_scaling = False
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "discriminant_analysis"
    _estimators = CustomDict({"class": "LinearDiscriminantAnalysis"})

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        if self._get_param(params, "solver") == "svd":
            params.pop("shrinkage", None)

        return params

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Categorical(["svd", "lsqr", "eigen"], name="solver"),
            Categorical([None, "auto", *half_to_one_inc], name="shrinkage"),
        ]


class LinearSVM(BaseModel):
    """Linear Support Vector Machine."""

    acronym = "lSVM"
    fullname = "Linear SVM"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "cuml"]

    _module = "svm"
    _estimators = CustomDict({"class": "LinearSVC", "reg": "LinearSVR"})

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        if self.T.goal == "class":
            if self._get_param(params, "loss") == "hinge":
                # l1 regularization can't be combined with hinge
                params.replace_value("penalty", "l2")
                # l2 regularization can't be combined with hinge when dual=False
                params.replace_value("dual", True)
            elif self._get_param(params, "loss") == "squared_hinge":
                # l1 regularization can't be combined with squared_hinge when dual=True
                if self._get_param(params, "penalty") == "l1":
                    params.replace_value("dual", False)
        elif self._get_param(params, "loss") == "epsilon_insensitive":
            params.replace_value("dual", True)

        return params

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self._gpu and self.T.goal == "class":
            return self.est_class(probability=True, **params)
        else:
            return super().get_estimator(**params)

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
            ]
        )

        if not self._gpu:
            dimensions.append(Categorical([True, False], name="dual"))

        return dimensions


class LogisticRegression(BaseModel):
    """Logistic Regression."""

    acronym = "LR"
    fullname = "Logistic Regression"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "LogisticRegression"})

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        # Limitations on penalty + solver combinations
        penalty = self._get_param(params, "penalty")
        solver = self._get_param(params, "solver")
        cond_1 = penalty == "none" and solver == "liblinear"
        cond_2 = penalty == "l1" and solver not in ("liblinear", "saga")
        cond_3 = penalty == "elasticnet" and solver != "saga"

        if cond_1 or cond_2 or cond_3:
            params.replace_value("penalty", "l2")  # Change to default value

        if self._get_param(params, "penalty") != "elasticnet":
            params.pop("l1_ratio", None)
        elif self._get_param(params, "l1_ratio") is None:
            # l1_ratio can't be None with elasticnet (select value randomly)
            params.replace_value("l1_ratio", random.choice(zero_to_one_exc))

        if self._get_param(params, "penalty") == "none":
            params.pop("C", None)

        return params

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        solvers = ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]

        dimensions = [
            Categorical(["l1", "l2", "elasticnet", "none"], name="penalty"),
            Real(1e-3, 100, "log-uniform", name="C"),
            Categorical(solvers, name="solver"),
            Integer(100, 1000, name="max_iter"),
            Categorical([None, *zero_to_one_exc], name="l1_ratio"),
        ]

        if self._gpu:
            del dimensions[2]

        return dimensions


class MultilayerPerceptron(BaseModel):
    """Multi-layer Perceptron."""

    acronym = "MLP"
    fullname = "Multi-layer Perceptron"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn"]

    _module = "neural_network"
    _estimators = CustomDict({"class": "MLPClassifier", "reg": "MLPRegressor"})

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
        if self._get_param(params, "solver") != "sgd":
            params.pop("learning_rate", None)
            params.pop("power_t", None)
        else:
            params.pop("learning_rate_init", None)

        return params

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        dimensions = [
            Integer(10, 100, name="hidden_layer_1"),
            Integer(0, 100, name="hidden_layer_2"),
            Integer(0, 100, name="hidden_layer_3"),
            Categorical(["identity", "logistic", "tanh", "relu"], name="activation"),
            Categorical(["lbfgs", "sgd", "adam"], name="solver"),
            Real(1e-4, 0.1, "log-uniform", name="alpha"),
            Categorical(["auto", 8, 16, 32, 64, 128, 256], name="batch_size"),
            Categorical(["constant", "invscaling", "adaptive"], name="learning_rate"),
            Real(1e-3, 0.1, "log-uniform", name="learning_rate_init"),
            Categorical(zero_to_one_exc, name="power_t"),
            Integer(50, 500, name="max_iter"),
        ]

        # Drop layers if sizes are specified by user
        if "hidden_layer_sizes" in self._est_params:
            return [d for d in dimensions if not d.name.startswith("hidden_layer")]
        else:
            return dimensions


class MultinomialNaiveBayes(BaseModel):
    """Multinomial Naive Bayes."""

    acronym = "MNB"
    fullname = "Multinomial Naive Bayes"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "MultinomialNB"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Real(0.01, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]


class OrdinaryLeastSquares(BaseModel):
    """Linear Regression (without regularization)."""

    acronym = "OLS"
    fullname = "Ordinary Least Squares"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "LinearRegression"})


class PassiveAggressive(BaseModel):
    """Passive Aggressive."""

    acronym = "PA"
    fullname = "Passive Aggressive"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict(
        {"class": "PassiveAggressiveClassifier", "reg": "PassiveAggressiveRegressor"}
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


class Perceptron(BaseModel):
    """Linear Perceptron classification."""

    acronym = "Perc"
    fullname = "Perceptron"
    needs_scaling = True
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "Perceptron"})

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        if self._get_param(params, "penalty") != "elasticnet":
            params.pop("l1_ratio", None)

        return params

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [
            Categorical([None, "l2", "l1", "elasticnet"], name="penalty"),
            Categorical([1e-4, 1e-3, 1e-2, 0.1, 1, 10], name="alpha"),
            Categorical([0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90], name="l1_ratio"),
            Integer(500, 1500, name="max_iter"),
            Real(1e-2, 10, "log-uniform", name="eta0"),
        ]


class QuadraticDiscriminantAnalysis(BaseModel):
    """Quadratic Discriminant Analysis."""

    acronym = "QDA"
    fullname = "Quadratic Discriminant Analysis"
    needs_scaling = False
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "discriminant_analysis"
    _estimators = CustomDict({"class": "QuadraticDiscriminantAnalysis"})

    @staticmethod
    def get_dimensions():
        """Return a list of the bounds for the hyperparameters."""
        return [Categorical(zero_to_one_inc, name="reg_param")]


class RadiusNearestNeighbors(BaseModel):
    """Radius Nearest Neighbors."""

    acronym = "RNN"
    fullname = "Radius Nearest Neighbors"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn"]

    _module = "neighbors"
    _estimators = CustomDict(
        {"class": "RadiusNeighborsClassifier", "reg": "RadiusNeighborsRegressor"}
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distances = None

    @property
    def distances(self):
        """Return distances between a random subsample of rows."""
        if self._distances is None:
            # If called only for estimator, there's no data to calculate
            # distances so return the estimator's default radius value: 1
            if hasattr(self.T, "_branches"):
                numerical_cols = self.X_train.select_dtypes("number")
                self._distances = cdist(
                    numerical_cols.sample(
                        n=min(50, len(self.X_train)),
                        random_state=self.T.random_state,
                    ),
                    numerical_cols.sample(
                        n=min(50, len(self.X_train)),
                        random_state=self.T.random_state,
                    ),
                ).flatten()
            else:
                self._distances = 1

        return self._distances

    def _get_default_params(self):
        """Custom method to return a valid radius."""
        x0 = super()._get_default_params()

        # Replace sklearn's default value for the mean of the distances
        x0.replace_value("radius", np.mean(self.distances))

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


class RandomForest(BaseModel):
    """Random Forest."""

    acronym = "RF"
    fullname = "Random Forest"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "RandomForestClassifier", "reg": "RandomForestRegressor"}
    )

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        if not self._get_param(params, "bootstrap"):
            params.pop("max_samples", None)

        return params

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self.T.goal == "class":
            criterion = ["gini", "entropy"]
        elif self._gpu:
            criterion = ["mse", "poisson", "gamma", "inverse_gaussian"]
        else:
            criterion = ["squared_error", "absolute_error", "poisson"]

        dimensions = [
            Integer(10, 500, name="n_estimators"),
            Categorical(criterion, name="criterion"),
            Categorical([None, *range(1, 17)], name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 20, name="min_samples_leaf"),
            Categorical(
                categories=["sqrt", "log2", *half_to_one_exc, None],
                name="max_features",
            ),
            Categorical([True, False], name="bootstrap"),
            Categorical([None, *half_to_one_exc], name="max_samples"),
        ]

        if self._gpu:
            dimensions[1].name = "split_criterion"
            dimensions[2].categories = list(range(1, 17))
            dimensions[5].categories = ["auto", "sqrt", "log2", *half_to_one_exc]
            dimensions[7].categories = half_to_one_exc
        else:
            dimensions.append(Real(0, 0.035, name="ccp_alpha"))

        return dimensions


class Ridge(BaseModel):
    """Linear least squares with l2 regularization."""

    acronym = "Ridge"
    fullname = "Ridge Estimator"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "RidgeClassifier", "reg": "Ridge"})

    def get_dimensions(self):
        """Return a list of the bounds for the hyperparameters."""
        if self._gpu:
            solvers = ["eig", "svd", "cd"]
        else:
            solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]

        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(solvers, name="solver"),
        ]


class StochasticGradientDescent(BaseModel):
    """Stochastic Gradient Descent."""

    acronym = "SGD"
    fullname = "Stochastic Gradient Descent"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "SGDClassifier", "reg": "SGDRegressor"})

    def get_parameters(self, x):
        """Return a dictionary of the model's hyperparameters."""
        params = super().get_parameters(x)

        if self._get_param(params, "penalty") != "elasticnet":
            params.pop("l1_ratio", None)

        if self._get_param(params, "learning_rate") == "optimal":
            params.pop("eta0", None)

        return params

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
            Real(1e-5, 1.0, "log-uniform", name="alpha"),
            Categorical([0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90], name="l1_ratio"),
            Integer(500, 1500, name="max_iter"),
            Real(1e-4, 1.0, "log-uniform", name="epsilon"),
            Categorical(learning_rate, name="learning_rate"),
            Real(1e-2, 10, "log-uniform", name="eta0"),
            Categorical(zero_to_one_exc, name="power_t"),
            Categorical([True, False], name="average"),
        ]


class XGBoost(BaseModel):
    """Extreme Gradient Boosting."""

    acronym = "XGB"
    fullname = "XGBoost"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = []

    _module = "xgboost"
    _estimators = CustomDict({"class": "XGBClassifier", "reg": "XGBRegressor"})

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self.T.random_state is None:  # XGBoost can't handle random_state to be None
            random_state = params.pop("random_state", random.randint(0, 1e5))
        else:
            random_state = params.pop("random_state", self.T.random_state)
        return self.est_class(
            use_label_encoder=params.pop("use_label_encoder", False),
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            tree_method=params.pop("tree_method", "gpu_hist" if self._gpu else None),
            gpu_id=self.T._device_id,
            verbosity=params.pop("verbosity", 0),
            random_state=random_state,
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
            Integer(1, 20, name="max_depth"),
            Real(0, 1.0, name="gamma"),
            Integer(1, 10, name="min_child_weight"),
            Categorical(half_to_one_inc, name="subsample"),
            Categorical([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], name="colsample_bytree"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_alpha"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_lambda"),
        ]


# Ensembles ======================================================== >>

class Stacking(BaseModel):
    """Stacking ensemble."""

    acronym = "Stack"
    fullname = "Stacking"
    needs_scaling = False

    _module = "atom.ensembles"
    _estimators = CustomDict(
        {"class": "StackingClassifier", "reg": "StackingRegressor"}
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._models = kwargs.pop("models")
        self._est_params = kwargs

        if any(m.branch is not self.branch for m in self._models.values()):
            raise ValueError(
                "Invalid value for the models parameter. All "
                "models must have been fitted on the current branch."
            )

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
    """Voting ensemble."""

    acronym = "Vote"
    fullname = "Voting"
    needs_scaling = False

    _module = "atom.ensembles"
    _estimators = CustomDict({"class": "VotingClassifier", "reg": "VotingRegressor"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._models = kwargs.pop("models")
        self._est_params = kwargs

        if any(m.branch is not self.branch for m in self._models.values()):
            raise ValueError(
                "Invalid value for the models parameter. All "
                "models must have been fitted on the current branch."
            )

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
    AdaB=AdaBoost,
    ARD=AutomaticRelevanceDetermination,
    Bag=Bagging,
    BR=BayesianRidge,
    BNB=BernoulliNaiveBayes,
    CatB=CatBoost,
    CatNB=CategoricalNaiveBayes,
    CNB=ComplementNaiveBayes,
    Tree=DecisionTree,
    Dummy=Dummy,
    EN=ElasticNet,
    ET=ExtraTrees,
    GNB=GaussianNaiveBayes,
    GP=GaussianProcess,
    GBM=GradientBoostingMachine,
    Huber=HuberRegression,
    hGBM=HistGBM,
    KNN=KNearestNeighbors,
    kSVM=KernelSVM,
    Lasso=Lasso,
    Lars=LeastAngleRegression,
    LGB=LightGBM,
    LDA=LinearDiscriminantAnalysis,
    lSVM=LinearSVM,
    LR=LogisticRegression,
    MLP=MultilayerPerceptron,
    MNB=MultinomialNaiveBayes,
    OLS=OrdinaryLeastSquares,
    PA=PassiveAggressive,
    Perc=Perceptron,
    QDA=QuadraticDiscriminantAnalysis,
    RNN=RadiusNearestNeighbors,
    RF=RandomForest,
    Ridge=Ridge,
    SGD=StochasticGradientDescent,
    XGB=XGBoost,
)

# List of available ensembles
ENSEMBLES = CustomDict(Stack=Stacking, Vote=Voting)

# List of all models + ensembles
MODELS_ENSEMBLES = CustomDict(**MODELS, **ENSEMBLES)

# Model types as list of all model classes
MODEL_TYPES = Union[tuple(MODELS_ENSEMBLES.values())]

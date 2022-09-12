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

        has_validation: str or None
            Whether the model allows in-training validation. If str,
            name of the estimator's parameter that states the number
            of iterations. If None, no support for in-training
            evaluation.

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

        Methods
        -------
        _get_parameters(self, x):
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

        _get_distributions(self):
            Return a list of the hyperparameter space for optimization.


List of available models:

- "AdaB" for AdaBoost
- "ARD" for AutomatedRelevanceDetermination
- "Bag" for Bagging
- "BR" for BayesianRidge
- "BNB" for BernoulliNB
- "CatB" for CatBoost (if package is available)
- "CatNB" for CategoricalNB
- "CNB" for ComplementNB
- "Tree" for DecisionTree
- "Dummy" for Dummy
- "EN" for ElasticNet
- "ET" for ExtraTrees
- "GNB" for GaussianNB (no hyperparameter tuning)
- "GP" for GaussianProcess (no hyperparameter tuning)
- "GBM" for GradientBoosting
- "Huber" for HuberRegression
- "hGBM" for HistGradientBoosting
- "KNN" for KNearestNeighbors
- "Lasso" for Lasso
- "Lars" for LeastAngleRegression
- "LGB" for LightGBM (if package is available)
- "LDA" for LinearDiscriminantAnalysis
- "lSVM" for LinearSVM
- "LR" for LogisticRegression
- "MLP" for MultiLayerPerceptron
- "MNB" for MultinomialNB
- "OLS" for OrdinaryLeastSquares (no hyperparameter tuning)
- "PA" for PassiveAggressive
- "Perc" for Perceptron
- "QDA" for QuadraticDiscriminantAnalysis
- "RNN" for RadiusNearestNeighbors
- "RF" for RandomForest
- "Ridge" for Ridge
- "SGD" for StochasticGradientDescent
- "SVM" for SupportVectorMachine
- "XGB" for XGBoost (if package is available)

Additionally, ATOM implements two ensemble models:

- "Stack" for Stacking
- "Vote" for Voting

"""

import random
from typing import Union

import numpy as np
from optuna.distributions import CategoricalDistribution as Categorical
from optuna.distributions import FloatDistribution as Float
from optuna.distributions import IntDistribution as Int
from optuna.trial import Trial
from scipy.spatial.distance import cdist
from skopt.space.space import Integer, Real

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

    needs_scaling = None
    accepts_sparse = None
    has_validation = None
    supports_engines = []

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
    def _est_class(self):
        """Return the estimator's class."""
        if callable(self.est):
            return self.est
        else:
            return self.est.__class__

    def _get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if callable(self.est):
            # Add n_jobs and random_state to the estimator (if available)
            for p in ("n_jobs", "random_state"):
                if p in self._sign():
                    params[p] = params.pop(p, getattr(self.T, p))

            return self.est(**params)
        else:
            # If the class has the param and it's the default value, change it
            for p in ("n_jobs", "random_state"):
                if p in self._sign(self.est):
                    if self.est.get_params()[p] == self._sign()[p]._default:
                        params[p] = params.pop(p, getattr(self.T, p))

            return self.est.set_params(**params)


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
    atom.models:GradientBoosting
    atom.models:RandomForest
    atom.models:XGBoost

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="AdaB", metric="f1", verbose=2)

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
    Total time: 0.109s
    -------------------------------------
    AdaBoost --> f1: 0.9722

    ```

    """

    acronym = "AdaB"
    fullname = "AdaBoost"
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict({"class": "AdaBoostClassifier", "reg": "AdaBoostRegressor"})

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        dist = CustomDict(
            n_estimators=Int(50, 500, step=10),
            learning_rate=Float(0.01, 10, log=True),
        )

        if self.T.goal == "class":
            dist["algorithm"] = Categorical(["SAMME.R", "SAMME"])
        else:
            dist["loss"] = Categorical(["linear", "square", "exponential"])

        return dist


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
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
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
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
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
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
        return [
            Integer(100, 1000, name="n_iter"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="alpha_1"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="alpha_2"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="lambda_1"),
            Categorical([1e-6, 1e-4, 1e-2, 1e-1, 1], name="lambda_2"),
        ]


class BernoulliNB(BaseModel):
    """Bernoulli Naive Bayes."""

    acronym = "BNB"
    fullname = "BernoulliNB"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "BernoulliNB"})

    @staticmethod
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
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
    supports_engines = ["catboost"]

    _module = "catboost"
    _estimators = CustomDict({"class": "CatBoostClassifier", "reg": "CatBoostRegressor"})

    def _get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self._est_class(
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
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
        return [
            Integer(20, 500, name="n_estimators"),
            Real(0.01, 1.0, "log-uniform", name="learning_rate"),
            Categorical([None, *range(1, 17)], name="max_depth"),
            Integer(1, 30, name="min_child_samples"),
            Categorical(half_to_one_inc, name="subsample"),
            Categorical([0, 0.01, 0.1, 1, 10, 100], name="reg_lambda"),
        ]


class CategoricalNB(BaseModel):
    """Categorical Naive Bayes."""

    acronym = "CatNB"
    fullname = "CategoricalNB"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "CategoricalNB"})

    @staticmethod
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
        return [
            Real(0.01, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]


class ComplementNB(BaseModel):
    """Complement Naive Bayes."""

    acronym = "CNB"
    fullname = "ComplementNB"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "ComplementNB"})

    @staticmethod
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
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

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
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
    fullname = "Dummy"
    needs_scaling = False
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "dummy"
    _estimators = CustomDict({"class": "DummyClassifier", "reg": "DummyRegressor"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

        if self._get_param(params, "strategy") != "quantile":
            params.pop("quantile", None)
        elif params.get("quantile") is None:
            # quantile can't be None with strategy="quantile" (select value randomly)
            params.replace_value("quantile", random.choice(zero_to_one_inc))

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        if self.T.goal == "class":
            categories = ["most_frequent", "prior", "stratified", "uniform"]
            dist = [Categorical(categories, name="strategy")]
        else:
            dist = [
                Categorical(["mean", "median", "quantile"], name="strategy"),
                Categorical([None, *zero_to_one_inc], name="quantile"),
            ]

        return dist


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
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
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

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

        if not self._get_param(params, "bootstrap"):
            params.pop("max_samples", None)

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
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


class GaussianNB(BaseModel):
    """Gaussian Naive Bayes."""

    acronym = "GNB"
    fullname = "GaussianNB"
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


class GradientBoosting(BaseModel):
    """Gradient Boosting Machine."""

    acronym = "GBM"
    fullname = "GradientBoosting"
    needs_scaling = False
    accepts_sparse = True
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "GradientBoostingClassifier", "reg": "GradientBoostingRegressor"}
    )

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

        if self._get_param(params, "loss") not in ("huber", "quantile"):
            params.pop("alpha", None)

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        dist = []  # Multiclass classification only works with deviance loss
        if self.T.task.startswith("bin"):
            dist.append(Categorical(["deviance", "exponential"], name="loss"))
        elif self.T.task.startswith("reg"):
            loss = ["squared_error", "absolute_error", "huber", "quantile"]
            dist.append(Categorical(loss, name="loss"))

        dist.extend(
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
            dist.append(Categorical(half_to_one_exc, name="alpha"))

        return dist


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
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
        return [
            Real(1, 10, "log-uniform", name="epsilon"),
            Integer(50, 500, name="max_iter"),
            Categorical([1e-4, 1e-3, 1e-2, 1e-1, 1], name="alpha"),
        ]


class HistGradientBoosting(BaseModel):
    """Histogram-based Gradient Boosting Machine."""

    acronym = "hGBM"
    fullname = "HistGradientBoosting"
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

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        dist = []
        if self.T.goal == "reg":
            loss = ["squared_error", "absolute_error", "poisson"]
            dist.append(Categorical(loss, name="loss"))

        dist.extend(
            [
                Real(0.01, 1.0, "log-uniform", name="learning_rate"),
                Integer(10, 500, name="max_iter"),
                Integer(10, 50, name="max_leaf_nodes"),
                Categorical([None, *range(1, 17)], name="max_depth"),
                Integer(10, 30, name="min_samples_leaf"),
                Categorical(zero_to_one_inc, name="l2_regularization"),
            ]
        )

        return dist


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

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        dist = [Integer(1, 100, name="n_neighbors")]

        if not self._gpu:
            dist.extend(
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

        return dist


class SupportVectorMachine(BaseModel):
    """Support Vector Machine."""

    acronym = "SVM"
    fullname = "SupportVectorMachine"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "svm"
    _estimators = CustomDict({"class": "SVC", "reg": "SVR"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

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

    def _get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self._gpu and self.T.goal == "class":
            return self._est_class(
                probability=True,
                random_state=params.pop("random_state", self.T.random_state),
                **params)
        else:
            return super()._get_estimator(**params)

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        dist = [
            Real(1e-3, 100, "log-uniform", name="C"),
            Categorical(["linear", "poly", "rbf", "sigmoid"], name="kernel"),
            Integer(2, 5, name="degree"),
            Categorical(["scale", "auto"], name="gamma"),
            Real(-1.0, 1.0, name="coef0"),
        ]

        if not self._gpu:
            dist.extend(
                [
                    Real(1e-3, 100, "log-uniform", name="epsilon"),
                    Categorical([True, False], name="shrinking"),
                ]
            )

        return dist


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
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(["cyclic", "random"], name="selection"),
        ]


class LeastAngleRegression(BaseModel):
    """Least Angle Regression."""

    acronym = "Lars"
    fullname = "LeastAngleRegression"
    needs_scaling = True
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "Lars"})


class LightGBM(BaseModel):
    """Light Gradient Boosting Machine.

    !!! info
        Using LightGBM's [GPU acceleration][] requires
        [additional software dependencies][lgb_gpu].

    """

    acronym = "LGB"
    fullname = "LightGBM"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["lightgbm"]

    _module = "lightgbm.sklearn"
    _estimators = CustomDict({"class": "LGBMClassifier", "reg": "LGBMRegressor"})

    def _get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        return self._est_class(
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
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
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

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

        if self._get_param(params, "solver") == "svd":
            params.pop("shrinkage", None)

        return params

    @staticmethod
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
        return [
            Categorical(["svd", "lsqr", "eigen"], name="solver"),
            Categorical([None, "auto", *half_to_one_inc], name="shrinkage"),
        ]


class LinearSVM(BaseModel):
    """Linear Support Vector Machine."""

    acronym = "lSVM"
    fullname = "LinearSVM"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "cuml"]

    _module = "svm"
    _estimators = CustomDict({"class": "LinearSVC", "reg": "LinearSVR"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

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

    def _get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self._gpu and self.T.goal == "class":
            return self._est_class(probability=True, **params)
        else:
            return super()._get_estimator(**params)

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        dist = []
        if self.T.goal == "class":
            loss = ["hinge", "squared_hinge"]
            dist.append(Categorical(["l1", "l2"], name="penalty"))
        else:
            loss = ["epsilon_insensitive", "squared_epsilon_insensitive"]

        dist.extend(
            [
                Categorical(loss, name="loss"),
                Real(1e-3, 100, "log-uniform", name="C"),
            ]
        )

        if not self._gpu:
            dist.append(Categorical([True, False], name="dual"))

        return dist


class LogisticRegression(BaseModel):
    """Logistic Regression.

    Logistic regression, despite its name, is a linear model for
    classification rather than regression. Logistic regression is also
    known in the literature as logit regression, maximum-entropy
    classification (MaxEnt) or the log-linear classifier. In this model,
    the probabilities describing the possible outcomes of a single trial
    are modeled using a logistic function.

    Corresponding estimators are:

    - [LogisticRegression][] for classification tasks.

    Read more in sklearn's [documentation][lrdocs].

    See Also
    --------
    atom.models:GaussianProcess
    atom.models:LinearDiscriminantAnalysis
    atom.models:PassiveAggressive

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="RF", metric="f1", verbose=2)

    Training ========================= >>
    Models: LR
    Metric: f1


    Results for LogisticRegression:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.9895
    Test evaluation --> f1: 0.993
    Time elapsed: 0.028s
    -------------------------------------------------
    Total time: 0.028s


    Final results ==================== >>
    Total time: 0.028s
    -------------------------------------
    LogisticRegression --> f1: 0.993

    ```

    """

    acronym = "LR"
    fullname = "LogisticRegression"
    needs_scaling = True
    accepts_sparse = True
    has_validation = False
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "LogisticRegression"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

        # Limitations on penalty + solver combinations
        penalty = self._get_param("penalty", params)
        solver = self._get_param("solver", params)
        cond_1 = penalty == "none" and solver == "liblinear"
        cond_2 = penalty == "l1" and solver not in ("liblinear", "saga")
        cond_3 = penalty == "elasticnet" and solver != "saga"

        if cond_1 or cond_2 or cond_3:
            params.replace_value("penalty", "l2")  # Change to default value

        if self._get_param("penalty", params) != "elasticnet":
            params.pop("l1_ratio", None)

        if self._get_param("penalty", params) == "none":
            params.pop("C", None)

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        dist = CustomDict(
            penalty=Categorical(["l1", "l2", "elasticnet", "none"]),
            C=Float(1e-3, 100, log=True),
            solver=Categorical(["lbfgs", "newton-cg", "liblinear", "sag", "saga"]),
            max_iter=Int(100, 1000, step=10),
            l1_ratio=Categorical(zero_to_one_exc),
        )

        if self._gpu:
            dist.pop("solver")
            if self.T.engine == "sklearnex":
                dist.pop("penalty")  # Only 'l2' is supported
        elif self.T.engine == "sklearnex":
            dist["solver"] = Categorical(["lbfgs", "newton-cg"])

        return dist


class MultiLayerPerceptron(BaseModel):
    """Multi-layer Perceptron."""

    acronym = "MLP"
    fullname = "MultiLayerPerceptron"
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

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

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

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        dist = [
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
            return [d for d in dist if not d.name.startswith("hidden_layer")]
        else:
            return dist


class MultinomialNB(BaseModel):
    """Multinomial Naive Bayes.

    MultinomialNB implements the Naive Bayes algorithm for multinomially
    distributed data, and is one of the two classic Naive Bayes variants
    used in text classification (where the data are typically
    represented as word vector counts, although tf-idf vectors are also
    known to work well in practice).

    Corresponding estimators are:

    - [MultinomialNB][] for classification tasks.

    Read more in sklearn's [documentation][mnbdocs].

    See Also
    --------
    atom.models:BernoulliNB
    atom.models:ComplementNB
    atom.models:GaussianNB

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="MNB", metric="f1", verbose=2)

    Training ========================= >>
    Models: MNB
    Metric: f1


    Results for MultinomialNB:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.916
    Test evaluation --> f1: 0.9371
    Time elapsed: 0.011s
    -------------------------------------------------
    Total time: 0.011s


    Final results ==================== >>
    Total time: 0.011s
    -------------------------------------
    MultinomialNB --> f1: 0.9371

    ```

    """

    acronym = "MNB"
    fullname = "MultinomialNB"
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "MultinomialNB"})

    @staticmethod
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
        return CustomDict(
            alpha=Float(0.01, 10, log=True),
            fit_prior=Categorical([True, False]),
        )


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

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
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

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

        if self._get_param(params, "penalty") != "elasticnet":
            params.pop("l1_ratio", None)

        return params

    @staticmethod
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
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
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
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

    def _get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self.T.goal == "class":
            return self._est_class(
                outlier_label=params.pop("outlier_label", "most_frequent"),
                radius=params.pop("radius", np.mean(self.distances)),
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )
        else:
            return self._est_class(
                radius=params.pop("radius", np.mean(self.distances)),
                n_jobs=params.pop("n_jobs", self.T.n_jobs),
                **params,
            )

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        return [
            Real(min(self.distances), max(self.distances), name="radius"),
            Categorical(["uniform", "distance"], name="weights"),
            Categorical(["auto", "ball_tree", "kd_tree", "brute"], name="algorithm"),
            Integer(20, 40, name="leaf_size"),
            Integer(1, 2, name="p"),
        ]


class RandomForest(BaseModel):
    """Random Forest.

    Random forests are an ensemble learning method that operate by
    constructing a multitude of decision trees at training time and
    outputting the class that is the mode of the classes
    (classification) or mean prediction (regression) of the individual
    trees. Random forests correct for decision trees' habit of
    overfitting to their training set.

    Corresponding estimators are:

    - [RandomForestClassifier][] for classification tasks.
    - [RandomForestRegressor][] for regression tasks.

    Read more in sklearn's [documentation][adabdocs].

    !!! warning
        cuML's implementation of [RandomForestClassifier][cumlrf] only
        supports predictions on dtype `float32`. Convert all dtypes
        before calling atom's [run][atomclassifier-run] method to avoid
        exceptions.

    See Also
    --------
    atom.models:DecisionTree
    atom.models:ExtraTrees
    atom.models:HistGradientBoosting

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="RF", metric="f1", verbose=2)

    Training ========================= >>
    Models: RF
    Metric: f1


    Results for Random Forest:
    Fit ---------------------------------------------
    Train evaluation --> f1: 1.0
    Test evaluation --> f1: 0.9722
    Time elapsed: 2.815s
    -------------------------------------------------
    Total time: 2.815s


    Final results ==================== >>
    Total time: 2.968s
    -------------------------------------
    Random Forest --> f1: 0.9722

    ```

    """

    acronym = "RF"
    fullname = "RandomForest"
    needs_scaling = False
    accepts_sparse = True
    has_validation = False
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "RandomForestClassifier", "reg": "RandomForestRegressor"}
    )

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

        if not self._get_param("bootstrap", params):
            params.pop("max_samples", None)

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        if self.T.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            if self.T.engine == "cuml":
                criterion = ["mse", "poisson", "gamma", "inverse_gaussian"]
            else:
                criterion = ["squared_error", "absolute_error", "poisson"]

        dist = CustomDict(
            n_estimators=Int(10, 500, step=10),
            criterion=Categorical(criterion),
            max_depth=Categorical([None, *range(1, 17)]),
            min_samples_split=Int(2, 20),
            min_samples_leaf=Int(1, 20),
            max_features=Categorical([None, "sqrt", "log2", *half_to_one_exc]),
            bootstrap=Categorical([True, False]),
            max_samples=Categorical([None, *half_to_one_exc]),
            ccp_alpha=Float(0, 0.035, step=0.005),
        )

        if self.T.engine == "sklearnex":
            dist.pop("criterion")
            dist.pop("ccp_alpha")
        elif self.T.engine == "cuml":
            dist.replace_key("criterion", "split_criterion")
            dist["max_depth"].choices = tuple(range(1, 17))
            dist["max_features"].choices = tuple(["sqrt", "log2", *half_to_one_exc])
            dist["max_samples"].choices = tuple(half_to_one_exc)
            dist.pop("ccp_alpha")

        return dist


class Ridge(BaseModel):
    """Linear least squares with l2 regularization."""

    acronym = "Ridge"
    fullname = "Ridge Estimator"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "RidgeClassifier", "reg": "Ridge"})

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        if self._gpu:
            solvers = ["eig", "svd", "cd"]
        else:
            solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]

        return [
            Real(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(solvers, name="solver"),
        ]


class StochasticGradientDescent(BaseModel):
    """Stochastic Gradient Descent.

    Stochastic Gradient Descent is a simple yet very efficient approach
    to fitting linear classifiers and regressors under convex loss
    functions. Even though SGD has been around in the machine learning
    community for a long time, it has received a considerable amount of
    attention just recently in the context of large-scale learning.

    Corresponding estimators are:

    - [SGDClassifier][] for classification tasks.
    - [SGDRegressor][] for regression tasks.

    Read more in sklearn's [documentation][sgddocs].

    See Also
    --------
    atom.models:MultiLayerPerceptron
    atom.models:PassiveAggressive
    atom.models:SupportVectorMachine

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="SGD", metric="f1", verbose=2)

    Training ========================= >>
    Models: SGD
    Metric: f1


    Results for StochasticGradientDescent:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.993
    Test evaluation --> f1: 0.9929
    Time elapsed: 4.561s
    -------------------------------------------------
    Total time: 4.561s


    Final results ==================== >>
    Total time: 4.562s
    -------------------------------------
    StochasticGradientDescent --> f1: 0.9929

    ```

    """

    acronym = "SGD"
    fullname = "StochasticGradientDescent"
    needs_scaling = True
    accepts_sparse = True
    has_validation = "max_iter"
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "SGDClassifier", "reg": "SGDRegressor"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters."""
        params = super()._get_parameters(trial)

        if self._get_param("penalty", params) != "elasticnet":
            params.pop("l1_ratio", None)

        if self._get_param("learning_rate", params) == "optimal":
            params.pop("eta0", None)

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions."""
        loss = [
            "hinge",
            "log_loss",
            "log",
            "modified_huber",
            "squared_hinge",
            "perceptron",
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ]

        return CustomDict(
            loss=Categorical(loss if self.T.goal == "class" else loss[-4:]),
            penalty=Categorical(["none", "l1", "l2", "elasticnet"]),
            alpha=Float(1e-5, 1.0, log=True),
            l1_ratio=Float(0.05, 0.9, step=0.05),
            max_iter=Int(500, 1500, step=50),
            epsilon=Float(1e-4, 1.0, log=True),
            learning_rate=Categorical(["constant", "invscaling", "optimal", "adaptive"]),
            eta0=Float(1e-2, 10, log=True),
            power_t=Float(0.1, 0.9, step=0.1),
            average=Categorical([True, False]),
        )


class XGBoost(BaseModel):
    """Extreme Gradient Boosting."""

    acronym = "XGB"
    fullname = "XGBoost"
    needs_scaling = True
    accepts_sparse = True
    supports_engines = ["xgboost"]

    _module = "xgboost"
    _estimators = CustomDict({"class": "XGBClassifier", "reg": "XGBRegressor"})

    def _get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        if self.T.random_state is None:  # XGBoost can't handle random_state to be None
            random_state = params.pop("random_state", random.randint(0, 1e5))
        else:
            random_state = params.pop("random_state", self.T.random_state)
        return self._est_class(
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
    def _get_distributions():
        """Get the predefined hyperparameter distributions."""
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
    _estimators = CustomDict({"class": "StackingClassifier", "reg": "StackingRegressor"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._models = kwargs.pop("models")
        self._est_params = kwargs

        if any(m.branch is not self.branch for m in self._models.values()):
            raise ValueError(
                "Invalid value for the models parameter. All "
                "models must have been fitted on the current branch."
            )

    def _get_estimator(self, **params):
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

        return self._est_class(
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

    def _get_estimator(self, **params):
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

        return self._est_class(
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
    BNB=BernoulliNB,
    CatB=CatBoost,
    CatNB=CategoricalNB,
    CNB=ComplementNB,
    Tree=DecisionTree,
    Dummy=Dummy,
    EN=ElasticNet,
    ET=ExtraTrees,
    GNB=GaussianNB,
    GP=GaussianProcess,
    GBM=GradientBoosting,
    Huber=HuberRegression,
    hGBM=HistGradientBoosting,
    KNN=KNearestNeighbors,
    Lasso=Lasso,
    Lars=LeastAngleRegression,
    LGB=LightGBM,
    LDA=LinearDiscriminantAnalysis,
    lSVM=LinearSVM,
    LR=LogisticRegression,
    MLP=MultiLayerPerceptron,
    MNB=MultinomialNB,
    OLS=OrdinaryLeastSquares,
    PA=PassiveAggressive,
    Perc=Perceptron,
    QDA=QuadraticDiscriminantAnalysis,
    RNN=RadiusNearestNeighbors,
    RF=RandomForest,
    Ridge=Ridge,
    SGD=StochasticGradientDescent,
    SVM=SupportVectorMachine,
    XGB=XGBoost,
)

# List of available ensembles
ENSEMBLES = CustomDict(Stack=Stacking, Vote=Voting)

# List of all models + ensembles
MODELS_ENSEMBLES = CustomDict(**MODELS, **ENSEMBLES)

# Model types as list of all model classes
MODEL_TYPES = Union[tuple(MODELS_ENSEMBLES.values())]

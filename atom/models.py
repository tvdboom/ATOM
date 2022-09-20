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
            Acronym of the model's name.

        needs_scaling: bool
            Whether the model needs scaled features.

        accepts_sparse: bool
            Whether the model has native support for sparse matrices.

        has_validation: str or None
            Whether the model allows in-training validation. If str,
            name of the estimator's parameter that states the number
            of iterations. If None, no support for in-training
            validation.

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
        _get_parameters(self, x) -> CustomDict:
            Return the trial's suggestions with rounded decimals and
            (optionally) custom changes to the params. Don't implement
            if the parent's implementation is sufficient.

        _fit_estimator(estimator, data, est_params_fit, validation, trial):
            This method is called to fit the estimator. Implement only
            to customize the fit.

        _get_distributions(self) -> CustomDict:
            Return a list of the hyperparameter distributions for
            optimization.


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

from random import choice
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from optuna.distributions import CategoricalDistribution as Categorical
from optuna.distributions import FloatDistribution as Float
from optuna.distributions import IntDistribution as Int
from optuna.exceptions import TrialPruned
from optuna.integration import (
    CatBoostPruningCallback, LightGBMPruningCallback, XGBoostPruningCallback,
)
from optuna.trial import Trial
from copy import deepcopy
from atom.basemodel import BaseModel
from atom.pipeline import Pipeline
from atom.utils import (
    CatBMetric, CustomDict, LGBMetric, Predictor, XGBMetric, create_acronym,
)


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

        # If no acronym is provided, use capital letters in the class' name
        if hasattr(self.est, "acronym"):
            self.acronym = self.est.acronym
        else:
            self.acronym = create_acronym(self._fullname)

        self.needs_scaling = getattr(self.est, "needs_scaling", False)
        super().__init__(*args)

    @property
    def _fullname(self) -> str:
        """Return the estimator's class name."""
        return self._est_class.__name__

    @property
    def _est_class(self):
        """Return the estimator's class."""
        if callable(self.est):
            return self.est
        else:
            return self.est.__class__

    def _get_est(self, **params) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
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
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict({"class": "AdaBoostClassifier", "reg": "AdaBoostRegressor"})

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
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
    needs_scaling = True
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "ARDRegression"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            n_iter=Int(100, 1000, step=10),
            alpha_1=Float(1e-4, 1, log=True),
            alpha_2=Float(1e-4, 1, log=True),
            lambda_1=Float(1e-4, 1, log=True),
            lambda_2=Float(1e-4, 1, log=True),
        )


class Bagging(BaseModel):
    """Bagging model (with decision tree as base estimator).

    Bagging uses an ensemble meta-estimator that fits base predictors
    on random subsets of the original dataset and then aggregate their
    individual predictions (either by voting or by averaging) to form a
    final prediction. Such a meta-estimator can typically be used as a
    way to reduce the variance of a black-box estimator by introducing
    randomization into its construction procedure and then making an
    ensemble out of it.

    Corresponding estimators are:

    - [BaggingClassifier][] for classification tasks.
    - [BaggingRegressor][] for regression tasks.

    Read more in sklearn's [documentation][bagdocs].

    See Also
    --------
    atom.models:DecisionTree
    atom.models:LogisticRegression
    atom.models:RandomForest

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="Bag", metric="f1", verbose=2)

    Training ========================= >>
    Models: Bag
    Metric: f1


    Results for Bagging:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.9965
    Test evaluation --> f1: 0.9722
    Time elapsed: 0.051s
    -------------------------------------------------
    Total time: 0.051s


    Final results ==================== >>
    Total time: 0.051s
    -------------------------------------
    Bagging --> f1: 0.9722

    ```

    """

    acronym = "Bag"
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict({"class": "BaggingClassifier", "reg": "BaggingRegressor"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            n_estimators=Int(10, 500, step=10),
            max_samples=Categorical(half_to_one_inc),
            max_features=Categorical(half_to_one_inc),
            bootstrap=Categorical([True, False]),
            bootstrap_features=Categorical([True, False]),
        )


class BayesianRidge(BaseModel):
    """Bayesian ridge regression."""

    acronym = "BR"
    needs_scaling = True
    accepts_sparse = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "BayesianRidge"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            n_iter=Int(100, 1000, step=10),
            alpha_1=Float(1e-4, 1, log=True),
            alpha_2=Float(1e-4, 1, log=True),
            lambda_1=Float(1e-4, 1, log=True),
            lambda_2=Float(1e-4, 1, log=True),
        )


class BernoulliNB(BaseModel):
    """Bernoulli Naive Bayes.

    BernoulliNB implements the Naive Bayes algorithm for multivariate
    Bernoulli models. Like [MultinomialNB][], this classifier is
    suitable for discrete data. The difference is that while MNB works
    with occurrence counts, BNB is designed for binary/boolean features.

    Corresponding estimators are:

    - [BernoulliNB][bernoullinbclass] for classification tasks.

    Read more in sklearn's [documentation][bnbdocs].

    See Also
    --------
    atom.models:ComplementNB
    atom.models:CategoricalNB
    atom.models:MultinomialNB

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="BNB", metric="f1", verbose=2)

    Training ========================= >>
    Models: BNB
    Metric: f1


    Results for BernoulliNB:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.7709
    Test evaluation --> f1: 0.7717
    Time elapsed: 0.014s
    -------------------------------------------------
    Total time: 0.014s

    Final results ==================== >>
    Total time: 0.015s
    -------------------------------------
    BernoulliNB --> f1: 0.7717

    ```

    """

    acronym = "BNB"
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "BernoulliNB"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            alpha=Float(0.01, 10, log=True),
            fit_prior=Categorical([True, False]),
        )


class CatBoost(BaseModel):
    """Categorical Boosting Machine.

    CatBoost is a machine learning method based on gradient boosting
    over decision trees. Main advantages of CatBoost:

    - Superior quality when compared with other GBDT models on many datasets.
    - Best in class prediction speed.

    Corresponding estimators are:

    - [CatBoostClassifier][] for classification tasks.
    - [CatBoostRegressor][] for regression tasks.

    Read more in sklearn's [documentation][catbdocs].

    !!! note
        ATOM uses CatBoost's `n_estimators` parameter instead of
        `iterations` to indicate the number of trees to fit. This is
        done to have consistent naming with the [XGBoost][] and
        [LightGBM][] models.

    See Also
    --------
    atom.models:GradientBoosting
    atom.models:LightGBM
    atom.models:XGBoost

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="CatB", metric="f1", verbose=2)

    Training ========================= >>
    Models: CatB
    Metric: f1


    Results for CatBoost:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.981
    Test evaluation --> f1: 0.9859
    Time elapsed: 14.789s
    -------------------------------------------------
    Total time: 14.789s


    Final results ==================== >>
    Total time: 14.789s
    -------------------------------------
    CatBoost --> f1: 0.9859

    ```

    """

    acronym = "CatB"
    needs_scaling = True
    accepts_sparse = True
    has_validation = "n_estimators"
    supports_engines = ["catboost"]

    _module = "catboost"
    _estimators = CustomDict({"class": "CatBoostClassifier", "reg": "CatBoostRegressor"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self._get_param("bootstrap_type", params) == "Bernoulli":
            params.pop("bagging_temperature", None)
        elif self._get_param("bootstrap_type", params) == "Bayesian":
            params.pop("subsample", None)

        return params

    def _get_est(self, **params) -> Predictor:
        """Get the estimator instance.

        Parameters
        ----------
        **params
            Unpacked hyperparameters for the estimator.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        return self._est_class(
            eval_metric=CatBMetric(self.T._metric[0], task=self.T.task),
            train_dir=params.pop("train_dir", ""),
            allow_writing_files=params.pop("allow_writing_files", False),
            thread_count=params.pop("n_jobs", self.T.n_jobs),
            task_type=params.pop("task_type", "GPU" if self._gpu else "CPU"),
            devices=str(self.T._device_id),
            verbose=params.pop("verbose", False),
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def _fit_estimator(
        self,
        estimator: Predictor,
        data: Tuple[pd.DataFrame, pd.Series],
        est_params_fit: dict,
        validation: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        trial: Optional[Trial] = None,
    ):
        """Fit the estimator and perform in-training validation.

        Parameters
        ----------
        estimator: Predictor
            Instance to fit.

        data: tuple
            Training data of the form (X, y).

        validation: tuple or None
            Validation data of the form (X, y). If None, no validation
            is performed.

        est_params_fit: dict
            Additional parameters for the estimator's fit method.

        trial: [Trial][] or None
            Active trial (during hyperparameter tuning).

        Returns
        -------
        Predictor
            Fitted instance.

        """
        params = est_params_fit.copy()

        callbacks = params.pop("callbacks", [])
        if trial and len(self.T._metric) == 1:
            callbacks.append(cb := CatBoostPruningCallback(trial, "CatBMetric"))

        estimator.fit(
            *data,
            eval_set=validation,
            callbacks=callbacks,
            **params,
        )

        # Create evals attribute with train and validation scores
        m = self.T._metric[0].name
        self._evals[f"{m}_train"] = estimator.evals_result_["learn"]["CatBMetric"]
        self._evals[f"{m}_test"] = estimator.evals_result_["validation"]["CatBMetric"]

        if trial and len(self.T._metric) == 1 and cb._pruned:
            # Hacky solution to add the pruned step to the output
            steps = estimator.get_params()[self.has_validation]
            p = trial.storage.get_trial_user_attrs(trial.number)["params"]
            p[self.has_validation] = f"{len(self.evals[f'{m}_train'])}/{steps}"

            trial.set_user_attr("estimator", estimator)
            raise TrialPruned(cb._message)

        return estimator

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            n_estimators=Int(20, 500, step=10),
            learning_rate=Float(0.01, 1.0, log=True),
            max_depth=Categorical([None, *range(1, 17)]),
            min_child_samples=Int(1, 30),
            bootstrap_type=Categorical(["Bayesian", "Bernoulli"]),
            bagging_temperature=Float(0, 10),
            subsample=Categorical(half_to_one_inc),
            reg_lambda=Float(0.001, 100, log=True),
        )


class CategoricalNB(BaseModel):
    """Categorical Naive Bayes."""

    acronym = "CatNB"
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "CategoricalNB"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return [
            Float(0.01, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
        ]


class ComplementNB(BaseModel):
    """Complement Naive Bayes."""

    acronym = "CNB"
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "ComplementNB"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return [
            Float(0.01, 10, "log-uniform", name="alpha"),
            Categorical([True, False], name="fit_prior"),
            Categorical([True, False], name="norm"),
        ]


class DecisionTree(BaseModel):
    """Single Decision Tree."""

    acronym = "Tree"
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "tree"
    _estimators = CustomDict(
        {"class": "DecisionTreeClassifier", "reg": "DecisionTreeRegressor"}
    )

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        if self.T.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error", "friedman_mse", "poisson"]

        return [
            Categorical(criterion, name="criterion"),
            Categorical(["best", "random"], name="splitter"),
            Categorical([None, *range(1, 17)], name="max_depth"),
            Int(2, 20, name="min_samples_split"),
            Int(1, 20, name="min_samples_leaf"),
            Categorical(
                categories=["auto", "sqrt", "log2", *half_to_one_exc, None],
                name="max_features",
            ),
            Float(0, 0.035, name="ccp_alpha"),
        ]


class Dummy(BaseModel):
    """Dummy classifier/regressor."""

    acronym = "Dummy"
    needs_scaling = False
    accepts_sparse = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "dummy"
    _estimators = CustomDict({"class": "DummyClassifier", "reg": "DummyRegressor"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self._get_param(params, "strategy") != "quantile":
            params.pop("quantile", None)
        elif params.get("quantile") is None:
            # quantile can't be None with strategy="quantile" (select value randomly)
            params.replace_value("quantile", choice(zero_to_one_inc))

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
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
    needs_scaling = True
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "ElasticNet"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return [
            Float(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(half_to_one_exc, name="l1_ratio"),
            Categorical(["cyclic", "random"], name="selection"),
        ]


class ExtraTrees(BaseModel):
    """Extremely Randomized Trees."""

    acronym = "ET"
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "ExtraTreesClassifier", "reg": "ExtraTreesRegressor"}
    )

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if not self._get_param(params, "bootstrap"):
            params.pop("max_samples", None)

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        if self.T.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error"]

        return [
            Int(10, 500, name="n_estimators"),
            Categorical(criterion, name="criterion"),
            Categorical([None, *range(1, 17)], name="max_depth"),
            Int(2, 20, name="min_samples_split"),
            Int(1, 20, name="min_samples_leaf"),
            Categorical(
                categories=["sqrt", "log2", *half_to_one_exc, None],
                name="max_features",
            ),
            Categorical([True, False], name="bootstrap"),
            Categorical([None, *half_to_one_exc], name="max_samples"),
            Float(0, 0.035, name="ccp_alpha"),
        ]


class GaussianNB(BaseModel):
    """Gaussian Naive Bayes."""

    acronym = "GNB"
    needs_scaling = False
    accepts_sparse = False
    has_validation = None
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "GaussianNB"})


class GaussianProcess(BaseModel):
    """Gaussian process."""

    acronym = "GP"
    needs_scaling = False
    accepts_sparse = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "gaussian_process"
    _estimators = CustomDict(
        {"class": "GaussianProcessClassifier", "reg": "GaussianProcessRegressor"}
    )


class GradientBoosting(BaseModel):
    """Gradient Boosting Machine."""

    acronym = "GBM"
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "GradientBoostingClassifier", "reg": "GradientBoostingRegressor"}
    )

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self._get_param(params, "loss") not in ("huber", "quantile"):
            params.pop("alpha", None)

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        dist = []  # Multiclass classification only works with deviance loss
        if self.T.task.startswith("bin"):
            dist.append(Categorical(["deviance", "exponential"], name="loss"))
        elif self.T.task.startswith("reg"):
            loss = ["squared_error", "absolute_error", "huber", "quantile"]
            dist.append(Categorical(loss, name="loss"))

        dist.extend(
            [
                Float(0.01, 1.0, "log-uniform", name="learning_rate"),
                Int(10, 500, name="n_estimators"),
                Categorical(half_to_one_inc, name="subsample"),
                Categorical(["friedman_mse", "squared_error"], name="criterion"),
                Int(2, 20, name="min_samples_split"),
                Int(1, 20, name="min_samples_leaf"),
                Int(1, 21, name="max_depth"),
                Categorical(
                    categories=["auto", "sqrt", "log2", *half_to_one_exc, None],
                    name="max_features",
                ),
                Float(0, 0.035, name="ccp_alpha"),
            ]
        )

        if self.T.goal == "reg":
            dist.append(Categorical(half_to_one_exc, name="alpha"))

        return dist


class HuberRegression(BaseModel):
    """Huber regression."""

    acronym = "Huber"
    needs_scaling = True
    accepts_sparse = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "HuberRegressor"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return [
            Float(1, 10, "log-uniform", name="epsilon"),
            Int(50, 500, name="max_iter"),
            Categorical([1e-4, 1e-3, 1e-2, 1e-1, 1], name="alpha"),
        ]


class HistGradientBoosting(BaseModel):
    """Histogram-based Gradient Boosting Machine."""

    acronym = "hGBM"
    needs_scaling = False
    accepts_sparse = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict(
        {
            "class": "HistGradientBoostingClassifier",
            "reg": "HistGradientBoostingRegressor",
        }
    )

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        dist = []
        if self.T.goal == "reg":
            loss = ["squared_error", "absolute_error", "poisson"]
            dist.append(Categorical(loss, name="loss"))

        dist.extend(
            [
                Float(0.01, 1.0, "log-uniform", name="learning_rate"),
                Int(10, 500, name="max_iter"),
                Int(10, 50, name="max_leaf_nodes"),
                Categorical([None, *range(1, 17)], name="max_depth"),
                Int(10, 30, name="min_samples_leaf"),
                Categorical(zero_to_one_inc, name="l2_regularization"),
            ]
        )

        return dist


class KNearestNeighbors(BaseModel):
    """K-Nearest Neighbors.

    K-Nearest Neighbors, as the name clearly indicates, implements the
    k-nearest neighbors vote. For regression, the target is predicted
    by local interpolation of the targets associated of the nearest
    neighbors in the training set.

    Corresponding estimators are:

    - [KNeighborsClassifier][] for classification tasks.
    - [KNeighborsRegressor][] for classification tasks.

    Read more in sklearn's [documentation][knndocs].

    See Also
    --------
    atom.models:LinearDiscriminantAnalysis
    atom.models:QuadraticDiscriminantAnalysis
    atom.models:RadiusNearestNeighbors

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="KNN", metric="f1", verbose=2)

    Training ========================= >>
    Models: KNN
    Metric: f1


    Results for KNearestNeighbors:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.9845
    Test evaluation --> f1: 0.9861
    Time elapsed: 0.187s
    -------------------------------------------------
    Total time: 0.187s


    Final results ==================== >>
    Total time: 0.189s
    -------------------------------------
    KNearestNeighbors --> f1: 0.9861

    ```

    """

    acronym = "KNN"
    needs_scaling = True
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "neighbors"
    _estimators = CustomDict(
        {"class": "KNeighborsClassifier", "reg": "KNeighborsRegressor"}
    )

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        dist = CustomDict(
            n_neighbors=Int(1, 100),
            weights=Categorical(["uniform", "distance"]),
            algorithm=Categorical(["auto", "ball_tree", "kd_tree", "brute"]),
            leaf_size=Int(20, 40),
            p=Int(1, 2),
        )

        if self._gpu:
            dist.pop("algorithm")  # Only 'brute' is supported
            if self.T.engine == "cuml":
                dist.pop("weights")  # Only 'uniform' is supported
                dist.pop("leaf_size")
                dist.pop("p")

        return dist


class SupportVectorMachine(BaseModel):
    """Support Vector Machine.

    The implementation of the Support Vector Machine is based on libsvm.
    The fit time scales at least quadratically with the number of
    samples and may be impractical beyond tens of thousands of samples.
    For large datasets consider using a [LinearSVM][] or a
    [StochasticGradientDescent][] model instead.

    Corresponding estimators are:

    - [SVC][] for classification tasks.
    - [SVR][] for classification tasks.

    Read more in sklearn's [documentation][svmdocs].

    See Also
    --------
    atom.models:LinearSVM
    atom.models:MultiLayerPerceptron
    atom.models:StochasticGradientDescent

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="SVM", metric="f1", verbose=2)

    Training ========================= >>
    Models: SVM
    Metric: f1


    Results for SupportVectorMachine:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.9896
    Test evaluation --> f1: 0.9645
    Time elapsed: 0.027s
    -------------------------------------------------
    Total time: 0.027s


    Final results ==================== >>
    Total time: 0.027s
    -------------------------------------
    SupportVectorMachine --> f1: 0.9645

    ```

    """

    acronym = "SVM"
    needs_scaling = True
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "svm"
    _estimators = CustomDict({"class": "SVC", "reg": "SVR"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self.T.goal == "class":
            params.pop("epsilon", None)

        kernel = self._get_param("kernel", params)
        if kernel == "poly":
            params.replace_value("gamma", "scale")  # Crashes in combination with "auto"
        else:
            params.pop("degree", None)

        if kernel not in ("rbf", "poly", "sigmoid"):
            params.pop("gamma", None)

        if kernel not in ("poly", "sigmoid"):
            params.pop("coef0", None)

        return params

    def _get_est(self, **params) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        if self.T.engine == "cuml" and self.T.goal == "class":
            return self._est_class(
                probability=params.pop("probability", True),
                random_state=params.pop("random_state", self.T.random_state),
                **params)
        else:
            return super()._get_est(**params)

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        dist = CustomDict(
            C=Float(1e-3, 100, log=True),
            kernel=Categorical(["linear", "poly", "rbf", "sigmoid"]),
            degree=Int(2, 5),
            gamma=Categorical(["scale", "auto"]),
            coef0=Float(-1.0, 1.0),
            epsilon=Float(1e-3, 100, log=True),
            shrinking=Categorical([True, False]),
        )

        if self.T.engine == "cuml":
            dist.pop("epsilon")
            dist.pop("shrinking")

        return dist


class Lasso(BaseModel):
    """Linear Regression with lasso regularization."""

    acronym = "Lasso"
    needs_scaling = True
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "sklearnex"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "Lasso"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return [
            Float(1e-3, 10, "log-uniform", name="alpha"),
            Categorical(["cyclic", "random"], name="selection"),
        ]


class LeastAngleRegression(BaseModel):
    """Least Angle Regression."""

    acronym = "Lars"
    needs_scaling = True
    accepts_sparse = False
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "Lars"})


class LightGBM(BaseModel):
    """Light Gradient Boosting Machine.

    LightGBM is a gradient boosting model that uses tree based learning
    algorithms. It is designed to be distributed and efficient with the
    following advantages:

    - Faster training speed and higher efficiency.
    - Lower memory usage.
    - Better accuracy.
    - Capable of handling large-scale data.

    Corresponding estimators are:

    - [LGBMClassifier][] for classification tasks.
    - [LGBMRegressor][] for regression tasks.

    Read more in sklearn's [documentation][lgbdocs].

    !!! info
        Using LightGBM's [GPU acceleration][] requires
        [additional software dependencies][lgb_gpu].

    See Also
    --------
    atom.models:CatBoost
    atom.models:GradientBoosting
    atom.models:XGBoost

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="LGB", metric="f1", verbose=2)

    Training ========================= >>
    Models: LGB
    Metric: f1


    Results for LightGBM:
    Fit ---------------------------------------------
    Train evaluation --> f1: 1.0
    Test evaluation --> f1: 0.979
    Time elapsed: 0.465s
    -------------------------------------------------
    Total time: 0.465s
    Final results ==================== >>


    Total time: 0.466s
    -------------------------------------
    LightGBM --> f1: 0.979

    ```

    """

    acronym = "LGB"
    needs_scaling = True
    accepts_sparse = True
    has_validation = "n_estimators"
    supports_engines = ["lightgbm"]

    _module = "lightgbm.sklearn"
    _estimators = CustomDict({"class": "LGBMClassifier", "reg": "LGBMRegressor"})

    def _get_est(self, **params) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        return self._est_class(
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            device=params.pop("device", "gpu" if self._gpu else "cpu"),
            gpu_device_id=params.pop("gpu_device_id", self.T._device_id or -1),
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def _fit_estimator(
        self,
        estimator: Predictor,
        data: Tuple[pd.DataFrame, pd.Series],
        est_params_fit: dict,
        validation: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        trial: Optional[Trial] = None,
    ):
        """Fit the estimator and perform in-training validation.

        Parameters
        ----------
        estimator: Predictor
            Instance to fit.

        data: tuple
            Training data of the form (X, y).

        validation: tuple or None
            Validation data of the form (X, y). If None, no validation
            is performed.

        est_params_fit: dict
            Additional parameters for the estimator's fit method.

        trial: [Trial][] or None
            Active trial (during hyperparameter tuning).

        Returns
        -------
        Predictor
            Fitted instance.

        """
        m = self.T._metric[0].name
        params = est_params_fit.copy()

        callbacks = params.pop("callbacks", [])
        if trial and len(self.T._metric) == 1:
            callbacks.append(LightGBMPruningCallback(trial, m, "valid_1"))

        try:
            estimator.fit(
                *data,
                eval_set=[data, validation] if validation else None,
                eval_metric=LGBMetric(self.T._metric[0], task=self.T.task),
                callbacks=callbacks,
                verbose=params.get("verbose", False),
                **params,
            )
        except TrialPruned as ex:
            # Hacky solution to add the pruned step to the output
            step = str(ex).split(" ")[-1][:-1]
            steps = estimator.get_params()[self.has_validation]
            p = trial.storage.get_trial_user_attrs(trial.number)["params"]
            p[self.has_validation] = f"{step}/{steps}"

            trial.set_user_attr("estimator", estimator)
            raise ex

        # Create evals attribute with train and validation scores
        self._evals[f"{m}_train"] = estimator.evals_result_["training"][m]
        self._evals[f"{m}_test"] = estimator.evals_result_["valid_1"][m]

        return estimator

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            n_estimators=Int(20, 500, step=10),
            learning_rate=Float(0.01, 1.0, log=True),
            max_depth=Int(-1, 17, step=2),
            num_leaves=Int(20, 40),
            min_child_weight=Float(1e-4, 100, log=True),
            min_child_samples=Int(1, 30),
            subsample=Float(0.5, 1.0, step=0.1),
            colsample_bytree=Float(0.4, 1.0, step=0.1),
            reg_alpha=Float(1e-4, 100, log=True),
            reg_lambda=Float(1e-4, 100, log=True),
        )


class LinearDiscriminantAnalysis(BaseModel):
    """Linear Discriminant Analysis.

    Linear Discriminant Analysis is a classifier with a linear
    decision boundary, generated by fitting class conditional densities
    to the data and using Bayes’ rule. The model fits a Gaussian
    density to each class, assuming that all classes share the same
    covariance matrix.

    Corresponding estimators are:

    - [LinearDiscriminantAnalysis][ldaclassifier] for classification tasks.

    Read more in sklearn's [documentation][ldadocs].

    See Also
    --------
    atom.models:LogisticRegression
    atom.models:RadiusNearestNeighbors
    atom.models:QuadraticDiscriminantAnalysis

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="LDA", metric="f1", verbose=2)

    Training ========================= >>
    Models: LDA
    Metric: f1


    Results for LinearDiscriminantAnalysis:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.976
    Test evaluation --> f1: 0.953
    Time elapsed: 0.025s
    -------------------------------------------------
    Total time: 0.025s


    Final results ==================== >>
    Total time: 0.025s
    -------------------------------------
    LinearDiscriminantAnalysis --> f1: 0.953

    ```

    """

    acronym = "LDA"
    needs_scaling = False
    accepts_sparse = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "discriminant_analysis"
    _estimators = CustomDict({"class": "LinearDiscriminantAnalysis"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self._get_param("solver", params) == "svd":
            params.pop("shrinkage", None)

        return params

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            solver=Categorical(["svd", "lsqr", "eigen"]),
            shrinkage=Categorical([None, "auto", 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        )


class LinearSVM(BaseModel):
    """Linear Support Vector Machine.

    Similar to [SupportVectorMachine][] but with a linear kernel.
    Implemented in terms of liblinear rather than libsvm, so it has
    more flexibility in the choice of penalties and loss functions and
    should scale better to large numbers of samples.

    Corresponding estimators are:

    - [LinearSVC][] for classification tasks.
    - [LinearSVR][] for classification tasks.

    Read more in sklearn's [documentation][svmdocs].

    See Also
    --------
    atom.models:KNearestNeighbors
    atom.models:StochasticGradientDescent
    atom.models:SupportVectorMachine

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="lSVM", metric="f1", verbose=2)

    Training ========================= >>
    Models: lSVM
    Metric: f1


    Results for LinearSVM:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.9913
    Test evaluation --> f1: 0.9861
    Time elapsed: 0.021s
    -------------------------------------------------
    Total time: 0.021s


    Final results ==================== >>
    Total time: 0.021s
    -------------------------------------
    LinearSVM --> f1: 0.9861

    ```

    """

    acronym = "lSVM"
    needs_scaling = True
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "cuml"]

    _module = "svm"
    _estimators = CustomDict({"class": "LinearSVC", "reg": "LinearSVR"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self.T.goal == "class":
            if self._get_param("loss", params) == "hinge":
                # l1 regularization can't be combined with hinge
                params.replace_value("penalty", "l2")
                # l2 regularization can't be combined with hinge when dual=False
                params.replace_value("dual", True)
            elif self._get_param("loss", params) == "squared_hinge":
                # l1 regularization can't be combined with squared_hinge when dual=True
                if self._get_param("penalty", params) == "l1":
                    params.replace_value("dual", False)
        elif self._get_param("loss", params) == "epsilon_insensitive":
            params.replace_value("dual", True)

        return params

    def _get_est(self, **params) -> Predictor:
        """Get the estimator instance.

        Parameters
        ----------
        **params
            Unpacked hyperparameters for the estimator.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        if self.T.engine == "cuml" and self.T.goal == "class":
            return self._est_class(probability=params.pop("probability", True), **params)
        else:
            return super()._get_est(**params)

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        dist = CustomDict()
        if self.T.goal == "class":
            dist["penalty"] = Categorical(["l1", "l2"])
            dist["loss"] = Categorical(["hinge", "squared_hinge"])
        else:
            dist["loss"] = Categorical(
                ["epsilon_insensitive", "squared_epsilon_insensitive"]
            )

        dist["C"] = Float(1e-3, 100, log=True)
        dist["dual"] = Categorical([True, False])

        if self.T.engine == "cuml":
            dist.pop("dual")

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
    needs_scaling = True
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "LogisticRegression"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
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
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
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
    """Multi-layer Perceptron.

    Multi-layer Perceptron is a supervised learning algorithm that
    learns a function by training on a dataset. Given a set of features
    and a target, it can learn a non-linear function approximator for
    either classification or regression. It is different from logistic
    regression, in that between the input and the output layer, there
    can be one or more non-linear layers, called hidden layers.

    Corresponding estimators are:

    - [MLPClassifier][] for classification tasks.
    - [MLPRegressor][] for regression tasks.

    Read more in sklearn's [documentation][mlpdocs].

    See Also
    --------
    atom.models:PassiveAggressive
    atom.models:Perceptron
    atom.models:StochasticGradientDescent

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="MLP", metric="f1", verbose=2)

    Training ========================= >>
    Models: MLP
    Metric: f1


    Results for MultiLayerPerceptron:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.9965
    Test evaluation --> f1: 0.979
    Time elapsed: 1.600s
    -------------------------------------------------
    Total time: 1.600s


    Final results ==================== >>
    Total time: 1.600s
    -------------------------------------
    MultiLayerPerceptron --> f1: 0.979

    ```

    """

    acronym = "MLP"
    needs_scaling = True
    accepts_sparse = True
    has_validation = "max_iter"
    supports_engines = ["sklearn"]

    _module = "neural_network"
    _estimators = CustomDict({"class": "MLPClassifier", "reg": "MLPRegressor"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self._get_param("solver", params) != "sgd":
            params.pop("learning_rate", None)
            params.pop("power_t", None)
        else:
            params.pop("learning_rate_init", None)

        return params

    def _trial_to_est(self, params: CustomDict) -> CustomDict:
        """Convert trial's hyperparameters to parameters for the estimator.

        Parameters
        ----------
        params: CustomDict
            Trial's hyperparameters.

        Returns
        -------
        CustomDict
            Estimator's hyperparameters.

        """
        params = super()._trial_to_est(params)

        hidden_layer_sizes = []
        for param in [p for p in sorted(params) if p.startswith("hidden_layer")]:
            if params[param] > 0:
                hidden_layer_sizes.append(params.pop(param))
            else:
                params.pop(param)

        if hidden_layer_sizes:
            params.insert(0, "hidden_layer_sizes", tuple(hidden_layer_sizes))

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        dist = CustomDict(
            hidden_layer_1=Int(10, 100),
            hidden_layer_2=Int(0, 100),
            hidden_layer_3=Int(0, 100),
            activation=Categorical(["identity", "logistic", "tanh", "relu"]),
            solver=Categorical(["lbfgs", "sgd", "adam"]),
            alpha=Float(1e-4, 0.1, log=True),
            batch_size=Categorical(["auto", 8, 16, 32, 64, 128, 256]),
            learning_rate=Categorical(["constant", "invscaling", "adaptive"]),
            learning_rate_init=Float(1e-3, 0.1, log=True),
            power_t=Float(0.1, 0.9, step=0.1),
            max_iter=Int(50, 500, step=10),
        )

        # Drop layers if sizes are specified by user
        return dist[3:] if "hidden_layer_sizes" in self._est_params else dist


class MultinomialNB(BaseModel):
    """Multinomial Naive Bayes.

    MultinomialNB implements the Naive Bayes algorithm for multinomially
    distributed data, and is one of the two classic Naive Bayes variants
    used in text classification (where the data are typically
    represented as word vector counts, although tf-idf vectors are also
    known to work well in practice).

    Corresponding estimators are:

    - [MultinomialNB][multinomialnbclass] for classification tasks.

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
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "MultinomialNB"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            alpha=Float(0.01, 10, log=True),
            fit_prior=Categorical([True, False]),
        )


class OrdinaryLeastSquares(BaseModel):
    """Linear Regression (without regularization)."""

    acronym = "OLS"
    needs_scaling = True
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "LinearRegression"})


class PassiveAggressive(BaseModel):
    """Passive Aggressive.

    The passive-aggressive algorithms are a family of algorithms for
    large-scale learning. They are similar to the Perceptron in that
    they do not require a learning rate. However, contrary to the
    [Perceptron][], they include a regularization parameter `C`.

    Corresponding estimators are:

    - [PassiveAggressiveClassifier][] for classification tasks.
    - [PassiveAggressiveRegressor][] for classification tasks.

    Read more in sklearn's [documentation][padocs].

    See Also
    --------
    atom.models:MultiLayerPerceptron
    atom.models:Perceptron
    atom.models:StochasticGradientDescent

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="PA", metric="f1", verbose=2)

    Training ========================= >>
    Models: PA
    Metric: f1


    Results for PassiveAggressive:
    Fit ---------------------------------------------
    Train evaluation --> f1: 1.0
    Test evaluation --> f1: 0.9655
    Time elapsed: 4.549s
    -------------------------------------------------
    Total time: 4.549s


    Final results ==================== >>
    Total time: 4.550s
    -------------------------------------
    PassiveAggressive --> f1: 0.9655

    ```

    """

    acronym = "PA"
    needs_scaling = True
    accepts_sparse = True
    has_validation = "max_iter"
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict(
        {"class": "PassiveAggressiveClassifier", "reg": "PassiveAggressiveRegressor"}
    )

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        if self.T.goal == "class":
            loss = ["hinge", "squared_hinge"]
        else:
            loss = ["epsilon_insensitive", "squared_epsilon_insensitive"]

        return CustomDict(
            C=Float(1e-3, 100, log=True),
            loss=Categorical(loss),
            average=Categorical([True, False]),
        )


class Perceptron(BaseModel):
    """Linear Perceptron classification.

    The Perceptron is a simple classification algorithm suitable for
    large scale learning. By default:

    * It does not require a learning rate.
    * It is not regularized (penalized).
    * It updates its model only on mistakes.

    The last characteristic implies that the Perceptron is slightly
    faster to train than [StochasticGradientDescent][] with the hinge
    loss and that the resulting models are sparser.

    Corresponding estimators are:

    - [Perceptron][percclassifier] for classification tasks.

    Read more in sklearn's [documentation][percdocs].

    See Also
    --------
    atom.models:MultiLayerPerceptron
    atom.models:PassiveAggressive
    atom.models:StochasticGradientDescent

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="Perc", metric="f1", verbose=2)

    Training ========================= >>
    Models: Perc
    Metric: f1


    Results for Perceptron:
    Fit ---------------------------------------------
    Train evaluation --> f1: 1.0
    Test evaluation --> f1: 0.9787
    Time elapsed: 4.109s
    -------------------------------------------------
    Total time: 4.109s


    Final results ==================== >>
    Total time: 4.110s
    -------------------------------------
    Perceptron --> f1: 0.9787

    ```

    """

    acronym = "Perc"
    needs_scaling = True
    accepts_sparse = False
    has_validation = "max_iter"
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "Perceptron"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self._get_param("penalty", params) != "elasticnet":
            params.pop("l1_ratio", None)

        return params

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            penalty=Categorical([None, "l2", "l1", "elasticnet"]),
            alpha=Float(1e-4, 10, log=True),
            l1_ratio=Float(0.05, 0.95, step=0.05),
            max_iter=Int(500, 1500, step=50),
            eta0=Float(1e-2, 10, log=True),
        )


class QuadraticDiscriminantAnalysis(BaseModel):
    """Quadratic Discriminant Analysis.

    Quadratic Discriminant Analysis is a classifier with a quadratic
    decision boundary, generated by fitting class conditional densities
    to the data and using Bayes’ rule. The model fits a Gaussian
    density to each class, assuming that all classes share the same
    covariance matrix.

    Corresponding estimators are:

    - [QuadraticDiscriminantAnalysis][qdaclassifier] for classification tasks.

    Read more in sklearn's [documentation][ldadocs].

    See Also
    --------
    atom.models:LinearDiscriminantAnalysis
    atom.models:LogisticRegression
    atom.models:RadiusNearestNeighbors

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="QDA", metric="f1", verbose=2)

    Training ========================= >>
    Models: QDA
    Metric: f1


    Results for QuadraticDiscriminantAnalysis:
    Fit ---------------------------------------------
    Train evaluation --> f1: 0.9758
    Test evaluation --> f1: 0.9718
    Time elapsed: 0.019s
    -------------------------------------------------
    Total time: 0.019s


    Final results ==================== >>
    Total time: 0.020s
    -------------------------------------
    QuadraticDiscriminantAnalysis --> f1: 0.9718

    ```

    """

    acronym = "QDA"
    needs_scaling = False
    accepts_sparse = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "discriminant_analysis"
    _estimators = CustomDict({"class": "QuadraticDiscriminantAnalysis"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(reg_param=Float(0, 1.0, step=0.1))


class RadiusNearestNeighbors(BaseModel):
    """Radius Nearest Neighbors.

    Radius Nearest Neighbors implements the nearest neighbors vote,
    where the neighbors are selected from within a given radius. For
    regression, the target is predicted by local interpolation of the
    targets associated of the nearest neighbors in the training set.

    !!! warning
        * The `radius` parameter should be tuned to the data at hand or
          the model will perform poorly.
        * If outliers are detected, the estimator raises an exception
          unless `est_params={"outlier_label": "most_frequent"}` is used.

    Corresponding estimators are:

    - [RadiusNeighborsClassifier][] for classification tasks.
    - [RadiusNeighborsRegressor][] for regression tasks.

    Read more in sklearn's [documentation][knndocs].

    See Also
    --------
    atom.models:KNearestNeighbors
    atom.models:LinearDiscriminantAnalysis
    atom.models:QuadraticDiscriminantAnalysis

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(
    ...     models="RNN",
    ...     metric="f1",
    ...     est_params={"outlier_label": "most_frequent"},
    ...     verbose=2,
    ... )

    Training ========================= >>
    Models: RNN
    Metric: f1


    Results for RadiusNearestNeighbors:
    Fit ---------------------------------------------
    Train evaluation --> f1: 1.0
    Test evaluation --> f1: 0.7717
    Time elapsed: 0.032s
    -------------------------------------------------
    Total time: 0.032s


    Final results ==================== >>
    Total time: 0.032s
    -------------------------------------
    RadiusNearestNeighbors --> f1: 0.7717 ~

    ```

    """

    acronym = "RNN"
    needs_scaling = True
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "neighbors"
    _estimators = CustomDict(
        {"class": "RadiusNeighborsClassifier", "reg": "RadiusNeighborsRegressor"}
    )

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            radius=Float(1e-2, 100),
            weights=Categorical(["uniform", "distance"]),
            algorithm=Categorical(["auto", "ball_tree", "kd_tree", "brute"]),
            leaf_size=Int(20, 40),
            p=Int(1, 2),
        )


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
    needs_scaling = False
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "RandomForestClassifier", "reg": "RandomForestRegressor"}
    )

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if not self._get_param("bootstrap", params):
            params.pop("max_samples", None)

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
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
    needs_scaling = True
    accepts_sparse = True
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "RidgeClassifier", "reg": "Ridge"})

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        if self._gpu:
            solvers = ["eig", "svd", "cd"]
        else:
            solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]

        return [
            Float(1e-3, 10, "log-uniform", name="alpha"),
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
    needs_scaling = True
    accepts_sparse = True
    has_validation = "max_iter"
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "SGDClassifier", "reg": "SGDRegressor"})

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self._get_param("penalty", params) != "elasticnet":
            params.pop("l1_ratio", None)

        if self._get_param("learning_rate", params) == "optimal":
            params.pop("eta0", None)

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
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
    """Extreme Gradient Boosting.

    XGBoost is an optimized distributed gradient boosting model
    designed to be highly efficient, flexible and portable. XGBoost
    provides a parallel tree boosting that solve many data science
    problems in a fast and accurate way.

    Corresponding estimators are:

    - [XGBClassifier][] for classification tasks.
    - [XGBRegressor][] for regression tasks.

    Read more in sklearn's [documentation][xgbdocs].

    See Also
    --------
    atom.models:CatBoost
    atom.models:GradientBoosting
    atom.models:LightGBM

    Examples
    --------

    ```pycon
    >>> from atom import ATOMClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    >>> atom = ATOMClassifier(X, y)
    >>> atom.run(models="XGB", metric="f1", verbose=2)

    Training ========================= >>
    Models: XGB
    Metric: f1


    Results for XGBoost:
    Fit ---------------------------------------------
    Train evaluation --> f1: 1.0
    Test evaluation --> f1: 0.9726
    Time elapsed: 0.359s
    -------------------------------------------------
    Total time: 0.359s


    Final results ==================== >>
    Total time: 0.359s
    -------------------------------------
    XGBoost --> f1: 0.9726

    ```

    """

    acronym = "XGB"
    needs_scaling = True
    accepts_sparse = True
    has_validation = "n_estimators"
    supports_engines = ["xgboost"]

    _module = "xgboost"
    _estimators = CustomDict({"class": "XGBClassifier", "reg": "XGBRegressor"})

    def _get_est(self, **params) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        return self._est_class(
            eval_metric=XGBMetric(self.T._metric[0], task=self.T.task),
            use_label_encoder=params.pop("use_label_encoder", False),
            n_jobs=params.pop("n_jobs", self.T.n_jobs),
            tree_method=params.pop("tree_method", "gpu_hist" if self._gpu else None),
            gpu_id=self.T._device_id,
            verbosity=params.pop("verbosity", 0),
            random_state=params.pop("random_state", self.T.random_state),
            **params,
        )

    def _fit_estimator(
        self,
        estimator: Predictor,
        data: Tuple[pd.DataFrame, pd.Series],
        est_params_fit: dict,
        validation: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        trial: Optional[Trial] = None,
    ):
        """Fit the estimator and perform in-training validation.

        Parameters
        ----------
        estimator: Predictor
            Instance to fit.

        data: tuple
            Training data of the form (X, y).

        validation: tuple or None
            Validation data of the form (X, y). If None, no validation
            is performed.

        est_params_fit: dict
            Additional parameters for the estimator's fit method.

        trial: [Trial][] or None
            Active trial (during hyperparameter tuning).

        Returns
        -------
        Predictor
            Fitted instance.

        """
        m = self.T._metric[0].name
        params = est_params_fit.copy()

        callbacks = params.pop("callbacks", [])
        if trial and len(self.T._metric) == 1:
            callbacks.append(XGBoostPruningCallback(trial, f"validation_1-{m}"))

        try:
            estimator.set_params(callbacks=callbacks)
            estimator.fit(
                *data,
                eval_set=[data, validation] if validation else None,
                verbose=params.get("verbose", False),
                **params,
            )
        except TrialPruned as ex:
            # Hacky solution to add the pruned step to the output
            step = str(ex).split(" ")[-1][:-1]
            steps = estimator.get_params()[self.has_validation]
            p = trial.storage.get_trial_user_attrs(trial.number)["params"]
            p[self.has_validation] = f"{step}/{steps}"

            trial.set_user_attr("estimator", estimator)
            raise ex

        # Create evals attribute with train and validation scores
        # Negative because minimizes the function
        results = estimator.evals_result()
        self._evals[f"{m}_train"] = np.negative(results["validation_0"][m])
        self._evals[f"{m}_test"] = np.negative(results["validation_1"][m])

        return estimator

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            n_estimators=Int(20, 500, step=10),
            learning_rate=Float(0.01, 1.0, log=True),
            max_depth=Int(1, 20),
            gamma=Float(0, 1.0),
            min_child_weight=Int(1, 10),
            subsample=Float(0.5, 1.0, step=0.1),
            colsample_bytree=Float(0.4, 1.0, step=0.1),
            reg_alpha=Float(1e-4, 100, log=True),
            reg_lambda=Float(1e-4, 100, log=True),
        )


# Ensembles ======================================================== >>

class Stacking(BaseModel):
    """Stacking ensemble."""

    acronym = "Stack"
    needs_scaling = False
    has_validation = None

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

    def _get_est(self, **params) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
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
    needs_scaling = False
    has_validation = None

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

    def _get_est(self, **params) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
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

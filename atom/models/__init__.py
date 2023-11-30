# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module for models.

To add new models, note the following:

1. Add the class in the right file depending on the task.
2. Models are ordered alphabetically.
3. Models have the following structure:

Class attributes
----------------
acronym: str
    Acronym of the model's name.

needs_scaling: bool
    Whether the model needs scaled features.

accepts_sparse: bool
    Whether the model has native support for sparse matrices.

native_multilabel: bool
    Whether the model has native support for multilabel tasks.

native_multioutput: bool
    Whether the model has native support for multioutput tasks.

has_validation: str or None
    Whether the model allows in-training validation. If str,
    name of the estimator's parameter that states the number
    of iterations. If None, no support for in-training
    validation.

supports_engines: list
    Engines that can be used to run this model.

_module: str
    Module from which to load the class. If one of engines,
    ignore the engine name, i.e., use "ensemble" instead of
    "sklearn.ensemble".

_estimators: dict
    Name of the estimators per goal.

Instance attributes
-------------------
name: str
    Name of the model. Defaults to the same as the acronym
    but can be different if the same model is called multiple
    times. The name is assigned in the basemodel.py module.

Methods
-------
_get_parameters(self, x) -> dict:
    Return the trial's suggestions with (optionally) custom changes
    to the params. Don't implement if the parent's implementation
    is sufficient.

_trial_to_est(self, params) -> dict:
    Convert trial's hyperparameters to parameters for the
    estimator. Only implement for models whose study params are
    different from those for the estimator.

_fit_estimator(self, estimator, data, est_params_fit, validation, trial):
    This method is called to fit the estimator. Implement only
    to customize the fit.

_get_distributions(self) -> dict:
    Return a list of the hyperparameter distributions for
    optimization.

"""

from atom.models.classreg import (
    AdaBoost, AutomaticRelevanceDetermination, Bagging, BayesianRidge,
    BernoulliNB, CatBoost, CategoricalNB, ComplementNB, DecisionTree, Dummy,
    ElasticNet, ExtraTree, ExtraTrees, GaussianNB, GaussianProcess,
    GradientBoostingMachine, HistGradientBoosting, HuberRegression,
    KNearestNeighbors, Lasso, LeastAngleRegression, LightGBM,
    LinearDiscriminantAnalysis, LinearSVM, LogisticRegression,
    MultiLayerPerceptron, MultinomialNB, OrdinaryLeastSquares,
    OrthogonalMatchingPursuit, PassiveAggressive, Perceptron,
    QuadraticDiscriminantAnalysis, RadiusNearestNeighbors, RandomForest, Ridge,
    StochasticGradientDescent, SupportVectorMachine, XGBoost,
)
from atom.models.custom import CustomModel
from atom.models.ensembles import Stacking, Voting
from atom.models.ts import (
    ARIMA, ETS, AutoARIMA, ExponentialSmoothing, NaiveForecaster,
    PolynomialTrend,
)
from atom.utils.types import Predictor
from atom.utils.utils import ClassMap


# Available models
MODELS = ClassMap(
    AdaBoost,
    ARIMA,
    AutoARIMA,
    AutomaticRelevanceDetermination,
    Bagging,
    BayesianRidge,
    BernoulliNB,
    CatBoost,
    CategoricalNB,
    ComplementNB,
    DecisionTree,
    Dummy,
    ElasticNet,
    ETS,
    ExponentialSmoothing,
    ExtraTree,
    ExtraTrees,
    GaussianNB,
    GaussianProcess,
    GradientBoostingMachine,
    HuberRegression,
    HistGradientBoosting,
    KNearestNeighbors,
    Lasso,
    LeastAngleRegression,
    LightGBM,
    LinearDiscriminantAnalysis,
    LinearSVM,
    LogisticRegression,
    MultiLayerPerceptron,
    MultinomialNB,
    NaiveForecaster,
    OrdinaryLeastSquares,
    OrthogonalMatchingPursuit,
    PassiveAggressive,
    Perceptron,
    PolynomialTrend,
    QuadraticDiscriminantAnalysis,
    RadiusNearestNeighbors,
    RandomForest,
    Ridge,
    StochasticGradientDescent,
    SupportVectorMachine,
    XGBoost,
    key="acronym",
)

# Available ensembles
ENSEMBLES = ClassMap(Stacking, Voting, key="acronym")

# Available models + ensembles
MODELS_ENSEMBLES = ClassMap(*MODELS, *ENSEMBLES, key="acronym")

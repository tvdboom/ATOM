"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module for models.

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
from atom.models.custom import create_custom_model
from atom.models.ensembles import create_stacking_model, create_voting_model
from atom.models.ts import (
    ARIMA, BATS, ETS, MSTL, SARIMAX, STL, TBATS, VAR, VARMAX, AutoARIMA,
    AutoETS, Croston, DynamicFactor, ExponentialSmoothing, NaiveForecaster,
    PolynomialTrend, Prophet, Theta,
)
from atom.utils.utils import ClassMap


# Available models
MODELS = ClassMap(
    AdaBoost,
    ARIMA,
    AutoARIMA,
    AutoETS,
    AutomaticRelevanceDetermination,
    Bagging,
    BATS,
    BayesianRidge,
    BernoulliNB,
    CatBoost,
    CategoricalNB,
    ComplementNB,
    Croston,
    DecisionTree,
    Dummy,
    DynamicFactor,
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
    MSTL,
    MultiLayerPerceptron,
    MultinomialNB,
    NaiveForecaster,
    OrdinaryLeastSquares,
    OrthogonalMatchingPursuit,
    PassiveAggressive,
    Perceptron,
    Prophet,
    PolynomialTrend,
    QuadraticDiscriminantAnalysis,
    RadiusNearestNeighbors,
    RandomForest,
    Ridge,
    SARIMAX,
    STL,
    StochasticGradientDescent,
    SupportVectorMachine,
    TBATS,
    Theta,
    VAR,
    VARMAX,
    XGBoost,
    key="acronym",
)

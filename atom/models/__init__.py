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
from atom.models.custom import CustomModel
from atom.models.ensembles import Stacking, Voting
from atom.models.ts import (
    ARIMA, BATS, ETS, STL, TBATS, AutoARIMA, Croston, ExponentialSmoothing,
    NaiveForecaster, PolynomialTrend, Theta,
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
    BATS,
    BayesianRidge,
    BernoulliNB,
    CatBoost,
    CategoricalNB,
    ComplementNB,
    Croston,
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
    STL,
    StochasticGradientDescent,
    SupportVectorMachine,
    TBATS,
    Theta,
    XGBoost,
    key="acronym",
)

# Available ensembles
ENSEMBLES = ClassMap(Stacking, Voting, key="acronym")

# Available models + ensembles
MODELS_ENSEMBLES = ClassMap(*MODELS, *ENSEMBLES, key="acronym")

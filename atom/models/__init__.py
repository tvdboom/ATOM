# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module for models.

To add new models note the following:

1. Add the class in the right file depending on task.
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
        ignore the engine name, i.e. use "ensemble" instead of
        "sklearn.ensemble".

    _estimators: CustomDict
        Name of the estimators per goal.

    Instance attributes
    -------------------
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

    _trial_to_est(self, params) -> CustomDict:
        Convert trial's hyperparameters to parameters for the
        estimator. Only implement for models whose study params are
        different from those for the estimator.

    _fit_estimator(self, estimator, data, est_params_fit, validation, trial):
        This method is called to fit the estimator. Implement only
        to customize the fit.

    _get_distributions(self) -> CustomDict:
        Return a list of the hyperparameter distributions for
        optimization.

"""

from atom.basemodel import ClassRegModel
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
from atom.models.ensembles import Stacking, Voting
from atom.models.ts import (
    ARIMA, ETS, AutoARIMA, ExponentialSmoothing, NaiveForecaster,
    PolynomialTrend,
)
from atom.utils.types import PREDICTOR
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


class CustomModel(ClassRegModel):
    """Model with estimator provided by user."""

    def __init__(self, **kwargs):
        if callable(est := kwargs.pop("estimator")):  # Estimator provided by the user
            self._est = est
            self._params = {}
        else:
            self._est = est.__class__
            self._params = est.get_params()  # Store the provided parameters

        if hasattr(est, "name"):
            name = est.name
        else:
            # If no name is provided, use the name of the class
            name = self._fullname
            if len(n := list(filter(str.isupper, name))) >= 2 and n not in MODELS:
                name = "".join(n)

        self.acronym = getattr(est, "acronym", name)
        if not name.startswith(self.acronym):
            raise ValueError(
                f"The name ({name}) and acronym ({self.acronym}) of model "
                f"{self._fullname} do not match. The name should start with "
                f"the model's acronym."
            )

        self.needs_scaling = getattr(est, "needs_scaling", False)
        self.native_multilabel = getattr(est, "native_multilabel", False)
        self.native_multioutput = getattr(est, "native_multioutput", False)
        self.has_validation = getattr(est, "has_validation", None)

        super().__init__(name=name, **kwargs)

    @property
    def _fullname(self) -> str:
        """Return the estimator's class name."""
        return self._est_class.__name__

    @property
    def _est_class(self):
        """Return the estimator's class."""
        return self._est

    def _get_est(self, **params) -> PREDICTOR:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        PREDICTOR
            Estimator instance.

        """
        return super()._get_est(**{**self._params, **params})

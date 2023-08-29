# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing all available models. The models are
             ordered alphabetically. Classes must have the following
             structure:

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
            different than those for the estimator.

        _fit_estimator(self, estimator, data, est_params_fit, validation, trial):
            This method is called to fit the estimator. Implement only
            to customize the fit.

        _get_distributions(self) -> CustomDict:
            Return a list of the hyperparameter distributions for
            optimization.

"""

from __future__ import annotations

import numpy as np
from optuna.distributions import CategoricalDistribution as Cat
from optuna.distributions import FloatDistribution as Float
from optuna.distributions import IntDistribution as Int
from optuna.exceptions import TrialPruned
from optuna.integration import (
    CatBoostPruningCallback, LightGBMPruningCallback, XGBoostPruningCallback,
)
from optuna.trial import Trial

from atom.basemodel import ClassRegModel, ForecastModel
from atom.pipeline import Pipeline
from atom.utils.types import DATAFRAME, PREDICTOR, SERIES
from atom.utils.utils import (
    CatBMetric, ClassMap, CustomDict, LGBMetric, XGBMetric, sign,
)


# Custom models ==================================================== >>

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


# Classification and Regression models ============================= >>

class AdaBoost(ClassRegModel):
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
    atom.models:GradientBoostingMachine
    atom.models:RandomForest
    atom.models:XGBoost

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="AdaB", metric="f1", verbose=2)
    ```

    """

    acronym = "AdaB"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
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

        if self.goal == "class":
            dist["algorithm"] = Cat(["SAMME.R", "SAMME"])
        else:
            dist["loss"] = Cat(["linear", "square", "exponential"])

        return dist


class AutomaticRelevanceDetermination(ClassRegModel):
    """Automatic Relevance Determination.

    Automatic Relevance Determination is very similar to
    [BayesianRidge][], but can lead to sparser coefficients. Fit the
    weights of a regression model, using an ARD prior. The weights of
    the regression model are assumed to be in Gaussian distributions.

    Corresponding estimators are:

    - [ARDRegression][] for regression tasks.

    Read more in sklearn's [documentation][arddocs].

    See Also
    --------
    atom.models:BayesianRidge
    atom.models:GaussianProcess
    atom.models:LeastAngleRegression

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    atom = ATOMRegressor(X, y, random_state=1)
    atom.run(models="ARD", metric="r2", verbose=2)
    ```

    """

    acronym = "ARD"
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    has_validation = None
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


class Bagging(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="Bag", metric="f1", verbose=2)
    ```

    """

    acronym = "Bag"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
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
            max_samples=Float(0.5, 1.0, step=0.1),
            max_features=Float(0.5, 1.0, step=0.1),
            bootstrap=Cat([True, False]),
            bootstrap_features=Cat([True, False]),
        )


class BayesianRidge(ClassRegModel):
    """Bayesian ridge regression.

    Bayesian regression techniques can be used to include regularization
    parameters in the estimation procedure: the regularization parameter
    is not set in a hard sense but tuned to the data at hand.

    Corresponding estimators are:

    - [BayesianRidge][bayesianridgeclass] for regression tasks.

    Read more in sklearn's [documentation][brdocs].

    See Also
    --------
    atom.models:AutomaticRelevanceDetermination
    atom.models:GaussianProcess
    atom.models:LeastAngleRegression

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    atom = ATOMRegressor(X, y, random_state=1)
    atom.run(models="BR", metric="r2", verbose=2)
    ```

    """

    acronym = "BR"
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
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


class BernoulliNB(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="BNB", metric="f1", verbose=2)
    ```

    """

    acronym = "BNB"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
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
            fit_prior=Cat([True, False]),
        )


class CatBoost(ClassRegModel):
    """Cat Boosting Machine.

    CatBoost is a machine learning method based on gradient boosting
    over decision trees. Main advantages of CatBoost:

    - Superior quality when compared with other GBDT models on many
      datasets.
    - Best in class prediction speed.

    Corresponding estimators are:

    - [CatBoostClassifier][] for classification tasks.
    - [CatBoostRegressor][] for regression tasks.

    Read more in CatBoost's [documentation][catbdocs].

    !!! warning
        * CatBoost selects the weights achieved by the best evaluation
          on the test set after training. This means that, by default,
          there is some minor data leakage in the test set. Use the
          `use_best_model=False` parameter to avoid this behavior or use
          a [holdout set][data-sets] to evaluate the final estimator.
        * [In-training validation][] and [pruning][] are disabled when
          `#!python device="gpu"`.

    !!! note
        ATOM uses CatBoost's `n_estimators` parameter instead of
        `iterations` to indicate the number of trees to fit. This is
        done to have consistent naming with the [XGBoost][] and
        [LightGBM][] models.

    See Also
    --------
    atom.models:GradientBoostingMachine
    atom.models:LightGBM
    atom.models:XGBoost

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="CatB", metric="f1", verbose=2)
    ```

    """

    acronym = "CatB"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = "n_estimators"
    supports_engines = ["catboost"]

    _module = "catboost"
    _estimators = CustomDict({"class": "CatBoostClassifier", "reg": "CatBoostRegressor"})

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

        if self._get_param("bootstrap_type", params) == "Bernoulli":
            params.pop("bagging_temperature")
        elif self._get_param("bootstrap_type", params) == "Bayesian":
            params.pop("subsample")

        return params

    def _get_est(self, **params) -> PREDICTOR:
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
        eval_metric = None
        if getattr(self, "_metric", None) and not self._gpu:
            eval_metric = CatBMetric(self._metric[0], task=self.task)

        return self._est_class(
            eval_metric=params.pop("eval_metric", eval_metric),
            train_dir=params.pop("train_dir", ""),
            allow_writing_files=params.pop("allow_writing_files", False),
            thread_count=params.pop("n_jobs", self.n_jobs),
            task_type=params.pop("task_type", "GPU" if self._gpu else "CPU"),
            devices=str(self._device_id),
            verbose=params.pop("verbose", False),
            random_state=params.pop("random_state", self.random_state),
            **params,
        )

    def _fit_estimator(
        self,
        estimator: PREDICTOR,
        data: tuple[DATAFRAME, SERIES],
        est_params_fit: dict,
        validation: tuple[DATAFRAME, SERIES] | None = None,
        trial: Trial | None = None,
    ):
        """Fit the estimator and perform in-training validation.

        Parameters
        ----------
        estimator: Predictor
            Instance to fit.

        data: tuple
            Training data of the form (X, y).

        est_params_fit: dict
            Additional parameters for the estimator's fit method.

        validation: tuple or None
            Validation data of the form (X, y). If None, no validation
            is performed.

        trial: [Trial][] or None
            Active trial (during hyperparameter tuning).

        Returns
        -------
        Predictor
            Fitted instance.

        """
        params = est_params_fit.copy()

        callbacks = params.pop("callbacks", [])
        if trial and len(self._metric) == 1 and not self._gpu:
            callbacks.append(cb := CatBoostPruningCallback(trial, "CatBMetric"))

        # gpu implementation fails if callbacks!=None
        estimator.fit(*data, eval_set=validation, callbacks=callbacks or None, **params)

        if not self._gpu:
            if validation:
                # Create evals attribute with train and validation scores
                m = self._metric[0].name
                evals = estimator.evals_result_
                self._evals[f"{m}_train"] = evals["learn"]["CatBMetric"]
                self._evals[f"{m}_test"] = evals["validation"]["CatBMetric"]

            if trial and len(self._metric) == 1 and cb._pruned:
                # Add the pruned step to the output
                step = len(self.evals[f'{m}_train'])
                steps = estimator.get_params()[self.has_validation]
                trial.params[self.has_validation] = f"{step}/{steps}"

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
            max_depth=Cat([None, *range(1, 17)]),
            min_child_samples=Int(1, 30),
            bootstrap_type=Cat(["Bayesian", "Bernoulli"]),
            bagging_temperature=Float(0, 10),
            subsample=Float(0.5, 1.0, step=0.1),
            reg_lambda=Float(0.001, 100, log=True),
        )


class CategoricalNB(ClassRegModel):
    """Categorical Naive Bayes.

    Categorical Naive Bayes implements the Naive Bayes algorithm for
    categorical features.

    Corresponding estimators are:

    - [CategoricalNB][categoricalnbclass] for classification tasks.

    Read more in sklearn's [documentation][catnbdocs].

    See Also
    --------
    atom.models:BernoulliNB
    atom.models:ComplementNB
    atom.models:GaussianNB

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    import numpy as np

    X = np.random.randint(5, size=(100, 100))
    y = np.random.randint(2, size=100)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="CatNB", metric="f1", verbose=2)
    ```

    """

    acronym = "CatNB"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
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
        return CustomDict(
            alpha=Float(0.01, 10, log=True),
            fit_prior=Cat([True, False]),
        )


class ComplementNB(ClassRegModel):
    """Complement Naive Bayes.

    The Complement Naive Bayes classifier was designed to correct the
    "severe assumptions" made by the standard [MultinomialNB][]
    classifier. It is particularly suited for imbalanced datasets.

    Corresponding estimators are:

    - [ComplementNB][complementnbclass] for classification tasks.

    Read more in sklearn's [documentation][cnbdocs].

    See Also
    --------
    atom.models:BernoulliNB
    atom.models:CategoricalNB
    atom.models:MultinomialNB

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="CNB", metric="f1", verbose=2)
    ```

    """

    acronym = "CNB"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
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
        return CustomDict(
            alpha=Float(0.01, 10, log=True),
            fit_prior=Cat([True, False]),
            norm=Cat([True, False]),
        )


class DecisionTree(ClassRegModel):
    """Single Decision Tree.

    A single decision tree classifier/regressor.

    Corresponding estimators are:

    - [DecisionTreeClassifier][] for classification tasks.
    - [DecisionTreeRegressor][] for regression tasks.

    Read more in sklearn's [documentation][treedocs].

    See Also
    --------
    atom.models:ExtraTree
    atom.models:ExtraTrees
    atom.models:RandomForest

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="Tree", metric="f1", verbose=2)
    ```

    """

    acronym = "Tree"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
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
        if self.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error", "friedman_mse", "poisson"]

        return CustomDict(
            criterion=Cat(criterion),
            splitter=Cat(["best", "random"]),
            max_depth=Cat([None, *range(1, 17)]),
            min_samples_split=Int(2, 20),
            min_samples_leaf=Int(1, 20),
            max_features=Cat([None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9]),
            ccp_alpha=Float(0, 0.035, step=0.005),
        )


class Dummy(ClassRegModel):
    """Dummy classifier/regressor.

    When doing supervised learning, a simple sanity check consists of
    comparing one's estimator against simple rules of thumb. The
    prediction methods completely ignore the input data. Do not use
    this model for real problems. Use it only as a simple baseline
    to compare with other models.

    Corresponding estimators are:

    - [DummyClassifier][] for classification tasks.
    - [DummyRegressor][] for regression tasks.

    Read more in sklearn's [documentation][dummydocs].

    See Also
    --------
    atom.models:DecisionTree
    atom.models:ExtraTree
    atom.models:NaiveForecaster

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="Dummy", metric="f1", verbose=2)
    ```

    """

    acronym = "Dummy"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "dummy"
    _estimators = CustomDict({"class": "DummyClassifier", "reg": "DummyRegressor"})

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

        if self._get_param("strategy", params) != "quantile":
            params.pop("quantile")

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        dist = CustomDict()
        if self.goal == "class":
            dist["strategy"] = Cat(["most_frequent", "prior", "stratified", "uniform"])
        else:
            dist["strategy"] = Cat(["mean", "median", "quantile"])
            dist["quantile"] = Float(0, 1.0, step=0.1)

        return dist


class ElasticNet(ClassRegModel):
    """Linear Regression with elasticnet regularization.

    Linear least squares with l1 and l2 regularization.

    Corresponding estimators are:

    - [ElasticNet][elasticnetreg] for regression tasks.

    Read more in sklearn's [documentation][endocs].

    See Also
    --------
    atom.models:Lasso
    atom.models:OrdinaryLeastSquares
    atom.models:Ridge

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    atom = ATOMRegressor(X, y, random_state=1)
    atom.run(models="EN", metric="r2", verbose=2)
    ```

    """

    acronym = "EN"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

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
        return CustomDict(
            alpha=Float(1e-3, 10, log=True),
            l1_ratio=Float(0.1, 0.9, step=0.1),
            selection=Cat(["cyclic", "random"]),
        )


class ExtraTree(ClassRegModel):
    """Extremely Randomized Tree.

    Extra-trees differ from classic decision trees in the way they are
    built. When looking for the best split to separate the samples of a
    node into two groups, random splits are drawn for each of the
    max_features randomly selected features and the best split among
    those is chosen. When max_features is set 1, this amounts to
    building a totally random decision tree.

    Corresponding estimators are:

    - [ExtraTreeClassifier][] for classification tasks.
    - [ExtraTreeRegressor][] for regression tasks.

    Read more in sklearn's [documentation][treedocs].

    See Also
    --------
    atom.models:DecisionTree
    atom.models:ExtraTrees
    atom.models:RandomForest

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="ETree", metric="f1", verbose=2)
    ```

    """

    acronym = "ETree"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "tree"
    _estimators = CustomDict(
        {"class": "ExtraTreeClassifier", "reg": "ExtraTreeRegressor"}
    )

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

        if not self._get_param("bootstrap", params):
            params.pop("max_samples")

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        if self.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error"]

        return CustomDict(
            criterion=Cat(criterion),
            splitter=Cat(["random", "best"]),
            max_depth=Cat([None, *range(1, 17)]),
            min_samples_split=Int(2, 20),
            min_samples_leaf=Int(1, 20),
            max_features=Cat([None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9]),
            ccp_alpha=Float(0, 0.035, step=0.005),
        )


class ExtraTrees(ClassRegModel):
    """Extremely Randomized Trees.

    Extra-Trees use a meta estimator that fits a number of randomized
    decision trees (a.k.a. [extra-trees][extratree]) on various
    sub-samples of the dataset and uses averaging to improve the
    predictive accuracy and control over-fitting.

    Corresponding estimators are:

    - [ExtraTreesClassifier][] for classification tasks.
    - [ExtraTreesRegressor][] for regression tasks.

    Read more in sklearn's [documentation][etdocs].

    See Also
    --------
    atom.models:DecisionTree
    atom.models:ExtraTree
    atom.models:RandomForest

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="ET", metric="f1", verbose=2)
    ```

    """

    acronym = "ET"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "ExtraTreesClassifier", "reg": "ExtraTreesRegressor"}
    )

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

        if not self._get_param("bootstrap", params):
            params.pop("max_samples")

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        if self.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error"]

        return CustomDict(
            n_estimators=Int(10, 500, step=10),
            criterion=Cat(criterion),
            max_depth=Cat([None, *range(1, 17)]),
            min_samples_split=Int(2, 20),
            min_samples_leaf=Int(1, 20),
            max_features=Cat([None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9]),
            bootstrap=Cat([True, False]),
            max_samples=Cat([None, 0.5, 0.6, 0.7, 0.8, 0.9]),
            ccp_alpha=Float(0, 0.035, step=0.005),
        )


class GaussianNB(ClassRegModel):
    """Gaussian Naive Bayes.

    Gaussian Naive Bayes implements the Naive Bayes algorithm for
    classification. The likelihood of the features is assumed to
    be Gaussian.

    Corresponding estimators are:

    - [GaussianNB][gaussiannbclass] for classification tasks.

    Read more in sklearn's [documentation][gnbdocs].

    See Also
    --------
    atom.models:BernoulliNB
    atom.models:CategoricalNB
    atom.models:ComplementNB

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="GNB", metric="f1", verbose=2)
    ```

    """

    acronym = "GNB"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn", "cuml"]

    _module = "naive_bayes"
    _estimators = CustomDict({"class": "GaussianNB"})


class GaussianProcess(ClassRegModel):
    """Gaussian process.

    Gaussian Processes are a generic supervised learning method
    designed to solve regression and probabilistic classification
    problems. The advantages of Gaussian processes are:

    * The prediction interpolates the observations.
    * The prediction is probabilistic (Gaussian) so that one can compute
      empirical confidence intervals and decide based on those if one
      should refit (online fitting, adaptive fitting) the prediction in
      some region of interest.

    The disadvantages of Gaussian processes include:

    * They are not sparse, i.e. they use the whole samples/features
      information to perform the prediction.
    * They lose efficiency in high dimensional spaces, namely when the
      number of features exceeds a few dozens.

    Corresponding estimators are:

    - [GaussianProcessClassifier][] for classification tasks.
    - [GaussianProcessRegressor][] for regression tasks.

    Read more in sklearn's [documentation][gpdocs].

    See Also
    --------
    atom.models:GaussianNB
    atom.models:LinearDiscriminantAnalysis
    atom.models:PassiveAggressive

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="GP", metric="f1", verbose=2)
    ```

    """

    acronym = "GP"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "gaussian_process"
    _estimators = CustomDict(
        {"class": "GaussianProcessClassifier", "reg": "GaussianProcessRegressor"}
    )


class GradientBoostingMachine(ClassRegModel):
    """Gradient Boosting Machine.

    A Gradient Boosting Machine builds an additive model in a forward
    stage-wise fashion; it allows for the optimization of arbitrary
    differentiable loss functions. In each stage `n_classes_` regression
    trees are fit on the negative gradient of the loss function, e.g.
    binary or multiclass log loss. Binary classification is a special
    case where only a single regression tree is induced.

    Corresponding estimators are:

    - [GradientBoostingClassifier][] for classification tasks.
    - [GradientBoostingRegressor][] for regression tasks.

    Read more in sklearn's [documentation][gbmdocs].

    !!! tip
        [HistGradientBoosting][] is a much faster variant of this
        algorithm for intermediate datasets (n_samples >= 10k).

    See Also
    --------
    atom.models:CatBoost
    atom.models:HistGradientBoosting
    atom.models:LightGBM

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="GBM", metric="f1", verbose=2)
    ```

    """

    acronym = "GBM"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "GradientBoostingClassifier", "reg": "GradientBoostingRegressor"}
    )

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

        if self._get_param("loss", params) not in ("huber", "quantile"):
            params.pop("alpha")

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        dist = CustomDict(
            loss=Cat(["log_loss", "exponential"]),
            learning_rate=Float(0.01, 1.0, log=True),
            n_estimators=Int(10, 500, step=10),
            subsample=Float(0.5, 1.0, step=0.1),
            criterion=Cat(["friedman_mse", "squared_error"]),
            min_samples_split=Int(2, 20),
            min_samples_leaf=Int(1, 20),
            max_depth=Int(1, 21),
            max_features=Cat([None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9]),
            ccp_alpha=Float(0, 0.035, step=0.005),
        )

        if self.task.startswith("multiclass"):
            dist.pop("loss")  # Multiclass only supports log_loss
        elif self.goal.startswith("reg"):
            dist["loss"] = Cat(["squared_error", "absolute_error", "huber", "quantile"])
            dist["alpha"] = Float(0.1, 0.9, step=0.1)

        return dist


class HuberRegression(ClassRegModel):
    """Huber regressor.

    Huber is a linear regression model that is robust to outliers. It
    makes sure that the loss function is not heavily influenced by the
    outliers while not completely ignoring their effect.

    Corresponding estimators are:

    - [HuberRegressor][] for regression tasks.

    Read more in sklearn's [documentation][huberdocs].

    See Also
    --------
    atom.models:AutomaticRelevanceDetermination
    atom.models:LeastAngleRegression
    atom.models:OrdinaryLeastSquares

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    atom = ATOMRegressor(X, y, random_state=1)
    atom.run(models="Huber", metric="r2", verbose=2)
    ```

    """

    acronym = "Huber"
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
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
        return CustomDict(
            epsilon=Float(1, 10, log=True),
            max_iter=Int(50, 500, step=10),
            alpha=Float(1e-4, 1, log=True),
        )


class HistGradientBoosting(ClassRegModel):
    """Histogram-based Gradient Boosting Machine.

    This Histogram-based Gradient Boosting Machine is much faster than
    the standard [GradientBoostingMachine][] for big datasets
    (n_samples>=10k). This variation first bins the input samples into
    integer-valued bins which tremendously reduces the number of
    splitting points to consider, and allows the algorithm to leverage
    integer-based data structures (histograms) instead of relying on
    sorted continuous values when building the trees.

    Corresponding estimators are:

    - [HistGradientBoostingClassifier][] for classification tasks.
    - [HistGradientBoostingRegressor][] for regression tasks.

    Read more in sklearn's [documentation][hgbmdocs].

    See Also
    --------
    atom.models:CatBoost
    atom.models:GradientBoostingMachine
    atom.models:XGBoost

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="hGBM", metric="f1", verbose=2)
    ```

    """

    acronym = "hGBM"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
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
        dist = CustomDict(
            loss=Cat(["squared_error", "absolute_error", "poisson", "quantile", "gamma"]),
            learning_rate=Float(0.01, 1.0, log=True),
            max_iter=Int(10, 500, step=10),
            max_leaf_nodes=Int(10, 50),
            max_depth=Cat([None, *range(1, 17)]),
            min_samples_leaf=Int(10, 30),
            l2_regularization=Float(0, 1.0, step=0.1),
        )

        if self.goal == "class":
            dist.pop("loss")

        return dist


class KNearestNeighbors(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="KNN", metric="f1", verbose=2)
    ```

    """

    acronym = "KNN"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
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
            weights=Cat(["uniform", "distance"]),
            algorithm=Cat(["auto", "ball_tree", "kd_tree", "brute"]),
            leaf_size=Int(20, 40),
            p=Int(1, 2),
        )

        if self._gpu:
            dist.pop("algorithm")  # Only 'brute' is supported
            if self.engine["estimator"] == "cuml":
                dist.pop("weights")  # Only 'uniform' is supported
                dist.pop("leaf_size")
                dist.pop("p")

        return dist


class Lasso(ClassRegModel):
    """Linear Regression with lasso regularization.

    Linear least squares with l1 regularization.

    Corresponding estimators are:

    - [Lasso][lassoreg] for regression tasks.

    Read more in sklearn's [documentation][lassodocs].

    See Also
    --------
    atom.models:ElasticNet
    atom.models:OrdinaryLeastSquares
    atom.models:Ridge

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    atom = ATOMRegressor(X, y, random_state=1)
    atom.run(models="Lasso", metric="r2", verbose=2)
    ```

    """

    acronym = "Lasso"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

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
        return CustomDict(
            alpha=Float(1e-3, 10, log=True),
            selection=Cat(["cyclic", "random"]),
        )


class LeastAngleRegression(ClassRegModel):
    """Least Angle Regression.

    Least-Angle Regression is a regression algorithm for
    high-dimensional data. Lars is similar to forward stepwise
    regression. At each step, it finds the feature most correlated
    with the target. When there are multiple features having equal
    correlation, instead of continuing along the same feature, it
    proceeds in a direction equiangular between the features.

    Corresponding estimators are:

    - [Lars][] for regression tasks.

    Read more in sklearn's [documentation][larsdocs].

    See Also
    --------
    atom.models:BayesianRidge
    atom.models:HuberRegression
    atom.models:OrdinaryLeastSquares

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    atom = ATOMRegressor(X, y, random_state=1)
    atom.run(models="Lars", metric="r2", verbose=2)
    ```

    """

    acronym = "Lars"
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "Lars"})


class LightGBM(ClassRegModel):
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

    Read more in LightGBM's [documentation][lgbdocs].

    !!! info
        Using LightGBM's [GPU acceleration][estimator-acceleration]
        requires [additional software dependencies][lgb_gpu].

    See Also
    --------
    atom.models:CatBoost
    atom.models:GradientBoostingMachine
    atom.models:XGBoost

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="LGB", metric="f1", verbose=2)
    ```

    """

    acronym = "LGB"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = "n_estimators"
    supports_engines = ["lightgbm"]

    _module = "lightgbm.sklearn"
    _estimators = CustomDict({"class": "LGBMClassifier", "reg": "LGBMRegressor"})

    def _get_est(self, **params) -> PREDICTOR:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        # Custom lightgbm mapping for warnings
        # PYTHONWARNINGS doesn't work since they go from C/C++ code to stdout
        warns = dict(always=2, default=1, error=0, ignore=-1)

        return self._est_class(
            verbose=params.pop("verbose", warns.get(self.warnings, -1)),
            n_jobs=params.pop("n_jobs", self.n_jobs),
            device=params.pop("device", "gpu" if self._gpu else "cpu"),
            gpu_device_id=params.pop("gpu_device_id", self._device_id or -1),
            random_state=params.pop("random_state", self.random_state),
            **params,
        )

    def _fit_estimator(
        self,
        estimator: PREDICTOR,
        data: tuple[DATAFRAME, SERIES],
        est_params_fit: dict,
        validation: tuple[DATAFRAME, SERIES] | None = None,
        trial: Trial | None = None,
    ):
        """Fit the estimator and perform in-training validation.

        Parameters
        ----------
        estimator: Predictor
            Instance to fit.

        data: tuple
            Training data of the form (X, y).

        est_params_fit: dict
            Additional parameters for the estimator's fit method.

        validation: tuple or None
            Validation data of the form (X, y). If None, no validation
            is performed.

        trial: [Trial][] or None
            Active trial (during hyperparameter tuning).

        Returns
        -------
        Predictor
            Fitted instance.

        """
        from lightgbm.callback import log_evaluation

        m = self._metric[0].name
        params = est_params_fit.copy()

        callbacks = params.pop("callbacks", []) + [log_evaluation(-1)]
        if trial and len(self._metric) == 1:
            callbacks.append(LightGBMPruningCallback(trial, m, "valid_1"))

        eval_metric = None
        if getattr(self, "_metric", None):
            eval_metric = LGBMetric(self._metric[0], task=self.task)

        try:
            estimator.fit(
                *data,
                eval_set=[data, validation] if validation else None,
                eval_metric=params.pop("eval_metric", eval_metric),
                callbacks=callbacks,
                **params,
            )
        except TrialPruned as ex:
            # Add the pruned step to the output
            step = str(ex).split(" ")[-1][:-1]
            steps = estimator.get_params()[self.has_validation]
            trial.params[self.has_validation] = f"{step}/{steps}"

            trial.set_user_attr("estimator", estimator)
            raise ex

        if validation:
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


class LinearDiscriminantAnalysis(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="LDA", metric="f1", verbose=2)
    ```

    """

    acronym = "LDA"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "discriminant_analysis"
    _estimators = CustomDict({"class": "LinearDiscriminantAnalysis"})

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

        if self._get_param("solver", params) == "svd":
            params.pop("shrinkage")

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
            solver=Cat(["svd", "lsqr", "eigen"]),
            shrinkage=Cat([None, "auto", 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        )


class LinearSVM(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="lSVM", metric="f1", verbose=2)
    ```

    """

    acronym = "lSVM"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn", "cuml"]

    _module = "svm"
    _estimators = CustomDict({"class": "LinearSVC", "reg": "LinearSVR"})

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

        if self.goal == "class":
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

    def _get_est(self, **params) -> PREDICTOR:
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
        if self.engine["estimator"] == "cuml" and self.goal == "class":
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
        if self.goal == "class":
            dist["penalty"] = Cat(["l1", "l2"])
            dist["loss"] = Cat(["hinge", "squared_hinge"])
        else:
            dist["loss"] = Cat(["epsilon_insensitive", "squared_epsilon_insensitive"])

        dist["C"] = Float(1e-3, 100, log=True)
        dist["dual"] = Cat([True, False])

        if self.engine["estimator"] == "cuml":
            dist.pop("dual")

        return dist


class LogisticRegression(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="RF", metric="f1", verbose=2)
    ```

    """

    acronym = "LR"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "LogisticRegression"})

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

        # Limitations on penalty + solver combinations
        penalty = self._get_param("penalty", params)
        solver = self._get_param("solver", params)
        cond_1 = penalty is None and solver == "liblinear"
        cond_2 = penalty == "l1" and solver not in ("liblinear", "saga")
        cond_3 = penalty == "elasticnet" and solver != "saga"

        if cond_1 or cond_2 or cond_3:
            params.replace_value("penalty", "l2")  # Change to default value

        if self._get_param("penalty", params) != "elasticnet":
            params.pop("l1_ratio")

        if self._get_param("penalty", params) is None:
            params.pop("C")

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        dist = CustomDict(
            penalty=Cat([None, "l1", "l2", "elasticnet"]),
            C=Float(1e-3, 100, log=True),
            solver=Cat(["lbfgs", "newton-cg", "liblinear", "sag", "saga"]),
            max_iter=Int(100, 1000, step=10),
            l1_ratio=Float(0, 1.0, step=0.1),
        )

        if self._gpu:
            dist.pop("solver")
            dist.pop("penalty")  # Only 'l2' is supported
        elif self.engine["estimator"] == "sklearnex":
            dist["solver"] = Cat(["lbfgs", "newton-cg"])

        return dist


class MultiLayerPerceptron(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="MLP", metric="f1", verbose=2)
    ```

    """

    acronym = "MLP"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = False
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

        # Drop layers when a previous layer has 0 neurons
        drop = False
        for param in [p for p in sorted(params) if p.startswith("hidden_layer")]:
            if params[param] == 0 or drop:
                drop = True
                params.pop(param)

        if self._get_param("solver", params) != "sgd":
            params.pop("learning_rate")
            params.pop("power_t")
        else:
            params.pop("learning_rate_init")

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
            hidden_layer_sizes.append(params.pop(param))

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
            hidden_layer_3=Int(0, 10),
            activation=Cat(["identity", "logistic", "tanh", "relu"]),
            solver=Cat(["lbfgs", "sgd", "adam"]),
            alpha=Float(1e-4, 0.1, log=True),
            batch_size=Cat(["auto", 8, 16, 32, 64, 128, 256]),
            learning_rate=Cat(["constant", "invscaling", "adaptive"]),
            learning_rate_init=Float(1e-3, 0.1, log=True),
            power_t=Float(0.1, 0.9, step=0.1),
            max_iter=Int(50, 500, step=10),
        )

        # Drop layers if sizes are specified by user
        return dist[3:] if "hidden_layer_sizes" in self._est_params else dist


class MultinomialNB(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="MNB", metric="f1", verbose=2)
    ```

    """

    acronym = "MNB"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
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
            fit_prior=Cat([True, False]),
        )


class OrdinaryLeastSquares(ClassRegModel):
    """Linear Regression.

    Ordinary Least Squares is just linear regression without any
    regularization. It fits a linear model with coefficients `w=(w1,
     ..., wp)` to minimize the residual sum of squares between the
    observed targets in the dataset, and the targets predicted by the
    linear approximation.

    Corresponding estimators are:

    - [LinearRegression][] for regression tasks.

    Read more in sklearn's [documentation][olsdocs].

    See Also
    --------
    atom.models:ElasticNet
    atom.models:Lasso
    atom.models:Ridge

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    atom = ATOMRegressor(X, y, random_state=1)
    atom.run(models="OLS", metric="r2", verbose=2)
    ```

    """

    acronym = "OLS"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "LinearRegression"})


class OrthogonalMatchingPursuit(ClassRegModel):
    """Orthogonal Matching Pursuit.

    Orthogonal Matching Pursuit implements the OMP algorithm for
    approximating the fit of a linear model with constraints imposed
    on the number of non-zero coefficients.

    Corresponding estimators are:

    - [OrthogonalMatchingPursuit][] for regression tasks.

    Read more in sklearn's [documentation][ompdocs].

    See Also
    --------
    atom.models:Lasso
    atom.models:LeastAngleRegression
    atom.models:OrdinaryLeastSquares

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    atom = ATOMRegressor(X, y, random_state=1)
    atom.run(models="OMP", metric="r2", verbose=2)
    ```

    """

    acronym = "OMP"
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"reg": "OrthogonalMatchingPursuit"})


class PassiveAggressive(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="PA", metric="f1", verbose=2)
    ```

    """

    acronym = "PA"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
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
        if self.goal == "class":
            loss = ["hinge", "squared_hinge"]
        else:
            loss = ["epsilon_insensitive", "squared_epsilon_insensitive"]

        return CustomDict(
            C=Float(1e-3, 100, log=True),
            max_iter=Int(500, 1500, step=50),
            loss=Cat(loss),
            average=Cat([True, False]),
        )


class Perceptron(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="Perc", metric="f1", verbose=2)
    ```

    """

    acronym = "Perc"
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    has_validation = "max_iter"
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "Perceptron"})

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

        if self._get_param("penalty", params) != "elasticnet":
            params.pop("l1_ratio")

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
            penalty=Cat([None, "l2", "l1", "elasticnet"]),
            alpha=Float(1e-4, 10, log=True),
            l1_ratio=Float(0.1, 0.9, step=0.1),
            max_iter=Int(500, 1500, step=50),
            eta0=Float(1e-2, 10, log=True),
        )


class QuadraticDiscriminantAnalysis(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="QDA", metric="f1", verbose=2)
    ```

    """

    acronym = "QDA"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
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


class RadiusNearestNeighbors(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(
        models="RNN",
        metric="f1",
        est_params={"outlier_label": "most_frequent"},
        verbose=2,
    )
    ```

    """

    acronym = "RNN"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
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
            weights=Cat(["uniform", "distance"]),
            algorithm=Cat(["auto", "ball_tree", "kd_tree", "brute"]),
            leaf_size=Int(20, 40),
            p=Int(1, 2),
        )


class RandomForest(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="RF", metric="f1", verbose=2)
    ```

    """

    acronym = "RF"
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "ensemble"
    _estimators = CustomDict(
        {"class": "RandomForestClassifier", "reg": "RandomForestRegressor"}
    )

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

        if not self._get_param("bootstrap", params):
            params.pop("max_samples")

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        if self.goal == "class":
            criterion = ["gini", "entropy"]
        else:
            if self.engine["estimator"] == "cuml":
                criterion = ["mse", "poisson", "gamma", "inverse_gaussian"]
            else:
                criterion = ["squared_error", "absolute_error", "poisson"]

        dist = CustomDict(
            n_estimators=Int(10, 500, step=10),
            criterion=Cat(criterion),
            max_depth=Cat([None, *range(1, 17)]),
            min_samples_split=Int(2, 20),
            min_samples_leaf=Int(1, 20),
            max_features=Cat([None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9]),
            bootstrap=Cat([True, False]),
            max_samples=Cat([None, 0.5, 0.6, 0.7, 0.8, 0.9]),
            ccp_alpha=Float(0, 0.035, step=0.005),
        )

        if self.engine["estimator"] == "sklearnex":
            dist.pop("criterion")
            dist.pop("ccp_alpha")
        elif self.engine["estimator"] == "cuml":
            dist.replace_key("criterion", "split_criterion")
            dist["max_depth"] = Int(1, 17)
            dist["max_features"] = Cat(["sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9])
            dist["max_samples"] = Float(0.5, 0.9, step=0.1)
            dist.pop("ccp_alpha")

        return dist


class Ridge(ClassRegModel):
    """Linear least squares with l2 regularization.

    If classifier, it first converts the target values into {-1, 1}
    and then treats the problem as a regression task.

    Corresponding estimators are:

    - [RidgeClassifier][] for classification tasks.
    - [Ridge][ridgeregressor] for regression tasks.

    Read more in sklearn's [documentation][ridgedocs].

    !!! warning
        Engines `sklearnex` and `cuml` are only available for regression
        tasks.

    See Also
    --------
    atom.models:BayesianRidge
    atom.models:ElasticNet
    atom.models:Lasso

    Examples
    --------
    ```pycon
    from atom import ATOMRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    atom = ATOMRegressor(X, y, random_state=1)
    atom.run(models="Ridge", metric="r2", verbose=2)
    ```

    """

    acronym = "Ridge"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = False
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
        dist = CustomDict(
            alpha=Float(1e-3, 10, log=True),
            solver=Cat(["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]),
        )

        if self.goal == "reg":
            if self.engine["estimator"] == "sklearnex":
                dist.pop("solver")  # Only supports 'auto'
            elif self.engine["estimator"] == "cuml":
                dist["solver"] = Cat(["eig", "svd", "cd"])

        return dist


class StochasticGradientDescent(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="SGD", metric="f1", verbose=2)
    ```

    """

    acronym = "SGD"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = "max_iter"
    supports_engines = ["sklearn"]

    _module = "linear_model"
    _estimators = CustomDict({"class": "SGDClassifier", "reg": "SGDRegressor"})

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

        if self._get_param("penalty", params) != "elasticnet":
            params.pop("l1_ratio")

        if self._get_param("learning_rate", params) == "optimal":
            params.pop("eta0")

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
            "modified_huber",
            "squared_hinge",
            "perceptron",
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ]

        return CustomDict(
            loss=Cat(loss if self.goal == "class" else loss[-4:]),
            penalty=Cat([None, "l1", "l2", "elasticnet"]),
            alpha=Float(1e-4, 1.0, log=True),
            l1_ratio=Float(0.1, 0.9, step=0.1),
            max_iter=Int(500, 1500, step=50),
            epsilon=Float(1e-4, 1.0, log=True),
            learning_rate=Cat(["constant", "invscaling", "optimal", "adaptive"]),
            eta0=Float(1e-2, 10, log=True),
            power_t=Float(0.1, 0.9, step=0.1),
            average=Cat([True, False]),
        )


class SupportVectorMachine(ClassRegModel):
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
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="SVM", metric="f1", verbose=2)
    ```

    """

    acronym = "SVM"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = None
    supports_engines = ["sklearn", "sklearnex", "cuml"]

    _module = "svm"
    _estimators = CustomDict({"class": "SVC", "reg": "SVR"})

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

        if self.goal == "class":
            params.pop("epsilon")

        kernel = self._get_param("kernel", params)
        if kernel == "poly":
            params.replace_value("gamma", "scale")  # Crashes in combination with "auto"
        else:
            params.pop("degree")

        if kernel not in ("rbf", "poly", "sigmoid"):
            params.pop("gamma")

        if kernel not in ("poly", "sigmoid"):
            params.pop("coef0")

        return params

    def _get_est(self, **params) -> PREDICTOR:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        if self.engine["estimator"] == "cuml" and self.goal == "class":
            return self._est_class(
                probability=params.pop("probability", True),
                random_state=params.pop("random_state", self.random_state),
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
            kernel=Cat(["linear", "poly", "rbf", "sigmoid"]),
            degree=Int(2, 5),
            gamma=Cat(["scale", "auto"]),
            coef0=Float(-1.0, 1.0),
            epsilon=Float(1e-3, 100, log=True),
            shrinking=Cat([True, False]),
        )

        if self.engine["estimator"] == "cuml":
            dist.pop("epsilon")
            dist.pop("shrinking")

        return dist


class XGBoost(ClassRegModel):
    """Extreme Gradient Boosting.

    XGBoost is an optimized distributed gradient boosting model
    designed to be highly efficient, flexible and portable. XGBoost
    provides a parallel tree boosting that solve many data science
    problems in a fast and accurate way.

    Corresponding estimators are:

    - [XGBClassifier][] for classification tasks.
    - [XGBRegressor][] for regression tasks.

    Read more in XGBoost's [documentation][xgbdocs].

    See Also
    --------
    atom.models:CatBoost
    atom.models:GradientBoostingMachine
    atom.models:LightGBM

    Examples
    --------
    ```pycon
    from atom import ATOMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    atom = ATOMClassifier(X, y, random_state=1)
    atom.run(models="XGB", metric="f1", verbose=2)
    ```

    """

    acronym = "XGB"
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    has_validation = "n_estimators"
    supports_engines = ["xgboost"]

    _module = "xgboost"
    _estimators = CustomDict({"class": "XGBClassifier", "reg": "XGBRegressor"})

    def _get_est(self, **params) -> PREDICTOR:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        eval_metric = None
        if getattr(self, "_metric", None):
            eval_metric = XGBMetric(self._metric[0], task=self.task)

        return self._est_class(
            eval_metric=params.pop("eval_metric", eval_metric),
            n_jobs=params.pop("n_jobs", self.n_jobs),
            tree_method=params.pop("tree_method", "gpu_hist" if self._gpu else None),
            gpu_id=self._device_id,
            verbosity=params.pop("verbosity", 0),
            random_state=params.pop("random_state", self.random_state),
            **params,
        )

    def _fit_estimator(
        self,
        estimator: PREDICTOR,
        data: tuple[DATAFRAME, SERIES],
        est_params_fit: dict,
        validation: tuple[DATAFRAME, SERIES] | None = None,
        trial: Trial | None = None,
    ):
        """Fit the estimator and perform in-training validation.

        Parameters
        ----------
        estimator: Predictor
            Instance to fit.

        data: tuple
            Training data of the form (X, y).

        est_params_fit: dict
            Additional parameters for the estimator's fit method.

        validation: tuple or None
            Validation data of the form (X, y). If None, no validation
            is performed.

        trial: [Trial][] or None
            Active trial (during hyperparameter tuning).

        Returns
        -------
        Predictor
            Fitted instance.

        """
        m = self._metric[0].name
        params = est_params_fit.copy()

        callbacks = params.pop("callbacks", [])
        if trial and len(self._metric) == 1:
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
            # Add the pruned step to the output
            step = str(ex).split(" ")[-1][:-1]
            steps = estimator.get_params()[self.has_validation]
            trial.params[self.has_validation] = f"{step}/{steps}"

            trial.set_user_attr("estimator", estimator)
            raise ex

        if validation:
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


# Time series ====================================================== >>

class ARIMA(ForecastModel):
    """Autoregressive Integrated Moving Average Model.

    Seasonal ARIMA models and exogeneous input is supported, hence this
    estimator is capable of fitting SARIMA, ARIMAX, and SARIMAX.

    An ARIMA model, is a generalization of an autoregressive moving
    average (ARMA) model, and is fitted to time-series data in an effort
    to forecast future points. ARIMA models can be especially
    efficacious in cases where data shows evidence of non-stationarity.

    The "AR" part of ARIMA indicates that the evolving variable of
    interest is regressed on its own lagged (i.e., prior observed)
    values. The "MA" part indicates that the regression error is
    actually a linear combination of error terms whose values occurred
    contemporaneously and at various times in the past. The "I" (for
    "integrated") indicates that the data values have been replaced with
    the difference between their values and the previous values (and this
    differencing process may have been performed more than once).

    Corresponding estimators are:

    - [ARIMA][arimaclass] for forecasting tasks.

    !!! warning
        ARIMA often runs into numerical errors when optimizing the
        hyperparameters. Possible solutions are:

        - Use the [AutoARIMA][] model instead.
        - Use [`est_params`][directforecaster-est_params] to specify the
          orders manually, e.g. `#!python atom.run("arima", n_trials=5,
          est_params={"order": (1, 1, 0)})`.
        - Use the `catch` parameter in [`ht_params`][directforecaster-ht_params]
          to avoid raising every exception, e.g. `#!python atom.run("arima",
          n_trials=5, ht_params={"catch": (Exception,)})`.

    See Also
    --------
    atom.models:AutoARIMA

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    atom = ATOMForecaster(X)
    atom.run(models="ARIMA", verbose=2)
    ```

    """

    acronym = "ARIMA"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
    supports_engines = ["sktime"]

    _module = "sktime.forecasting.arima"
    _estimators = CustomDict({"fc": "ARIMA"})

    _order = ("p", "d", "q")
    _sorder = ("Ps", "Ds", "Qs", "S")

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

        # If no seasonal periodicity, set seasonal components to zero
        if self._get_param("S", params) == 0:
            for p in self._sorder:
                params.replace_value(p, 0)

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

        # Convert params to hyperparameters order and seasonal_order
        if all(p in params for p in self._sorder):
            params.insert(0, "seasonal_order", tuple(params.pop(p) for p in self._sorder))
        if all(p in params for p in self._order):
            params.insert(0, "order", tuple(params.pop(p) for p in self._order))

        return params

    def _get_distributions(self) -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        methods = ["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]

        dist = CustomDict(
            p=Int(0, 2),
            d=Int(0, 1),
            q=Int(0, 2),
            Ps=Int(0, 2),
            Ds=Int(0, 1),
            Qs=Int(0, 2),
            S=Cat([0, 4, 6, 7, 12]),
            method=Cat(methods),
            maxiter=Int(50, 200, step=10),
            with_intercept=Cat([True, False]),
        )

        # Drop order and seasonal_order params if specified by user
        if "order" in self._est_params:
            for p in self._order:
                dist.pop(p)
        if "seasonal_order" in self._est_params:
            for p in self._sorder:
                dist.pop(p)

        return dist


class AutoARIMA(ForecastModel):
    """Automatic Autoregressive Integrated Moving Average Model.

    [ARIMA][] implementation that includes automated fitting of
    (S)ARIMA(X) hyperparameters (p, d, q, P, D, Q). The AutoARIMA
    algorithm seeks to identify the most optimal parameters for an
    ARIMA model, settling on a single fitted ARIMA model. This process
    is based on the commonly-used R function.

    AutoARIMA works by conducting differencing tests (i.e.,
    Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller or
    Phillips–Perron) to determine the order of differencing, d, and
    then fitting models within defined ranges. AutoARIMA also seeks
    to identify the optimal P and Q hyperparameters after conducting
    the Canova-Hansen to determine the optimal order of seasonal
    differencing.

    Note that due to stationarity issues, AutoARIMA might not find a
    suitable model that will converge. If this is the case, a ValueError
    is thrown suggesting stationarity-inducing measures be taken prior
    to re-fitting or that a new range of order values be selected.

    Corresponding estimators are:

    - [AutoARIMA][autoarimaclass] for forecasting tasks.

    See Also
    --------
    atom.models:ARIMA
    atom.models:ETS

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    atom = ATOMForecaster(X, random_state=1)
    atom.run(models="autoarima", verbose=2)
    ```

    """

    acronym = "AutoARIMA"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
    supports_engines = ["sktime"]

    _module = "sktime.forecasting.arima"
    _estimators = CustomDict({"fc": "AutoARIMA"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        methods = ["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]

        return CustomDict(
            method=Cat(methods),
            maxiter=Int(50, 200, step=10),
            with_intercept=Cat([True, False]),
        )


class ExponentialSmoothing(ForecastModel):
    """Exponential Smoothing forecaster.

    Holt-Winters exponential smoothing forecaster. The default settings
    use simple exponential smoothing, without trend and seasonality
    components.

    Corresponding estimators are:

    - [ExponentialSmoothing][esclass] for forecasting tasks.

    See Also
    --------
    atom.models:ARIMA
    atom.models:ETS
    atom.models:PolynomialTrend

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="ES", verbose=2)
    ```

    """

    acronym = "ES"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
    supports_engines = ["sktime"]

    _module = "sktime.forecasting.exp_smoothing"
    _estimators = CustomDict({"fc": "ExponentialSmoothing"})

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

        if self._get_param("trend", params) is None:
            params.pop("damped_trend")

        if self._get_param("sp", params) is None:
            params.pop("seasonal")

        return params

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        methods = ["L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr", "bh", "ls"]

        return CustomDict(
            trend=Cat(["add", "mul", None]),
            damped_trend=Cat([True, False]),
            seasonal=Cat(["add", "mul", None]),
            sp=Cat([4, 6, 7, 12, None]),
            use_boxcox=Cat([True, False]),
            initialization_method=Cat(["estimated", "heuristic"]),
            method=Cat(methods),
        )


class ETS(ForecastModel):
    """ETS model with automatic fitting capabilities.

    The ETS models are a family of time series models with an
    underlying state space model consisting of a level component,
    a trend component (T), a seasonal component (S), and an error
    term (E).

    Corresponding estimators are:

    - [AutoETS][] for forecasting tasks.

    See Also
    --------
    atom.models:ARIMA
    atom.models:ExponentialSmoothing
    atom.models:PolynomialTrend

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="ETS", verbose=2)

    ```

    """

    acronym = "ETS"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
    supports_engines = ["sktime"]

    _module = "sktime.forecasting.ets"
    _estimators = CustomDict({"fc": "AutoETS"})

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

        # If no seasonal periodicity, set seasonal components to zero
        if self._get_param("sp", params) == 1:
            params.pop("seasonal")

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
            error=Cat(["add", "mul"]),
            trend=Cat(["add", "mul", None]),
            damped_trend=Cat([True, False]),
            seasonal=Cat(["add", "mul", None]),
            sp=Cat([1, 4, 6, 7, 12]),
            initialization_method=Cat(["estimated", "heuristic"]),
            maxiter=Int(500, 2000, step=100),
            auto=Cat([True, False]),
            information_criterion=Cat(["aic", "bic", "aicc"]),
        )


class NaiveForecaster(ForecastModel):
    """Naive Forecaster.

    NaiveForecaster is a dummy forecaster that makes forecasts using
    simple strategies based on naive assumptions about past trends
    continuing. When used in [multivariate][] tasks, each column is
    forecasted with the same strategy.

    Corresponding estimators are:

    - [NaiveForecaster][naiveforecasterclass] for forecasting tasks.

    See Also
    --------
    atom.models:ExponentialSmoothing
    atom.models:Dummy
    atom.models:PolynomialTrend

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="NF", verbose=2)

    ```

    """

    acronym = "NF"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
    supports_engines = ["sktime"]

    _module = "sktime.forecasting.naive"
    _estimators = CustomDict({"fc": "NaiveForecaster"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(strategy=Cat(["last", "mean", "drift"]))


class PolynomialTrend(ForecastModel):
    """Polynomial Trend forecaster.

    Forecast time series data with a polynomial trend, using a sklearn
    [LinearRegression][] class to regress values of time series on
    index, after extraction of polynomial features.

    Corresponding estimators are:

    - [PolynomialTrendForecaster][] for forecasting tasks.

    See Also
    --------
    atom.models:ARIMA
    atom.models:ETS
    atom.models:NaiveForecaster

    Examples
    --------
    ```pycon
    from atom import ATOMForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    atom = ATOMForecaster(y, random_state=1)
    atom.run(models="PT", verbose=2)
    ```

    """

    acronym = "PT"
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = True
    has_validation = None
    supports_engines = ["sktime"]

    _module = "sktime.forecasting.trend"
    _estimators = CustomDict({"fc": "PolynomialTrendForecaster"})

    @staticmethod
    def _get_distributions() -> CustomDict:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        CustomDict
            Hyperparameter distributions.

        """
        return CustomDict(
            degree=Int(1, 5),
            with_intercept=Cat([True, False]),
        )


# Ensembles ======================================================== >>

class Stacking(ClassRegModel):
    """Stacking ensemble.

    Parameters
    ----------
    models: ClassMap
        Models from which to build the ensemble.

    **kwargs
        Additional keyword arguments for the estimator.

    """

    acronym = "Stack"
    needs_scaling = False
    has_validation = None
    native_multilabel = False
    native_multioutput = False
    supports_engines = []

    _module = "atom.ensembles"
    _estimators = CustomDict({"class": "StackingClassifier", "reg": "StackingRegressor"})

    def __init__(self, models: ClassMap, **kwargs):
        self._models = models
        kw_model = {k: v for k, v in kwargs.items() if k in sign(ClassRegModel.__init__)}
        super().__init__(**kw_model)
        self._est_params = {k: v for k, v in kwargs.items() if k not in kw_model}

    def _get_est(self, **params) -> PREDICTOR:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        estimators = []
        for m in self._models:
            if m.scaler:
                name = f"pipeline_{m.name}"
                est = Pipeline([("scaler", m.scaler), (m.name, m.estimator)])
            else:
                name = m.name
                est = m.estimator

            estimators.append((name, est))

        return self._est_class(
            estimators=estimators,
            n_jobs=params.pop("n_jobs", self.n_jobs),
            **params,
        )


class Voting(ClassRegModel):
    """Voting ensemble.

    Parameters
    ----------
    models: ClassMap
        Models from which to build the ensemble.

    **kwargs
        Additional keyword arguments for the estimator.

    """

    acronym = "Vote"
    needs_scaling = False
    has_validation = None
    native_multilabel = False
    native_multioutput = False
    supports_engines = []

    _module = "atom.ensembles"
    _estimators = CustomDict({"class": "VotingClassifier", "reg": "VotingRegressor"})

    def __init__(self, models: ClassMap, **kwargs):
        self._models = models
        kw_model = {k: v for k, v in kwargs.items() if k in sign(ClassRegModel.__init__)}
        super().__init__(**kw_model)
        self._est_params = {k: v for k, v in kwargs.items() if k not in kw_model}

        if self._est_params.get("voting") == "soft":
            for m in self._models:
                if not hasattr(m.estimator, "predict_proba"):
                    raise ValueError(
                        "Invalid value for the voting parameter. If "
                        "'soft', all models in the ensemble should have "
                        f"a predict_proba method, got {m._fullname}."
                    )

    def _get_est(self, **params) -> PREDICTOR:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        estimators = []
        for m in self._models:
            if m.scaler:
                name = f"pipeline_{m.name}"
                est = Pipeline([("scaler", m.scaler), (m.name, m.estimator)])
            else:
                name = m.name
                est = m.estimator

            estimators.append((name, est))

        return self._est_class(
            estimators=estimators,
            n_jobs=params.pop("n_jobs", self.n_jobs),
            **params,
        )


# Variables ======================================================== >>

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

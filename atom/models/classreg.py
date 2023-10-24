# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing classification and regression models.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from beartype.typing import Any, cast
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution as Cat
from optuna.distributions import FloatDistribution as Float
from optuna.distributions import IntDistribution as Int
from optuna.exceptions import TrialPruned
from optuna.integration import (
    CatBoostPruningCallback, LightGBMPruningCallback, XGBoostPruningCallback,
)
from optuna.trial import Trial

from atom.basemodel import ClassRegModel
from atom.utils.types import DataFrame, Pandas, Predictor
from atom.utils.utils import CatBMetric, LGBMetric, XGBMetric


class AdaBoost(ClassRegModel):
    """Adaptive Boosting.

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
    _estimators = {
        "classification": "AdaBoostClassifier",
        "regression": "AdaBoostRegressor",
    }

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = dict(
            n_estimators=Int(50, 500, step=10),
            learning_rate=Float(0.01, 10, log=True),
        )

        if self.task.is_classification:
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
    _estimators = {"regression": "ARDRegression"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {
        "classification": "BaggingClassifier",
        "regression": "BaggingRegressor",
    }

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {"regression": "BayesianRidge"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {"classification": "BernoulliNB"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {
        "classification": "CatBoostClassifier",
        "regression": "CatBoostRegressor",
    }

    def _get_parameters(self, trial: Trial) -> dict:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        dict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self._get_param("bootstrap_type", params) == "Bernoulli":
            if "bagging_temperature" in params:
                params["bagging_temperature"] = None
        elif self._get_param("bootstrap_type", params) == "Bayesian":
            if "subsample" in params:
                params["subsample"] = None

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
        estimator: Predictor,
        data: tuple[DataFrame, Pandas],
        est_params_fit: dict,
        validation: tuple[DataFrame, Pandas] | None = None,
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
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {"classification": "CategoricalNB"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {"classification": "ComplementNB"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {
        "classification": "DecisionTreeClassifier",
        "regression": "DecisionTreeRegressor",
    }

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        if self.task.is_classification:
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error", "friedman_mse", "poisson"]

        return dict(
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
    _estimators = {"classification": "DummyClassifier", "regression": "DummyRegressor"}

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = dict(
            strategy=Cat(["most_frequent", "prior", "stratified", "uniform"]),
            quantile=Float(0, 1.0, step=0.1),
        )

        if self.task.is_classification:
            dist.pop("quantile")
        else:
            dist["strategy"] = Cat(["mean", "median", "quantile"])

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
    _estimators = {"regression": "ElasticNet"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {
        "classification": "ExtraTreeClassifier",
        "regression": "ExtraTreeRegressor",
    }

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        if self.task.is_classification:
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error"]

        return dict(
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
    _estimators = {
        "classification": "ExtraTreesClassifier",
        "regression": "ExtraTreesRegressor",
    }

    def _get_parameters(self, trial: Trial) -> dict:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        dict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if not self._get_param("bootstrap", params):
            if "max_samples" in params:
                params["max_samples"] = None

        return params

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        if self.task.is_classification:
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error"]

        return dict(
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
    _estimators = {"classification": "GaussianNB"}


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

    * They are not sparse, i.e., they use the whole samples/features
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
    _estimators = {
        "classification": "GaussianProcessClassifier",
        "regression": "GaussianProcessRegressor",
    }


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
    _estimators = {
        "classification": "GradientBoostingClassifier",
        "regression": "GradientBoostingRegressor",
    }

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = dict(
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

        if self.task.is_multiclass:
            dist.pop("loss")  # Multiclass only supports log_loss
        elif self.task.is_regression:
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
    _estimators = {"regression": "HuberRegressor"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {
        "classification": "HistGradientBoostingClassifier",
        "regression": "HistGradientBoostingRegressor",
    }

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = dict(
            loss=Cat(["squared_error", "absolute_error", "poisson", "quantile", "gamma"]),
            quantile=Float(0, 1, step=0.1),
            learning_rate=Float(0.01, 1.0, log=True),
            max_iter=Int(10, 500, step=10),
            max_leaf_nodes=Int(10, 50),
            max_depth=Cat([None, *range(1, 17)]),
            min_samples_leaf=Int(10, 30),
            l2_regularization=Float(0, 1.0, step=0.1),
        )

        if self.task.is_classification:
            dist.pop("loss")
            dist.pop("quantile")

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
    _estimators = {
        "classification": "KNeighborsClassifier",
        "regression": "KNeighborsRegressor",
    }

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = dict(
            n_neighbors=Int(1, 100),
            weights=Cat(["uniform", "distance"]),
            algorithm=Cat(["auto", "ball_tree", "kd_tree", "brute"]),
            leaf_size=Int(20, 40),
            p=Int(1, 2),
        )

        if self._gpu:
            dist.pop("algorithm")  # Only 'brute' is supported
            if self.engine.get("estimator") == "cuml":
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
    _estimators = {"regression": "Lasso"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {"regression": "Lars"}


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
    _estimators = {"classification": "LGBMClassifier", "regression": "LGBMRegressor"}

    def _get_est(self, **params) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        # Custom lightgbm mapping for warnings
        # PYTHONWarnings doesn't work since they go from C/C++ code to stdout
        warns = dict(always=2, default=1, once=0, error=0, ignore=-1)

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
        estimator: Predictor,
        data: tuple[DataFrame, Pandas],
        est_params_fit: dict,
        validation: tuple[DataFrame, Pandas] | None = None,
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
            trial = cast(Trial, trial)  # If pruned, trial can't be None

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
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {"classification": "LinearDiscriminantAnalysis"}

    def _get_parameters(self, trial: Trial) -> dict:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        dict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self._get_param("solver", params) == "svd":
            if "shrinkage" in params:
                params["shrinkage"] = None

        return params

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {"classification": "LinearSVC", "regression": "LinearSVR"}

    def _get_parameters(self, trial: Trial) -> dict:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        dict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self.task.is_classification:
            if self._get_param("loss", params) == "hinge":
                # l1 regularization can't be combined with hinge
                if "penalty" in params:
                    params["penalty"] = "l2"
                # l2 regularization can't be combined with hinge when dual=False
                if "dual" in params:
                    params["dual"] = True
            elif self._get_param("loss", params) == "squared_hinge":
                # l1 regularization can't be combined with squared_hinge when dual=True
                if self._get_param("penalty", params) == "l1":
                    if "dual" in params:
                        params["dual"] = False
        elif self._get_param("loss", params) == "epsilon_insensitive":
            if "dual" in params:
                params["dual"] = True

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
        if self.engine.get("estimator") == "cuml" and self.task.is_classification:
            return self._est_class(probability=params.pop("probability", True), **params)
        else:
            return super()._get_est(**params)

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = dict(
            penalty=Cat(["l1", "l2"]),
            loss=Cat(["hinge", "squared_hinge"]),
            C=Float(1e-3, 100, log=True),
            dual=Cat([True, False]),
        )

        if not self.task.is_classification:
            dist["loss"] = Cat(["epsilon_insensitive", "squared_epsilon_insensitive"])

        if self.engine.get("estimator") == "cuml":
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
    _estimators = {"classification": "LogisticRegression"}

    def _get_parameters(self, trial: Trial) -> dict:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        dict
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
            if "penalty" in params:
                params["penalty"] = "l2"  # Change to default value

        return params

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = dict(
            penalty=Cat([None, "l1", "l2", "elasticnet"]),
            C=Float(1e-3, 100, log=True),
            solver=Cat(["lbfgs", "newton-cg", "liblinear", "sag", "saga"]),
            max_iter=Int(100, 1000, step=10),
            l1_ratio=Float(0, 1.0, step=0.1),
        )

        if self._gpu:
            if self.engine.get("estimator") == "cuml":
                dist["penalty"] = Cat(["none", "l1", "l2", "elasticnet"])
                dist.pop("solver")  # Only `qn` is supported
            elif self.engine.get("estimator") == "sklearnex":
                dist["penalty"] = Cat(["none", "l1", "elasticnet"])
                dist["solver"] = Cat(["lbfgs", "liblinear", "sag", "saga"])
        elif self.engine.get("estimator") == "sklearnex":
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
    _estimators = {"classification": "MLPClassifier", "regression": "MLPRegressor"}

    def _trial_to_est(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert trial's hyperparameters to parameters for the estimator.

        Parameters
        ----------
        params: dict
            Trial's hyperparameters.

        Returns
        -------
        dict
            Estimator's hyperparameters.

        """
        params = super()._trial_to_est(params)

        hidden_layer_sizes = []
        for param in [p for p in sorted(params) if p.startswith("hidden_layer")]:
            if value := params.pop(param):  # Neurons should be more than zero
                hidden_layer_sizes.append(value)

        if hidden_layer_sizes:
            params["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        return params

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = dict(
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

        # Drop layers if user specifies sizes
        if "hidden_layer_sizes" in self._est_params:
            return {k: v for k, v in dist.items() if "hidden_layer" not in k}
        else:
            return dist


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
    _estimators = {"classification": "MultinomialNB"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {"regression": "LinearRegression"}


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
    _estimators = {"regression": "OrthogonalMatchingPursuit"}


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
    _estimators = {
        "classification": "PassiveAggressiveClassifier",
        "regression": "PassiveAggressiveRegressor",
    }

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        if self.task.is_classification:
            loss = ["hinge", "squared_hinge"]
        else:
            loss = ["epsilon_insensitive", "squared_epsilon_insensitive"]

        return dict(
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
    _estimators = {"classification": "Perceptron"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {"classification": "QuadraticDiscriminantAnalysis"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(reg_param=Float(0, 1.0, step=0.1))


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
    _estimators = {
        "classification": "RadiusNeighborsClassifier",
        "regression": "RadiusNeighborsRegressor",
    }

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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
    _estimators = {
        "classification": "RandomForestClassifier",
        "regression": "RandomForestRegressor",
    }

    def _get_parameters(self, trial: Trial) -> dict:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        dict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if not self._get_param("bootstrap", params):
            if "max_samples" in params:
                params["max_samples"] = None

        return params

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        if self.task.is_classification:
            criterion = ["gini", "entropy"]
        else:
            if self.engine.get("estimator") == "cuml":
                criterion = ["mse", "poisson", "gamma", "inverse_gaussian"]
            else:
                criterion = ["squared_error", "absolute_error", "poisson"]

        dist = dict(
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

        if self.engine.get("estimator") == "sklearnex":
            dist.pop("criterion")
            dist.pop("ccp_alpha")
        elif self.engine.get("estimator") == "cuml":
            dist["split_criterion"] = dist.pop("criterion")
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
    _estimators = {"classification": "RidgeClassifier", "regression": "Ridge"}

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = dict(
            alpha=Float(1e-3, 10, log=True),
            solver=Cat(["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]),
        )

        if self.task.is_regression:
            if self.engine.get("estimator") == "sklearnex":
                dist.pop("solver")  # Only supports 'auto'
            elif self.engine.get("estimator") == "cuml":
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
    _estimators = {"classification": "SGDClassifier", "regression": "SGDRegressor"}

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
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

        return dict(
            loss=Cat(loss if self.task.is_classification else loss[-4:]),
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
    _estimators = {"classification": "SVC", "regression": "SVR"}

    def _get_parameters(self, trial: Trial) -> dict:
        """Get the trial's hyperparameters.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        dict
            Trial's hyperparameters.

        """
        params = super()._get_parameters(trial)

        if self._get_param("kernel", params) == "poly":
            if "gamma" in params:
                params["gamma"] = "scale"  # Crashes in combination with "auto"

        return params

    def _get_est(self, **params) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        if self.engine.get("estimator") == "cuml" and self.task.is_classification:
            return self._est_class(
                probability=params.pop("probability", True),
                random_state=params.pop("random_state", self.random_state),
                **params)
        else:
            return super()._get_est(**params)

    def _get_distributions(self) -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = dict(
            C=Float(1e-3, 100, log=True),
            kernel=Cat(["linear", "poly", "rbf", "sigmoid"]),
            degree=Int(2, 5),
            gamma=Cat(["scale", "auto"]),
            coef0=Float(-1.0, 1.0),
            epsilon=Float(1e-3, 100, log=True),
            shrinking=Cat([True, False]),
        )

        if self.task.is_classification:
            dist.pop("epsilon")

        if self.engine.get("estimator") == "cuml":
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
    _estimators = {"classification": "XGBClassifier", "regression": "XGBRegressor"}

    @property
    def trials(self) -> pd.DataFrame:
        """Overview of the trials' results.

        This property is only available for models that ran
        [hyperparameter tuning][]. All durations are in seconds.
        Columns include:

        - **[param_name]:** Parameter value used in this trial.
        - **estimator:** Estimator used in this trial.
        - **[metric_name]:** Metric score of the trial.
        - **[best_metric_name]:** Best score so far in this study.
        - **time_trial:** Duration of the trial.
        - **time_ht:** Duration of the hyperparameter tuning.
        - **state:** Trial's state (COMPLETE, PRUNED, FAIL).

        """
        trials = super().trials

        # XGBoost always minimizes metric, so flip sign
        for met in self._metric.keys():
            trials[met] = trials.apply(
                lambda row: -row[met] if row["state"] == "PRUNED" else row[met], axis=1
            )

        return trials

    def _get_est(self, **params) -> Predictor:
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
            device=params.pop("device", self.device),
            verbosity=params.pop("verbosity", 0),
            random_state=params.pop("random_state", self.random_state),
            **params,
        )

    def _fit_estimator(
        self,
        estimator: Predictor,
        data: tuple[DataFrame, Pandas],
        est_params_fit: dict,
        validation: tuple[DataFrame, Pandas] | None = None,
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
            trial = cast(Trial, trial)  # If pruned, trial can't be None

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
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Get the predefined hyperparameter distributions.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return dict(
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

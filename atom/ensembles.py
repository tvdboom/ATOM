# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the VotingModel and VotingEstimator classes.

"""

# Standard packages
import numpy as np
from copy import copy
from typing import Optional, Union
from typeguard import typechecked

# Own modules
from .models import MODEL_LIST
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, lst, check_is_fitted, get_acronym,
    get_best_score, catch_return, transform, composed, crash, method_to_log
)


class Voting(object):
    """Class for voting with the models in the pipeline."""

    def __init__(self, *args):
        self.T = args[0]
        self._weights = None
        self._pred_attrs = [None] * 8
        self._exclude = []  # Models excluded from voting
        if self.T.goal.startswith("class"):
            self.fullname = "VotingClassifier"
        else:
            self.fullname = "VotingRegressor"

    def __repr__(self):
        out = f"{self.fullname}"
        out += f"\n --> Models: {self.models}"
        out += f"\n --> Weights: {self.weights}"

        return out

    @property
    def weights(self):
        return self._weights

    @weights.setter
    @typechecked
    def weights(self, weights: Optional[SEQUENCE_TYPES]):
        if weights is not None and len(weights) != len(self.models):
            raise ValueError(
                "The weights should have the same length as the number of "
                f"models in the {self.fullname}, got len(weights)={len(weights)} "
                f"and len(models)={len(self.models)}."
            )

        self._weights = weights

    @property
    def exclude(self):
        return self._exclude

    @exclude.setter
    @typechecked
    def exclude(self, exclude: Union[str, SEQUENCE_TYPES]):
        self._exclude = [self.T._get_model(m) for m in lst(exclude)]

    @property
    def models_(self):
        return [m for m in self.T.models_ if m.name not in self.exclude]

    @property
    def models(self):
        return [m.name for m in self.models_]

    def _final_output(self):
        """Returns the average final output of the models."""

        def get_average_score(index):
            """Return the average score of the models on a metric."""
            scores = [get_best_score(m, index) for m in self.T.models_]
            return np.average(scores, weights=self.weights)

        check_is_fitted(self.T, "results")
        out = "   ".join([
            f"{m.name}: {get_average_score(i):.3f}"
            for i, m, in enumerate(self.T.metric_)
        ])
        return out

    @composed(crash, method_to_log, typechecked)
    def scoring(self, metric: Optional[str] = None, dataset: str = "test", **kwargs):
        """Get the average scoring of a specific metric.

        Parameters
        ----------
        metric: str, optional (default=None)
            Name of the metric to calculate. Choose from any of
            sklearn's SCORERS or one of the CUSTOM_METRICS. If None,
            returns the model's final results (ignores `dataset`).

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train" or "test".

        **kwargs
            Additional keyword arguments for the metric function.

        """
        if metric is None:
            return self._final_output()

        pred = [m.scoring(metric, dataset, **kwargs) for m in self.models_]
        return np.average(pred, axis=0, weights=self.weights)

    # Prediction methods =========================================== >>

    def _prediction(self, X, y=None, sample_weight=None, method="predict", **kwargs):
        """Get the mean of the prediction methods on new data.

        First transform the new data and apply the attribute on all
        the models. All models need to have the provided attribute.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        sample_weight: sequence or None, optional (default=None)
            Sample weights for the score method.

        method: str, optional (default="predict")
            Method of the model to be applied.

        **kwargs
            Keyword arguments for the transform method.

        Returns
        -------
        np.ndarray
            Return of the attribute.

        """
        check_is_fitted(self.T, "results")

        # Attribute check also done in BaseModel but better to
        # do it before all the data transformations
        for m in self.models_:
            if not hasattr(m.estimator, method):
                raise AttributeError(
                    f"{m.estimator.__class__.__name__} doesn't have a {method} method!"
                )

        # Verbosity is set to trainer's verbosity if left to default
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = self.T.verbose

        # Apply transformations per branch. Shared transformations are
        # not repeated. A unique copy of the data is stored per branch
        data = {k: (copy(X), copy(y)) for k in self.T._branches}
        step = {}  # Current step in the pipeline per branch
        for b1, v1 in self.T._branches.items():
            self.T.log(f"Transforming data for branch {b1}...", 1)

            for i, est1 in enumerate(v1.pipeline):
                # Skip if the transformation was already applied
                if step.get(b1, -1) < i:
                    kwargs["_one_trans"] = i
                    data[b1] = catch_return(transform(v1.pipeline, *data[b1], **kwargs))

                    for b2, v2 in self.T._branches.items():
                        try:  # Can fail if pipeline is shorter than i
                            est2 = v2.pipeline[i]
                            if b1 != b2 and est1 is est2:
                                # Update the data and step for the other branch
                                data[b2] = copy(data[b1])
                                step[b2] = i
                        except KeyError:
                            continue

        # Use pipeline=[] to skip the transformations
        if method == "predict":
            pred = np.array([
                m.predict(data[m.branch.name][0], pipeline=[])
                for m in self.T.models_
            ])
            majority = np.apply_along_axis(
                func1d=lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=0,
                arr=pred.astype("int")
            )
            return majority

        elif method in ("predict_proba", "predict_log_proba"):
            pred = np.array([
                getattr(m, method)(data[m.branch.name][0], pipeline=[])
                for m in self.models_
            ])
            return np.average(pred, axis=0, weights=self.weights)

        else:
            pred = np.array([
                m.score(*data[m.branch.name], sample_weight, pipeline=[])
                for m in self.models_
            ])
            return np.average(pred, axis=0, weights=self.weights)

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Majority voting of class labels for X."""
        return self._prediction(X, method="predict", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Average probability prediction of class labels for X."""
        return self._prediction(X, method="predict_proba", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Average log probability prediction of class labels for X."""
        return self._prediction(X, method="predict_log_proba", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: X_TYPES,
        y: Y_TYPES,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        **kwargs,
    ):
        """Average score of class labels."""
        return self._prediction(X, y, sample_weight, method="score", **kwargs)

    # Prediction properties ======================================== >>

    @composed(crash, method_to_log)
    def reset_predictions(self):
        """Clear all the prediction attributes."""
        self._pred_attrs = [None] * 8

    @property
    def predict_train(self):
        if self._pred_attrs[0] is None:
            pred = np.array([m.predict_train for m in self.models_])
            self._pred_attrs[0] = np.apply_along_axis(
                func1d=lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=0,
                arr=pred.astype("int"),
            )
        return self._pred_attrs[0]

    @property
    def predict_test(self):
        if self._pred_attrs[1] is None:
            pred = np.array([m.predict_test for m in self.models_])
            self._pred_attrs[1] = np.apply_along_axis(
                func1d=lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=0,
                arr=pred.astype("int"),
            )
        return self._pred_attrs[1]

    @property
    def predict_proba_train(self):
        if self._pred_attrs[2] is None:
            pred = np.array([m.predict_proba_train for m in self.models_])
            self._pred_attrs[2] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[2]

    @property
    def predict_proba_test(self):
        if self._pred_attrs[3] is None:
            pred = np.array([m.predict_proba_test for m in self.models_])
            self._pred_attrs[3] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[3]

    @property
    def predict_log_proba_train(self):
        if self._pred_attrs[4] is None:
            pred = np.array([m.predict_log_proba_train for m in self.models_])
            self._pred_attrs[4] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[4]

    @property
    def predict_log_proba_test(self):
        if self._pred_attrs[5] is None:
            pred = np.array([m.predict_log_proba_test for m in self.models_])
            self._pred_attrs[5] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[5]

    @property
    def score_train(self):
        if self._pred_attrs[6] is None:
            pred = np.array([m.score_train for m in self.models_])
            self._pred_attrs[6] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[6]

    @property
    def score_test(self):
        if self._pred_attrs[7] is None:
            pred = np.array([m.score_test for m in self.models_])
            self._pred_attrs[7] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[7]


class Stacking(object):
    """Class for stacking the models in the pipeline."""

    def __init__(self, *args):
        self.T = args[0]
        self._final_estimator = None
        self._pred_attrs = [None] * 10
        self._exclude = []  # Models excluded from stacking
        if self.T.goal.startswith("class"):
            self.fullname = "StackingClassifier"
            self.final_estimator = "LR"
        else:
            self.fullname = "StackingRegressor"
            self.final_estimator = "Ridge"

    def __repr__(self):
        out = f"{self.fullname}"
        out += f"\n --> Models: {self.models}"
        out += f"\n --> Final estimator: {self._final_estimator.__class__.__name__}"

        return out

    @property
    def n_jobs(self):
        return self.T.n_jobs

    @property
    def random_state(self):
        return self.T.random_state

    @property
    def final_estimator(self):
        return self._final_estimator

    @final_estimator.setter
    @typechecked
    def final_estimator(self, final_estimator: Union[str, callable]):
        if isinstance(final_estimator, str):
            # Set to right acronym and get the model's estimator
            model = MODEL_LIST[get_acronym(final_estimator)](self)
            self._final_estimator = model.get_estimator()
        else:
            self._final_estimator = final_estimator

    @property
    def exclude(self):
        return self._exclude

    @exclude.setter
    @typechecked
    def exclude(self, exclude: Union[str, SEQUENCE_TYPES]):
        self._exclude = [self.T._get_model(m) for m in lst(exclude)]

    @property
    def models_(self):
        return [m for m in self.T.models_ if m.name not in self.exclude]

    @property
    def models(self):
        return [m.name for m in self.models_]

    def _fit(self):
        self._in_fit_models = copy(self.models)

        X_train = []
        for m in self.models_:
            if hasattr(m.estimator, "predict_proba"):
                if self.T.task.startswith("bin"):
                    pred = m.predict_proba_train[:, 1]
                else:
                    pred = m.predict_proba_train
            elif hasattr(m.estimator, "decision_function"):
                pred = m.decision_function_train
            else:
                pred = m.predict_train

            if pred.ndim == 1:
                X_train.append(pred.reshape(-1, 1))
            else:
                X_train.append(pred)

        self.final_estimator.fit(np.hstack(X_train), self.T.y_train)

    def _final_output(self):
        """Returns the model's final output as a string."""
        # If bagging was used, we use a different format
        if self.mean_bagging is None:
            out = "   ".join([
                f"{m.name}: {lst(self.metric_test)[i]:.3f}"
                for i, m, in enumerate(self.T.metric_)
            ])
        else:
            out = "   ".join([
                f"{m.name}: {lst(self.mean_bagging)[i]:.3f}"
                " \u00B1 "
                f"{lst(self.std_bagging)[i]:.3f}"
                for i, m in enumerate(self.T.metric_)
            ])

        # Annotate if model overfitted when train 20% > test
        metric_train = lst(self.metric_train)
        metric_test = lst(self.metric_test)
        if metric_train[0] - 0.2 * metric_train[0] > metric_test[0]:
            out += " ~"

        return out

    @composed(crash, method_to_log, typechecked)
    def scoring(self, metric: Optional[str] = None, dataset: str = "test", **kwargs):
        """Get the average scoring of a specific metric.

        Parameters
        ----------
        metric: str, optional (default=None)
            Name of the metric to calculate. Choose from any of
            sklearn's SCORERS or one of the CUSTOM_METRICS. If None,
            returns the model's final results (ignores `dataset`).

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train" or "test".

        **kwargs
            Additional keyword arguments for the metric function.

        """
        if metric is None:
            return self._final_output()

        pred = [m.scoring(metric, dataset, **kwargs) for m in self.models_]
        return np.average(pred, axis=0, weights=self.weights)

    # Prediction methods =========================================== >>

    def _prediction(self, X, y=None, sample_weight=None, method="predict", **kwargs):
        """Get the mean of the prediction methods on new data.

        First transform the new data and apply the attribute on all
        the models. All models need to have the provided attribute.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        sample_weight: sequence or None, optional (default=None)
            Sample weights for the score method.

        method: str, optional (default="predict")
            Method of the model to be applied.

        **kwargs
            Keyword arguments for the transform method.

        Returns
        -------
        np.ndarray
            Return of the attribute.

        """
        check_is_fitted(self.T, "results")

        # Verbosity is set to trainer's verbosity if left to default
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = self.T.verbose

        # Apply transformations per branch. Shared transformations are
        # not repeated. A unique copy of the data is stored per branch
        data = {k: (copy(X), copy(y)) for k in self.T._branches}
        step = {}  # Current step in the pipeline per branch
        for b1, v1 in self.T._branches.items():
            self.T.log(f"Transforming data for branch {b1}...", 1)

            for i, est1 in enumerate(v1.pipeline):
                # Skip if the transformation was already applied
                if step.get(b1, -1) < i:
                    kwargs["_one_trans"] = i
                    data[b1] = catch_return(transform(v1.pipeline, *data[b1], **kwargs))

                    for b2, v2 in self.T._branches.items():
                        try:  # Can fail if pipeline is shorter than i
                            est2 = v2.pipeline[i]
                            if b1 != b2 and est1 is est2:
                                # Update the data and step for the other branch
                                data[b2] = copy(data[b1])
                                step[b2] = i
                        except KeyError:
                            continue

        # Use pipeline=[] to skip the transformations
        if method == "predict":
            pred = np.array([
                m.predict(data[m.branch.name][0], pipeline=[])
                for m in self.T.models_
            ])
            majority = np.apply_along_axis(
                func1d=lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=0,
                arr=pred.astype("int")
            )
            return majority

        elif method in ("predict_proba", "predict_log_proba"):
            pred = np.array([
                getattr(m, method)(data[m.branch.name][0], pipeline=[])
                for m in self.models_
            ])
            return np.average(pred, axis=0, weights=self.weights)

        else:
            pred = np.array([
                m.score(*data[m.branch.name], sample_weight, pipeline=[])
                for m in self.models_
            ])
            return np.average(pred, axis=0, weights=self.weights)

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Majority voting of class labels for X."""
        return self._prediction(X, method="predict", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Average probability prediction of class labels for X."""
        return self._prediction(X, method="predict_proba", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Average log probability prediction of class labels for X."""
        return self._prediction(X, method="predict_log_proba", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: X_TYPES,
        y: Y_TYPES,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        **kwargs,
    ):
        """Average score of class labels."""
        return self._prediction(X, y, sample_weight, method="score", **kwargs)

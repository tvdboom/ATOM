# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the VotingModel and VotingEstimator classes.

"""

# Standard packages
import numpy as np
from typing import Optional
from typeguard import typechecked

# Own modules
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, check_is_fitted, check_scaling,
    get_best_score, catch_return, transform, composed, crash, method_to_log
)


class Voting(object):
    """Model subclass for voting with the models in the pipeline."""

    def __init__(self, *args):
        super().__init__(
            estimator=VotingEstimator(args[0]),
            T=args[0],
            name="vote",
            acronym="vote",
            needs_scaling=False,
            type="kernel"
        )
        if self.T.goal.startswith("class"):
            self.fullname = "VotingClassifier"
        else:
            self.fullname = "VotingRegressor"

    def __repr__(self):
        out = f"{self.fullname}"
        out += f"\n --> Models: {self.T.models}"
        out += f"\n --> Weights: {self.weights}"

        return out

    @property
    def weights(self):
        return self.weights

    @weights.setter
    @typechecked
    def weights(self, weights: Optional[SEQUENCE_TYPES]):
        if weights is not None and len(weights) != len(self.T.models):
            raise ValueError(
                "The weights should have the same length as the number of "
                f"models in then the pipeline, got len(weights)={len(weights)}"
                f" and len(models)={len(self.T.models)}.")

        self.estimator.weights = weights

    def _get_average_score(self, index):
        """Return the average score of the models on a metric.

        Parameters
        ----------
        index: int
            Position of the trainer's metric.

        """
        scores = [get_best_score(m, index) for m in self.T.models_]
        return np.average(scores, weights=self.weights)

    def _final_output(self):
        """Returns the average final output of the models."""
        check_is_fitted(self.T, "results")
        out = "   ".join([
            f"{m.name}: {self._get_average_score(i):.3f}"
            for i, m, in enumerate(self.T.metric_)
        ])
        return out

    def _prediction(self, X, y=None, sample_weight=None, method="score", **kwargs):
        """Get the average of the prediction methods.

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

        method: str, optional (default="score")
            Method of the model to be applied.

        **kwargs
            Keyword arguments for the transform method.

        Returns
        -------
        np.ndarray
            Return of the attribute.

        """
        check_is_fitted(self.T, "results")

        # When there is a pipeline, apply all data transformations first
        if not self.T.branch.estimators.empty:
            if kwargs.get("verbose") is None:
                kwargs["verbose"] = self.T.verbose
            X, y = catch_return(transform(self.T.branch.estimators, X, y, **kwargs))

        results = np.array([])
        for m in self.T.models_:
            # Scale the data if needed
            if self.needs_scaling and not check_scaling(X):
                data = self.T.scaler.transform(X)
                print(data is X)
            else:
                data = X

            if y is None:
                results.append(getattr(m.estimator, method)(data))
            else:
                return getattr(m.estimator, method)(data, y, sample_weight)

        return np.average(results, axis=0, weights=self.weights)

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Majority voting of class labels for X."""
        check_is_fitted(self.T, "results")
        pred = np.asarray([est.predict(X, **kwargs) for est in self.T.models_]).T
        majority = np.apply_along_axis(
            func1d=lambda x: np.argmax(np.bincount(x, weights=self.weights)),
            axis=1,
            arr=pred.astype("int")
        )
        return majority

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Average probability prediction of class labels for X."""
        return self._prediction(X, method="predict_proba", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Average log probability prediction of class labels for X."""
        return self._prediction(X, method="predict_log_proba", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def decision_function(self, X: X_TYPES, **kwargs):
        """Average decision function of class labels for X."""
        return self._prediction(X, method="decision_function", **kwargs)

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


class VotingEstimator(object):
    """Voting estimator.

     This variation on sklearn's Voting estimators uses prefit
     estimators and applies hard voting via the predict method
     and soft voting via the predict_proba method on the models
     in the trainer's pipeline.

    Parameters
    ----------
    T: class
        Trainer from which the estimator is called.

     """

    def __init__(self, *args):
        self.T = args[0]  # Trainer instance
        self.weights = None

    def _prediction(self, X, y=None, sample_weight=None, method="score", **kwargs):
        """Get the average of the prediction methods.

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

        method: str, optional (default="score")
            Method of the model to be applied.

        **kwargs
            Keyword arguments for the transform method.

        Returns
        -------
        np.ndarray
            Return of the attribute.

        """
        check_is_fitted(self.T, "results")

        # When there is a pipeline, apply all data transformations first
        if not self.T.branch.estimators.empty:
            if kwargs.get("verbose") is None:
                kwargs["verbose"] = self.T.verbose
            X, y = catch_return(transform(self.T.branch.estimators, X, y, **kwargs))

        results = np.array([])
        for m in self.T.models_:
            # Scale the data if needed
            if self.needs_scaling and not check_scaling(X):
                data = self.T.scaler.transform(X)
                print(data is X)
            else:
                data = X

            if y is None:
                results.append(getattr(m.estimator, method)(data))
            else:
                return getattr(m.estimator, method)(data, y, sample_weight)

        return np.average(results, axis=0, weights=self.weights)

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Majority voting of class labels for X."""
        check_is_fitted(self.T, "results")
        pred = np.asarray([est.predict(X, **kwargs) for est in self.T.models_]).T
        majority = np.apply_along_axis(
            func1d=lambda x: np.argmax(np.bincount(x, weights=self.weights)),
            axis=1,
            arr=pred.astype("int")
        )
        return majority

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Average probability prediction of class labels for X."""
        return self._prediction(X, method="predict_proba", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Average log probability prediction of class labels for X."""
        return self._prediction(X, method="predict_log_proba", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def decision_function(self, X: X_TYPES, **kwargs):
        """Average decision function of class labels for X."""
        return self._prediction(X, method="decision_function", **kwargs)

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

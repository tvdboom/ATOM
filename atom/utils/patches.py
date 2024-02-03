"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing patches for external libraries.

"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier as VC
from sklearn.ensemble import VotingRegressor as VR
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.utils import Bunch
from sklearn.utils._set_output import _wrap_method_output
from sklearn.utils.multiclass import check_classification_targets
from sktime.forecasting.compose import EnsembleForecaster as EF
from sktime.forecasting.compose import StackingForecaster as SF
from typing_extensions import Self

from atom.utils.types import (
    Bool, Float, Int, Predictor, Scalar, Sequence, XSelector,
)
from atom.utils.utils import check_is_fitted


# Functions ======================================================== >>

def wrap_method_output(f: Callable, method: str) -> Callable:
    """Wrap sklearn's _wrap_method_output function.

    Custom implementation to avoid errors for transformers that allow
    only providing `y`. Is used internally by _SetOutputMixin.

    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        try:
            return _wrap_method_output(f, method)(self, *args, **kwargs)
        except TypeError:
            return f(self, *args, **kwargs)

    return wrapper


def fit_and_score(*args, **kwargs) -> dict[str, Any]:
    """Wrap sklearn's _fit_and_score function.

    Wrap the function sklearn.model_selection._validation._fit_and_score
    to, in turn, path sklearn's _score function to accept pipelines that
    drop samples during transforming, within a joblib parallel context.

    """

    def wrapper(*args, **kwargs) -> dict[str, Any]:
        with patch("sklearn.model_selection._validation._score", score(_score)):
            return _fit_and_score(*args, **kwargs)

    return wrapper(*args, **kwargs)


def score(f: Callable) -> Callable:
    """Patch decorator for sklearn's _score function.

    Monkey patch for sklearn.model_selection._validation._score
    function to score pipelines that drop samples during transforming.

    """

    def wrapper(*args, **kwargs) -> Float | dict[str, Float]:
        args_c = list(args)  # Convert to a list for item assignment
        if len(args[0]) > 1:  # Has transformers
            args_c[1], args_c[2] = args_c[0][:-1].transform(args_c[1], args_c[2])

        # Return f(final_estimator, X_transformed, y_transformed, ...)
        return f(args_c[0][-1], *args_c[1:], **kwargs)

    return wrapper


# Ensembles ======================================================== >>

class BaseVoting:
    """Base class for the patched voting estimators."""

    def _get_fitted_attrs(self):
        """Update the fit attributes (end with underscore)."""
        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            if est == "drop" or check_is_fitted(est, exception=False):
                self.named_estimators_[name] = est
            else:
                self.named_estimators_[name] = next(est_iter)

            if hasattr(est, "feature_names_in_"):
                self.feature_names_in_ = est.feature_names_in_

    def fit(
        self,
        X: XSelector,
        y: Sequence[Any],
        sample_weight: Sequence[Scalar] | None = None,
    ) -> Self:
        """Fit the estimators in the ensemble.

        Largely same code as sklearn's implementation with one major
        difference: estimators that are already fitted are skipped.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape (n_samples, n_features)

        y: sequence
            Target column.

        sample_weight: sequence or None, default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self
            Estimator instance.

        """
        names, all_estimators = self._validate_estimators()

        # Difference with sklearn's implementation, skip fitted estimators
        estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                clone(clf),
                X,
                y,
                sample_weight=sample_weight,
                message_clsname="Voting",
                message=self._log_message(names[idx], idx + 1, len(all_estimators)),
            )
            for idx, clf in enumerate(all_estimators)
            if clf != "drop" and not check_is_fitted(clf, exception=False)
        )

        self.estimators_ = []
        estimators = iter(estimators)
        for est in self.estimators:
            if est[1] != "drop":
                if check_is_fitted(est[1], exception=False):
                    self.estimators_.append(est[1])
                else:
                    self.estimators_.append(next(estimators))

        self._get_fitted_attrs()  # Update the fit attrs

        return self


class VotingClassifier(BaseVoting, VC):
    """Soft Voting/Majority Rule classifier.

    Modified version of sklearn's VotingClassifier. The differences
    are:

    - Doesn't fit estimators if they're already fitted.
    - Is considered fitted when all estimators are.
    - Doesn't implement a LabelEncoder to encode the target column.

    See sklearn's [VotingClassifier][] for a description of the
    parameters and attributes.

    """

    __module__ = VC.__module__
    __name__ = VC.__name__
    __qualname__ = VC.__qualname__
    __doc__ = VC.__doc__
    __annotations__ = VC.__annotations__

    def __init__(
        self,
        estimators: list[tuple[str, Predictor]],
        *,
        voting: str = "hard",
        weights: Sequence[Scalar] | None = None,
        n_jobs: Int | None = None,
        flatten_transform: Bool = True,
        verbose: Bool = False,
    ):
        super().__init__(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=n_jobs,
            flatten_transform=flatten_transform,
            verbose=verbose,
        )

        # If all estimators are prefit, create fitted attrs
        if all(
            est[1] == "drop" or check_is_fitted(est[1], exception=False)
            for est in self.estimators
        ):
            self.estimators_ = [est[1] for est in self.estimators if est[1] != "drop"]
            self._get_fitted_attrs()

    def fit(
        self,
        X: XSelector,
        y: Sequence[Any],
        sample_weight: Sequence[Scalar] | None = None,
    ) -> Self:
        """Fit the estimators, skipping prefit ones.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: sequence
            Target column.

        sample_weight: sequence or None, default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self
            Estimator instance.

        """
        check_classification_targets(y)
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError(
                "Multilabel and multioutput classification is not supported."
            )

        if self.voting not in ("soft", "hard"):
            raise ValueError(f"Voting must be 'soft' or 'hard', got (voting={self.voting}).")

        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(
                "Number of estimators and weights must be equal, got "
                f"{len(self.weights)} weights, {len(self.estimators)} estimators."
            )

        return super().fit(X, y, sample_weight)

    def predict(self, X: XSelector) -> np.ndarray:
        """Predict class labels for X.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        Returns
        -------
        np.array
            Predicted class labels.

        """
        check_is_fitted(self)
        if self.voting == "soft":
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            return np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self._weights_not_none)),
                axis=1,
                arr=self._predict(X),
            )


class VotingRegressor(BaseVoting, VR):
    """Soft Voting/Majority Rule regressor.

    Modified version of sklearn's VotingRegressor. Differences are:

    - Doesn't fit estimators if they're already fitted.
    - Is considered fitted when all estimators are.

    See sklearn's [VotingRegressor][] for a description of the
    parameters and attributes.

    """

    __module__ = VR.__module__
    __name__ = VR.__name__
    __qualname__ = VR.__qualname__
    __doc__ = VR.__doc__
    __annotations__ = VR.__annotations__

    def __init__(
        self,
        estimators: list[tuple[str, Predictor]],
        *,
        weights: Sequence[Scalar] | None = None,
        n_jobs: Int | None = None,
        verbose: Bool = False,
    ):
        super().__init__(
            estimators,
            weights=weights,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # If all estimators are prefit, create fitted attrs
        if all(
            est[1] == "drop" or check_is_fitted(est[1], exception=False) for est in self.estimators
        ):
            self.estimators_ = [est[1] for est in self.estimators if est[1] != "drop"]
            self._get_fitted_attrs()


class BaseForecaster:
    """Base class for the patched ensemble forecasters."""

    def _fit_forecasters(self, forecasters, y, X, fh):
        """Fit all forecasters in parallel.

        Patched to skip already fitted forecasters from refitting.

        """
        if all(check_is_fitted(fc, exception=False) for fc in forecasters):
            self.forecasters_ = [deepcopy(fc) for fc in forecasters]
        else:
            self.forecasters_ = Parallel(n_jobs=self.n_jobs)(
                delayed(lambda fc, y, X, fh: fc.fit(y, X, fh))(fc.clone(), y, X, fh)
                for fc in forecasters
            )

    def _predict_forecasters(self, fh=None, X=None):
        """Collect results from forecaster.predict() calls.

        Patched to convert all prediction to pd.DataFrame, which is
        done normally during fit(). If absent, the prediction fails
        when trying to get a multilevel index.

        """
        return [pd.DataFrame(forecaster.predict(fh=fh, X=X)) for forecaster in self.forecasters_]


class EnsembleForecaster(BaseForecaster, EF):
    """Ensemble of voting forecasters.

    Modified version of sktime's EnsembleForecaster. The differences
    are:

    - Doesn't fit estimators if they're already fitted.

    See sktime's [EnsembleForecaster][] for a description of the
    parameters and attributes.

    """

    __module__ = EF.__module__
    __name__ = EF.__name__
    __qualname__ = EF.__qualname__
    __doc__ = EF.__doc__
    __annotations__ = EF.__annotations__


class StackingForecaster(BaseForecaster, SF):
    """Ensemble of stacking forecasters.

    Modified version of sktime's StackingForecaster. The differences
    are:

    - Doesn't fit estimators if they're already fitted.

    See sktime's [StackingForecaster][] for a description of the
    parameters and attributes.

    """

    __module__ = SF.__module__
    __name__ = SF.__name__
    __qualname__ = SF.__qualname__
    __doc__ = SF.__doc__
    __annotations__ = SF.__annotations__

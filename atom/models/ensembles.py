"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing all ensemble models.

"""

from __future__ import annotations

from typing import Any, ClassVar

from atom.basemodel import BaseModel
from atom.utils.types import Model, Predictor


class Stacking(BaseModel):
    """Stacking ensemble.

    Parameters
    ----------
    models: list of Model
        Models from which to build the ensemble.

    **kwargs
        Additional keyword arguments for BaseModel's constructor.

    """

    acronym = "Stack"
    handles_missing = False
    needs_scaling = False
    validation = None
    native_multilabel = False
    native_multioutput = False
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.ensemble.StackingClassifier",
        "regression": "sklearn.ensemble.StackingRegressor",
        "forecast": "atom.utils.patches.StackingForecaster",
    }

    def __init__(self, models: list[Model], **kwargs):
        super().__init__(**kwargs)
        self._models = models

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Parameters
        ----------
        params: dict
            Hyperparameters for the estimator.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        default = {
            "estimators" if not self.task.is_forecast else "forecasters": [
                (m.name, m.export_pipeline()[-2:] if m.scaler else m.estimator)
                for m in self._models
            ]
        }

        return super()._get_est(default | params)


class Voting(BaseModel):
    """Voting ensemble.

    Parameters
    ----------
    models: list of Model
        Models from which to build the ensemble.

    **kwargs
        Additional keyword arguments for BaseModel's constructor.

    """

    acronym = "Vote"
    handles_missing = False
    needs_scaling = False
    validation = None
    native_multilabel = False
    native_multioutput = False
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "atom.utils.patches.VotingClassifier",
        "regression": "atom.utils.patches.VotingRegressor",
        "forecast": "atom.utils.patches.EnsembleForecaster",
    }

    def __init__(self, models: list[Model], **kwargs):
        super().__init__(**kwargs)
        self._models = models

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Get the model's estimator with unpacked parameters.

        Parameters
        ----------
        params: dict
            Hyperparameters for the estimator.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        default = {
            "estimators" if not self.task.is_forecast else "forecasters": [
                (m.name, m.export_pipeline()[-2:] if m.scaler else m.estimator)
                for m in self._models
            ]
        }

        return super()._get_est(default | params)

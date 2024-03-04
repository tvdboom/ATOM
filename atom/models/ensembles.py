"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing all ensemble models.

"""

from __future__ import annotations

from typing import Any, ClassVar

from atom.basemodel import BaseModel, ClassRegModel, ForecastModel
from atom.utils.types import Model, Predictor
from atom.utils.utils import Goal


def create_stacking_model(**kwargs) -> BaseModel:
    """Create a stacking model.

    This function dynamically assigns the parent to the class.

    Parameters
    ----------
    kwargs
        Additional keyword arguments passed to the model's constructor.

    Returns
    -------
    Stacking
        Ensemble model.

    """
    base = ForecastModel if kwargs["goal"] is Goal.forecast else ClassRegModel

    class Stacking(base):  # type: ignore[valid-type]
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
        multiple_seasonality = False
        native_multilabel = False
        native_multioutput = False
        supports_engines = ("sklearn",)

        _estimators: ClassVar[dict[str, str]] = {
            "classification": "sklearn.ensemble.StackingClassifier",
            "regression": "sklearn.ensemble.StackingRegressor",
            "forecast": "sktime.forecasting.compose.StackingForecaster",
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
            # We use _est_class with get_params instead of just a dict
            # to also fix the parameters of the models in the ensemble
            estimator = self._est_class(
                **{
                    "estimators" if not self.task.is_forecast else "forecasters": [
                        (m.name, m.export_pipeline()[-2:] if m.scaler else m.estimator)
                        for m in self._models
                    ]
                }
            )

            # Drop the model names from params since those
            # are not direct parameters of the ensemble
            default = {
                k: v
                for k, v in estimator.get_params().items()
                if k not in (m.name for m in self._models)
            }

            return super()._get_est(default | params)

    return Stacking(**kwargs)


def create_voting_model(**kwargs) -> BaseModel:
    """Create a voting model.

    This function dynamically assigns the parent to the class.

    Parameters
    ----------
    kwargs
        Additional keyword arguments passed to the model's constructor.

    Returns
    -------
    Stacking
        Ensemble model.

    """
    base = ForecastModel if kwargs["goal"] is Goal.forecast else ClassRegModel

    class Voting(base):   # type: ignore[valid-type]
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
        multiple_seasonality = False
        native_multilabel = False
        native_multioutput = False
        supports_engines = ("sklearn",)

        _estimators: ClassVar[dict[str, str]] = {
            "classification": "sklearn.ensemble.VotingClassifier",
            "regression": "sklearn.ensemble.VotingRegressor",
            "forecast": "sktime.forecasting.compose.EnsembleForecaster",
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
            # We use _est_class with get_params instead of just a dict
            # to also fix the parameters of the models in the ensemble
            estimator = self._est_class(
                **{
                    "estimators" if not self.task.is_forecast else "forecasters": [
                        (m.name, m.export_pipeline()[-2:] if m.scaler else m.estimator)
                        for m in self._models
                    ]
                }
            )

            # Drop the model names from params since those
            # are not direct parameters of the ensemble
            default = {
                k: v
                for k, v in estimator.get_params().items()
                if k not in (m.name for m in self._models)
            }

            return super()._get_est(default | params)

    return Voting(**kwargs)

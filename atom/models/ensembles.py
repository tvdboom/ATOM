"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing all ensemble models.

"""

from __future__ import annotations

from typing import Any, ClassVar

from atom.basemodel import ClassRegModel
from atom.utils.types import Model, Predictor
from atom.utils.utils import sign


class Stacking(ClassRegModel):
    """Stacking ensemble.

    Parameters
    ----------
    models: list of Model
        Models from which to build the ensemble.

    **kwargs
        Additional keyword arguments for the estimator.

    """

    acronym = "Stack"
    needs_scaling = False
    has_validation = None
    native_multilabel = False
    native_multioutput = False
    supports_engines = ()

    _module = "atom.ensembles"
    _estimators: ClassVar[dict[str, str]] = {
        "classification": "StackingClassifier",
        "regression": "StackingRegressor",
    }

    def __init__(self, models: list[Model], **kwargs):
        self._models = models
        kw_model = {k: v for k, v in kwargs.items() if k in sign(ClassRegModel.__init__)}
        super().__init__(**kw_model)
        self._est_params = {k: v for k, v in kwargs.items() if k not in kw_model}

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
        return self._est_class(
            estimators=[
                (m.name, m.export_pipeline() if m.scaler else m.estimator) for m in self._models
            ],
            n_jobs=params.pop("n_jobs", self.n_jobs),
            **params,
        )


class Voting(ClassRegModel):
    """Voting ensemble.

    Parameters
    ----------
    models: list of Model
        Models from which to build the ensemble.

    **kwargs
        Additional keyword arguments for the estimator.

    """

    acronym = "Vote"
    needs_scaling = False
    has_validation = None
    native_multilabel = False
    native_multioutput = False
    supports_engines = ()

    _module = "atom.ensembles"
    _estimators: ClassVar[dict[str, str]] = {
        "classification": "VotingClassifier",
        "regression": "VotingRegressor",
    }

    def __init__(self, models: list[Model], **kwargs):
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
                        f"a predict_proba method, got {m.fullname}."
                    )

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
        return self._est_class(
            estimators=[
                (m.name, m.export_pipeline() if m.scaler else m.estimator) for m in self._models
            ],
            n_jobs=params.pop("n_jobs", self.n_jobs),
            **params,
        )

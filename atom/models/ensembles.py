# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing all ensemble models.

"""

from __future__ import annotations

from atom.basemodel import ClassRegModel
from atom.pipeline import Pipeline
from atom.utils.types import Predictor
from atom.utils.utils import ClassMap, sign


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
    supports_engines: list[str] = []

    _module = "atom.ensembles"
    _estimators = {
        "classification": "StackingClassifier",
        "regression": "StackingRegressor",
    }

    def __init__(self, models: ClassMap, **kwargs):
        self._models = models
        kw_model = {k: v for k, v in kwargs.items() if k in sign(ClassRegModel.__init__)}
        super().__init__(**kw_model)
        self._est_params = {k: v for k, v in kwargs.items() if k not in kw_model}

    def _get_est(self, **params) -> Predictor:
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
    supports_engines: list[str] = []

    _module = "atom.ensembles"
    _estimators = {"classification": "VotingClassifier", "regression": "VotingRegressor"}

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
                        f"a predict_proba method, got {m.fullname}."
                    )

    def _get_est(self, **params) -> Predictor:
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

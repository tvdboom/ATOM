# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the CustomModel class.

"""

from typing import Any

from atom.basemodel import ClassRegModel
from atom.utils.types import Predictor


class CustomModel(ClassRegModel):
    """Model with estimator provided by user."""

    def __init__(self, **kwargs):
        # Assign the estimator and store the provided parameters
        if callable(est := kwargs.pop("estimator")):
            self._est = est
            self._params = {}
        else:
            self._est = est.__class__
            self._params = est.get_params()

        if hasattr(est, "name"):
            name = est.name
        else:
            from atom.models import MODELS

            # If no name is provided, use the name of the class
            name = self.fullname
            if len(n := list(filter(str.isupper, name))) >= 2 and n not in MODELS:
                name = "".join(n)

        self.acronym = getattr(est, "acronym", name)
        if not name.startswith(self.acronym):
            raise ValueError(
                f"The name ({name}) and acronym ({self.acronym}) of model "
                f"{self.fullname} do not match. The name should start with "
                f"the model's acronym."
            )

        self.needs_scaling = getattr(est, "needs_scaling", False)
        self.native_multilabel = getattr(est, "native_multilabel", False)
        self.native_multioutput = getattr(est, "native_multioutput", False)
        self.has_validation = getattr(est, "has_validation", None)

        super().__init__(name=name, **kwargs)

    @property
    def fullname(self) -> str:
        """Return the estimator's class name."""
        return self._est_class.__name__

    @property
    def _est_class(self) -> type[Predictor]:
        """Return the estimator's class."""
        return self._est

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
        return super()._get_est(self._params | params)

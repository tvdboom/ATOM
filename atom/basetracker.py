# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the BaseTracker class.

"""

from __future__ import annotations

from atom.utils.types import Bool
from atom.utils.utils import TrackingParams


class BaseTracker:

    # Tracking parameters for mlflow
    _tracking_params = TrackingParams(
        log_ht=True,
        log_plots=True,
        log_data=False,
        log_pipeline=False,
    )

    @property
    def log_ht(self) -> bool:
        """Whether to track every trial of the hyperparameter tuning."""
        return self._tracking_params.log_ht

    @log_ht.setter
    def log_ht(self, value: Bool):
        self._tracking_params.log_ht = bool(value)

    @property
    def log_plots(self) -> bool:
        """Whether to save plots as artifacts."""
        return self._tracking_params.log_plots

    @log_plots.setter
    def log_plots(self, value: Bool):
        self._tracking_params.log_plots = bool(value)

    @property
    def log_data(self) -> bool:
        """Whether to save the train and test sets."""
        return self._tracking_params.log_data

    @log_data.setter
    def log_data(self, value: Bool):
        self._tracking_params.log_data = bool(value)

    @property
    def log_pipeline(self) -> bool:
        """Whether to save the model's pipeline."""
        return self._tracking_params.log_pipeline

    @log_pipeline.setter
    def log_pipeline(self, value: Bool):
        self._tracking_params.log_pipeline = bool(value)

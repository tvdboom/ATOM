# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseTracker class.

"""

from __future__ import annotations

from dataclasses import dataclass



from atom.utils.types import BOOL


@dataclass
class TrackingParams:
    """Tracking parameters for a mlflow experiment."""

    log_ht: BOOL  # Track every trial of the hyperparameter tuning
    log_model: BOOL  # Save the model estimator after fitting
    log_plots: BOOL  # Save plot artifacts
    log_data: BOOL  # Save the train and test sets
    log_pipeline: BOOL  # Save the model's pipeline


class BaseTracker:

    # Tracking parameters for mlflow
    _tracking_params = TrackingParams(
        log_ht=True,
        log_model=True,
        log_plots=True,
        log_data=False,
        log_pipeline=False,
    )

    @property
    def log_ht(self) -> BOOL:
        """Whether to track every trial of the hyperparameter tuning."""
        return self._tracking_params.log_ht

    @log_ht.setter
    def log_ht(self, value: BOOL):
        self._tracking_params.log_ht = value

    @property
    def log_model(self) -> BOOL:
        """Whether to save the model's estimator after fitting."""
        return self._tracking_params.log_model

    @log_model.setter
    def log_model(self, value: BOOL):
        self._tracking_params.log_model = value

    @property
    def log_plots(self) -> BOOL:
        """Whether to save plots as artifacts."""
        return self._tracking_params.log_plots

    @log_plots.setter
    def log_plots(self, value: BOOL):
        self._tracking_params.log_plots = value

    @property
    def log_data(self) -> BOOL:
        """Whether to save the train and test sets."""
        return self._tracking_params.log_data

    @log_data.setter
    def log_data(self, value: BOOL):
        self._tracking_params.log_data = value

    @property
    def log_pipeline(self) -> BOOL:
        """Whether to save the model's pipeline."""
        return self._tracking_params.log_pipeline

    @log_pipeline.setter
    def log_pipeline(self, value: BOOL):
        self._tracking_params.log_pipeline = value

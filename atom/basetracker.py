# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseTracker class.

"""

from dataclasses import dataclass

from typeguard import typechecked


@dataclass
class TrackingParams:
    """Tracking parameters for a mlflow experiment."""

    log_ht: bool  # Track every trial of the hyperparameter tuning
    log_model: bool  # Save the model estimator after fitting
    log_plots: bool  # Save plot artifacts
    log_data: bool  # Save the train and test sets
    log_pipeline: bool  # Save the model's pipeline


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
    def log_ht(self) -> bool:
        """Whether to track every trial of the hyperparameter tuning."""
        return self._tracking_params.log_ht

    @log_ht.setter
    @typechecked
    def log_ht(self, value: bool):
        self._tracking_params.log_ht = value

    @property
    def log_model(self) -> bool:
        """Whether to save the model's estimator after fitting."""
        return self._tracking_params.log_model

    @log_model.setter
    @typechecked
    def log_model(self, value: bool):
        self._tracking_params.log_model = value

    @property
    def log_plots(self) -> bool:
        """Whether to save plots as artifacts."""
        return self._tracking_params.log_plots

    @log_plots.setter
    @typechecked
    def log_plots(self, value: bool):
        self._tracking_params.log_plots = value

    @property
    def log_data(self) -> bool:
        """Whether to save the train and test sets."""
        return self._tracking_params.log_data

    @log_data.setter
    @typechecked
    def log_data(self, value: bool):
        self._tracking_params.log_data = value

    @property
    def log_pipeline(self) -> bool:
        """Whether to save the model's pipeline."""
        return self._tracking_params.log_pipeline

    @log_pipeline.setter
    @typechecked
    def log_pipeline(self, value: bool):
        self._tracking_params.log_pipeline = value

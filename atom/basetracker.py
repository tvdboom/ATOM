# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the BaseTracker class.

"""

from __future__ import annotations

from dataclasses import dataclass

from atom.utils.types import Bool


@dataclass
class TrackingParams:
    """Tracking parameters for a mlflow experiment."""

    log_ht: Bool  # Track every trial of the hyperparameter tuning
    log_plots: Bool  # Save plot artifacts
    log_data: Bool  # Save the train and test sets
    log_pipeline: Bool  # Save the model's pipeline


class BaseTracker:

    # Tracking parameters for mlflow
    _tracking_params = TrackingParams(
        log_ht=True,
        log_plots=True,
        log_data=False,
        log_pipeline=False,
    )

    @property
    def log_ht(self) -> Bool:
        """Whether to track every trial of the hyperparameter tuning."""
        return self._tracking_params.log_ht

    @log_ht.setter
    def log_ht(self, value: Bool):
        self._tracking_params.log_ht = value

    @property
    def log_plots(self) -> Bool:
        """Whether to save plots as artifacts."""
        return self._tracking_params.log_plots

    @log_plots.setter
    def log_plots(self, value: Bool):
        self._tracking_params.log_plots = value

    @property
    def log_data(self) -> Bool:
        """Whether to save the train and test sets."""
        return self._tracking_params.log_data

    @log_data.setter
    def log_data(self, value: Bool):
        self._tracking_params.log_data = value

    @property
    def log_pipeline(self) -> Bool:
        """Whether to save the model's pipeline."""
        return self._tracking_params.log_pipeline

    @log_pipeline.setter
    def log_pipeline(self, value: Bool):
        self._tracking_params.log_pipeline = value

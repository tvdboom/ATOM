# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module for plots.

"""

from abc import ABCMeta

from atom.plots.dataplot import DataPlot
from atom.plots.hyperparametertuningplot import HyperparameterTuningPlot
from atom.plots.predictionplot import PredictionPlot
from atom.plots.shapplot import ShapPlot


class ATOMPlot(
    DataPlot,
    HyperparameterTuningPlot,
    PredictionPlot,
    ShapPlot,
    metaclass=ABCMeta,
):
    """Plot classes inherited by main ATOM classes."""


class RunnerPlot(HyperparameterTuningPlot, PredictionPlot, ShapPlot, metaclass=ABCMeta):
    """Plot classes inherited by the runners and callable from models."""

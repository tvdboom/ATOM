# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module for plots.

"""

from abc import ABC

from atom.plots.dataplot import DataPlot
from atom.plots.featureselectionplot import FeatureSelectionPlot
from atom.plots.hyperparametertuningplot import HyperparameterTuningPlot
from atom.plots.predictionplot import PredictionPlot
from atom.plots.shapplot import ShapPlot


class ATOMPlot(
    FeatureSelectionPlot,
    DataPlot,
    HyperparameterTuningPlot,
    PredictionPlot,
    ShapPlot,
    ABC,
):
    """Plot classes inherited by main ATOM classes."""
    pass


class RunnerPlot(HyperparameterTuningPlot, PredictionPlot, ShapPlot, ABC):
    """Plot classes inherited by the runners and callable from models."""
    pass

# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing utilities for typing analysis.

"""

from __future__ import annotations

from typing import Callable, Literal, Protocol, TypedDict, Union

import modin.pandas as md
import numpy as np
import pandas as pd
import scipy.sparse as sps


# Variable types for isinstance ==================================== >>

# TODO: From Python 3.10, isinstance accepts pipe operator (change by then)
BOOL_TYPES = (bool, np.bool_)
INT_TYPES = (int, np.integer)
FLOAT_TYPES = (float, np.floating)
SCALAR_TYPES = (*INT_TYPES, *FLOAT_TYPES)
INDEX_TYPES = (pd.Index, md.Index)
TS_INDEX_TYPES = (
    pd.PeriodIndex,
    md.PeriodIndex,
    pd.DatetimeIndex,
    md.DatetimeIndex,
    pd.TimedeltaIndex,
    md.TimedeltaIndex,
)
SERIES_TYPES = (pd.Series, md.Series)
DATAFRAME_TYPES = (pd.DataFrame, md.DataFrame)
PANDAS_TYPES = (*SERIES_TYPES, *DATAFRAME_TYPES)
SEQUENCE_TYPES = (list, tuple, np.ndarray, *INDEX_TYPES, *SERIES_TYPES)


# Variable types for type hinting ================================== >>

BOOL = Union[BOOL_TYPES]
INT = Union[INT_TYPES]
FLOAT = Union[FLOAT_TYPES]
SCALAR = Union[SCALAR_TYPES]
INDEX = Union[INDEX_TYPES]
SERIES = Union[SERIES_TYPES]
DATAFRAME = Union[DATAFRAME_TYPES]
PANDAS = Union[PANDAS_TYPES]
SEQUENCE = Union[SEQUENCE_TYPES]

# Types for X and y
FEATURES = Union[iter, dict, list, tuple, np.ndarray, sps.spmatrix, DATAFRAME]
TARGET = Union[INT, str, dict, SEQUENCE, DATAFRAME]

BACKEND = Literal["loky", "multiprocessing", "threading", "ray"]

DATASET = Literal[
    "dataset",
    "train",
    "test",
    "holdout",
    "X",
    "y",
    "X_train",
    "X_test",
    "X_holdout",
    "y_train",
    "y_test",
    "y_holdout",
]

# Selection of rows or columns by name or position
SLICE = Union[INT | str | slice | SEQUENCE]

# Assignment of index or stratify parameter
INDEX_SELECTOR = Union[bool | INT | str | SEQUENCE]

# Allowed values for the goal attribute
GOAL = Literal["class", "reg", "fc"]

# Metric selectors
METRIC_SELECTOR = Union[str, Callable[..., SCALAR], SEQUENCE | None]

# Pruning strategies
PRUNING = Literal["zscore", "iforest", "ee", "lof", "svm", "dbscan", "hdbscan", "optics"]


# Classes for type hinting ========================================= >>

class ENGINE(TypedDict, total=False):
    """Types for the `engine` parameter."""
    data: Literal["numpy", "pyarrow", "modin"]
    estimator: Literal["sklearn", "sklearnex", "cuml"]


class SCORER(Protocol):
    """Protocol for all scorers."""
    def _score(self, method_caller, clf, X, y, sample_weight=None): ...


class TRANSFORMER(Protocol):
    """Protocol for all predictors."""
    def fit(self, **params): ...
    def transform(self, **params): ...


class PREDICTOR(Protocol):
    """Protocol for all predictors."""
    def fit(self, **params): ...
    def predict(self, **params): ...


class ESTIMATOR(Protocol):
    """Protocol for all estimators."""
    def fit(self, **params): ...


class BRANCH(Protocol):
    """Protocol for the Branch class."""
    def _get_rows(self, **params): ...
    def _get_columns(self, **params): ...
    def _get_target(self, **params): ...


class MODEL(Protocol):
    """Protocol for all models."""
    def est_class(self): ...
    def get_estimator(self, **params): ...


class RUNNER(Protocol):
    """Protocol for all runners."""
    def run(self, **params): ...

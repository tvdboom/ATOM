# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing utilities for typing analysis.

"""

from __future__ import annotations

from typing import (
    Callable, Literal, Protocol, TypedDict, Union, runtime_checkable, Any
)
from beartype.vale import Is
from beartype.typing import Annotated, TypeVar, Protocol, Iterable
from beartype.door import is_bearable
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

T = TypeVar("T")

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

DATASETS = Literal[
    "dataset", "train", "test", "holdout", "X", "y", "X_train",
    "X_test", "X_holdout", "y_train", "y_test", "y_holdout",
]

# Selection of rows or columns by name or position
SLICE = Union[INT, str, slice, SEQUENCE]

# Assignment of index or stratify parameter
INDEX_SELECTOR = Union[bool, INT, str, SEQUENCE]

# Types to initialize a metric
METRIC_SELECTOR = (str, Callable[..., SCALAR], SEQUENCE, None)

# Allowed values for method selection
METHOD_SELECTOR = Literal["predict", "predict_proba", "decision_function", "thresh"]

# Allowed values for BaseTransformer parameter
BACKEND = Literal["loky", "multiprocessing", "threading", "ray"]
WARNINGS = Literal["default", "error", "ignore", "always", "module", "once"]
SEVERITY = Literal["debug", "info", "warning", "error", "critical"]

# Data cleaning parameters
STRAT_NUM = SCALAR | Literal["drop", "mean", "median", "knn", "most_frequent"]
DISCRETIZER_STRATS = Literal["uniform", "quantile", "kmeans", "custom"]
PRUNER_STRATS = Literal[
    "zscore", "iforest", "ee", "lof", "svm", "dbscan", "hdbscan", "optics"
]
SCALER_STRATS = Literal["standard", "minmax", "maxabs", "robust"]

# Feature engineering parameters
OPERATORS = Literal["add", "mul", "div", "abs", "sqrt", "log", "sin", "cos", "tan"]
FS_STRATS = Literal[
    "univariate", "pca", "sfm", "sfs", "rfe", "rfecv", "pso", "hho", "gwo", "dfo", "go"
]

# Plotting parameters
LEGEND = Literal[
    "upper left", "lower left", "upper right", "lower right", "upper center",
    "lower center", "center left", "center right", "center", "out",
]

# Mlflow stages
STAGES = Literal["None", "Staging", "Production", "Archived"]


# Classes for type hinting ========================================= >>

class ENGINE(TypedDict, total=False):
    """Types for the `engine` parameter."""
    data: Literal["numpy", "pyarrow", "modin"]
    estimator: Literal["sklearn", "sklearnex", "cuml"]


class SeqProtocol(Protocol[T]):
    """Protocol for all sequences."""
    def __iter__(self) -> Iterable[T]: ...
    def __getitem__(self, item) -> T: ...
    def __len__(self) -> INT: ...


class SEQUENCE:
    """Type hint factory for sequences with subscripted types.

    Dynamically creates new `Annotated[SeqProtocol[...], ...]` type
    hints, subscripted by the passed type.

    Parameters
    ----------
    X : object
        Arbitrary child type hint with which to subscript the
        `SeqProtocol` protocol.

    Returns
    ----------
    Annotated
        `Annotated[SeqProtocol[X], ...]` type hint validating that all
        items of this sequence satisfy this child type hint.

    """

    @classmethod
    def __class_getitem__(cls, X: Any) -> Annotated[SeqProtocol, Is]:
        return Annotated[SeqProtocol[X], Is[
            lambda l: not isinstance(l, str) and all(is_bearable(i, X) for i in l)]
        ]


@runtime_checkable
class SCORER(Protocol):
    """Protocol for all scorers."""
    def _score(self, method_caller, clf, X, y, sample_weight=None): ...


@runtime_checkable
class TRANSFORMER(Protocol):
    """Protocol for all predictors."""
    def transform(self, **params): ...


@runtime_checkable
class PREDICTOR(Protocol):
    """Protocol for all predictors."""
    def fit(self, **params): ...
    def predict(self, **params): ...


@runtime_checkable
class ESTIMATOR(Protocol):
    """Protocol for all estimators."""
    def fit(self, **params): ...


@runtime_checkable
class BRANCH(Protocol):
    """Protocol for the Branch class."""
    def _get_rows(self, **params): ...
    def _get_columns(self, **params): ...
    def _get_target(self, **params): ...


@runtime_checkable
class MODEL(Protocol):
    """Protocol for all models."""
    def _est_class(self): ...
    def _get_est(self, **params): ...


@runtime_checkable
class RUNNER(Protocol):
    """Protocol for all runners."""
    def run(self, **params): ...

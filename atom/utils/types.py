# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing utilities for typing analysis.

"""

from __future__ import annotations

from typing import (
    Any, Callable, Hashable, Literal, TypedDict, Union, runtime_checkable,
)

import modin.pandas as md
import numpy as np
import pandas as pd
import scipy.sparse as sps
from beartype.door import is_bearable
from beartype.typing import Annotated, Iterable, Protocol, TypeVar
from beartype.vale import Is


# Variable types for isinstance ==================================== >>

# TODO: From Python 3.10, isinstance accepts pipe operator (change by then)
BoolTypes = (bool, np.bool_)
IntTypes = (int, np.integer)
FloatTypes = (float, np.floating)
ScalarTypes = (*IntTypes, *FloatTypes)
IndexTypes = (pd.Index, md.Index)
TSIndexTypes = (
    pd.PeriodIndex,
    md.PeriodIndex,
    pd.DatetimeIndex,
    md.DatetimeIndex,
    pd.TimedeltaIndex,
    md.TimedeltaIndex,
)
SeriesTypes = (pd.Series, md.Series)
DataFrameTypes = (pd.DataFrame, md.DataFrame)
PandasTypes = (*SeriesTypes, *DataFrameTypes)
SequenceTypes = (list, tuple, np.ndarray, *IndexTypes, *SeriesTypes)


# Classes for type hinting ========================================= >>

T = TypeVar("T")


class Engine(TypedDict, total=False):
    """Types for the `engine` parameter."""
    data: Literal["numpy", "pyarrow", "modin"]
    estimator: Literal["sklearn", "sklearnex", "cuml"]


class SeqProtocol(Protocol[T]):
    """Protocol for all sequences."""
    def __iter__(self) -> Iterable[T]: ...
    def __getitem__(self, item) -> T: ...
    def __len__(self) -> Int: ...


class Sequence(Protocol[T]):
    """Type hint factory for sequences with subscripted types.

    Dynamically creates new `Annotated[SeqProtocol[...], ...]` type
    hints, subscripted by the passed type.

    Parameters
    ----------
    X: object
        Arbitrary child type hint with which to subscript the protocol.

    Returns
    ----------
    Annotated
        Type hint validating that all items of this sequence satisfy
        this child type hint.

    """

    def __iter__(self) -> Iterable[T]: ...
    def __getitem__(self, item) -> T: ...
    def __len__(self) -> Int: ...

    @classmethod
    def __class_getitem__(cls, X: Any) -> Annotated[SeqProtocol, Is]:
        return Annotated[cls, Is[lambda lst: all(is_bearable(i, X) for i in lst)]]


@runtime_checkable
class Scorer(Protocol):
    """Protocol for all scorers."""
    def _score(self, method_caller, clf, X, y, sample_weight=None): ...


@runtime_checkable
class Transformer(Protocol):
    """Protocol for all predictors."""
    def transform(self, **params): ...


@runtime_checkable
class Predictor(Protocol):
    """Protocol for all predictors."""
    def fit(self, **params): ...
    def predict(self, **params): ...


@runtime_checkable
class Estimator(Protocol):
    """Protocol for all estimators."""
    def fit(self, **params): ...


@runtime_checkable
class Model(Protocol):
    """Protocol for all models."""
    def _est_class(self): ...
    def _get_est(self, **params): ...


@runtime_checkable
class Runner(Protocol):
    """Protocol for all runners."""
    def run(self, **params): ...


# Variable types for type hinting ================================== >>

Bool = Union[BoolTypes]
Int = Union[IntTypes]
Float = Union[FloatTypes]
Scalar = Union[ScalarTypes]
Index = Union[IndexTypes]
Series = Union[SeriesTypes]
DataFrame = Union[DataFrameTypes]
Pandas = Union[PandasTypes]

# Types for X and y
Features = Union[iter, dict, list, tuple, np.ndarray, sps.spmatrix, DataFrame]
Target = Union[Int, str, dict, Sequence, DataFrame]

Datasets = Literal[
    "dataset", "train", "test", "holdout", "X", "y", "X_train",
    "X_test", "X_holdout", "y_train", "y_test", "y_holdout",
]

# Selection of rows or columns by name or position
ColumnSelector = Union[Int, str, range, slice, Sequence]
RowSelector = Union[Hashable, ColumnSelector]

# Assignment of index or stratify parameter
IndexSelector = Union[Bool, Int, str, Sequence]

# Types to initialize models and metric
ModelSelector = Union[Int, str, Model, range, slice, Sequence, None]
MetricSelector = Union[str, Callable[..., Scalar], Sequence, None]

# Allowed values for method selection
MethodSelector = Literal["predict", "predict_proba", "decision_function", "thresh"]

# Allowed values for BaseTransformer parameter
Backend = Literal["loky", "multiprocessing", "threading", "ray"]
Warnings = Literal["default", "error", "ignore", "always", "module", "once"]
Severity = Literal["debug", "info", "warning", "error", "critical"]
Verbose = Literal[0, 1, 2]

# Data cleaning parameters
NumericalStrats = Scalar | Literal["drop", "mean", "median", "knn", "most_frequent"]
DiscretizerStrats = Literal["uniform", "quantile", "kmeans", "custom"]
PrunerStrats = Literal[
    "zscore", "iforest", "ee", "lof", "svm", "dbscan", "hdbscan", "optics"
]
ScalerStrats = Literal["standard", "minmax", "maxabs", "robust"]

# Feature engineering parameters
Operators = Literal["add", "mul", "div", "abs", "sqrt", "log", "sin", "cos", "tan"]
FeatureSelectionStrats = Literal[
    "univariate", "pca", "sfm", "sfs", "rfe", "rfecv", "pso", "hho", "gwo", "dfo", "go"
]

# Plotting parameters
Legend = Literal[
    "upper left", "lower left", "upper right", "lower right", "upper center",
    "lower center", "center left", "center right", "center", "out",
]

# Mlflow stages
Stages = Literal["None", "Staging", "Production", "Archived"]

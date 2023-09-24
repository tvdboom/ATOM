# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing utilities for typing analysis.

"""

from __future__ import annotations

from typing import (
    Any, Callable, Hashable, Literal, NotRequired, TypedDict, Union,
    runtime_checkable,
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


# class Literal(typing.Literal):
#
#     def __class_getitem__(cls, X: tuple[str]) -> Annotated[str, Is]:
#         return Annotated[str, Is[lambda x: x.lower() in X]]


class Sequence(Protocol[T]):
    """Type hint factory for sequences with subscripted types.

    Dynamically creates new `Annotated[Sequence[...], ...]` type hints,
    subscripted by the passed type. For subscripted types, it passes
    when the type is no string (since strings has the required dunder
    methods), is one-dimensional, and all items in the sequence are of
    the subscripted type.

    Parameters
    ----------
    T: object
        Arbitrary child type hint with which to subscript the protocol.

    Returns
    ----------
    Annotated
        Type hint validating that all items of this sequence satisfy
        this child type hint.

    Notes
    -----
    This implementation only works because beartype.Protocol doesn't
    check the `@classmethod`. This could fail in the future if fixed.
    See https://github.com/beartype/beartype/discussions/277#discussioncomment-7086878

    """

    def __iter__(self) -> Iterable[T]: ...
    def __getitem__(self, item) -> T: ...
    def __len__(self) -> Int: ...

    @classmethod
    def __class_getitem__(cls, item: Any) -> Annotated[Sequence, Is]:
        return Annotated[
            cls,
            Is[lambda lst: not isinstance(lst, str)]
            & Is[lambda lst: np.array(lst).ndim == 1]
            & Is[lambda lst: all(is_bearable(i, item) for i in lst)]
        ]


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
    name: NotRequired[str]
    acronym: NotRequired[str]
    needs_scaling: NotRequired[Bool]
    native_multilabel: NotRequired[Bool]
    native_multioutput: NotRequired[Bool]
    has_validation: NotRequired[str | None]

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

Bool = Union[bool, np.bool_]
Int = Union[int, np.integer]
Float = Union[float, np.floating]
Scalar = Union[Int, Float]
Index = Union[pd.Index, md.Index]
Series = Union[pd.Series, md.Series]
DataFrame = Union[pd.DataFrame, md.DataFrame]
Pandas = Union[Series, DataFrame]

# Types for X and y
Features = Union[Iterable, dict, list, tuple, np.ndarray, sps.spmatrix, DataFrame]
Target = Union[Int, str, dict, Sequence[Int, str], DataFrame]

# Selection of rows or columns by name or position
ColumnSelector = Union[Int, str, range, slice, Sequence[Int, str], DataFrame]
RowSelector = Union[Hashable, Sequence[Hashable], ColumnSelector]

# Assignment of index or stratify parameter
IndexSelector = Union[Bool, Int, str, Sequence[Hashable]]

# Types to initialize models and metric
ModelSelector = Union[Int, str, Model, range, slice, Sequence[Int, str, Model], None]
MetricFunction = Callable[[Sequence[Scalar], Sequence[Scalar], ...], Scalar]
MetricSelector = Union[str, MetricFunction, Sequence[str, MetricFunction], None]

# Allowed values for method selection
MethodSelector = Literal["predict", "predict_proba", "decision_function", "thresh"]

# Allowed values for BaseTransformer parameter
NJobs = Annotated[Int, Is[lambda x: x >= 0]]
Backend = Literal["loky", "multiprocessing", "threading", "ray"]
Warnings = Literal["default", "error", "ignore", "always", "module", "once"]
Severity = Literal["debug", "info", "warning", "error", "critical"]
Verbose = Literal[0, 1, 2]

# Data cleaning parameters
NumericalStrats = Union[Scalar, Literal["drop", "mean", "median", "knn", "most_frequent"]]
CategoricalStrats = Union[str, Literal["drop", "most_frequent"]]
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

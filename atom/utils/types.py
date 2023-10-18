# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing utilities for typing analysis.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import modin.pandas as md
import numpy as np
import pandas as pd
import scipy.sparse as sps
from beartype.door import is_bearable
from beartype.typing import (
    Any, Callable, Hashable, Iterable, Iterator, Literal, Protocol, TypeAlias,
    TypedDict, TypeVar, Union, runtime_checkable,
)
from beartype.vale import Is
from optuna.distributions import BaseDistribution


if TYPE_CHECKING:
    from atom.branch import Branch
    from atom.utils.utils import ClassMap, Goal, ShapExplanation


# Variable types for isinstance ==================================== >>

# TODO: From Python 3.11, use Self type hint to return classes
# TODO: From Python 3.10, isinstance accepts pipe operator (change to TypeAlias)
BoolTypes = (bool, np.bool_)
IntTypes = (int, np.integer)
FloatTypes = (float, np.floating)
ScalarTypes = (*IntTypes, *FloatTypes)
SegmentTypes = (range, slice)
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
T_cov = TypeVar("T_cov", covariant=True)


class Engine(TypedDict, total=False):
    """Types for the `engine` parameter."""
    data: Literal["numpy", "pyarrow", "modin"]
    estimator: Literal["sklearn", "sklearnex", "cuml"]


class HT(TypedDict, total=False):
    """Types for the `_ht` attribute of Model."""
    distributions: dict[str, BaseDistribution]
    cv: Int
    plot: Bool
    tags: dict[str, str]


class Style(TypedDict):
    """Types for the plotting styles."""
    palette: dict[str, str]
    marker: dict[str, str]
    dash: dict[str, str]
    shape: dict[str, str]


@runtime_checkable
class Sequence(Protocol[T_cov]):
    """Type hint factory for sequences with subscripted types.

    Dynamically creates new `Annotated[Sequence[...], ...]` type hints,
    subscripted by the passed type. For subscripted types, it passes
    when the type is of a sequence type and all items in the sequence
    are of the subscripted type.

    Parameters
    ----------
    T: object
        Arbitrary child type hint with which to subscript the protocol.

    Returns
    -------
    Annotated
        Type hint validating that all items of this sequence satisfy
        this child type hint.

    Notes
    -----
    This implementation only works because beartype.Protocol doesn't
    check the `@classmethod`. This could fail in the future if fixed.
    See https://github.com/beartype/beartype/discussions/277#discussioncomment-7086878

    """

    def __iter__(self) -> Iterator[T_cov]: ...
    def __getitem__(self, item) -> T_cov: ...
    def __len__(self) -> Int: ...

    @classmethod
    def __class_getitem__(cls, item: Any) -> Annotated[Any, Is]:
        return Annotated[
            cls,
            Is[lambda lst: isinstance(lst, SequenceTypes)]
            & Is[lambda lst: all(is_bearable(i, item) for i in lst)]  # type: ignore
        ]


@runtime_checkable
class SkScorer(Protocol):
    """Protocol for sklearn's scorers."""
    def __call__(self, *args, **kwargs): ...
    def _score(self, *args, **kwargs): ...


@runtime_checkable
class Scorer(SkScorer, Protocol):
    """Protocol for ATOM's scorers.

    ATOM's scorers are the same objects as sklearn's scorers
    but with an extra 'name' and 'fullname' attribute.

    """
    name: str
    fullname: str


@runtime_checkable
class Estimator(Protocol):
    """Protocol for sklearn-like estimators."""
    def __init__(self, *args, **kwargs): ...
    def get_params(self, *args, **kwargs): ...
    def set_params(self, *args, **kwargs): ...


@runtime_checkable
class Transformer(Estimator, Protocol):
    """Protocol for sklearn-like transformers."""
    def transform(self, *args, **kwargs): ...


@runtime_checkable
class Predictor(Estimator, Protocol):
    """Protocol for sklearn-like predictors."""
    def fit(self, *args, **kwargs): ...
    def predict(self, *args, **kwargs): ...


@runtime_checkable
class Model(Protocol):
    """Protocol for all models."""
    _goal: Goal
    _metric: ClassMap
    _ht: dict[str, Any]
    _shap: ShapExplanation

    @property
    def name(self) -> str: ...
    @property
    def branch(self) -> Branch: ...
    @property
    def estimator(self) -> Predictor: ...
    @property
    def _est_class(self) -> type[Predictor]: ...
    @property
    def evals(self) -> dict[str, list]: ...
    @property
    def feature_importance(self) -> pd.Series: ...
    # @property
    # def run(self) -> Run: ...
    # @property
    # def study(self) -> Study: ...
    # @property
    # def best_trial(self) -> FrozenTrial: ...
    # @property
    # def trials(self) -> pd.DataFrame: ...

    def _get_pred(self, *args, **kwargs) -> tuple[Pandas, Pandas]: ...
    def predict(self, *args, **kwargs) -> Pandas: ...
    def predict_interval(self, *args, **kwargs) -> DataFrame: ...


# Variable types for type hinting ================================== >>

# General types
Bool = Union[bool, np.bool_]
Int = Union[int, np.integer]
Float = Union[float, np.floating]
Scalar = Union[Int, Float]
Segment = Union[range, slice]
Index = Union[pd.Index, md.Index]
Series = Union[pd.Series, md.Series]
DataFrame = Union[pd.DataFrame, md.DataFrame]
Pandas = Union[Series, DataFrame]

# Numerical types
IntLargerZero: TypeAlias = Annotated[Int, Is[lambda x: x > 0]]
IntLargerEqualZero: TypeAlias = Annotated[Int, Is[lambda x: x >= 0]]
IntLargerTwo: TypeAlias = Annotated[Int, Is[lambda x: x > 2]]
IntLargerFour: TypeAlias = Annotated[Int, Is[lambda x: x > 4]]
FloatLargerZero: TypeAlias = Annotated[Scalar, Is[lambda x: x > 0]]
FloatLargerEqualZero: TypeAlias = Annotated[Scalar, Is[lambda x: x >= 0]]
FloatZeroToOneInc: TypeAlias = Annotated[Float, Is[lambda x: 0 <= x <= 1]]
FloatZeroToOneExc: TypeAlias = Annotated[Float, Is[lambda x: 0 < x < 1]]

# Types for X and y
Features = Union[
    dict[str, Sequence[Any]],
    Sequence[Sequence[Any]],
    Iterable[Sequence[Any], tuple[Hashable, Sequence[Any]], dict[str, Sequence[Any]]],
    np.ndarray,
    sps.spmatrix,
    DataFrame,
]
Target = Union[Int, str, dict[str, Any], Sequence[Any], DataFrame]

# Return types for transform methods
TReturn = Union[np.ndarray, sps.spmatrix, Series, DataFrame]
TReturns = Union[TReturn, tuple[TReturn, TReturn]]

# Selection of rows or columns by name or position
ColumnSelector = Union[Int, str, Segment, Sequence[Union[Int, str]], DataFrame]
RowSelector = Union[Hashable, Sequence[Hashable], ColumnSelector]

# Assignment of index or stratify parameter
IndexSelector = Union[Bool, Int, str, Sequence[Hashable]]

# Types to initialize and select models and metric
ModelsConstructor = Union[str, Predictor, Sequence[Union[str, Predictor]], None]
ModelSelector = Union[Int, str, Model]
ModelsSelector = Union[ModelSelector, Segment, Sequence[ModelSelector], None]
MetricFunction = Callable[[Sequence[Scalar], Sequence[Scalar]], Scalar]
MetricConstructor = Union[
    str,
    MetricFunction,
    Scorer,
    Sequence[Union[str, MetricFunction, Scorer]],
    None,
]
MetricSelector = Union[
    IntLargerEqualZero,
    str,
    Sequence[Union[IntLargerEqualZero, str]],
    None,
]

# Allowed values for BaseTransformer parameter
NJobs = Annotated[Int, Is[lambda x: x != 0]]
Backend = Literal["loky", "multiprocessing", "threading", "ray"]
Warnings = Literal["default", "error", "ignore", "always", "module", "once"]
Severity = Literal["debug", "info", "warning", "error", "critical"]
Verbose = Literal[0, 1, 2]

# Data cleaning parameters
NumericalStrats = Union[Scalar, Literal["drop", "mean", "median", "knn", "most_frequent"]]
CategoricalStrats = Union[str, Literal["drop", "most_frequent"]]
DiscretizerStrats = Literal["uniform", "quantile", "kmeans", "custom"]
Bins = Union[
    IntLargerZero,
    Sequence[IntLargerZero],
    dict[str, Union[IntLargerZero, Sequence[IntLargerZero]]],
]
NormalizerStrats = Literal["yeojohnson", "boxcox", "quantile"]
PrunerStrats = Literal[
    "zscore", "iforest", "ee", "lof", "svm", "dbscan", "hdbscan", "optics"
]
ScalerStrats = Literal["standard", "minmax", "maxabs", "robust"]

# NLP parameters
VectorizerStarts = Literal["bow", "tfidf", "hashing"]

# Feature engineering parameters
Operators = Literal["add", "mul", "div", "abs", "sqrt", "log", "sin", "cos", "tan"]
FeatureSelectionStrats = Literal[
    "univariate", "pca", "sfm", "sfs", "rfe", "rfecv", "pso", "hho", "gwo", "dfo", "go"
]
FeatureSelectionSolvers = Union[
    str,
    Callable[..., tuple[Sequence[Scalar], Sequence[Scalar]]],  # e.g., f_classif
    Estimator,
    None,
]

# Runner parameters
NItems = Union[
    IntLargerEqualZero,
    dict[str, IntLargerEqualZero],
    Sequence[IntLargerEqualZero],
]

# Allowed values for method selection
PredictionMethod = Literal["decision_function", "predict_proba", "predict", "thresh"]

# Plotting parameters
PlotBackend = Literal["plotly", "matplotlib"]
ParamsSelector = Union[str, Segment, Sequence[Union[IntLargerEqualZero, str]]]
TargetSelector = Union[IntLargerEqualZero, str]
TargetsSelector = Union[TargetSelector, tuple[TargetSelector, ...]]
Kind = Literal["average", "individual", "average+individual", "individual+average"]
Legend = Literal[
    "upper left", "lower left", "upper right", "lower right", "upper center",
    "lower center", "center left", "center right", "center", "out",
]

# Mlflow stages
Stages = Literal["None", "Staging", "Production", "Archived"]

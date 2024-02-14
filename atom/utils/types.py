"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing utilities for typing analysis.

"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import (
    TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, SupportsIndex,
    TypeAlias, TypedDict, TypeVar, overload, runtime_checkable,
)

import modin.pandas as md
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import scipy.sparse as sps
from beartype.door import is_bearable
from beartype.typing import Protocol
from beartype.vale import Is
from optuna.distributions import BaseDistribution
from sktime.forecasting.base import ForecastingHorizon


if TYPE_CHECKING:
    from atom.branch.dataengines import DataEngine
    from atom.utils.utils import Goal


# Classes for type hinting ========================================= >>

_T = TypeVar("_T")


class Sequence(Protocol[_T]):
    """Type hint factory for sequences with subscripted types.

    Dynamically creates new `Annotated[Sequence[...], ...]` type hints,
    subscripted by the passed type. For subscripted types, it passes
    when the type is an array-like and all items in the sequence are of
    the subscripted type.

    Parameters
    ----------
    _T: object
        Arbitrary child type hint to subscript the protocol.

    Notes
    -----
    See https://github.com/beartype/beartype/discussions/277#discussioncomment-7086878

    """

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_T]: ...
    @overload
    def __getitem__(self, __i: SupportsIndex, /) -> _T: ...
    @overload
    def __getitem__(self, __s: slice, /) -> Sequence[_T]: ...

    @classmethod
    def __class_getitem__(cls, item: Any) -> Annotated[Any, Is]:
        """Get the sequence annotation for type and content."""
        return Annotated[
            cls,
            Is[lambda lst: isinstance(lst, sequence_t)]
            & Is[lambda lst: all(is_bearable(i, item) for i in lst)],
        ]


class SPDict(TypedDict, total=False):
    """Dictionary type for the `sp` parameter."""

    sp: Seasonality
    seasonal_model: SeasonalityModels
    trend_model: SeasonalityModels


class EngineDict(TypedDict, total=False):
    """Dictionary type for the `engine` parameter."""

    data: EngineDataOptions
    estimator: EngineEstimatorOptions


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


class EngineTuple(NamedTuple):
    """Return type of the `engine` parameter.

    We use a namedtuple for immutability to be able to clone
    the estimator since sklearn does a `is` instance check.

    """

    data: EngineDataOptions = "pandas"
    estimator: EngineEstimatorOptions = "sklearn"

    def __repr__(self) -> str:
        """Print representation as dictionary."""
        return self._asdict().__repr__()

    @property
    def data_engine(self) -> DataEngine:
        """Return the data engine."""
        from atom.branch.dataengines import DATA_ENGINES

        return DATA_ENGINES[self.data]()


class SPTuple(NamedTuple):
    """Return type of the `sp` parameter."""

    sp: int | list[int] | None = None
    seasonal_model: SeasonalityModels = "additive"
    trend_model: SeasonalityModels = "additive"


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

    def fit(self, *args, **kwargs): ...
    def get_params(self, *args, **kwargs): ...
    def set_params(self, *args, **kwargs): ...


@runtime_checkable
class Transformer(Estimator, Protocol):
    """Protocol for sklearn-like transformers."""

    def transform(self, *args, **kwargs): ...


@runtime_checkable
class Predictor(Estimator, Protocol):
    """Protocol for sklearn-like predictors."""

    def predict(self, *args, **kwargs): ...


@runtime_checkable
class Model(Protocol):
    """Protocol for all models."""

    _goal: Goal
    # _metric: ClassMap
    # _ht: dict[str, Any]

    def predict(self, *args, **kwargs) -> Pandas: ...


# Variable types for type hinting ================================== >>

# General types
Bool: TypeAlias = bool | np.bool_
Int: TypeAlias = int | np.integer
Float: TypeAlias = float | np.floating
Scalar: TypeAlias = Int | Float
Segment: TypeAlias = slice | range
Pandas: TypeAlias = pd.Series | pd.DataFrame

# Numerical types
IntLargerZero: TypeAlias = Annotated[Int, Is[lambda x: x > 0]]
IntLargerEqualZero: TypeAlias = Annotated[Int, Is[lambda x: x >= 0]]
IntLargerOne: TypeAlias = Annotated[Int, Is[lambda x: x > 1]]
IntLargerTwo: TypeAlias = Annotated[Int, Is[lambda x: x > 2]]
IntLargerFour: TypeAlias = Annotated[Int, Is[lambda x: x > 4]]
FloatLargerZero: TypeAlias = Annotated[Scalar, Is[lambda x: x > 0]]
FloatLargerEqualZero: TypeAlias = Annotated[Scalar, Is[lambda x: x >= 0]]
FloatZeroToOneInc: TypeAlias = Annotated[Scalar, Is[lambda x: 0 <= x <= 1]]
FloatZeroToOneExc: TypeAlias = Annotated[Float, Is[lambda x: 0 < x < 1]]

# Types for X, y and fh
XConstructor: TypeAlias = (
    dict[str, Sequence[Any]]
    | Sequence[Sequence[Any]]
    | Iterable[Sequence[Any] | tuple[Hashable, Sequence[Any]] | dict[str, Sequence[Any]]]
    | np.ndarray
    | sps.spmatrix
    | pd.DataFrame
)
XSelector: TypeAlias = XConstructor | Callable[..., XConstructor]
YConstructor: TypeAlias = dict[str, Any] | Sequence[Any] | XConstructor
YSelector: TypeAlias = Int | str | YConstructor
FHConstructor: TypeAlias = Int | Sequence[Int] | ForecastingHorizon

# Return types for transform methods
TReturn: TypeAlias = np.ndarray | sps.spmatrix | Sequence[Any] | pd.DataFrame
TReturns: TypeAlias = TReturn | tuple[TReturn, TReturn]

# Selection of rows or columns by name or position
ColumnSelector: TypeAlias = Int | str | Segment | Sequence[Int | str] | pd.DataFrame
RowSelector: TypeAlias = Hashable | Sequence[Hashable] | ColumnSelector

# Assignment of index or stratify parameter
IndexSelector: TypeAlias = Bool | Int | str | Sequence[Hashable]

# Construct and select models and metric
ModelsConstructor: TypeAlias = str | Predictor | Sequence[str | Predictor] | None
ModelSelector: TypeAlias = Int | str | Model
ModelsSelector: TypeAlias = ModelSelector | Segment | Sequence[ModelSelector] | None
MetricFunction: TypeAlias = Callable[[Sequence[Scalar], Sequence[Scalar]], Scalar]
MetricConstructor: TypeAlias = (
    str
    | MetricFunction
    | Scorer
    | Sequence[str | MetricFunction | Scorer]
    | None
)
MetricSelector: TypeAlias = IntLargerEqualZero | str | Sequence[IntLargerEqualZero | str] | None

# BaseTransformer parameters
NJobs: TypeAlias = Annotated[Int, Is[lambda x: x != 0]]
EngineDataOptions: TypeAlias = Literal[
    "numpy",
    "pandas",
    "pandas-pyarrow",
    "polars",
    "polars-lazy",
    "pyarrow",
    "modin",
    "dask",
    "pyspark",
    "pyspark-pandas",
]
EngineEstimatorOptions: TypeAlias = Literal["sklearn", "sklearnex", "cuml"]
Engine: TypeAlias = EngineDataOptions | EngineEstimatorOptions | EngineDict | EngineTuple | None
Backend: TypeAlias = Literal["loky", "multiprocessing", "threading", "ray", "dask"]
Warnings: TypeAlias = Literal["default", "error", "ignore", "always", "module", "once"]
Severity: TypeAlias = Literal["debug", "info", "warning", "error", "critical"]
Verbose: TypeAlias = Literal[0, 1, 2]

# Data cleaning parameters
NumericalStrats: TypeAlias = Literal[
    "drop",
    "mean",
    "median",
    "knn",
    "iterative",
    "most_frequent",
    "drift",
    "linear",
    "nearest",
    "bfill",
    "ffill",
    "random",
]
CategoricalStrats: TypeAlias = Literal["drop", "most_frequent"]
DiscretizerStrats: TypeAlias = Literal["uniform", "quantile", "kmeans", "custom"]
Bins: TypeAlias = IntLargerOne | Sequence[Scalar] | dict[str, IntLargerOne | Sequence[Scalar]]
NormalizerStrats: TypeAlias = Literal["yeojohnson", "boxcox", "quantile"]
PrunerStrats: TypeAlias = Literal[
    "zscore", "iforest", "ee", "lof", "svm", "dbscan", "hdbscan", "optics"
]
ScalerStrats: TypeAlias = Literal["standard", "minmax", "maxabs", "robust"]

# NLP parameters
VectorizerStarts: TypeAlias = Literal["bow", "tfidf", "hashing"]

# Feature engineering parameters
Operators: TypeAlias = Literal[
    "add", "sub", "mul", "div", "abs", "sqrt", "log", "sin", "cos", "tan"
]
FeatureSelectionStrats: TypeAlias = Literal[
    "univariate", "pca", "sfm", "sfs", "rfe", "rfecv", "pso", "hho", "gwo", "dfo", "go"
]
FeatureSelectionSolvers: TypeAlias = (
    str
    | Callable[..., tuple[Sequence[Scalar], Sequence[Scalar]]]  # e.g., f_classif
    | Predictor
    | None
)

# Allowed values for method selection
PredictionMethods: TypeAlias = Literal[
    "decision_function", "predict", "predict_log_proba", "predict_proba", "score"
]
PredictionMethodsTS: TypeAlias = Literal[
    "predict",
    "predict_interval",
    "predict_proba",
    "predict_quantiles",
    "predict_residuals",
    "predict_var",
    "score",
]

# Plotting parameters
PlotBackend: TypeAlias = Literal["plotly", "matplotlib"]
ParamsSelector: TypeAlias = str | Segment | Sequence[IntLargerEqualZero | str]
TargetSelector: TypeAlias = IntLargerEqualZero | str
TargetsSelector: TypeAlias = TargetSelector | tuple[TargetSelector, ...]
Kind: TypeAlias = Literal["average", "individual", "average+individual", "individual+average"]
Legend: TypeAlias = Literal[
    "upper left",
    "lower left",
    "upper right",
    "lower right",
    "upper center",
    "lower center",
    "center left",
    "center right",
    "center",
    "out",
]

# Others
Seasonality: TypeAlias = IntLargerOne | str | Sequence[IntLargerOne | str] | None
SeasonalityModels: TypeAlias = Literal["additive", "multiplicative"]
FeatureNamesOut: TypeAlias = (
    Callable[[Any, Sequence[str]], Sequence[str]]
    | Literal["one-to-one"]
    | None
)
HarmonicsSelector: TypeAlias = Literal["drop", "raw_strength", "harmonic_strength"]
Stages: TypeAlias = Literal["None", "Staging", "Production", "Archived"]
PACFMethods: TypeAlias = Literal[
    "yw",
    "ywadjusted",
    "ywm",
    "ywmle",
    "ols",
    "ols-inefficient",
    "ols-adjusted",
    "ld",
    "ldadjusted",
    "ldb",
    "ldbiased",
    "burg",
]
NItems: TypeAlias = (
    IntLargerEqualZero
    | dict[str, IntLargerEqualZero]
    | Sequence[IntLargerEqualZero]
)


# Variable types for isinstance ================================== >>

# Although injecting the type hints directly to isinstance works, mypy fails
# https://github.com/python/mypy/issues/11673
# https://github.com/python/mypy/issues/16358
bool_t = (bool, np.bool_)
int_t = (int, np.integer)
float_t = (float, np.floating)
segment_t = (slice, range)
sequence_t = (range, list, tuple, np.ndarray, pd.Index, pd.Series)
pandas_t = (pd.Series, pd.DataFrame)

"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the data engines.

"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from polars.dependencies import _lazy_import

import numpy as np
import pandas as pd
import polars as pl

from atom.utils.types import Any, Pandas, Sequence
from atom.utils.utils import get_cols

import os


os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

dd, _ = _lazy_import("dask.dataframe")
md, _ = _lazy_import("modin.pandas")
pa, _ = _lazy_import("pyarrow")
ps, _ = _lazy_import("pyspark")


class DataEngine(metaclass=ABCMeta):
    """Abstract class for data engines.

    Data engines convert a pandas object to a specific type.
    The type is determined by the data engine.

    """

    @staticmethod
    @abstractmethod
    def convert(obj: Pandas) -> np.ndarray | Sequence[Any] | pd.DataFrame: ...


class NumpyEngine(DataEngine):
    """Numpy data engine."""

    library = "numpy"

    @staticmethod
    def convert(obj: Pandas) -> np.ndarray:
        """Convert to numpy array."""
        return obj.to_numpy()


class PandasEngine(DataEngine):
    """Pandas numpy data engine."""

    library = "pandas"

    @staticmethod
    def convert(obj: Pandas) -> Pandas:
        """Leave as is."""
        return obj


class PandasPyarrowEngine(DataEngine):
    """Pandas pyarrow data engine."""

    library = "pandas"

    @staticmethod
    def convert(obj: Pandas) -> Pandas:
        """Convert to pyarrow dtypes."""
        return obj.astype(
            {
                col.name: pd.ArrowDtype(
                    pa.from_numpy_dtype(getattr(col.dtype, "numpy_dtype", col.dtype))
                )
                for col in get_cols(obj)
            }
        )


class PolarsEngine(DataEngine):
    """Polars data engine."""

    library = "polars"

    @staticmethod
    def convert(obj: Pandas) -> pl.Series | pl.DataFrame:
        """Convert to polars objects."""
        if isinstance(obj, pd.DataFrame):
            return pl.DataFrame(obj)
        elif isinstance(obj, pd.Series):
            return pl.Series(obj)


class PolarsLazyEngine(DataEngine):
    """Polars lazy data engine."""

    library = "polars"

    @staticmethod
    def convert(obj: Pandas) -> pl.Series | pl.DataFrame:
        """Convert to lazy polars objects."""
        if isinstance(obj, pd.DataFrame):
            return pl.LazyFrame(obj)
        elif isinstance(obj, pd.Series):
            return pl.Series(obj)


class PyArrowEngine(DataEngine):
    """PyArrow data engine."""

    library = "pyarrow"

    @staticmethod
    def convert(obj: Pandas) -> pa.Array | pa.Table:
        """Convert to pyarrow objects."""
        if isinstance(obj, pd.DataFrame):
            return pa.Table.from_pandas(obj)
        elif isinstance(obj, pd.Series):
            return pa.Array.from_pandas(obj)


class ModinEngine(DataEngine):
    """Modin data engine."""

    library = "modin"

    @staticmethod
    def convert(obj: Pandas) -> md.Series | md.DataFrame:
        """Convert to modin objects."""
        if isinstance(obj, pd.DataFrame):
            return md.DataFrame(obj)
        elif isinstance(obj, pd.Series):
            return md.Series(obj)


class DaskEngine(DataEngine):
    """Dask data engine."""

    library = "dask"

    @staticmethod
    def convert(obj: Pandas) -> dd.Series | dd.DataFrame:
        """Convert to dask objects."""
        return dd.from_pandas(obj, npartitions=max(1, len(obj) // 1e6))


class PySparkEngine(DataEngine):
    """PySpark data engine."""

    library = "pyspark"

    @staticmethod
    def convert(obj: Pandas) -> ps.sql.DataFrame:
        """Convert to pyspark objects."""
        spark = ps.sql.SparkSession.builder.appName("atom-ml").getOrCreate()
        return spark.createDataFrame(obj)


class PySparkPandasEngine(DataEngine):
    """PySpark data engine with pandas API."""

    library = "pyspark"

    @staticmethod
    def convert(obj: Pandas) -> ps.pandas.Series | ps.pandas.DataFrame:
        """Convert to pyspark objects."""
        if isinstance(obj, pd.DataFrame):
            return ps.pandas.DataFrame(obj)
        elif isinstance(obj, pd.Series):
            return ps.pandas.Series(obj)


DATA_ENGINES = {
    "numpy": NumpyEngine,
    "pandas": PandasEngine,
    "pandas-pyarrow": PandasPyarrowEngine,
    "polars": PolarsEngine,
    "polars-lazy": PolarsLazyEngine,
    "pyarrow": PyArrowEngine,
    "modin": ModinEngine,
    "dask": DaskEngine,
    "pyspark": PySparkEngine,
    "pyspark-pandas": PySparkPandasEngine,
}

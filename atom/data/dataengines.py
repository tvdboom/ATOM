"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the data engines.

"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from atom.utils.types import Any, Pandas


if TYPE_CHECKING:
    import dask.dataframe as dd
    import modin.pandas as md
    import polars as pl
    import pyarrow as pa
    import pyspark.sql as psql
    import pyspark.pandas as ps


class DataEngine(metaclass=ABCMeta):
    """Abstract class for data engines.

    Data engines convert a pandas object to a specific type.
    The type is determined by the data engine.

    """

    @staticmethod
    @abstractmethod
    def convert(obj: Pandas) -> Any:
        """Convert to data engine output types."""


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
        from pyarrow import from_numpy_dtype

        if isinstance(obj, pd.DataFrame):
            return obj.astype(
                {
                    c: pd.ArrowDtype(from_numpy_dtype(getattr(d, "numpy_dtype", d)))
                    for c, d in obj.dtypes.items()
                }
            )
        else:
            return obj.astype(
                pd.ArrowDtype(from_numpy_dtype(obj.dtype))
                if isinstance(obj.dtype, np.dtype) else obj.dtype
            )


class PolarsEngine(DataEngine):
    """Polars data engine."""

    library = "polars"

    @staticmethod
    def convert(obj: Pandas) -> pl.Series | pl.DataFrame:
        """Convert to polars objects."""
        import polars as pl

        if isinstance(obj, pd.DataFrame):
            return pl.DataFrame(obj)
        else:
            return pl.Series(obj)


class PolarsLazyEngine(DataEngine):
    """Polars lazy data engine."""

    library = "polars"

    @staticmethod
    def convert(obj: Pandas) -> pl.Series | pl.LazyFrame:
        """Convert to lazy polars objects."""
        import polars as pl

        if isinstance(obj, pd.DataFrame):
            return pl.LazyFrame(obj)
        else:
            return pl.Series(obj)


class PyArrowEngine(DataEngine):
    """PyArrow data engine."""

    library = "pyarrow"

    @staticmethod
    def convert(obj: Pandas) -> pa.Array | pa.Table:
        """Convert to pyarrow objects."""
        import pyarrow as pa

        if isinstance(obj, pd.DataFrame):
            return pa.Table.from_pandas(obj)
        else:
            return pa.Array.from_pandas(obj)


class ModinEngine(DataEngine):
    """Modin data engine."""

    library = "modin"

    @staticmethod
    def convert(obj: Pandas) -> md.Series | md.DataFrame:
        """Convert to modin objects."""
        import modin.pandas as md

        if isinstance(obj, pd.DataFrame):
            return md.DataFrame(obj)
        else:
            return md.Series(obj)


class DaskEngine(DataEngine):
    """Dask data engine."""

    library = "dask"

    @staticmethod
    def convert(obj: Pandas) -> dd.Series | dd.DataFrame:
        """Convert to dask objects."""
        import dask.dataframe as dd

        return dd.from_pandas(obj, npartitions=int(max(1, len(obj) // 1e6)))


class PySparkEngine(DataEngine):
    """PySpark data engine."""

    library = "pyspark"

    @staticmethod
    def convert(obj: Pandas) -> psql.DataFrame:
        """Convert to pyspark objects."""
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("atom-ml").getOrCreate()
        return spark.createDataFrame(obj)


class PySparkPandasEngine(DataEngine):
    """PySpark data engine with pandas API."""

    library = "pyspark"

    @staticmethod
    def convert(obj: Pandas) -> ps.Series | ps.DataFrame:
        """Convert to pyspark objects."""
        import pyspark.pandas as ps

        if isinstance(obj, pd.DataFrame):
            return ps.DataFrame(obj)
        else:
            return ps.Series(obj)


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

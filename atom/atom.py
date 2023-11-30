# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the ATOM class.

"""

from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator
from copy import deepcopy
from logging import Logger
from pathlib import Path
from platform import machine, platform, python_build, python_version
from types import MappingProxyType
from typing import Any, Literal, TypeVar

import dill as pickle
import numpy as np
import pandas as pd
from beartype import beartype
from joblib.memory import Memory
from pandas._typing import DtypeObj
from scipy import stats
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.utils.metaestimators import available_if

from atom.baserunner import BaseRunner
from atom.basetransformer import BaseTransformer
from atom.branch import Branch, BranchManager
from atom.data_cleaning import (
    Balancer, Cleaner, Discretizer, Encoder, Imputer, Normalizer, Pruner,
    Scaler, TransformerMixin,
)
from atom.feature_engineering import (
    FeatureExtractor, FeatureGenerator, FeatureGrouper, FeatureSelector,
)
from atom.nlp import TextCleaner, TextNormalizer, Tokenizer, Vectorizer
from atom.plots import ATOMPlot
from atom.training import (
    DirectClassifier, DirectForecaster, DirectRegressor,
    SuccessiveHalvingClassifier, SuccessiveHalvingForecaster,
    SuccessiveHalvingRegressor, TrainSizingClassifier, TrainSizingForecaster,
    TrainSizingRegressor,
)
from atom.utils.constants import CAT_TYPES, DEFAULT_MISSING, __version__
from atom.utils.types import (
    Backend, Bins, Bool, CategoricalStrats, ColumnSelector, DataFrame,
    DiscretizerStrats, Engine, Estimator, FeatureSelectionSolvers,
    FeatureSelectionStrats, FloatLargerEqualZero, FloatLargerZero,
    FloatZeroToOneInc, Index, IndexSelector, Int, IntLargerEqualZero,
    IntLargerTwo, IntLargerZero, MetricConstructor, ModelsConstructor, NItems,
    NJobs, NormalizerStrats, NumericalStrats, Operators, Pandas, PrunerStrats,
    RowSelector, Scalar, ScalerStrats, Sequence, Series, TargetSelector,
    Transformer, VectorizerStarts, Verbose, Warnings, XSelector, YSelector,
    sequence_t, tsindex_t,
)
from atom.utils.utils import (
    ClassMap, DataConfig, DataContainer, Goal, adjust_verbosity, bk,
    check_dependency, check_scaling, composed, crash, fit_one, flt, get_cols,
    get_custom_scorer, has_task, is_sparse, lst, merge, method_to_log,
    replace_missing, sign, to_pyarrow,
)


T_Transformer = TypeVar("T_Transformer", bound=Transformer)


@beartype
class ATOM(BaseRunner, ATOMPlot, metaclass=ABCMeta):
    """ATOM abstract base class.

    The ATOM class is a convenient wrapper for all data cleaning,
    feature engineering and trainer classes in this package. Provide
    the dataset to the class, and apply all transformations and model
    management from here.

    !!! warning
        This class cannot be called directly. Use the descendant
        classes in api.py instead.

    """

    @property
    @abstractmethod
    def _goal(self) -> Goal: ...

    def __init__(
        self,
        arrays,
        *,
        y: YSelector = -1,
        index: IndexSelector = False,
        shuffle: Bool = True,
        stratify: IndexSelector = True,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Path | Logger | None = None,
        experiment: str | None = None,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            backend=backend,
            memory=memory,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )

        self._config = DataConfig(
            index=index,
            shuffle=shuffle,
            stratify=stratify,
            n_rows=n_rows,
            test_size=test_size,
            holdout_size=holdout_size,
        )

        self._missing = DEFAULT_MISSING

        self._models = ClassMap()
        self._metric = ClassMap()

        self._log("<< ================== ATOM ================== >>", 1)

        # Initialize the branch system and fill with data
        self._branches = BranchManager(memory=self.memory)
        self._branches.fill(*self._get_data(arrays, y=y))

        self._log("\nConfiguration ==================== >>", 1)
        self._log(f"Algorithm task: {self.task}.", 1)
        if self.n_jobs > 1:
            self._log(f"Parallel processing with {self.n_jobs} cores.", 1)
        elif self.backend != "loky":
            self._log(
                "Leaving n_jobs=1 ignores all parallelization. Set n_jobs>1 to make use "
                f"of the {self.backend} parallelization backend.", 1, severity="warning"
            )
        if "cpu" not in self.device.lower():
            self._log(f"Device: {self.device}", 1)
        if (data := self.engine.get("data", "numpy")) != "numpy":
            self._log(f"Data engine: {data}", 1)
        if (models := self.engine.get("estimator", "sklearn")) != "sklearn":
            self._log(f"Estimator engine: {models}", 1)
        if self.backend == "ray" or self.n_jobs > 1:
            self._log(f"Parallelization backend: {self.backend}", 1)
        if self.memory.location is not None:
            self._log(f"Cache storage: {os.path.join(self.memory.location, 'joblib')}", 1)
        if self.experiment:
            self._log(f"Mlflow experiment: {self.experiment}", 1)

        # System settings only to logger
        self._log("\nSystem info ====================== >>", 3)
        self._log(f"Machine: {machine()}", 3)
        self._log(f"OS: {platform()}", 3)
        self._log(f"Python version: {python_version()}", 3)
        self._log(f"Python build: {python_build()}", 3)
        self._log(f"ATOM version: {__version__}", 3)

        # Add an empty rows around stats for a neater look
        self._log("", 1)
        self.stats(1)
        self._log("", 1)

    def __repr__(self) -> str:
        """Print an overview of branches, models, and metrics."""
        out = f"{self.__class__.__name__}"
        out += "\n --> Branches:"
        if len(branches := self._branches.branches) == 1:
            out += f" {self.branch.name}"
        else:
            for branch in branches:
                out += f"\n   --> {branch.name}{' !' if branch is self.branch else ''}"
        out += f"\n --> Models: {', '.join(lst(self.models)) if self.models else None}"
        out += f"\n --> Metric: {', '.join(lst(self.metric)) if self.metric else None}"

        return out

    def __iter__(self) -> Iterator[Transformer]:
        """Iterate over transformers in the pipeline."""
        yield from self.pipeline.named_steps.values()

    # Utility properties =========================================== >>

    @property
    def branch(self) -> Branch:
        """Current active branch.

        Use the property's `@setter` to change the branch or to create
        a new one. If the value is the name of an existing branch,
        switch to that one. Else, create a new branch using that name.
        The new branch is split from the current branch. Use `_from_`
        to split the new branch from any other existing branch. Read
        more in the [user guide][branches].

        """
        return super().branch

    @branch.setter
    def branch(self, name: str):
        """Change from branch or create a new one."""
        if name in self._branches:
            if self.branch is self._branches[name]:
                self._log(f"Already on branch {self.branch.name}.", 1)
            else:
                self._branches.current = name  # type: ignore[assignment]
                self._log(f"Switched to branch {self.branch.name}.", 1)
        else:
            # Branch can be created from current or another one
            if "_from_" in name:
                new_name, parent_name = name.split("_from_")

                # Check if the parent branch exists
                if parent_name not in self._branches:
                    raise ValueError(
                        "The selected branch to split from does not exist! Use "
                        "atom.status() for an overview of the available branches."
                    )
                else:
                    parent = self._branches[parent_name]

            else:
                new_name, parent = name, self.branch

            # Check if the new branch is not in existing
            if new_name in self._branches:  # Can happen when using _from_
                raise ValueError(
                    f"Branch {new_name} already exists. Try using a different "
                    "name. Note that branch names are case-insensitive."
                )

            self._branches.add(name=new_name, parent=parent)
            self._log(f"Successfully created new branch: {new_name}.", 1)

    @branch.deleter
    def branch(self):
        """Delete the current active branch."""
        if len(self._branches) == 1:
            raise PermissionError("Can't delete the last branch!")

        # Delete all depending models
        for model in self._models:
            if model.branch is self.branch:
                self._delete_models(model.name)

        current = self.branch.name
        self._branches.branches.remove(current)
        self._branches.current = self._branches[0].name
        self._log(
            f"Branch {current} successfully deleted. "
            f"Switched to branch {self.branch.name}.", 1
        )

    @property
    def missing(self) -> list[Any]:
        """Values that are considered "missing".

        These values are used by the [clean][self-clean] and
        [impute][self-impute] methods. Default values are: None, NaN,
        NA, NaT, +inf, -inf, "", "?", "NA", "nan", "NaN", "NaT", "none",
        "None", "inf", "-inf". Note that None, NaN, NA, +inf and -inf
        are always considered missing since they are incompatible with
        sklearn estimators.

        """
        return self._missing

    @missing.setter
    def missing(self, value: Sequence[Any]):
        self._missing = list(value)

    @property
    def scaled(self) -> bool:
        """Whether the feature set is scaled.

        A data set is considered scaled when it has mean=0 and std=1,
        or when there is a scaler in the pipeline. Binary columns (only
        zeros and ones) are excluded from the calculation.

        """
        return check_scaling(self.X, pipeline=self.pipeline)

    @property
    def duplicates(self) -> Int:
        """Number of duplicate rows in the dataset."""
        return self.branch.dataset.duplicated().sum()

    @property
    def nans(self) -> Series:
        """Columns with the number of missing values in them.

        This property is unavailable for [sparse datasets][].

        """
        if not is_sparse(self.X):
            return replace_missing(self.X, self.missing).isna().sum()

        raise AttributeError("This property is unavailable for sparse datasets.")

    @property
    def n_nans(self) -> int:
        """Number of rows containing missing values.

        This property is unavailable for [sparse datasets][].

        """
        if not is_sparse(self.X):
            nans = replace_missing(self.X, self.missing).isna().sum(axis=1)
            return len(nans[nans > 0])

        raise AttributeError("This property is unavailable for sparse datasets.")

    @property
    def numerical(self) -> Index:
        """Names of the numerical features in the dataset."""
        return self.X.select_dtypes(include=["number"]).columns

    @property
    def n_numerical(self) -> int:
        """Number of numerical features in the dataset."""
        return len(self.numerical)

    @property
    def categorical(self) -> Index:
        """Names of the categorical features in the dataset."""
        return self.X.select_dtypes(include=CAT_TYPES).columns

    @property
    def n_categorical(self) -> int:
        """Number of categorical features in the dataset."""
        return len(self.categorical)

    @property
    def outliers(self) -> pd.Series:
        """Columns in training set with number of outlier values.

        This property is unavailable for [sparse datasets][].

        """
        if not is_sparse(self.X):
            data = self.branch.train.select_dtypes(include=["number"])
            z_scores = (np.abs(stats.zscore(data.to_numpy(float, na_value=np.nan))) > 3)
            z_scores = pd.Series(z_scores.sum(axis=0), index=data.columns)
            return z_scores[z_scores > 0]

        raise AttributeError("This property is unavailable for sparse datasets.")

    @property
    def n_outliers(self) -> Int:
        """Number of samples in the training set containing outliers.

        This property is unavailable for [sparse datasets][].

        """
        if not is_sparse(self.X):
            data = self.branch.train.select_dtypes(include=["number"])
            z_scores = (np.abs(stats.zscore(data.to_numpy(float, na_value=np.nan))) > 3)
            return z_scores.any(axis=1).sum()

        raise AttributeError("This property is unavailable for sparse datasets.")

    @property
    def classes(self) -> pd.DataFrame:
        """Distribution of target classes per data set.

        This property is only available for classification tasks.

        """
        if self.task.is_classification:
            index = []
            data = defaultdict(list)

            for col in lst(self.target):
                for ds in ("dataset", "train", "test"):
                    values, counts = np.unique(getattr(self, ds)[col], return_counts=True)
                    data[ds].extend(list(counts))
                index.extend([(col, i) for i in values])

            df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index))

            # Non-multioutput has single level index (for simplicity)
            if not self.task.is_multioutput:
                df.index = df.index.droplevel(0)

            return df.fillna(0).astype(int)  # If no counts, returns a NaN -> fill with 0

        raise AttributeError("This property is unavailable for regression tasks.")

    @property
    def n_classes(self) -> Int | Series:
        """Number of classes in the target column(s).

        This property is only available for classification tasks.

        """
        if self.task.is_classification:
            return self.y.nunique(dropna=False)

        raise AttributeError("This property is unavailable for regression tasks.")

    # Utility methods =============================================== >>

    @crash
    def distribution(
        self,
        distributions: str | Sequence[str] | None = None,
        *,
        columns: ColumnSelector | None = None,
    ) -> pd.DataFrame:
        """Get statistics on column distributions.

        Compute the [Kolmogorov-Smirnov test][kstest] for various
        distributions against columns in the dataset. Only for numerical
        columns. Missing values are ignored.

        !!! tip
            Use the [plot_distribution][] method to plot a column's
            distribution.

        Parameters
        ----------
        distributions: str, sequence or None, default=None
            Names of the distributions in `scipy.stats` to get the
            statistics on. If None, a selection of the most common
            ones is used.

        columns: int, str, segment, sequence or None, default=None
            [Selection of columns][row-and-column-selection] to perform
            the test on. If None, select all numerical columns.

        Returns
        -------
        pd.DataFrame
            Statistic results with multiindex levels:

            - **dist:** Name of the distribution.
            - **stat:** Statistic results:
                - **score:** KS-test score.
                - **p_value:** Corresponding p-value.

        """
        if distributions is None:
            distributions_c = [
                "beta",
                "expon",
                "gamma",
                "invgauss",
                "lognorm",
                "norm",
                "pearson3",
                "triang",
                "uniform",
                "weibull_min",
                "weibull_max",
            ]
        else:
            distributions_c = lst(distributions)

        columns_c = self.branch._get_columns(columns, only_numerical=True)

        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                iterables=(distributions_c, ["score", "p_value"]),
                names=["dist", "stat"],
            ),
            columns=columns_c,
        )

        for col in columns_c:
            # Drop missing values from the column before fitting
            X = replace_missing(self[col], self.missing).dropna()
            X = X.to_numpy(dtype=float)

            for dist in distributions_c:
                # Get KS-statistic with fitted distribution parameters
                param = getattr(stats, dist).fit(X)
                stat = stats.kstest(X, dist, args=param)

                # Add as column to the dataframe
                df.at[(dist, "score"), col] = round(stat[0], 4)
                df.at[(dist, "p_value"), col] = round(stat[1], 4)

        return df

    @crash
    def eda(
        self,
        rows: str | Sequence[str] | dict[str, RowSelector] = "dataset",
        *,
        target: TargetSelector = 0,
        filename: str | Path | None = None,
    ):
        """Create an Exploratory Data Analysis report.

        ATOM uses the [sweetviz][] package for EDA. The [report][] is
        rendered directly in the notebook. It can also be accessed
        through the `report` attribute. It can either report one
        dataset or compare two datasets against each other.

        !!! warning
            This method can be slow for large datasets.

        Parameters
        ----------
        rows: str, sequence or dict, default="dataset"
            Selection of rows on which to calculate the metric.

            - If str: Name of the data set to report.
            - If sequence: Names of two data sets to compare.
            - If dict: Names of up to two data sets with corresponding
              [selection of rows][row-and-column-selection] to report.

        target: int or str, default=0
            Target column to look at. Only for [multilabel][] tasks. Only
            bool and numerical features can be used as target.

        filename: str, Path or None, default=None
            Filename or [pathlib.Path][] of the (html) file to save. If
            None, don't save anything.

        """
        check_dependency("sweetviz")
        import sweetviz as sv

        self._log("Creating EDA report...", 1)

        if isinstance(rows, str):
            rows_c = [(self.branch._get_rows(rows), rows)]
        elif isinstance(rows, sequence_t):
            rows_c = [(self.branch._get_rows(r), r) for r in rows]
        elif isinstance(rows, dict):
            rows_c = [(self.branch._get_rows(v), k) for k, v in rows.items()]

        if len(rows_c) == 1:
            self.report = sv.analyze(
                source=rows_c[0],
                target_feat=self.branch._get_target(target, only_columns=True),
            )
        elif len(rows_c) == 2:
            self.report = sv.compare(
                source=rows_c[0],
                compare=rows_c[1],
                target_feat=self.branch._get_target(target, only_columns=True),
            )
        else:
            raise ValueError(
                "Invalid value for the rows parameter. The maximum "
                f"number of data sets to use is 2, got {len(rows_c)}."
            )

        if filename:
            if (path := Path(filename)).suffix != ".html":
                path = path.with_suffix(".html")

        self.report.show_notebook(filepath=path if filename else None)

    @composed(crash, method_to_log)
    def inverse_transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        *,
        verbose: Verbose | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Inversely transform new data through the pipeline.

        Transformers that are only applied on the training set are
        skipped. The rest should all implement an `inverse_transform`
        method. If only `X` or only `y` is provided, it ignores
        transformers that require the other parameter. This can be
        used to transform only the target column.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Transformed feature set with shape=(n_samples, n_features).
            If None, X is ignored in the transformers.

        y: int, str, dict, sequence, dataframe or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput tasks.
            - If dataframe: Target columns for multioutput tasks.

        verbose: int or None, default=None
            Verbosity level for the transformers in the pipeline. If None,
            it uses the pipeline's verbosity.

        Returns
        -------
        dataframe
            Original feature set. Only returned if provided.

        series or dataframe
            Original target column. Only returned if provided.

        """
        with adjust_verbosity(self.pipeline, verbose) as pipeline:
            return pipeline.inverse_transform(X, y)

    @classmethod
    def load(cls, filename: str | Path, data: tuple[Any, ...] | None = None) -> ATOM:
        """Load an atom instance from a pickle file.

        If the instance was [saved][self-save] using `save_data=False`,
        it's possible to load new data into it and apply all data
        transformations.

        !!! info
            The loaded instance's current branch is the same branch as it
            was when saved.

        Parameters
        ----------
        filename: str or Path
            Filename or [pathlib.Path][] of the pickle file.

        data: tuple of indexables or None, default=None
            Original dataset as it was provided to the instance's
            constructor. Only use this parameter if the loaded file
            was saved using `save_data=False`. Allowed formats are:

            - X
            - X, y
            - train, test
            - train, test, holdout
            - X_train, X_test, y_train, y_test
            - X_train, X_test, X_holdout, y_train, y_test, y_holdout
            - (X_train, y_train), (X_test, y_test)
            - (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

            **X, train, test: dataframe-like**<br>
            Feature set with shape=(n_samples, n_features).

            **y: int, str or sequence**<br>
            Target column corresponding to X.

            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        Returns
        -------
        atom
            Unpickled atom instance.

        """
        if (path := Path(filename)).suffix != ".pkl":
            path = path.with_suffix(".pkl")

        with open(path, "rb") as f:
            atom = pickle.load(f)

        # Check if it's an atom instance
        if not atom.__class__.__name__.startswith("ATOM"):
            raise ValueError(
                "The loaded class is not a ATOMClassifier, ATOMRegressor nor "
                f"ATOMForecaster instance, got {atom.__class__.__name__}."
            )

        # Reassign the transformer attributes (warnings random_state, etc...)
        BaseTransformer.__init__(
            atom, **{x: getattr(atom, x) for x in BaseTransformer.attrs},
        )

        if data is not None:
            # Prepare the provided data
            container, holdout = atom._get_data(data)

            # Assign the data to the original branch
            if atom._branches._og is not None:
                atom._branches._og._container = container

            # Apply transformations per branch
            for branch in atom._branches:
                if branch._container is None:
                    branch._container = deepcopy(container)
                    branch._holdout = holdout
                else:
                    raise ValueError(
                        f"The loaded {atom.__class__.__name__} instance "
                        f"already contains data in branch {branch.name}."
                    )

                if len(atom._branches) > 2 and branch.pipeline:
                    atom._log(f"Transforming data for branch {branch.name}:", 1)

                X_train, y_train = branch.pipeline.transform(
                    X=branch.X_train,
                    y=branch.y_train,
                    filter_train_only=False,
                )
                X_test, y_test = branch.pipeline.transform(branch.X_test, branch.y_test)

                # Update complete dataset
                branch._container.data = bk.concat(
                    [merge(X_train, y_train), merge(X_test, y_test)]
                )

                if atom._config.index is False:
                    branch._container = DataContainer(
                        data=(dataset := branch._container.data.reset_index(drop=True)),
                        train_idx=dataset.index[:len(branch._container.train_idx)],
                        test_idx=dataset.index[-len(branch._container.test_idx):],
                        n_cols=branch._container.n_cols,
                    )

                # Store inactive branches in memory
                if branch is not atom.branch:
                    branch.store()

        atom._log(f"{atom.__class__.__name__} successfully loaded.", 1)

        return atom

    @composed(crash, method_to_log)
    def reset(self, hard: Bool = False):
        """Reset the instance to it's initial state.

        Deletes all branches and models. The dataset is also reset
        to its form after initialization.

        Parameters
        ----------
        hard: bool, default=False
            If True, flushes completely the cache.

        """
        self._delete_models(self._get_models())
        self._branches.reset(hard=hard)
        self._log(f"{self.__class__.__name__} successfully reset.", 1)

    @composed(crash, method_to_log)
    def save_data(
        self,
        filename: str | Path = "auto",
        *,
        rows: RowSelector = "dataset",
        **kwargs,
    ):
        """Save the data in the current branch to a `.csv` file.

        Parameters
        ----------
        filename: str or Path, default="auto"
            Filename or [pathlib.Path][] of the file to save. Use
            "auto" for automatic naming.

        rows: hashable, segment, sequence or dataframe, default="dataset"
            [Selection of rows][row-and-column-selection] to save.

        **kwargs
            Additional keyword arguments for pandas' [to_csv][] method.

        """
        if (path := Path(filename)).suffix != ".csv":
            path = path.with_suffix(".csv")

        if path.name == "auto.csv":
            if isinstance(rows, str):
                path = path.with_name(f"{self.__class__.__name__}_{rows}.csv")
            else:
                path = path.with_name(f"{self.__class__.__name__}.csv")

        self.branch._get_rows(rows).to_csv(path, **kwargs)
        self._log("Data set successfully saved.", 1)

    @composed(crash, method_to_log)
    def shrink(
        self,
        *,
        int2bool: Bool = False,
        int2uint: Bool = False,
        str2cat: Bool = False,
        dense2sparse: Bool = False,
        columns: ColumnSelector | None = None,
    ):
        """Convert the columns to the smallest possible matching dtype.

        Examples are: float64 -> float32, int64 -> int8, etc... Sparse
        arrays also transform their non-fill value. Use this method for
        memory optimization before [saving][self-save_data] the dataset.
        Note that applying transformers to the data may alter the types
        again.

        Parameters
        ----------
        int2bool: bool, default=False
            Whether to convert `int` columns to `bool` type. Only if the
            values in the column are strictly in (0, 1) or (-1, 1).

        int2uint: bool, default=False
            Whether to convert `int` to `uint` (unsigned integer). Only if
            the values in the column are strictly positive.

        str2cat: bool, default=False
            Whether to convert `string` to `category`. Only if the
            number of categories is less than 30% of the column's length.

        dense2sparse: bool, default=False
            Whether to convert all features to sparse format. The value
            that is compressed is the most frequent value in the column.

        columns: int, str, segment, sequence or None, default=None
            [Selection of columns][row-and-column-selection] to shrink. If
            None, transform all columns.

        """

        def get_data(new_t: DtypeObj) -> Series:
            """Get the series with the right data format.

            Also converts to sparse format if `dense2sparse=True`.

            Parameters
            ----------
            new_t: DtypeObj
                Data type object to convert to.

            Returns
            -------
            series
                Object with the new data type.

            """
            new_t_np = str(new_t).lower()

            # If already sparse array, cast directly to a new sparse type
            if isinstance(column.dtype, pd.SparseDtype):
                # SparseDtype subtype must be a numpy dtype
                return column.astype(pd.SparseDtype(new_t_np, column.dtype.fill_value))

            if dense2sparse and name not in lst(self.target):  # Skip target cols
                # Select the most frequent value to fill the sparse array
                fill_value = column.mode(dropna=False)[0]

                # Convert first to a sparse array, else fails for nullable pd types
                sparse_col = pd.arrays.SparseArray(column, fill_value=fill_value)

                return sparse_col.astype(pd.SparseDtype(new_t_np, fill_value=fill_value))
            else:
                return column.astype(new_t)

        t1 = (pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype)
        t2 = (pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype)
        t3 = (pd.Float32Dtype, pd.Float64Dtype)

        types: dict[str, list] = {
            "int": [(x.name, np.iinfo(x.type).min, np.iinfo(x.type).max) for x in t1],
            "uint": [(x.name, np.iinfo(x.type).min, np.iinfo(x.type).max) for x in t2],
            "float": [(x.name, np.finfo(x.type).min, np.finfo(x.type).max) for x in t3],
        }

        data = self.branch.dataset[self.branch._get_columns(columns)]

        # Convert back since convert_dtypes doesn't work properly for pyarrow dtypes
        data = data.astype({n: to_pyarrow(c, inverse=True) for n, c in data.items()})

        # Convert to the best nullable dtype
        data = data.convert_dtypes()

        for name, column in data.items():
            # Get subtype from sparse dtypes
            old_t = getattr(column.dtype, "subtype", column.dtype)

            if old_t.name.startswith("string"):
                if str2cat and column.nunique() <= int(len(column) * 0.3):
                    self.branch._data.data[name] = get_data(pd.CategoricalDtype())
                    continue

            try:
                # Get the types to look at
                t = next(v for k, v in types.items() if old_t.name.lower().startswith(k))
            except StopIteration:
                self.branch._data.data[name] = get_data(column.dtype)
                continue

            # Use bool if values are in (0, 1)
            if int2bool and (t == types["int"] or t == types["uint"]):
                if column.isin([0, 1]).all() or column.isin([-1, 1]).all():
                    self.branch._data.data[name] = get_data(pd.BooleanDtype())
                    continue

            # Use uint if values are strictly positive
            if int2uint and t == types["int"] and column.min() >= 0:
                t = types["uint"]

            # Find the smallest type that fits
            self.branch._data.data[name] = next(
                get_data(r[0]) for r in t if r[1] <= column.min() and r[2] >= column.max()
            )

        if self.engine.get("data") == "pyarrow":
            self.branch.dataset = self.dataset.astype(
                {name: to_pyarrow(col) for name, col in self.dataset.items()}
            )

        self._log("The column dtypes are successfully converted.", 1)

    @composed(crash, method_to_log)
    def stats(self, _vb: Int = -2, /):
        """Display basic information about the dataset.

        Parameters
        ----------
        _vb: int, default=-2
            Internal parameter to always print if called by user.

        """
        self._log("Dataset stats " + "=" * 20 + " >>", _vb)
        self._log(f"Shape: {self.shape}", _vb)

        for set_ in ("train", "test", "holdout"):
            if (data := getattr(self, set_)) is not None:
                self._log(f"{set_.capitalize()} set size: {len(data)}", _vb)
                if isinstance(self.branch.train.index, tsindex_t):
                    self._log(f" --> From: {min(data.index)}  To: {max(data.index)}", _vb)

        self._log("-" * 37, _vb)
        if (memory := self.dataset.memory_usage().sum()) < 1e6:
            self._log(f"Memory: {memory / 1e3:.2f} kB", _vb)
        else:
            self._log(f"Memory: {memory / 1e6:.2f} MB", _vb)

        if is_sparse(self.X):
            self._log("Sparse: True", _vb)
            if hasattr(self.X, "sparse"):  # All columns are sparse
                self._log(f"Density: {100. * self.X.sparse.density:.2f}%", _vb)
            else:  # Not all columns are sparse
                n_sparse = sum([pd.api.types.is_sparse(self.X[c]) for c in self.X])
                n_dense = self.n_features - n_sparse
                p_sparse = round(100 * n_sparse / self.n_features, 1)
                p_dense = round(100 * n_dense / self.n_features, 1)
                self._log(f"Dense features: {n_dense} ({p_dense}%)", _vb)
                self._log(f"Sparse features: {n_sparse} ({p_sparse}%)", _vb)
        else:
            nans = self.nans.sum()
            n_categorical = self.n_categorical
            outliers = self.outliers.sum()
            try:  # Can fail for unhashable columns (e.g., multilabel with lists)
                duplicates = self.dataset.duplicated().sum()
            except TypeError:
                duplicates = None
                self._log(
                    "Unable to calculate the number of duplicate "
                    "rows because a column is unhashable.", 3
                )

            if not self.X.empty:
                self._log(f"Scaled: {self.scaled}", _vb)
            if nans:
                p_nans = round(100 * nans / self.branch.dataset.size, 1)
                self._log(f"Missing values: {nans} ({p_nans}%)", _vb)
            if n_categorical:
                p_cat = round(100 * n_categorical / self.n_features, 1)
                self._log(f"Categorical features: {n_categorical} ({p_cat}%)", _vb)
            if outliers:
                p_out = round(100 * outliers / self.branch.train.size, 1)
                self._log(f"Outlier values: {outliers} ({p_out}%)", _vb)
            if duplicates:
                p_dup = round(100 * duplicates / len(self.branch.dataset), 1)
                self._log(f"Duplicates: {duplicates} ({p_dup}%)", _vb)

    @composed(crash, method_to_log)
    def status(self):
        r"""Get an overview of the branches and models.

        This method prints the same information as the \__repr__ and
        also saves it to the logger.

        """
        self._log(str(self))

    @composed(crash, method_to_log)
    def transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        *,
        verbose: Verbose | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Transform new data through the pipeline.

        Transformers that are only applied on the training set are
        skipped. If only `X` or only `y` is provided, it ignores
        transformers that require the other parameter. This can be
         of use to, for example, transform only the target column.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. If None,
            X is ignored in the transformers.

        y: int, str, dict, sequence, dataframe or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput tasks.
            - If dataframe: Target columns for multioutput tasks.

        verbose: int or None, default=None
            Verbosity level for the transformers in the pipeline. If None,
            it uses the pipeline's verbosity.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        with adjust_verbosity(self.pipeline, verbose) as pipeline:
            return pipeline.transform(X, y)

    # Base transformers ============================================ >>

    def _prepare_kwargs(
        self,
        kwargs: dict[str, Any],
        params: MappingProxyType | None = None,
    ) -> dict[str, Any]:
        """Return kwargs with atom's values if not specified.

        This method is used for all transformers and runners to pass
        atom's BaseTransformer's properties to the classes.

        Parameters
        ----------
        kwargs: dict
            Keyword arguments specified in the function call.

        params: mappingproxy or None, default=None
            Parameters in the class' signature.

        Returns
        -------
        dict
            Converted properties.

        """
        for attr in BaseTransformer.attrs:
            if (not params or attr in params) and attr not in kwargs:
                kwargs[attr] = getattr(self, attr)

        return kwargs

    def _add_transformer(
        self,
        transformer: T_Transformer,
        columns: ColumnSelector | None = None,
        train_only: Bool = False,
        **fit_params,
    ) -> T_Transformer:
        """Add a transformer to the pipeline.

        If the transformer is not fitted, it is fitted on the
        complete training set. Afterwards, the data set is
        transformed and the transformer is added to atom's
        pipeline.

        If the transformer has the n_jobs and/or random_state
        parameters and they are left to their default value,
        they adopt atom's values.

        Parameters
        ----------
        transformer: Transformer
            Estimator to add. Should implement a `transform` method.

        columns: int, str, segment, sequence or None, default=None
            Columns in the dataset to transform. If None, transform
            all features.

        train_only: bool, default=False
            Whether to apply the transformer only on the train set or
            on the complete dataset.

        **fit_params
            Additional keyword arguments for the transformer's fit method.

        Returns
        -------
        Transformer
            Fitted transformer.

        """
        if callable(transformer):
            transformer_c = transformer()
        else:
            transformer_c = transformer

        if any(m.branch is self.branch for m in self._models):
            raise PermissionError(
                "It's not allowed to add transformers to the branch "
                "after it has been used to train models. Create a "
                "new branch to continue the pipeline."
            )

        # Add BaseTransformer params to the estimator if left to default
        transformer_c = self._inherit(transformer_c)

        # Transformers remember the train_only and cols parameters
        if not hasattr(transformer_c, "_train_only"):
            transformer_c._train_only = train_only
        if columns is not None:
            inc = self.branch._get_columns(columns)
            fxs_in_inc = any(c in self.features for c in inc)
            target_in_inc = any(c in lst(self.target) for c in inc)
            if fxs_in_inc and target_in_inc:
                self._log(
                    "Features and target columns passed to transformer "
                    f"{transformer_c.__class__.__name__}. Either select features or "
                    "the target column, not both at the same time. The transformation "
                    "of the target column will be ignored.", 1, severity="warning"
                )
            transformer_c._cols = inc

        # Add custom cloning method to keep internal attrs
        transformer_c.__class__.__sklearn_clone__ = TransformerMixin.__sklearn_clone__

        if hasattr(transformer_c, "fit"):
            if not transformer_c.__module__.startswith("atom"):
                self._log(f"Fitting {transformer_c.__class__.__name__}...", 1)

            # Memoize the fitted transformer_c for repeated instantiations of atom
            fit = self._memory.cache(fit_one)
            kwargs = dict(
                estimator=transformer_c,
                X=self.X_train,
                y=self.y_train,
                **fit_params,
            )

            # Check if the fitted estimator is retrieved from cache to inform
            # the user, else user might notice the lack of printed messages
            if self.memory.location is not None:
                if fit._is_in_cache_and_valid([*fit._get_output_identifiers(**kwargs)]):
                    self._log(
                        "Retrieving cached results for "
                        f"{transformer_c.__class__.__name__}...", 1
                    )

            transformer_c = fit(**kwargs)

        # If this is the last empty branch, create a new og branch
        if len([b for b in self._branches if not b.pipeline.steps]) == 1:
            self._branches.add("og")

        if transformer_c._train_only:
            X, y = self.pipeline._mem_transform(transformer_c, self.X_train, self.y_train)
            self.train = merge(
                self.X_train if X is None else X,
                self.y_train if y is None else y,
            )
        else:
            X, y = self.pipeline._mem_transform(transformer_c, self.X, self.y)
            data = merge(self.X if X is None else X, self.y if y is None else y)

            # y can change the number of columns or remove rows -> reassign index
            self.branch._container = DataContainer(
                data=data,
                train_idx=self.branch._data.train_idx.intersection(data.index),
                test_idx=self.branch._data.test_idx.intersection(data.index),
                n_cols=self.branch._data.n_cols if y is None else len(get_cols(y)),
            )

        if self._config.index is False:
            self.branch._container = DataContainer(
                data=(data := self.dataset.reset_index(drop=True)),
                train_idx=data.index[:len(self.branch._data.train_idx)],
                test_idx=data.index[-len(self.branch._data.test_idx):],
                n_cols=self.branch._data.n_cols,
            )
            if self.branch._holdout is not None:
                self.branch._holdout.index = range(
                    len(data), len(data) + len(self.branch._holdout)
                )
        elif self.dataset.index.duplicated().any():
            raise ValueError(
                "Duplicate indices found in the dataset. "
                "Try initializing atom using `index=False`."
            )

        # Add the transformer to the pipeline
        # Check if there already exists an estimator with that
        # name. If so, add a counter at the end of the name
        counter = 1
        name = transformer_c.__class__.__name__.lower()
        while name in self.pipeline:
            counter += 1
            name = f"{transformer_c.__class__.__name__.lower()}-{counter}"

        self.branch.pipeline.steps.append((name, transformer_c))

        # Attach atom's transformer attributes to the branch
        if "atom" in transformer_c.__module__:
            attrs = ("mapping_", "feature_names_in_", "n_features_in_")
            for name, value in vars(transformer_c).items():
                if not name.startswith("_") and name.endswith("_") and name not in attrs:
                    setattr(self.branch, name, value)

        return transformer_c

    @composed(crash, method_to_log)
    def add(
        self,
        transformer: Transformer,
        *,
        columns: ColumnSelector | None = None,
        train_only: Bool = False,
        **fit_params,
    ):
        """Add a transformer to the pipeline.

        If the transformer is not fitted, it is fitted on the complete
        training set. Afterwards, the data set is transformed and the
        estimator is added to atom's pipeline. If the estimator is
        a sklearn Pipeline, every estimator is merged independently
        with atom.

        !!! warning

            * The transformer should have fit and/or transform methods
              with arguments `X` (accepting a dataframe-like object of
              shape=(n_samples, n_features)) and/or `y` (accepting a
              sequence of shape=(n_samples,)).
            * The transform method should return a feature set as a
              dataframe-like object of shape=(n_samples, n_features)
              and/or a target column as a sequence of shape=(n_samples,).

        !!! note
            If the transform method doesn't return a dataframe:

            * The column naming happens as follows. If the transformer
              has a `get_feature_names_out` method, it is used. If not,
              and it returns the same number of columns, the names are
              kept equal. If the number of columns changes, old columns
              will keep their name (as long as the column is unchanged)
              and new columns will receive the name `x[N-1]`, where N
              stands for the n-th feature. This means that a transformer
              should only transform, add or drop columns, not
              combinations of these.
            * The index remains the same as before the transformation.
              This means that the transformer should not add, remove or
              shuffle rows unless it returns a dataframe.

        !!! note
            If the transformer has a `n_jobs` and/or `random_state`
            parameter that is left to its default value, it adopts
            atom's value.

        Parameters
        ----------
        transformer: Transformer
            Estimator to add to the pipeline. Should implement a
            `transform` method.

        columns: int, str, segment, sequence or None, default=None
            [Selection of columns][row-and-column-selection] to
            transform. Only select features or the target column, not
            both at the same time (if that happens, the target column
            is ignored). If None, transform all columns.

        train_only: bool, default=False
            Whether to apply the estimator only on the training set or
            on the complete dataset. Note that if True, the transformation
            is skipped when making predictions on new data.

        **fit_params
            Additional keyword arguments for the transformer's fit method.

        """
        if isinstance(transformer, SkPipeline):
            # Recursively add all transformers to the pipeline
            for name, est in transformer.named_steps.items():
                self._log(f"Adding {est.__class__.__name__} to the pipeline...", 1)
                self._add_transformer(est, columns, train_only, **fit_params)
        else:
            self._log(
                f"Adding {transformer.__class__.__name__} to the pipeline...", 1)
            self._add_transformer(transformer, columns, train_only, **fit_params)

    @composed(crash, method_to_log)
    def apply(
        self,
        func: Callable[..., DataFrame],
        inverse_func: Callable[..., DataFrame] | None = None,
        *,
        kw_args: dict[str, Any] | None = None,
        inv_kw_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Apply a function to the dataset.

        This method is useful for stateless transformations such as
        taking the log, doing custom scaling, etc...

        !!! note
            This approach is preferred over changing the dataset directly
            through the property's `@setter` since the transformation is
            stored in the pipeline.

        !!! tip
            Use `#!python atom.apply(lambda df: df.drop("column_name",
            axis=1))` to store the removal of columns in the pipeline.

        Parameters
        ----------
        func: callable
            Function to apply with signature `func(dataset, **kw_args) ->
            dataset`.

        inverse_func: callable or None, default=None
            Inverse function of `func`. If None, the inverse_transform
            method returns the input unchanged.

        kw_args: dict or None, default=None
            Additional keyword arguments for the function.

        inv_kw_args: dict or None, default=None
            Additional keyword arguments for the inverse function.

        """
        FunctionTransformer = self._get_est_class("FunctionTransformer", "preprocessing")

        columns = kwargs.pop("columns", None)
        transformer = FunctionTransformer(
            func=func,
            inverse_func=inverse_func,
            kw_args=kw_args,
            inv_kw_args=inv_kw_args,
        )

        self._add_transformer(transformer, columns=columns)  # type: ignore[type-var]

    # Data cleaning transformers =================================== >>

    @available_if(has_task(["classification", "!multioutput"]))
    @composed(crash, method_to_log)
    def balance(self, strategy: str | Estimator = "adasyn", **kwargs):
        """Balance the number of rows per class in the target column.

        When oversampling, the newly created samples have an increasing
        integer index for numerical indices, and an index of the form
        [estimator]_N for non-numerical indices, where N stands for the
        N-th sample in the data set.

        See the [Balancer][] class for a description of the parameters.

        !!! warning
            * The balance method does not support [multioutput tasks][].
            * This transformation is only applied to the training set
              in order to maintain the original distribution of target
              classes in the test set.

        !!! tip
            Use atom's [classes][self-classes] attribute for an overview
            of the target class distribution per data set.

        """
        columns = kwargs.pop("columns", None)
        balancer = Balancer(
            strategy=strategy,
            **self._prepare_kwargs(kwargs, sign(Balancer)),
        )

        # Add target column mapping for cleaner printing
        if mapping := self.mapping.get(self.target):
            balancer.mapping_ = mapping

        self._add_transformer(balancer, columns=columns)

    @composed(crash, method_to_log)
    def clean(
        self,
        *,
        convert_dtypes: Bool = True,
        drop_dtypes: str | Sequence[str] | None = None,
        drop_chars: str | None = None,
        strip_categorical: Bool = True,
        drop_duplicates: Bool = False,
        drop_missing_target: Bool = True,
        encode_target: Bool = True,
        **kwargs,
    ):
        """Apply standard data cleaning steps on the dataset.

        Use the parameters to choose which transformations to perform.
        The available steps are:

        - Convert dtypes to the best possible types.
        - Drop columns with specific data types.
        - Remove characters from column names.
        - Strip categorical features from spaces.
        - Drop duplicate rows.
        - Drop rows with missing values in the target column.
        - Encode the target column (ignored for regression tasks).

        See the [Cleaner][] class for a description of the parameters.

        """
        columns = kwargs.pop("columns", None)
        cleaner = Cleaner(
            convert_dtypes=convert_dtypes,
            drop_dtypes=drop_dtypes,
            drop_chars=drop_chars,
            strip_categorical=strip_categorical,
            drop_duplicates=drop_duplicates,
            drop_missing_target=drop_missing_target,
            encode_target=encode_target if self.task.is_classification else False,
            **self._prepare_kwargs(kwargs, sign(Cleaner)),
        )

        # Pass atom's missing values to the cleaner before transforming
        cleaner.missing_ = self.missing

        cleaner = self._add_transformer(cleaner, columns=columns)
        self.branch._mapping.update(cleaner.mapping_)

    @composed(crash, method_to_log)
    def discretize(
        self,
        strategy: DiscretizerStrats = "quantile",
        *,
        bins: Bins = 5,
        labels: Sequence[str] | dict[str, Sequence[str]] | None = None,
        **kwargs,
    ):
        """Bin continuous data into intervals.

        For each feature, the bin edges are computed during fit
        and, together with the number of bins, they will define the
        intervals. Ignores numerical columns.

        See the [Discretizer][] class for a description of the parameters.

        !!! tip
            Use the [plot_distribution][] method to visualize a column's
            distribution and decide on the bins.

        """
        columns = kwargs.pop("columns", None)
        discretizer = Discretizer(
            strategy=strategy,
            bins=bins,
            labels=labels,
            **self._prepare_kwargs(kwargs, sign(Discretizer)),
        )

        self._add_transformer(discretizer, columns=columns)

    @composed(crash, method_to_log)
    def encode(
        self,
        strategy: str = "Target",
        *,
        max_onehot: IntLargerTwo | None = 10,
        ordinal: dict[str, Sequence[Any]] | None = None,
        infrequent_to_value: FloatLargerZero | None = None,
        value: str = "rare",
        **kwargs,
    ):
        """Perform encoding of categorical features.

        The encoding type depends on the number of classes in the
        column:

        - If n_classes=2 or ordinal feature, use Ordinal-encoding.
        - If 2 < n_classes <= `max_onehot`, use OneHot-encoding.
        - If n_classes > `max_onehot`, use `strategy`-encoding.

        Missing values are propagated to the output column. Unknown
        classes encountered during transforming are imputed according
        to the selected strategy. Rare classes can be replaced with a
        value in order to prevent too high cardinality.

        See the [Encoder][] class for a description of the parameters.

        !!! note
            This method only encodes the categorical features. It does
            not encode the target column! Use the [clean][self-clean]
            method for that.

        !!! tip
            Use the [categorical][self-categorical] attribute for a
            list of the categorical features in the dataset.

        """
        columns = kwargs.pop("columns", None)
        encoder = Encoder(
            strategy=strategy,
            max_onehot=max_onehot,
            ordinal=ordinal,
            infrequent_to_value=infrequent_to_value,
            value=value,
            **self._prepare_kwargs(kwargs, sign(Encoder)),
        )

        encoder = self._add_transformer(encoder, columns=columns)
        self.branch._mapping.update(encoder.mapping_)

    @composed(crash, method_to_log)
    def impute(
        self,
        strat_num: Scalar | NumericalStrats = "drop",
        strat_cat: str | CategoricalStrats = "drop",
        *,
        max_nan_rows: FloatLargerZero | None = None,
        max_nan_cols: FloatLargerZero | None = None,
        **kwargs,
    ):
        """Handle missing values in the dataset.

        Impute or remove missing values according to the selected
        strategy. Also removes rows and columns with too many missing
        values. Use the `missing` attribute to customize what are
        considered "missing values".

        See the [Imputer][] class for a description of the parameters.

        !!! tip
            Use the [nans][self-nans] attribute to check the amount of
            missing values per column.

        """
        columns = kwargs.pop("columns", None)
        imputer = Imputer(
            strat_num=strat_num,
            strat_cat=strat_cat,
            max_nan_rows=max_nan_rows,
            max_nan_cols=max_nan_cols,
            **self._prepare_kwargs(kwargs, sign(Imputer)),
        )

        # Pass atom's missing values to the imputer before transforming
        imputer.missing = self.missing

        self._add_transformer(imputer, columns=columns)

    @composed(crash, method_to_log)
    def normalize(self, strategy: NormalizerStrats = "yeojohnson", **kwargs):
        """Transform the data to follow a Normal/Gaussian distribution.

        This transformation is useful for modeling issues related
        to heteroscedasticity (non-constant variance), or other
        situations where normality is desired. Missing values are
        disregarded in fit and maintained in transform. Ignores
        categorical columns.

        See the [Normalizer][] class for a description of the parameters.

        !!! tip
            Use the [plot_distribution][] method to examine a column's
            distribution.

        """
        columns = kwargs.pop("columns", None)
        normalizer = Normalizer(
            strategy=strategy,
            **self._prepare_kwargs(kwargs, sign(Normalizer)),
        )

        self._add_transformer(normalizer, columns=columns)

    @composed(crash, method_to_log)
    def prune(
        self,
        strategy: PrunerStrats | Sequence[PrunerStrats] = "zscore",
        *,
        method: Scalar | Literal["drop", "minmax"] = "drop",
        max_sigma: FloatLargerZero = 3,
        include_target: Bool = False,
        **kwargs,
    ):
        """Prune outliers from the training set.

        Replace or remove outliers. The definition of outlier depends
        on the selected strategy and can greatly differ from one
        another. Ignores categorical columns.

        See the [Pruner][] class for a description of the parameters.

        !!! note
            This transformation is only applied to the training set in
            order to maintain the original distribution of samples in
            the test set.

        !!! tip
            Use the [outliers][self-outliers] attribute to check the
            number of outliers per column.

        """
        columns = kwargs.pop("columns", None)
        pruner = Pruner(
            strategy=strategy,
            method=method,
            max_sigma=max_sigma,
            include_target=include_target,
            **self._prepare_kwargs(kwargs, sign(Pruner)),
        )

        self._add_transformer(pruner, columns=columns)

    @composed(crash, method_to_log)
    def scale(
        self,
        strategy: ScalerStrats = "standard",
        include_binary: Bool = False,
        **kwargs,
    ):
        """Scale the data.

        Apply one of sklearn's scalers. Categorical columns are ignored.

        See the [Scaler][] class for a description of the parameters.

        !!! tip
            Use the [scaled][self-scaled] attribute to check whether
            the dataset is scaled.

        """
        columns = kwargs.pop("columns", None)
        scaler = Scaler(
            strategy=strategy,
            include_binary=include_binary,
            **self._prepare_kwargs(kwargs, sign(Scaler)),
        )

        self._add_transformer(scaler, columns=columns)

    # NLP transformers ============================================= >>

    @composed(crash, method_to_log)
    def textclean(
        self,
        *,
        decode: Bool = True,
        lower_case: Bool = True,
        drop_email: Bool = True,
        regex_email: str | None = None,
        drop_url: Bool = True,
        regex_url: str | None = None,
        drop_html: Bool = True,
        regex_html: str | None = None,
        drop_emoji: Bool = True,
        regex_emoji: str | None = None,
        drop_number: Bool = True,
        regex_number: str | None = None,
        drop_punctuation: Bool = True,
        **kwargs,
    ):
        """Apply standard text cleaning to the corpus.

        Transformations include normalizing characters and drop
        noise from the text (emails, HTML tags, URLs, etc...). The
        transformations are applied on the column named `corpus`, in
        the same order the parameters are presented. If there is no
        column with that name, an exception is raised.

        See the [TextCleaner][] class for a description of the
        parameters.

        """
        columns = kwargs.pop("columns", None)
        textcleaner = TextCleaner(
            decode=decode,
            lower_case=lower_case,
            drop_email=drop_email,
            regex_email=regex_email,
            drop_url=drop_url,
            regex_url=regex_url,
            drop_html=drop_html,
            regex_html=regex_html,
            drop_emoji=drop_emoji,
            regex_emoji=regex_emoji,
            drop_number=drop_number,
            regex_number=regex_number,
            drop_punctuation=drop_punctuation,
            **self._prepare_kwargs(kwargs, sign(TextCleaner)),
        )

        self._add_transformer(textcleaner, columns=columns)

    @composed(crash, method_to_log)
    def textnormalize(
        self,
        *,
        stopwords: Bool | str = True,
        custom_stopwords: Sequence[str] | None = None,
        stem: Bool | str = False,
        lemmatize: Bool = True,
        **kwargs,
    ):
        """Normalize the corpus.

        Convert words to a more uniform standard. The transformations
        are applied on the column named `corpus`, in the same order the
        parameters are presented. If there is no column with that name,
        an exception is raised. If the provided documents are strings,
        words are separated by spaces.

        See the [TextNormalizer][] class for a description of the
        parameters.

        """
        columns = kwargs.pop("columns", None)
        normalizer = TextNormalizer(
            stopwords=stopwords,
            custom_stopwords=custom_stopwords,
            stem=stem,
            lemmatize=lemmatize,
            **self._prepare_kwargs(kwargs, sign(TextNormalizer)),
        )

        self._add_transformer(normalizer, columns=columns)

    @composed(crash, method_to_log)
    def tokenize(
        self,
        bigram_freq: FloatLargerZero | None = None,
        trigram_freq: FloatLargerZero | None = None,
        quadgram_freq: FloatLargerZero | None = None,
        **kwargs,
    ):
        """Tokenize the corpus.

        Convert documents into sequences of words. Additionally,
        create n-grams (represented by words united with underscores,
        e.g., "New_York") based on their frequency in the corpus. The
        transformations are applied on the column named `corpus`. If
        there is no column with that name, an exception is raised.

        See the [Tokenizer][] class for a description of the parameters.

        """
        columns = kwargs.pop("columns", None)
        tokenizer = Tokenizer(
            bigram_freq=bigram_freq,
            trigram_freq=trigram_freq,
            quadgram_freq=quadgram_freq,
            **self._prepare_kwargs(kwargs, sign(Tokenizer)),
        )

        self._add_transformer(tokenizer, columns=columns)

    @composed(crash, method_to_log)
    def vectorize(
        self,
        strategy: VectorizerStarts = "bow",
        *,
        return_sparse: Bool = True,
        **kwargs,
    ):
        """Vectorize the corpus.

        Transform the corpus into meaningful vectors of numbers. The
        transformation is applied on the column named `corpus`. If
        there is no column with that name, an exception is raised.

        If strategy="bow" or "tfidf", the transformed columns are named
        after the word they are embedding with the prefix `corpus_`. If
        strategy="hashing", the columns are named hash[N], where N stands
        for the n-th hashed column.

        See the [Vectorizer][] class for a description of the
        parameters.

        """
        columns = kwargs.pop("columns", None)
        vectorizer = Vectorizer(
            strategy=strategy,
            return_sparse=return_sparse,
            **self._prepare_kwargs(kwargs, sign(Vectorizer)),
        )

        self._add_transformer(vectorizer, columns=columns)

    # Feature engineering transformers ============================= >>

    @composed(crash, method_to_log)
    def feature_extraction(
        self,
        features: str | Sequence[str] = ("day", "month", "year"),
        fmt: str | Sequence[str] | None = None,
        *,
        encoding_type: Literal["ordinal", "cyclic"] = "ordinal",
        drop_columns: Bool = True,
        **kwargs,
    ):
        """Extract features from datetime columns.

        Create new features extracting datetime elements (day, month,
        year, etc...) from the provided columns. Columns of dtype
        `datetime64` are used as is. Categorical columns that can be
        successfully converted to a datetime format (less than 30% NaT
        values after conversion) are also used.

        See the [FeatureExtractor][] class for a description of the
        parameters.

        """
        columns = kwargs.pop("columns", None)
        feature_extractor = FeatureExtractor(
            features=features,
            fmt=fmt,
            encoding_type=encoding_type,
            drop_columns=drop_columns,
            **self._prepare_kwargs(kwargs, sign(FeatureExtractor)),
        )

        self._add_transformer(feature_extractor, columns=columns)

    @composed(crash, method_to_log)
    def feature_generation(
        self,
        strategy: Literal["dfs", "gfg"] = "dfs",
        *,
        n_features: IntLargerZero | None = None,
        operators: Operators | Sequence[Operators] | None = None,
        **kwargs,
    ):
        """Generate new features.

        Create new combinations of existing features to capture the
        non-linear relations between the original features.

        See the [FeatureGenerator][] class for a description of the
        parameters.

        """
        columns = kwargs.pop("columns", None)
        feature_generator = FeatureGenerator(
            strategy=strategy,
            n_features=n_features,
            operators=operators,
            **self._prepare_kwargs(kwargs, sign(FeatureGenerator)),
        )

        self._add_transformer(feature_generator, columns=columns)

    @composed(crash, method_to_log)
    def feature_grouping(
        self,
        groups: dict[str, ColumnSelector],
        *,
        operators: str | Sequence[str] | None = None,
        drop_columns: Bool = True,
        **kwargs,
    ):
        """Extract statistics from similar features.

        Replace groups of features with related characteristics with new
        features that summarize statistical properties of the group. The
        statistical operators are calculated over every row of the group.
        The group names and features can be accessed through the `groups`
        method.

        See the [FeatureGrouper][] class for a description of the
        parameters.

        !!! tip
            Use a regex pattern with the `groups` parameter to select
            groups easier, e.g., `atom.feature_grouping({"group1": "var_.+")`
            to select all features that start with `var_`.

        """
        columns = kwargs.pop("columns", None)
        feature_grouper = FeatureGrouper(
            groups={
                name: self.branch._get_columns(fxs, include_target=False)
                for name, fxs in groups.items()
            },
            operators=operators,
            drop_columns=drop_columns,
            **self._prepare_kwargs(kwargs, sign(FeatureGrouper)),
        )

        self._add_transformer(feature_grouper, columns=columns)

    @composed(crash, method_to_log)
    def feature_selection(
        self,
        strategy: FeatureSelectionStrats | None = None,
        *,
        solver: FeatureSelectionSolvers = None,
        n_features: FloatLargerZero | None = None,
        min_repeated: FloatLargerEqualZero | None = 2,
        max_repeated: FloatLargerEqualZero | None = 1.0,
        max_correlation: FloatZeroToOneInc | None = 1.0,
        **kwargs,
    ):
        """Reduce the number of features in the data.

        Apply feature selection or dimensionality reduction, either to
        improve the estimators' accuracy or to boost their performance
        on very high-dimensional datasets. Additionally, remove
        multicollinear and low-variance features.

        See the [FeatureSelector][] class for a description of the
        parameters.

        !!! note
            * When strategy="univariate" and solver=None, [f_classif][]
              or [f_regression][] is used as default solver.
            * When strategy is "sfs", "rfecv" or any of the
              [advanced strategies][] and no scoring is specified,
              atom's metric (if it exists) is used as scoring.

        """
        if isinstance(strategy, str):
            if strategy == "univariate" and solver is None:
                solver = "f_classif" if self.task.is_classification else "f_regression"
            elif (
                strategy not in ("univariate", "pca")
                and isinstance(solver, str)
                and (not solver.endswith("_class") and not solver.endswith("_reg"))
            ):
                solver += f"_{'class' if self.task.is_classification else 'reg'}"

            # If the run method was called before, use the main metric
            if strategy not in ("univariate", "pca", "sfm", "rfe"):
                if self._metric and "scoring" not in kwargs:
                    kwargs["scoring"] = self._metric[0]

        columns = kwargs.pop("columns", None)
        feature_selector = FeatureSelector(
            strategy=strategy,
            solver=solver,
            n_features=n_features,
            min_repeated=min_repeated,
            max_repeated=max_repeated,
            max_correlation=max_correlation,
            **self._prepare_kwargs(kwargs, sign(FeatureSelector)),
        )

        self._add_transformer(feature_selector, columns=columns)

    # Training methods ============================================= >>

    def _check_metric(self, metric: MetricConstructor) -> MetricConstructor:
        """Check whether the provided metric is valid.

        If there was a previous run, check that the provided metric
        is the same.

        Parameters
        ----------
        metric: str, func, scorer, sequence or None
            Metric provided for the run.

        Returns
        -------
        str, func, scorer, sequence or None
            Metric for the run.

        """
        if self._metric:
            # If the metric is empty, assign the existing one
            if metric is None:
                metric = list(self._metric)
            else:
                # If there's a metric, it should be the same as previous run
                new_metric = [get_custom_scorer(m).name for m in lst(metric)]
                if new_metric != self._metric.keys():
                    raise ValueError(
                        "Invalid value for the metric parameter! The metric "
                        "should be the same as previous run. Expected "
                        f"{self.metric}, got {flt(new_metric)}."
                    )

        return metric

    def _run(self, trainer: BaseRunner):
        """Train and evaluate the models.

        If all models failed, catch the errors and pass them to the
        atom before raising the exception. If the run is successful,
        update all relevant attributes and methods.

        Parameters
        ----------
        trainer: Runner
            Instance that does the actual model training.

        """
        if any(col.dtype.kind not in "ifu" for col in get_cols(self.y)):
            raise ValueError(
                "The target column is not numerical. Use atom.clean() "
                "to encode the target column to numerical values."
            )

        # Transfer attributes
        trainer._config = self._config
        trainer._branches = self._branches

        trainer.run()

        # Overwrite models with the same name as new ones
        for model in trainer._models:
            if model.name in self._models:
                self._delete_models(model.name)
                self._log(
                    f"Consecutive runs of model {model.name}. "
                    "The former model has been overwritten.", 1
                )

        self._models.extend(trainer._models)
        self._metric = trainer._metric

    @composed(crash, method_to_log)
    def run(
        self,
        models: ModelsConstructor = None,
        metric: MetricConstructor = None,
        *,
        est_params: dict[str, Any] | None = None,
        n_trials: NItems = 0,
        ht_params: dict[str, Any] | None = None,
        n_bootstrap: NItems = 0,
        parallel: Bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        **kwargs,
    ):
        """Train and evaluate the models in a direct fashion.

        Contrary to [successive_halving][self-successive_halving] and
        [train_sizing][self-train_sizing], the direct approach only
        iterates once over the models, using the full dataset.

        The following steps are applied to every model:

        1. Apply [hyperparameter tuning][] (optional).
        2. Fit the model on the training set using the best combination
           of hyperparameters found.
        3. Evaluate the model on the test set.
        4. Train the estimator on various [bootstrapped][bootstrapping]
           samples of the training set and evaluate again on the test
           set (optional).

        See the [DirectClassifier][] or [DirectRegressor][] class for a
        description of the parameters.

        """
        trainer = {
            "classification": DirectClassifier,
            "regression": DirectRegressor,
            "forecast": DirectForecaster,
        }

        self._run(
            trainer[self._goal.name](
                models=models,
                metric=self._check_metric(metric),
                est_params=est_params,
                n_trials=n_trials,
                ht_params=ht_params,
                n_bootstrap=n_bootstrap,
                parallel=parallel,
                errors=errors,
                **self._prepare_kwargs(kwargs),
            )
        )

    @composed(crash, method_to_log)
    def successive_halving(
        self,
        models: ModelsConstructor = None,
        metric: MetricConstructor = None,
        *,
        skip_runs: IntLargerEqualZero = 0,
        est_params: dict[str, Any] | None = None,
        n_trials: NItems = 0,
        ht_params: dict[str, Any] | None = None,
        n_bootstrap: NItems = 0,
        parallel: Bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        **kwargs,
    ):
        """Fit the models in a successive halving fashion.

        The successive halving technique is a bandit-based algorithm
        that fits N models to 1/N of the data. The best half are
        selected to go to the next iteration where the process is
        repeated. This continues until only one model remains, which
        is fitted on the complete dataset. Beware that a model's
        performance can depend greatly on the amount of data on which
        it is trained. For this reason, it is recommended to only use
        this technique with similar models, e.g., only using tree-based
        models.

        The following steps are applied to every model (per iteration):

        1. Apply [hyperparameter tuning][] (optional).
        2. Fit the model on the training set using the best combination
           of hyperparameters found.
        3. Evaluate the model on the test set.
        4. Train the estimator on various [bootstrapped][bootstrapping]
           samples of the training set and evaluate again on the test
           set (optional).

        See the [SuccessiveHalvingClassifier][] or [SuccessiveHalvingRegressor][]
        class for a description of the parameters.

        """
        trainer = {
            "classification": SuccessiveHalvingClassifier,
            "regression": SuccessiveHalvingRegressor,
            "forecast": SuccessiveHalvingForecaster,
        }

        self._run(
            trainer[self._goal.name](
                models=models,
                metric=self._check_metric(metric),
                skip_runs=skip_runs,
                est_params=est_params,
                n_trials=n_trials,
                ht_params=ht_params,
                n_bootstrap=n_bootstrap,
                parallel=parallel,
                errors=errors,
                **self._prepare_kwargs(kwargs),
            )
        )

    @composed(crash, method_to_log)
    def train_sizing(
        self,
        models: ModelsConstructor = None,
        metric: MetricConstructor = None,
        *,
        train_sizes: FloatLargerZero | Sequence[FloatLargerZero] = 5,
        est_params: dict[str, Any] | None = None,
        n_trials: NItems = 0,
        ht_params: dict[str, Any] | None = None,
        n_bootstrap: NItems = 0,
        parallel: Bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        **kwargs,
    ):
        """Train and evaluate the models in a train sizing fashion.

        When training models, there is usually a trade-off between
        model performance and computation time; that is regulated by
        the number of samples in the training set. This method can be
        used to create insights in this trade-off, and help determine
        the optimal size of the training set. The models are fitted
        multiple times, ever-increasing the number of samples in the
        training set.

        The following steps are applied to every model (per iteration):

        1. Apply [hyperparameter tuning][] (optional).
        2. Fit the model on the training set using the best combination
           of hyperparameters found.
        3. Evaluate the model on the test set.
        4. Train the estimator on various [bootstrapped][bootstrapping]
           samples of the training set and evaluate again on the test
           set (optional).

        See the [TrainSizingClassifier][] or [TrainSizingRegressor][]
        class for a description of the parameters.

        """
        trainer = {
            "classification": TrainSizingClassifier,
            "regression": TrainSizingRegressor,
            "forecast": TrainSizingForecaster,
        }

        self._run(
            trainer[self._goal.name](
                models=models,
                metric=self._check_metric(metric),
                train_sizes=train_sizes,
                est_params=est_params,
                n_trials=n_trials,
                ht_params=ht_params,
                n_bootstrap=n_bootstrap,
                parallel=parallel,
                errors=errors,
                **self._prepare_kwargs(kwargs),
            )
        )

# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the ATOM class.

"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from copy import deepcopy
from platform import machine, platform, python_build, python_version
from typing import Callable

import dill as pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory

from atom.baserunner import BaseRunner
from atom.basetransformer import BaseTransformer
from atom.branch import Branch
from atom.data_cleaning import (
    Balancer, Cleaner, Discretizer, Encoder, Imputer, Normalizer, Pruner,
    Scaler,
)
from atom.feature_engineering import (
    FeatureExtractor, FeatureGenerator, FeatureGrouper, FeatureSelector,
)
from atom.models import MODELS
from atom.nlp import TextCleaner, TextNormalizer, Tokenizer, Vectorizer
from atom.plots import (
    DataPlot, FeatureSelectorPlot, HTPlot, PredictionPlot, ShapPlot,
)
from atom.training import (
    DirectClassifier, DirectForecaster, DirectRegressor,
    SuccessiveHalvingClassifier, SuccessiveHalvingForecaster,
    SuccessiveHalvingRegressor, TrainSizingClassifier, TrainSizingForecaster,
    TrainSizingRegressor,
)
from atom.utils import (
    DATAFRAME, FEATURES, INT, PANDAS, SCALAR, SEQUENCE, SERIES, TARGET,
    ClassMap, DataConfig, Predictor, Runner, Transformer, __version__, bk,
    check_dependency, check_is_fitted, check_scaling, composed, crash,
    custom_transform, fit_one, flt, get_cols, get_custom_scorer, has_task,
    infer_task, is_multioutput, is_sparse, lst, method_to_log, sign,
    variable_return,
)


class ATOM(BaseRunner, FeatureSelectorPlot, DataPlot, HTPlot, PredictionPlot, ShapPlot):
    """ATOM base class.

    The ATOM class is a convenient wrapper for all data cleaning,
    feature engineering and trainer classes in this package. Provide
    the dataset to the class, and apply all transformations and model
    management from here.

    !!! warning
        This class should not be called directly. Use descendant
        classes ATOMClassifier or ATOMRegressor instead.

    """

    @composed(crash, method_to_log)
    def __init__(
        self,
        arrays,
        *,
        y: TARGET = -1,
        index: bool | INT | str | SEQUENCE = False,
        shuffle: bool = True,
        stratify: bool | INT | str | SEQUENCE = True,
        n_rows: SCALAR = 1,
        test_size: SCALAR = 0.2,
        holdout_size: SCALAR | None = None,
    ):
        self.index = index
        self.shuffle = shuffle
        self.stratify = stratify
        self.n_rows = n_rows
        self.test_size = test_size
        self.holdout_size = holdout_size

        self._config = DataConfig(index, shuffle, stratify, test_size)
        self._memory = check_memory(tempfile.gettempdir())

        self._multioutput = "auto"
        self._missing = [
            None, np.nan, np.inf, -np.inf, "", "?", "NA",
            "nan", "NaN", "none", "None", "inf", "-inf"
        ]

        self._models = ClassMap()
        self._metric = ClassMap()

        self.log("<< ================== ATOM ================== >>", 1)

        self._og = None
        self._current = Branch("master")
        self._branches = ClassMap(self._current)

        self.branch._data, self.branch._idx, holdout = self._get_data(arrays, y=y)
        self.holdout = self.branch._holdout = holdout

        self.task = infer_task(self.y, goal=self.goal)
        self.log(f"Algorithm task: {self.task}.", 1)

        if self.n_jobs > 1:
            self.log(f"Parallel processing with {self.n_jobs} cores.", 1)
        elif self.backend not in ("loky", "ray"):
            self.log(
                "Leaving n_jobs=1 ignores all parallelization. Set n_jobs>1 to make use "
                f"of the {self.backend} parallelization backend.", 1, severity="warning"
            )
        if "gpu" in self.device.lower():
            self.log("GPU training enabled.", 1)
        if self.engine != "sklearn":
            self.log(f"Execution engine: {self.engine}.", 1)
        if self.backend == "ray" or self.n_jobs > 1:
            self.log(f"Parallelization backend: {self.backend}", 1)
        if self.experiment:
            self.log(f"Mlflow experiment: {self.experiment}.", 1)

        # System settings only to logger
        self.log("\nSystem info ====================== >>", 3)
        self.log(f"Machine: {machine()}", 3)
        self.log(f"OS: {platform()}", 3)
        self.log(f"Python version: {python_version()}", 3)
        self.log(f"Python build: {python_build()}", 3)
        self.log(f"ATOM version: {__version__}", 3)

        # Add empty rows around stats for cleaner look
        self.log("", 1)
        self.stats(1)
        self.log("", 1)

    def __repr__(self) -> str:
        out = f"{self.__class__.__name__}"
        out += "\n --> Branches:"
        if len(branches := self._branches) == 1:
            out += f" {self._current.name}"
        else:
            for branch in branches:
                out += f"\n   --> {branch.name}{' !' if branch is self._current else ''}"
        out += f"\n --> Models: {', '.join(lst(self.models)) if self.models else None}"
        out += f"\n --> Metric: {', '.join(lst(self.metric)) if self.metric else None}"

        return out

    def __iter__(self) -> Transformer:
        yield from self.pipeline.values

    # Utility properties =========================================== >>

    @BaseRunner.branch.setter
    def branch(self, name: str):
        """Change branch or create a new one."""
        if name in self._branches:
            if self.branch is self._branches[name]:
                self.log(f"Already on branch {self.branch.name}.", 1)
            else:
                self._current = self._branches[name]
                self.log(f"Switched to branch {self._current.name}.", 1)
        else:
            # Branch can be created from current or another one
            if "_from_" in name:
                new, parent = name.split("_from_")

                # Check if the parent branch exists
                if parent not in self._branches:
                    raise ValueError(
                        "The selected branch to split from does not exist! Use "
                        "atom.status() for an overview of the available branches."
                    )
                else:
                    parent = self._branches[parent]

            else:
                new, parent = name, self._current

            # Check if the new branch is not in existing
            if new in self._branches:  # Can happen when using _from_
                raise ValueError(f"Branch {self._branches[new].name} already exists!")

            self._current = Branch(name=new, parent=parent)
            self._branches.append(self._current)
            self.log(f"New branch {new} successfully created.", 1)

    @property
    def missing(self) -> list:
        """Values that are considered "missing".

        These values are used by the [clean][self-clean] and
        [impute][self-impute] methods. Default values are: None, NaN,
        +inf, -inf, "", "?", "None", "NA", "nan", "NaN" and "inf".
        Note that None, NaN, +inf and -inf are always considered
        missing since they are incompatible with sklearn estimators.

        """
        return self._missing

    @missing.setter
    def missing(self, value: SEQUENCE):
        self._missing = list(set(list(value) + [None, np.nan, np.inf, -np.inf]))

    @property
    def scaled(self) -> bool:
        """Whether the feature set is scaled.

        A data set is considered scaled when it has mean=0 and std=1,
        or when there is a scaler in the pipeline. Binary columns (only
        0s and 1s) are excluded from the calculation.

        """
        return check_scaling(self.X, pipeline=self.pipeline)

    @property
    def duplicates(self) -> SERIES:
        """Number of duplicate rows in the dataset."""
        return self.dataset.duplicated().sum()

    @property
    def nans(self) -> SERIES | None:
        """Columns with the number of missing values in them."""
        if not is_sparse(self.X):
            nans = self.dataset.replace(self.missing, np.NaN)
            nans = nans.isna().sum()
            return nans[nans > 0]

    @property
    def n_nans(self) -> int | None:
        """Number of samples containing missing values."""
        if not is_sparse(self.X):
            nans = self.dataset.replace(self.missing, np.NaN)
            nans = nans.isna().sum(axis=1)
            return len(nans[nans > 0])

    @property
    def numerical(self) -> SERIES:
        """Names of the numerical features in the dataset."""
        return self.X.select_dtypes(include=["number"]).columns

    @property
    def n_numerical(self) -> int:
        """Number of numerical features in the dataset."""
        return len(self.numerical)

    @property
    def categorical(self) -> SERIES:
        """Names of the categorical features in the dataset."""
        return self.X.select_dtypes(include=["object", "category"]).columns

    @property
    def n_categorical(self) -> int:
        """Number of categorical features in the dataset."""
        return len(self.categorical)

    @property
    def outliers(self) -> pd.series | None:
        """Columns in training set with amount of outlier values."""
        if not is_sparse(self.X):
            z_scores = self.train.select_dtypes(include=["number"]).apply(stats.zscore)
            z_scores = (z_scores.abs() > 3).sum(axis=0)
            return z_scores[z_scores > 0]

    @property
    def n_outliers(self) -> int | None:
        """Number of samples in the training set containing outliers."""
        if not is_sparse(self.X):
            z_scores = self.train.select_dtypes(include=["number"]).apply(stats.zscore)
            return (z_scores.abs() > 3).any(axis=1).sum()

    @property
    def classes(self) -> pd.DataFrame | None:
        """Distribution of target classes per data set."""
        if self.goal.startswith("class"):
            index = []
            data = defaultdict(list)

            for col in lst(self.target):
                for ds in ("dataset", "train", "test"):
                    values, counts = np.unique(getattr(self, ds)[col], return_counts=True)
                    data[ds].extend(list(counts))
                index.extend([(col, i) for i in values])

            df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index))

            # Non-multioutput has single level index (for simplicity)
            if not is_multioutput(self.task):
                df.index = df.index.droplevel(0)

            return df.fillna(0).astype(int)  # If no counts, returns a NaN -> fill with 0

    @property
    def n_classes(self) -> int | SERIES | None:
        """Number of classes in the target column(s)."""
        if self.goal.startswith("class"):
            return self.y.nunique(dropna=False)

    # Utility methods =============================================== >>

    @composed(crash, method_to_log)
    def automl(self, **kwargs):
        """Search for an optimized pipeline in an automated fashion.

        Automated machine learning (AutoML) automates the selection,
        composition and parameterization of machine learning pipelines.
        Automating the machine learning often provides faster, more
        accurate outputs than hand-coded algorithms. ATOM uses the
        [evalML][] package for AutoML optimization. The resulting
        transformers and final estimator are merged with atom's pipeline
        (check the [`pipeline`][self-pipeline] and [`models`][self-models]
        attributes after the method finishes running). The created
        [AutoMLSearch][] instance can be accessed through the `evalml`
        attribute.

        !!! warning
            AutoML algorithms aren't intended to run for only a few minutes.
            The method may need a very long time to achieve optimal results.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the AutoMLSearch instance.

        """
        check_dependency("evalml")
        from evalml import AutoMLSearch

        self.log("Searching for optimal pipeline...", 1)

        # Define the objective parameter
        if self._metric and not kwargs.get("objective"):
            kwargs["objective"] = self._metric[0].name
        elif kwargs.get("objective"):
            if not self._metric:
                self._metric = ClassMap(get_custom_scorer(kwargs["objective"]))
            elif kwargs["objective"] != self._metric[0].name:
                raise ValueError(
                    "Invalid value for the objective parameter! The metric "
                    "should be the same as the primary metric. Expected "
                    f"{self._metric[0].name}, got {kwargs['objective']}."
                )

        self.evalml = AutoMLSearch(
            X_train=self.X_train,
            y_train=self.y_train,
            X_holdout=self.X_test,
            y_holdout=self.y_test,
            problem_type=self.task.split(" ")[0],
            objective=kwargs.pop("objective") if "objective" in kwargs else "auto",
            automl_algorithm=kwargs.pop("automl_algorithm", "iterative"),
            n_jobs=kwargs.pop("n_jobs", self.n_jobs),
            verbose=kwargs.pop("verbose", self.verbose > 1),
            random_seed=kwargs.pop("random_seed", self.random_state),
            **kwargs,
        )
        self.evalml.search()

        self.log("\nMerging automl results with atom...", 1)

        # Add transformers and model to atom
        for est in self.evalml.best_pipeline:
            if hasattr(est, "transform"):
                self._add_transformer(est)
                self.log(f" --> Adding {est.__class__.__name__} to the pipeline...", 2)
            else:
                est_name = est._component_obj.__class__.__name__
                for m in MODELS:
                    if m._estimators.get(self.goal) == est_name:
                        model = m(
                            goal=self.goal,
                            metric=self._metric,
                            multioutput=self.multioutput,
                            og=self.og,
                            branch=self.branch,
                            **{x: getattr(self, x) for x in BaseTransformer.attrs},
                        )
                        model._estimator = est._component_obj
                        break

                # Save metric scores on train and test set
                for metric in self._metric:
                    model._get_score(metric, "train")
                    model._get_score(metric, "test")

                self._models.append(model)
                self.log(
                    f" --> Adding model {model._fullname} "
                    f"({model.name}) to the pipeline...", 2
                )
                break  # Avoid non-linear pipelines

    @crash
    def distribution(
        self,
        distributions: str | SEQUENCE | None = None,
        *,
        columns: INT | str | slice | SEQUENCE | None = None,
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

        columns: int, str, slice, sequence or None, default=None
            Names, positions or dtypes of the columns in the dataset to
            perform the test on. If None, select all numerical columns.

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
            distributions = [
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
            distributions = lst(distributions)

        columns = self.branch._get_columns(columns, only_numerical=True)

        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                iterables=(distributions, ["score", "p_value"]),
                names=["dist", "stat"],
            ),
            columns=columns,
        )

        for col in columns:
            # Drop missing values from the column before fitting
            X = self[col].replace(self.missing, np.NaN).dropna()

            for dist in distributions:
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
        dataset: str = "dataset",
        *,
        n_rows: SCALAR | None = None,
        filename: str | None = None,
        **kwargs,
    ):
        """Create an Exploratory Data Analysis report.

        ATOM uses the [ydata-profiling][profiling] package for the EDA.
        The report is rendered directly in the notebook. The created
        [ProfileReport][] instance can be accessed through the `report`
        attribute.

        !!! warning
            This method can be slow for large datasets.

        Parameters
        ----------
        dataset: str, default="dataset"
            Data set to get the report from.

        n_rows: int or None, default=None
            Number of (randomly picked) rows to process. None to use
            all rows.

        filename: str or None, default=None
            Name to save the file with (as .html). None to not save
            anything.

        **kwargs
            Additional keyword arguments for the [ProfileReport][]
            instance.

        """
        check_dependency("ydata-profiling")
        from ydata_profiling import ProfileReport

        self.log("Creating EDA report...", 1)

        n_rows = getattr(self, dataset).shape[0] if n_rows is None else int(n_rows)
        self.report = ProfileReport(getattr(self, dataset).sample(n_rows), **kwargs)

        if filename:
            if not filename.endswith(".html"):
                filename += ".html"
            self.report.to_file(filename)
            self.log("Report successfully saved.", 1)

        self.report.to_notebook_iframe()

    @composed(crash, method_to_log)
    def inverse_transform(
        self,
        X: FEATURES | None = None,
        y: TARGET | None = None,
        *,
        verbose: INT | None = None,
    ) -> PANDAS | tuple[DATAFRAME, SERIES]:
        """Inversely transform new data through the pipeline.

        Transformers that are only applied on the training set are
        skipped. The rest should all implement a `inverse_transform`
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
            - If sequence: Target array with shape=(n_samples,) or
              sequence of column names or positions for multioutput tasks.
            - If dataframe: Target columns for multioutput tasks.

        verbose: int or None, default=None
            Verbosity level for the transformers. If None, it uses the
            transformer's own verbosity.

        Returns
        -------
        dataframe
            Original feature set. Only returned if provided.

        series
            Original target column. Only returned if provided.

        """
        X, y = self._prepare_input(X, y, columns=self.og.features)

        for transformer in reversed(self.pipeline):
            if not transformer._train_only:
                X, y = custom_transform(
                    transformer=transformer,
                    branch=self.branch,
                    data=(X, y),
                    verbose=verbose,
                    method="inverse_transform",
                )

        return variable_return(X, y)

    @classmethod
    def load(
        cls,
        filename: str,
        data: SEQUENCE | None = None,
        *,
        transform_data: bool = True,
        verbose: INT | None = None,
    ):
        """Loads an atom instance from a pickle file.

        If the instance was [saved][self-save] using `save_data=False`,
        it's possible to load new data into it and apply all data
        transformations.

        !!! note
            The loaded instance's current branch is the same branch as it
            was when saved.

        Parameters
        ----------
        filename: str
            Name of the pickle file.

        data: sequence of indexables or None, default=None
            Original dataset. Only use this parameter if the loaded file
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
            - If sequence: Target array with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        transform_data: bool, default=True
            If False, the `data` is left as provided. If True, it's
            transformed through all the steps in the loaded instance's
            pipeline.

        verbose: int or None, default=None
            Verbosity level of the transformations applied on the new
            data. If None, use the verbosity from the loaded instance.
            This parameter is ignored if `transform_data=False`.

        Returns
        -------
        atom instance
            Unpickled atom instance.

        """
        with open(filename, "rb") as f:
            atom = pickle.load(f)

        # Check if atom instance
        if atom.__class__.__name__ not in ("ATOMClassifier", "ATOMRegressor"):
            raise ValueError(
                "The loaded class is not a ATOMClassifier nor "
                f"ATOMRegressor instance, got {atom.__class__.__name__}."
            )

        # Reassign the transformer attributes (warnings random_state, etc...)
        BaseTransformer.__init__(
            atom, **{x: getattr(atom, x) for x in BaseTransformer.attrs},
        )

        if data is not None:
            if any(branch._data is not None for branch in atom._branches):
                raise ValueError(
                    f"The loaded {atom.__class__.__name__} "
                    "instance already contains data."
                )

            # Prepare the provided data
            data, idx, atom.holdout = atom._get_data(data, use_n_rows=transform_data)

            # Apply transformations per branch
            step = {}  # Current step in the pipeline per branch
            for b1 in atom._branches:
                # Provide the input data if not already filled from another branch
                if b1._data is None:
                    b1._data, b1._idx = data, idx

                if transform_data:
                    if len(atom._branches) > 2 and not b1.pipeline.empty:
                        atom.log(f"Transforming data for branch {b1.name}:", 1)

                    for i, est1 in enumerate(b1.pipeline):
                        # Skip if the transformation was already applied
                        if step.get(b1.name, -1) < i:
                            custom_transform(est1, b1, verbose=verbose)

                            if atom.index is False:
                                b1._data = b1.dataset.reset_index(drop=True)
                                b1._idx = [
                                    b1._idx[0],
                                    b1._data.index[:len(b1._idx[1])],
                                    b1._data.index[-len(b1._idx[2]):],
                                ]

                            for b2 in atom._branches:
                                if b1.name != b2.name and b2.pipeline.get(i) is est1:
                                    # Update the data and step for the other branch
                                    atom._branches[b2.name]._data = b1._data.copy()
                                    atom._branches[b2.name]._idx = deepcopy(b1._idx)
                                    atom._branches[b2.name]._holdout = b1._holdout
                                    step[b2.name] = i

        atom.log(f"{atom.__class__.__name__} successfully loaded.", 1)

        return atom

    @composed(crash, method_to_log)
    def reset(self):
        """Reset the instance to it's initial state.

        Deletes all branches and models. The dataset is also reset
        to its form after initialization.

        """
        # Delete all models
        self._delete_models(self._get_models())

        # Recreate the master branch from original and drop rest
        self._current = self.og
        self._current.name = "master"
        self._branches = ClassMap(self._current)
        self._og = None  # Reset original branch

        self.log(f"{self.__class__.__name__} successfully reset.", 1)

    @composed(crash, method_to_log)
    def save_data(self, filename: str = "auto", *, dataset: str = "dataset", **kwargs):
        """Save the data in the current branch to a `.csv` file.

        Parameters
        ----------
        filename: str, default="auto"
            Name of the file. Use "auto" for automatic naming.

        dataset: str, default="dataset"
            Data set to save.

        **kwargs
            Additional keyword arguments for pandas' [to_csv][] method.

        """
        if filename.endswith("auto"):
            filename = filename.replace("auto", f"{self.__class__.__name__}_{dataset}")
        if not filename.endswith(".csv"):
            filename += ".csv"

        getattr(self, dataset).to_csv(filename, **kwargs)
        self.log("Data set successfully saved.", 1)

    @composed(crash, method_to_log)
    def shrink(
        self,
        *,
        obj2cat: bool = True,
        int2uint: bool = False,
        dense2sparse: bool = False,
        columns: INT | str | slice | SEQUENCE | None = None,
    ):
        """Converts the columns to the smallest possible matching dtype.

        Examples are: float64 -> float32, int64 -> int8, etc... Sparse
        arrays also transform their non-fill value. Use this method for
        memory optimization. Note that applying transformers to the
        data may alter the types again.

        Parameters
        ----------
        obj2cat: bool, default=True
            Whether to convert `object` to `category`. Only if the
            number of categories would be less than 30% of the length
            of the column.

        int2uint: bool, default=False
            Whether to convert `int` to `uint` (unsigned integer). Only if
            the values in the column are strictly positive.

        dense2sparse: bool, default=False
            Whether to convert all features to sparse format. The value
            that is compressed is the most frequent value in the column.

        columns: int, str, slice, sequence or None, default=None
            Names, positions or dtypes of the columns in the dataset to
            shrink. If None, transform all columns.

        Notes
        -----
        Partially from: https://github.com/fastai/fastai/blob/master/
        fastai/tabular/core.py

        """
        columns = self.branch._get_columns(columns)
        exclude_types = ["category", "datetime64[ns]", "bool"]

        # Build column filter and types
        types_1 = (np.int8, np.int16, np.int32, np.int64)
        types_2 = (np.uint8, np.uint16, np.uint32, np.uint64)
        types_3 = (np.float32, np.float64, np.longdouble)

        types = {
            "int": [(np.dtype(x), np.iinfo(x).min, np.iinfo(x).max) for x in types_1],
            "uint": [(np.dtype(x), np.iinfo(x).min, np.iinfo(x).max) for x in types_2],
            "float": [(np.dtype(x), np.finfo(x).min, np.finfo(x).max) for x in types_3],
        }

        if obj2cat:
            types["object"] = "category"
        else:
            exclude_types += ["object"]

        new_dtypes = {}
        for name, column in self.dataset.items():
            old_t = column.dtype
            if name not in columns or old_t.name in exclude_types:
                continue

            if pd.api.types.is_sparse(column):
                t = next(v for k, v in types.items() if old_t.subtype.name.startswith(k))
            else:
                t = next(v for k, v in types.items() if old_t.name.startswith(k))

            if isinstance(t, list):
                # Use uint if values are strictly positive
                if int2uint and t == types["int"] and column.min() >= 0:
                    t = types["uint"]

                # Find the smallest type that fits
                new_t = next(
                    r[0] for r in t if r[1] <= column.min() and r[2] >= column.max()
                )
                if new_t and new_t == old_t:
                    new_t = None  # Keep as is
            else:
                # Convert to category if number of categories less than 30% of column
                new_t = t if column.nunique() <= int(len(column) * 0.3) else "object"

            if new_t:
                if pd.api.types.is_sparse(column):
                    new_dtypes[name] = pd.SparseDtype(new_t, old_t.fill_value)
                else:
                    new_dtypes[name] = new_t

        self.branch.dataset = self.branch.dataset.astype(new_dtypes)

        if dense2sparse:
            new_cols = {}
            for name, column in self.X.items():
                new_cols[name] = pd.arrays.SparseArray(
                    data=column,
                    fill_value=column.mode(dropna=False)[0],
                    dtype=column.dtype,
                )

            self.branch.X = bk.DataFrame(new_cols, index=self.y.index)

        self.log("The column dtypes are successfully converted.", 1)

    @composed(crash, method_to_log)
    def stats(self, _vb: INT = -2, /):
        """Print basic information about the dataset.

        Parameters
        ----------
        _vb: int, default=-2
            Internal parameter to always print if called by user.

        """
        self.log("Dataset stats " + "=" * 20 + " >>", _vb)
        self.log(f"Shape: {self.shape}", _vb)
        self.log(f"Train set size: {len(self.train)}", _vb)
        self.log(f"Test set size: {len(self.test)}", _vb)
        if self.holdout is not None:
            self.log(f"Holdout set size: {len(self.holdout)}", _vb)

        self.log("-" * 37, _vb)
        if (memory := self.dataset.memory_usage(deep=True).sum()) < 1e6:
            self.log(f"Memory: {memory / 1e3:.2f} kB", _vb)
        else:
            self.log(f"Memory: {memory / 1e6:.2f} MB", _vb)

        if is_sparse(self.X):
            self.log("Sparse: True", _vb)
            if hasattr(self.X, "sparse"):  # All columns are sparse
                self.log(f"Density: {100. * self.X.sparse.density:.2f}%", _vb)
            else:  # Not all columns are sparse
                n_sparse = sum([pd.api.types.is_sparse(self.X[c]) for c in self.X])
                n_dense = self.n_features - n_sparse
                p_sparse = round(100 * n_sparse / self.n_features, 1)
                p_dense = round(100 * n_dense / self.n_features, 1)
                self.log(f"Dense features: {n_dense} ({p_dense}%)", _vb)
                self.log(f"Sparse features: {n_sparse} ({p_sparse}%)", _vb)
        else:
            nans = self.nans.sum()
            n_categorical = self.n_categorical
            outliers = self.outliers.sum()
            try:
                # Can fail for unhashable columns (e.g. multilabel with lists)
                duplicates = self.dataset.duplicated().sum()
            except TypeError:
                duplicates = None
                self.log(
                    "Unable to calculate the number of duplicate "
                    "rows because a column is unhashable.", 3
                )

            if not self.X.empty:
                self.log(f"Scaled: {self.scaled}", _vb)
            if nans:
                p_nans = round(100 * nans / self.dataset.size, 1)
                self.log(f"Missing values: {nans} ({p_nans}%)", _vb)
            if n_categorical:
                p_cat = round(100 * n_categorical / self.n_features, 1)
                self.log(f"Categorical features: {n_categorical} ({p_cat}%)", _vb)
            if outliers:
                p_out = round(100 * outliers / self.train.size, 1)
                self.log(f"Outlier values: {outliers} ({p_out}%)", _vb)
            if duplicates:
                p_dup = round(100 * duplicates / len(self.dataset), 1)
                self.log(f"Duplicates: {duplicates} ({p_dup}%)", _vb)

    @composed(crash, method_to_log)
    def status(self):
        """Get an overview of the branches and models.

        This method prints the same information as the \__repr__ and
        also saves it to the logger.

        """
        self.log(str(self))

    @composed(crash, method_to_log)
    def transform(
        self,
        X: FEATURES | None = None,
        y: TARGET | None = None,
        *,
        verbose: INT | None = None,
    ) -> PANDAS | tuple[DATAFRAME, SERIES]:
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
            - If sequence: Target array with shape=(n_samples,) or
              sequence of column names or positions for multioutput tasks.
            - If dataframe: Target columns for multioutput tasks.

        verbose: int or None, default=None
            Verbosity level for the transformers. If None, it uses the
            transformer's own verbosity.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series
            Transformed target column. Only returned if provided.

        """
        X, y = self._prepare_input(X, y, columns=self.og.features)

        for transformer in self.pipeline:
            if not transformer._train_only:
                X, y = custom_transform(transformer, self.branch, (X, y), verbose)

        return variable_return(X, y)

    # Base transformers ============================================ >>

    def _prepare_kwargs(self, kwargs: dict, params: list[str] | None = None) -> dict:
        """Return kwargs with atom's values if not specified.

        This method is used for all transformers and runners to pass
        atom's BaseTransformer's properties to the classes.

        Parameters
        ----------
        kwargs: dict
            Keyword arguments specified in the function call.

        params: list or None
            Parameters in the class' signature.

        """
        for attr in BaseTransformer.attrs:
            if (not params or attr in params) and attr not in kwargs:
                kwargs[attr] = getattr(self, attr)

        return kwargs

    def _add_transformer(
        self,
        transformer: Transformer,
        columns: INT | str | slice | SEQUENCE | None = None,
        train_only: bool = False,
        **fit_params,
    ):
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

        columns: int, str, slice, sequence or None, default=None
            Columns in the dataset to transform. If None, transform
            all features.

        train_only: bool, default=False
            Whether to apply the transformer only on the train set or
            on the complete dataset.

        **fit_params
            Additional keyword arguments for the transformer's fit method.

        """
        if any(m.branch is self.branch for m in self._models):
            raise PermissionError(
                "It's not allowed to add transformers to the branch "
                "after it has been used to train models. Create a "
                "new branch to continue the pipeline."
            )

        if not hasattr(transformer, "transform"):
            raise AttributeError("Added transformers should have a transform method!")

        # Add BaseTransformer params to the estimator if left to default
        sig = sign(transformer.__init__)
        for p in ("n_jobs", "random_state"):
            if p in sig and getattr(transformer, p, "<!>") == sig[p]._default:
                setattr(transformer, p, getattr(self, p))

        # Transformers remember the train_only and cols parameters
        if not hasattr(transformer, "_train_only"):
            transformer._train_only = train_only
        if columns is not None:
            inc = self.branch._get_columns(columns)
            fxs_in_inc = any(c in self.features for c in inc)
            target_in_inc = any(c in lst(self.target) for c in inc)
            if fxs_in_inc and target_in_inc:
                self.log(
                    "Features and target columns passed to transformer "
                    f"{transformer.__class__.__name__}. Only select features or the "
                    "target column, not both at the sametime. The transformation of "
                    "the target column will be ignored.", 1, severity="warning"
                )
            transformer._cols = inc

        if hasattr(transformer, "fit") and not check_is_fitted(transformer, False):
            if not transformer.__module__.startswith("atom"):
                self.log(f"Fitting {transformer.__class__.__name__}...", 1)

            fit_one(transformer, self.X_train, self.y_train, **fit_params)

        # Store data in og branch
        if not self._og:
            self._og = Branch(name="og", parent=self.branch)

        custom_transform(transformer, self.branch)

        if self.index is False:
            self.branch._data = self.branch.dataset.reset_index(drop=True)
            self.branch._idx = [
                self.branch._idx[0],
                self.branch._data.index[:len(self.branch._idx[1])],
                self.branch._data.index[-len(self.branch._idx[2]):],
            ]

        # Add the estimator to the pipeline
        self.branch._pipeline = pd.concat(
            [self.pipeline, pd.Series([transformer], dtype="object")],
            ignore_index=True,
        )

    @composed(crash, method_to_log)
    def add(
        self,
        transformer: Transformer,
        *,
        columns: INT | str | slice | SEQUENCE | None = None,
        train_only: bool = False,
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
              has a `get_feature_names_out` or `get_feature_names`
              method, it is used. If not, and it returns the same number
              of columns, the names are kept equal. If the number of
              columns change, old columns will keep their name (as long
              as the column is unchanged) and new columns will receive
              the name `x[N-1]`, where N stands for the n-th feature.
              This means that a transformer should only transform, add
              or drop columns, not combinations of these.
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

        columns: int, str, slice, sequence or None, default=None
            Names, indices or dtypes of the columns in the dataset to
            transform. Only select features or the target column, not
            both at the same time (if that happens, the target column
            is ignored). If None, transform all columns. Add `!` in
            front of a name or dtype to exclude that column, e.g.
            `atom.add(Transformer(), columns="!Location")`</code>`
            transforms all columns except `Location`. You can either
            include or exclude columns, not combinations of these.

        train_only: bool, default=False
            Whether to apply the estimator only on the training set or
            on the complete dataset. Note that if True, the transformation
            is skipped when making predictions on new data.

        **fit_params
            Additional keyword arguments for the transformer's fit method.

        """
        if transformer.__class__.__name__ == "Pipeline":
            # Recursively add all transformers to the pipeline
            for name, est in transformer.named_steps.items():
                self.log(f"Adding {est.__class__.__name__} to the pipeline...", 1)
                self._add_transformer(est, columns, train_only, **fit_params)
        else:
            self.log(
                f"Adding {transformer.__class__.__name__} to the pipeline...", 1)
            self._add_transformer(transformer, columns, train_only, **fit_params)

    @composed(crash, method_to_log)
    def apply(
        self,
        func: Callable[[DATAFRAME, ...], DATAFRAME],
        inverse_func: Callable[[DATAFRAME, ...], DATAFRAME] | None = None,
        *,
        kw_args: dict | None = None,
        inv_kw_args: dict | None = None,
        **kwargs,
    ):
        """Apply a function to the dataset.

        The function should have signature `func(dataset, **kw_args) ->
        dataset`. This method is useful for stateless transformations
        such as taking the log, doing custom scaling, etc...

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
            Function to apply.

        inverse_func: callable or None, default=None
            Inverse function of `func`. If None, the inverse_transform
            method returns the input unchanged.

        kw_args: dict or None, default=None
            Additional keyword arguments for the function.

        inv_kw_args: dict or None, default=None
            Additional keyword arguments for the inverse function.

        """
        columns = kwargs.pop("columns", None)
        function_transformer = FunctionTransformer(
            func=func,
            inverse_func=inverse_func,
            kw_args=kw_args,
            inv_kw_args=inv_kw_args,
        )

        self._add_transformer(function_transformer, columns=columns)

    # Data cleaning transformers =================================== >>

    @available_if(has_task("class"))
    @composed(crash, method_to_log)
    def balance(self, strategy: str = "adasyn", **kwargs):
        """Balance the number of rows per class in the target column.

        When oversampling, the newly created samples have an increasing
        integer index for numerical indices, and an index of the form
        [estimator]_N for non-numerical indices, where N stands for the
        N-th sample in the data set.

        See the [Balancer][] class for a description of the parameters.

        !!! note
            * The balance method does not support [multioutput tasks][].
            * This transformation is only applied to the training set
              in order to maintain the original distribution of target
              classes in the test set.

        !!! tip
            Use atom's [classes][self-classes] attribute for an overview
            of the target class distribution per data set.

        """
        if is_multioutput(self.task):
            raise ValueError("The balance method does not support multioutput tasks.")

        columns = kwargs.pop("columns", None)
        balancer = Balancer(
            strategy=strategy,
            **self._prepare_kwargs(kwargs, sign(Balancer)),
        )

        # Add target column mapping for cleaner printing
        balancer.mapping = self.mapping.get(self.target, {})

        self._add_transformer(balancer, columns=columns)

        # Attach the estimator attribute to atom's branch
        setattr(self.branch, strategy.lower(), getattr(balancer, strategy.lower()))

    @composed(crash, method_to_log)
    def clean(
        self,
        *,
        drop_types: str | SEQUENCE | None = None,
        drop_chars: str | None = None,
        strip_categorical: bool = True,
        drop_duplicates: bool = False,
        drop_missing_target: bool = True,
        encode_target: bool = True,
        **kwargs,
    ):
        """Applies standard data cleaning steps on the dataset.

        Use the parameters to choose which transformations to perform.
        The available steps are:

        - Drop columns with specific data types.
        - Remove characters from column names.
        - Strip categorical features from white spaces.
        - Drop duplicate rows.
        - Drop rows with missing values in the target column.
        - Encode the target column (ignored for regression tasks).

        See the [Cleaner][] class for a description of the parameters.

        """
        columns = kwargs.pop("columns", None)
        cleaner = Cleaner(
            drop_types=drop_types,
            drop_chars=drop_chars,
            strip_categorical=strip_categorical,
            drop_duplicates=drop_duplicates,
            drop_missing_target=drop_missing_target,
            encode_target=encode_target if self.goal == "class" else False,
            **self._prepare_kwargs(kwargs, sign(Cleaner)),
        )

        # Pass atom's missing values to the cleaner before transforming
        cleaner.missing = self.missing

        self._add_transformer(cleaner, columns=columns)
        self.mapping.update(cleaner.mapping)

    @composed(crash, method_to_log)
    def discretize(
        self,
        strategy: str = "quantile",
        *,
        bins: INT | SEQUENCE | dict = 5,
        labels: SEQUENCE | dict | None = None,
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
        max_onehot: INT | None = 10,
        ordinal: dict[str, SEQUENCE] | None = None,
        infrequent_to_value: SCALAR | None = None,
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
            Use the [categorical][self-categorical] attribute  for a
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

        self._add_transformer(encoder, columns=columns)

        # Add mapping of the encoded columns and reorder because of target col
        self.branch._mapping.update(encoder.mapping)
        self.branch._mapping.reorder(self.columns)

    @composed(crash, method_to_log)
    def impute(
        self,
        strat_num: SCALAR | str = "drop",
        strat_cat: str = "drop",
        *,
        max_nan_rows: SCALAR | None = None,
        max_nan_cols: SCALAR | None = None,
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
    def normalize(self, strategy: str = "yeojohnson", **kwargs):
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

        # Attach the estimator attribute to atom's branch
        for attr in ("yeojohnson", "boxcox", "quantile"):
            if hasattr(normalizer, attr):
                setattr(self.branch, attr, getattr(normalizer, attr))

    @composed(crash, method_to_log)
    def prune(
        self,
        strategy: str | SEQUENCE = "zscore",
        *,
        method: SCALAR | str = "drop",
        max_sigma: SCALAR = 3,
        include_target: bool = False,
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

        # Attach the estimator attribute to atom's branch
        for strat in lst(strategy):
            if strat.lower() != "zscore":
                setattr(self.branch, strat.lower(), getattr(pruner, strat.lower()))

    @composed(crash, method_to_log)
    def scale(self, strategy: str = "standard", include_binary: bool = False, **kwargs):
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

        # Attach the estimator attribute to atom's branch
        setattr(self.branch, strategy.lower(), getattr(scaler, strategy.lower()))

    # NLP transformers ============================================= >>

    @composed(crash, method_to_log)
    def textclean(
        self,
        *,
        decode: bool = True,
        lower_case: bool = True,
        drop_email: bool = True,
        regex_email: str | None = None,
        drop_url: bool = True,
        regex_url: str | None = None,
        drop_html: bool = True,
        regex_html: str | None = None,
        drop_emoji: bool = True,
        regex_emoji: str | None = None,
        drop_number: bool = True,
        regex_number: str | None = None,
        drop_punctuation: bool = True,
        **kwargs,
    ):
        """Applies standard text cleaning to the corpus.

        Transformations include normalizing characters and dropping
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

        setattr(self.branch, "drops", getattr(textcleaner, "drops"))

    @composed(crash, method_to_log)
    def textnormalize(
        self,
        *,
        stopwords: bool | str = True,
        custom_stopwords: SEQUENCE | None = None,
        stem: bool | str = False,
        lemmatize: bool = True,
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
        bigram_freq: SCALAR | None = None,
        trigram_freq: SCALAR | None = None,
        quadgram_freq: SCALAR | None = None,
        **kwargs,
    ):
        """Tokenize the corpus.

        Convert documents into sequences of words. Additionally,
        create n-grams (represented by words united with underscores,
        e.g. "New_York") based on their frequency in the corpus. The
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

        self.branch.bigrams = tokenizer.bigrams
        self.branch.trigrams = tokenizer.trigrams
        self.branch.quadgrams = tokenizer.quadgrams

    @composed(crash, method_to_log)
    def vectorize(self, strategy: str = "bow", *, return_sparse: bool = True, **kwargs):
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

        # Attach the estimator attribute to atom's branch
        for attr in ("bow", "tfidf", "hashing"):
            if hasattr(vectorizer, attr):
                setattr(self.branch, attr, getattr(vectorizer, attr))

    # Feature engineering transformers ============================= >>

    @composed(crash, method_to_log)
    def feature_extraction(
        self,
        features: str | SEQUENCE = ["day", "month", "year"],
        fmt: str | SEQUENCE | None = None,
        *,
        encoding_type: str = "ordinal",
        drop_columns: bool = True,
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
        strategy: str = "dfs",
        *,
        n_features: INT | None = None,
        operators: str | SEQUENCE | None = None,
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

        # Attach the genetic attributes to atom's branch
        if strategy.lower() == "gfg":
            self.branch.gfg = feature_generator.gfg
            self.branch.genetic_features = feature_generator.genetic_features

    @composed(crash, method_to_log)
    def feature_grouping(
        self,
        group: dict[str, str | SEQUENCE],
        *,
        operators: str | SEQUENCE | None = None,
        drop_columns: bool = True,
        **kwargs,
    ):
        """Extract statistics from similar features.

        Replace groups of features with related characteristics with new
        features that summarize statistical properties of te group. The
        statistical operators are calculated over every row of the group.
        The group names and features can be accessed through the `groups`
        method.

        See the [FeatureGrouper][] class for a description of the
        parameters.

        """
        columns = kwargs.pop("columns", None)
        feature_grouper = FeatureGrouper(
            group=group,
            operators=operators,
            drop_columns=drop_columns,
            **self._prepare_kwargs(kwargs, sign(FeatureGrouper)),
        )

        self._add_transformer(feature_grouper, columns=columns)
        self.branch.groups = feature_grouper.groups

    @composed(crash, method_to_log)
    def feature_selection(
        self,
        strategy: str | None = None,
        *,
        solver: str | Callable | None = None,
        n_features: SCALAR | None = None,
        min_repeated: SCALAR | None = 2,
        max_repeated: SCALAR | None = 1.0,
        max_correlation: SCALAR | None = 1.0,
        **kwargs,
    ):
        """Reduce the number of features in the data.

        Apply feature selection or dimensionality reduction, either to
        improve the estimators' accuracy or to boost their performance
        on very high-dimensional datasets. Additionally, remove
        multicollinear and low variance features.

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
            if strategy.lower() == "univariate" and solver is None:
                solver = "f_classif" if self.goal == "class" else "f_regression"
            elif (
                strategy.lower() not in ("univariate", "pca")
                and isinstance(solver, str)
                and (not solver.endswith("_class") and not solver.endswith("_reg"))
            ):
                solver += f"_{self.goal}"

            # If the run method was called before, use the main metric
            if strategy.lower() not in ("univariate", "pca", "sfm", "rfe"):
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

        # Add estimator to support multioutput tasks
        feature_selector._multioutput = self.multioutput

        self._add_transformer(feature_selector, columns=columns)

        # Attach used attributes to atom's branch
        for attr in ("collinear", str(strategy).lower()):
            if getattr(feature_selector, attr) is not None:
                setattr(self.branch, attr, getattr(feature_selector, attr))

    # Training methods ============================================= >>

    def _check(self, metric: str | Callable | SEQUENCE) -> str | Callable | SEQUENCE:
        """Check whether the provided metric is valid.

        If there was a previous run, check that the provided metric
        is the same.

        Parameters
        ----------
        metric: str, func, scorer or sequence
            Metric provided for the run.

        Returns
        -------
        str, func, scorer or sequence
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

    def _run(self, trainer: Runner):
        """Train and evaluate the models.

        If all models failed, catch the errors and pass them to the
        atom before raising the exception. If successful run, update
        all relevant attributes and methods.

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
        trainer._og = self._og
        trainer._current = self._current
        trainer._branches = self._branches
        trainer._multioutput = self._multioutput

        trainer.run()

        # Overwrite models with same name as new ones
        for model in trainer._models:
            if model.name in self._models:
                self._delete_models(model.name)
                self.log(
                    f"Consecutive runs of model {model.name}. "
                    "The former model has been overwritten.", 1
                )

        self._models.extend(trainer._models)
        self._metric = trainer._metric

    @composed(crash, method_to_log)
    def run(
        self,
        models: str | Predictor | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        est_params: dict | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
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
        if self.goal == "class":
            trainer = DirectClassifier
        elif self.goal == "reg":
            trainer = DirectRegressor
        else:
            trainer = DirectForecaster

        self._run(
            trainer(
                models=models,
                metric=self._check(metric),
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
        models: str | Predictor | SEQUENCE,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        skip_runs: INT = 0,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
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
        this technique with similar models, e.g. only using tree-based
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
        if self.goal == "class":
            trainer = SuccessiveHalvingClassifier
        elif self.goal == "reg":
            trainer = SuccessiveHalvingRegressor
        else:
            trainer = SuccessiveHalvingForecaster

        self._run(
            trainer(
                models=models,
                metric=self._check(metric),
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
        models: str | Callable | Predictor | SEQUENCE,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        train_sizes: INT | SEQUENCE = 5,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
        **kwargs,
    ):
        """Train and evaluate the models in a train sizing fashion.

        When training models, there is usually a trade-off between
        model performance and computation time, that is regulated by
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
        if self.goal == "class":
            trainer = TrainSizingClassifier
        elif self.goal == "reg":
            trainer = TrainSizingRegressor
        else:
            trainer = TrainSizingForecaster

        self._run(
            trainer(
                models=models,
                metric=self._check(metric),
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

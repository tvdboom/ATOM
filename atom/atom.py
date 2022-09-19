# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the ATOM class.

"""

import tempfile
from copy import deepcopy
from inspect import signature
from platform import machine, platform, python_build, python_version
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib.memory import Memory
from scipy import stats
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.metaestimators import available_if
from typeguard import typechecked

from atom.baserunner import BaseRunner
from atom.basetrainer import BaseTrainer
from atom.basetransformer import BaseTransformer
from atom.branch import Branch
from atom.data_cleaning import (
    Balancer, Cleaner, Discretizer, Encoder, Imputer, Normalizer, Pruner,
    Scaler,
)
from atom.feature_engineering import (
    FeatureExtractor, FeatureGenerator, FeatureGrouper, FeatureSelector,
)
from atom.models import MODELS, MODELS_ENSEMBLES, CustomModel
from atom.nlp import TextCleaner, TextNormalizer, Tokenizer, Vectorizer
from atom.pipeline import Pipeline
from atom.plots import DataPlot, FeatureSelectorPlot, ModelPlot, ShapPlot
from atom.training import (
    DirectClassifier, DirectRegressor, SuccessiveHalvingClassifier,
    SuccessiveHalvingRegressor, TrainSizingClassifier, TrainSizingRegressor,
)
from atom.utils import (
    INT, SCALAR, SEQUENCE_TYPES, X_TYPES, Y_TYPES, CustomDict, Predictor,
    Runner, Scorer, Table, Transformer, __version__, check_is_fitted,
    check_scaling, composed, crash, create_acronym, custom_transform, divide,
    fit_one, flt, get_custom_scorer, get_pl_name, has_task, infer_task,
    is_sparse, lst, method_to_log, variable_return,
)


class ATOM(BaseRunner, FeatureSelectorPlot, DataPlot, ModelPlot, ShapPlot):
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
        y: Y_TYPES = -1,
        index: Union[bool, INT, str, SEQUENCE_TYPES] = False,
        shuffle: bool = True,
        stratify: Union[bool, INT, str, SEQUENCE_TYPES] = True,
        n_rows: SCALAR = 1,
        test_size: SCALAR = 0.2,
        holdout_size: Optional[SCALAR] = None,
    ):
        self.index = index
        self.shuffle = shuffle
        self.stratify = stratify
        self.n_rows = n_rows
        self.test_size = test_size
        self.holdout_size = holdout_size

        self._missing = [
            None, np.nan, np.inf, -np.inf, "", "?", "NA", "nan", "NaN", "None", "inf"
        ]

        self._current = "master"  # Keeps track of the branch the user is in
        self._branches = CustomDict({self._current: Branch(self, self._current)})

        self._models = CustomDict()
        self._metric = CustomDict()
        self._errors = CustomDict()

        self.log("<< ================== ATOM ================== >>", 1)

        # Prepare the provided data
        self.branch._data, self.branch._idx, self.holdout = self._get_data(arrays, y=y)

        self.task = infer_task(self.y, goal=self.goal)
        self.log(f"Algorithm task: {self.task}.", 1)

        if self.n_jobs > 1:
            self.log(f"Parallel processing with {self.n_jobs} cores.", 1)
        if "gpu" in self.device.lower():
            self.log("GPU training enabled.", 1)
        if self.engine != "sklearn":
            self.log(f"Backend engine: {self.engine}.", 1)
        if self.experiment:
            self.log(f"Mlflow experiment: {self.experiment}.", 1)

        # System settings only to logger
        self.log("\nSystem info ====================== >>", 3)
        self.log(f"Machine: {machine()}", 3)
        self.log(f"OS: {platform()}", 3)
        self.log(f"Python version: {python_version()}", 3)
        self.log(f"Python build: {python_build()}", 3)
        self.log(f"ATOM version: {__version__}", 3)

        self.log("", 1)  # Add empty rows around stats for cleaner look
        self.stats(1)
        self.log("", 1)

    def __repr__(self) -> str:
        out = f"{self.__class__.__name__}"
        out += "\n --> Branches:"
        if len(self._branches.min("og")) == 1:
            out += f" {self._current}"
        else:
            for branch in self._branches.min("og"):
                out += f"\n   --> {branch}{' !' if branch == self._current else ''}"
        out += f"\n --> Models: {', '.join(lst(self.models)) if self.models else None}"
        out += f"\n --> Metric: {', '.join(lst(self.metric)) if self.metric else None}"
        out += f"\n --> Errors: {len(self.errors)}"

        return out

    def __iter__(self) -> Transformer:
        yield from self.pipeline.values

    # Utility properties =========================================== >>

    @BaseRunner.branch.setter
    @typechecked
    def branch(self, name: str):
        if name in self._branches and name.lower() != "og":
            if self.branch is self._branches[name]:
                self.log(f"Already on branch {self.branch.name}.", 1)
            else:
                self._current = self._branches[name].name
                self.log(f"Switched to branch {self._current}.", 1)
        else:
            # Branch can be created from current or another one
            if "_from_" in name:
                new, parent = name.split("_from_")
            else:
                new, parent = name, self._current

            # Check if the new name is valid
            if not new:
                raise ValueError("A branch can't have an empty name!")
            elif new in self._branches:  # Can happen when using _from_
                raise ValueError(f"Branch {self._branches[new].name} already exists!")
            else:
                for model in MODELS_ENSEMBLES.values():
                    if new.lower().startswith(model.acronym.lower()):
                        raise ValueError(
                            "Invalid name for the branch. The name of a branch may "
                            f"not begin with a model's acronym, and {model.acronym} "
                            f"is the acronym of the {model._fullname} model."
                        )

            # Check if the parent branch exists
            if parent not in self._branches:
                raise ValueError(
                    "The selected branch to split from does not exist! Use "
                    "atom.status() for an overview of the available branches."
                )

            self._branches[new] = Branch(self, new, parent=self._branches[parent])
            self._current = new
            self.log(f"New branch {self._current} successfully created.", 1)

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
    @typechecked
    def missing(self, value: SEQUENCE_TYPES):
        self._missing = list(set(list(value) + [None, np.nan, np.inf, -np.inf]))

    @property
    def scaled(self) -> bool:
        """Whether the feature set is scaled.

        A data set is considered scaled when it has mean=0 and std=1,
        or when atom has a scaler in the pipeline. Returns None for
        [sparse datasets][].

        """
        if not is_sparse(self.X):
            est_names = [est.__class__.__name__.lower() for est in self.pipeline]
            return check_scaling(self.X) or any("scaler" in name for name in est_names)

    @property
    def duplicates(self) -> pd.Series:
        """Number of duplicate rows in the dataset."""
        return self.dataset.duplicated().sum()

    @property
    def nans(self) -> pd.Series:
        """Columns with the number of missing values in them."""
        if not is_sparse(self.X):
            nans = self.dataset.replace(self.missing, np.NaN)
            nans = nans.isna().sum()
            return nans[nans > 0]

    @property
    def n_nans(self) -> int:
        """Number of samples containing missing values."""
        if not is_sparse(self.X):
            nans = self.dataset.replace(self.missing, np.NaN)
            nans = nans.isna().sum(axis=1)
            return len(nans[nans > 0])

    @property
    def numerical(self) -> pd.Series:
        """Names of the numerical features in the dataset."""
        return self.X.select_dtypes(include=["number"]).columns

    @property
    def n_numerical(self) -> int:
        """Number of numerical features in the dataset."""
        return len(self.numerical)

    @property
    def categorical(self) -> pd.Series:
        """Names of the categorical features in the dataset."""
        return self.X.select_dtypes(include=["object", "category"]).columns

    @property
    def n_categorical(self) -> int:
        """Number of categorical features in the dataset."""
        return len(self.categorical)

    @property
    def outliers(self) -> pd.Series:
        """Columns in training set with amount of outlier values."""
        if not is_sparse(self.X):
            num_and_target = self.dataset.select_dtypes(include=["number"]).columns
            z_scores = stats.zscore(self.train[num_and_target], nan_policy="propagate")
            srs = pd.Series((np.abs(z_scores) > 3).sum(axis=0), index=num_and_target)
            return srs[srs > 0]

    @property
    def n_outliers(self) -> int:
        """Number of samples in the training set containing outliers."""
        if not is_sparse(self.X):
            num_and_target = self.dataset.select_dtypes(include=["number"]).columns
            z_scores = stats.zscore(self.train[num_and_target], nan_policy="propagate")
            return len(np.where((np.abs(z_scores) > 3).any(axis=1))[0])

    @property
    def classes(self) -> pd.DataFrame:
        """Distribution of target classes per data set."""
        return pd.DataFrame(
            {
                "dataset": self.y.value_counts(sort=False, dropna=False),
                "train": self.y_train.value_counts(sort=False, dropna=False),
                "test": self.y_test.value_counts(sort=False, dropna=False),
            },
            index=self.mapping.get(self.target, self.y.sort_values().unique()),
        ).fillna(0).astype(int)  # If no counts, returns a NaN -> fill with 0

    @property
    def n_classes(self) -> int:
        """Number of classes in the target column."""
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
        from evalml import AutoMLSearch

        self.log("Searching for optimal pipeline...", 1)

        # Define the scoring parameter
        if self._metric and not kwargs.get("objective"):
            kwargs["objective"] = self._metric[0]
        elif kwargs.get("objective"):
            kwargs["objective"] = BaseTrainer._prepare_metric([kwargs["objective"]])
            if not self._metric:
                self._metric = kwargs["objective"]  # Update the pipeline's metric
            elif kwargs["objective"][0].name != self.metric[0]:
                raise ValueError(
                    "Invalid value for the objective parameter! The metric "
                    "should be the same as the primary metric. Expected "
                    f"{self.metric[0]}, got {kwargs['objective'][0].name}."
                )

        self.evalml = AutoMLSearch(
            X_train=self.X_train,
            y_train=self.y_train,
            X_holdout=self.X_test,
            y_holdout=self.y_test,
            problem_type=self.task.split(" ")[0],
            objective=kwargs.pop("objective", "auto"),
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
                for key, value in MODELS.items():
                    m = value(self, fast_init=True)
                    if m._est_class.__name__ == est._component_obj.__class__.__name__:
                        est.acronym = key

                # If it's not any of the predefined models, create a new acronym
                if not hasattr(est, "acronym"):
                    est.acronym = create_acronym(est.__class__.__name__)

                model = CustomModel(self, estimator=est)
                model.estimator = model.est

                # Save metric scores on train and test set
                for metric in self._metric.values():
                    model._get_score(metric, "train")
                    model._get_score(metric, "test")

                self._models.update({model.name: model})
                self.log(
                    f" --> Adding model {model._fullname} "
                    f"({model.name}) to the pipeline...", 2
                )
                break  # Avoid non-linear pipelines

    @composed(crash, typechecked)
    def distribution(
        self,
        distributions: Optional[Union[str, SEQUENCE_TYPES]] = None,
        *,
        columns: Optional[Union[INT, str, slice, SEQUENCE_TYPES]] = None,
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

        columns = self._get_columns(columns, only_numerical=True)

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

    @composed(crash, typechecked)
    def export_pipeline(
        self,
        model: Optional[str] = None,
        *,
        memory: Optional[Union[bool, str, Memory]] = None,
        verbose: Optional[INT] = None,
    ) -> Pipeline:
        """Export atom's pipeline.

        Optionally, you can add a model as final estimator. The
        returned pipeline is already fitted on the training set.

        !!! info
            The returned pipeline behaves similarly to sklearn's
            [Pipeline][], and additionally:

            - Accepts transformers that change the target column.
            - Accepts transformers that drop rows.
            - Accepts transformers that only are fitted on a subset of
              the provided dataset.
            - Always returns pandas objects.
            - Uses transformers that are only applied on the training
              set to fit the pipeline, not to make predictions.

        Parameters
        ----------
        model: str or None, default=None
            Name of the model to add as a final estimator to the
            pipeline. If the model used [automated feature scaling][],
            the scaler is added before the model. If None, only the
            transformers are added.

        memory: bool, str, Memory or None, default=None
            Used to cache the fitted transformers of the pipeline.
                - If None or False: No caching is performed.
                - If True: A default temp directory is used.
                - If str: Path to the caching directory.
                - If Memory: Object with the joblib.Memory interface.

        verbose: int or None, default=None
            Verbosity level of the transformers in the pipeline. If
            None, it leaves them to their original verbosity. Note
            that this is not the pipeline's own verbose parameter.
            To change that, use the `set_params` method.

        Returns
        -------
        Pipeline
            Sklearn-like Pipeline object with all transformers in the
            current branch.

        """
        if len(self.pipeline) == 0 and not model:
            raise ValueError("There is no pipeline to export!")

        steps = []
        for transformer in self.pipeline:
            est = deepcopy(transformer)  # Not clone to keep fitted

            # Set the new verbosity (if possible)
            if verbose is not None and hasattr(est, "verbose"):
                est.verbose = verbose

            steps.append((get_pl_name(est.__class__.__name__, steps), est))

        if model:
            model = getattr(self, self._get_models(model)[0])

            if self.branch is not model.branch:
                raise ValueError(
                    "The model to export is not fitted on the current "
                    f"branch. Change to the {model.branch.name} branch "
                    "or use the model's export_pipeline method."
                )

            if model.scaler:
                steps.append(("scaler", deepcopy(model.scaler)))

            steps.append((model.name, deepcopy(model.estimator)))

        if not memory:  # None or False
            memory = None
        elif memory is True:
            memory = tempfile.gettempdir()

        return Pipeline(steps, memory=memory)  # ATOM's pipeline, not sklearn

    @composed(crash, method_to_log, typechecked)
    def inverse_transform(
        self,
        X: Optional[X_TYPES] = None,
        /,
        y: Optional[Y_TYPES] = None,
        *,
        verbose: Optional[INT] = None,
    ) -> Union[pd.Series, pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
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

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.
                - If None: y is ignored in the transformers.
                - If int: Position of the target column in X.
                - If str: Name of the target column in X.
                - Else: Array with shape=(n_samples,) to use as target.

        verbose: int or None, default=None
            Verbosity level for the transformers. If None, it uses the
            transformer's own verbosity.

        Returns
        -------
        pd.DataFrame
            Original feature set. Only returned if provided.

        y: pd.Series
            Original target column. Only returned if provided.

        """
        X, y = self._prepare_input(X, y)

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

    @composed(crash, typechecked)
    def report(
        self,
        dataset: str = "dataset",
        *,
        n_rows: Optional[SCALAR] = None,
        filename: Optional[str] = None,
        **kwargs,
    ):
        """Create an extensive profile analysis report of the data.

        ATOM uses the [pandas-profiling][profiling] package for the
        analysis. The report is rendered directly in the notebook. The
        created [ProfileReport][] instance can be accessed through the
        `profile` attribute.

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
        from pandas_profiling import ProfileReport

        self.log("Creating profile report...", 1)

        n_rows = getattr(self, dataset).shape[0] if n_rows is None else int(n_rows)
        self.profile = ProfileReport(getattr(self, dataset).sample(n_rows), **kwargs)

        if filename:
            if not filename.endswith(".html"):
                filename += ".html"
            self.profile.to_file(filename)
            self.log("Report successfully saved.", 1)

        self.profile.to_notebook_iframe()

    @composed(crash, method_to_log)
    def reset(self):
        """Reset the instance to it's initial state.

        Deletes all branches and models. The dataset is also reset
        to its form after initialization.

        """
        # Delete all models
        self._delete_models(self._get_models())

        # Recreate the master branch from original and drop rest
        self._current = "master"
        self._branches = CustomDict({self._current: self._get_og_branches()[0]})

        self.log(f"{self.__class__.__name__} successfully reset.", 1)

    @composed(crash, method_to_log, typechecked)
    def save_data(self, filename: str = "auto", *, dataset: str = "dataset"):
        """Save the data in the current branch to a `.csv` file.

        Parameters
        ----------
        filename: str, default="auto"
            Name of the file. Use "auto" for automatic naming.

        dataset: str, default="dataset"
            Data set to save.

        """
        if filename.endswith("auto"):
            filename = filename.replace("auto", f"{self.__class__.__name__}_{dataset}")
        if not filename.endswith(".csv"):
            filename += ".csv"

        getattr(self, dataset).to_csv(filename, index=False)
        self.log("Data set successfully saved.", 1)

    @composed(crash, method_to_log, typechecked)
    def shrink(
        self,
        *,
        obj2cat: bool = True,
        int2uint: bool = False,
        dense2sparse: bool = False,
        columns: Optional[Union[INT, str, slice, SEQUENCE_TYPES]] = None,
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
        columns = self._get_columns(columns)
        exclude_types = ["category", "datetime64[ns]", "bool"]

        # Build column filter and type_map
        types_1 = (np.int8, np.int16, np.int32, np.int64)
        types_2 = (np.uint8, np.uint16, np.uint32, np.uint64)
        types_3 = (np.float32, np.float64, np.longdouble)

        type_map = {
            "int": [(np.dtype(x), np.iinfo(x).min, np.iinfo(x).max) for x in types_1],
            "uint": [(np.dtype(x), np.iinfo(x).min, np.iinfo(x).max) for x in types_2],
            "float": [(np.dtype(x), np.finfo(x).min, np.finfo(x).max) for x in types_3],
        }

        if obj2cat:
            type_map["object"] = "category"
        else:
            exclude_types += ["object"]

        new_dtypes = {}
        for name, column in self.dataset.items():
            old_t = column.dtype
            if name not in columns or old_t.name in exclude_types:
                continue

            if pd.api.types.is_sparse(column):
                t = next(
                    v for k, v in type_map.items() if old_t.subtype.name.startswith(k)
                )
            else:
                t = next(
                    v for k, v in type_map.items() if old_t.name.startswith(k)
                )

            if isinstance(t, list):
                # Use uint if values are strictly positive
                if int2uint and t == type_map["int"] and column.min() >= 0:
                    t = type_map["uint"]

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

            self.branch.X = pd.DataFrame(new_cols, index=self.y.index)

        self.log("The column dtypes are successfully converted.", 1)

    @composed(crash, method_to_log)
    def stats(self, _vb: INT = -2, /):
        """Print basic information about the dataset.

        !!! tip
            For classification tasks, the count and balance of classes
            is shown, followed by the ratio (between parentheses) of
            the class with respect to the rest of the classes in the
            same data set, i.e. the class with the fewest samples is
            followed by `(1.0)`. This information can be used to quickly
            assess if the data set is unbalanced.

        Parameters
        ----------
        _vb: int, default=-2
            Internal parameter to always print if called by user.

        """
        self.log("Dataset stats " + "=" * 20 + " >>", _vb)
        self.log(f"Shape: {self.shape}", _vb)

        memory = self.dataset.memory_usage(deep=True).sum()
        if memory < 1e6:
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
            duplicates = self.dataset.duplicated().sum()

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
                self.log(f"Duplicate samples: {duplicates} ({p_dup}%)", _vb)

        self.log("-" * 37, _vb)
        self.log(f"Train set size: {len(self.train)}", _vb)
        self.log(f"Test set size: {len(self.test)}", _vb)
        if self.holdout is not None:
            self.log(f"Holdout set size: {len(self.holdout)}", _vb)

        # Print count and balance of classes
        if self.task != "regression":
            self.log("-" * 37, _vb)
            cls = self.classes
            func = lambda i, col: f"{i} ({divide(i, min(cls[col])):.1f})"

            # Create custom Table object and print the content
            table = Table(
                headers=[("", "left"), *cls.columns],
                spaces=(
                    max(cls.index.astype(str).str.len()),
                    *[len(str(max(cls["dataset"]))) + 8] * len(cls.columns),
                )
            )
            self.log(table.print_header(), _vb + 1)
            self.log(table.print_line(), _vb + 1)
            for i, row in cls.iterrows():
                sequence = {"": i, **{c: func(row[c], c) for c in cls.columns}}
                self.log(table.print(sequence), _vb + 1)

    @composed(crash, method_to_log)
    def status(self):
        """Get an overview of the branches and models.

        This method prints the same information as the \__repr__ and
        also saves it to the logger.

        """
        self.log(str(self))

    @composed(crash, method_to_log, typechecked)
    def transform(
        self,
        X: Optional[X_TYPES] = None,
        /,
        y: Optional[Y_TYPES] = None,
        *,
        verbose: Optional[INT] = None,
    ) -> Union[pd.Series, pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
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

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.
                - If None: y is ignored in the transformers.
                - If int: Position of the target column in X.
                - If str: Name of the target column in X.
                - Else: Array with shape=(n_samples,) to use as target.

        verbose: int or None, default=None
            Verbosity level for the transformers. If None, it uses the
            transformer's own verbosity.

        Returns
        -------
        pd.DataFrame
            Transformed feature set. Only returned if provided.

        pd.Series
            Transformed target column. Only returned if provided.

        """
        X, y = self._prepare_input(X, y)

        for transformer in self.pipeline:
            if not transformer._train_only:
                X, y = custom_transform(transformer, self.branch, (X, y), verbose)

        return variable_return(X, y)

    # Base transformers ============================================ >>

    def _prepare_kwargs(self, kwargs: dict, params: Optional[List[str]] = None) -> dict:
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
        columns: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
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
            Names or indices of the columns in the dataset to transform.
            If None, transform all columns.

        train_only: bool, default=False
            Whether to apply the transformer only on the train set or
            on the complete dataset.

        **fit_params
            Additional keyword arguments for the transformer's fit method.

        """
        if self.branch._get_depending_models():
            raise PermissionError(
                "It's not allowed to add transformers to the branch "
                "after it has been used to train models. Create a "
                "new branch to continue the pipeline."
            )

        if not hasattr(transformer, "transform"):
            raise AttributeError("Added transformers should have a transform method!")

        # Add BaseTransformer params to the estimator if left to default
        sign = signature(transformer.__init__).parameters
        for p in ("n_jobs", "random_state"):
            if p in sign and transformer.get_params()[p] == sign[p]._default:
                transformer.set_params(**{p: getattr(self, p)})

        # Transformers remember the train_only and cols parameters
        transformer._train_only = train_only
        if columns is not None:
            inc, exc = self._get_columns(columns, return_inc_exc=True)
            transformer._cols = [[c for c in inc if c != self.target], exc]

        if hasattr(transformer, "fit") and not check_is_fitted(transformer, False):
            if not transformer.__module__.startswith("atom"):
                self.log(f"Fitting {transformer.__class__.__name__}...", 1)

            fit_one(transformer, self.X_train, self.y_train, **fit_params)

        # Create an og branch before transforming (if it doesn't exist already)
        if self._get_og_branches() == [self.branch]:
            self._branches.insert(0, "og", Branch(self, "og", parent=self.branch))

        custom_transform(transformer, self.branch)

        # Add the estimator to the pipeline
        self.branch.pipeline = pd.concat(
            [self.pipeline, pd.Series([transformer], dtype="object")],
            ignore_index=True,
        )

    @composed(crash, method_to_log, typechecked)
    def add(
        self,
        transformer: Transformer,
        *,
        columns: Optional[Union[INT, str, slice, SEQUENCE_TYPES]] = None,
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
              has a `get_feature_names` or `get_feature_names_out`
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
            transform. If None, transform all columns. Add `!` in front
            of a name or dtype to exclude that column, e.g.
            `atom.add(Transformer(), columns="!Location")`</code>`
            transforms all columns except `Location`. You can either
            include or exclude columns, not combinations of these. The
            target column is always included if required by the transformer.

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

    @composed(crash, method_to_log, typechecked)
    def apply(
        self,
        func: Callable,
        inverse_func: Optional[Callable] = None,
        *,
        kw_args: Optional[dict] = None,
        inv_kw_args: Optional[dict] = None,
        **kwargs,
    ):
        """Apply a function to the dataset.

        The function should have signature `func(dataset, **kw_args)
        -> dataset`. This method is useful for stateless transformations
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
    @composed(crash, method_to_log, typechecked)
    def balance(self, strategy: str = "adasyn", **kwargs):
        """Balance the number of rows per class in the target column.

        When oversampling, the newly created samples have an increasing
        integer index for numerical indices, and an index of the form
        [estimator]_N for non-numerical indices, where N stands for the
        N-th sample in the data set.

        See the [Balancer][] class for a description of the parameters.

        !!! note
            This transformation is only applied to the training set in
            order to maintain the original distribution of target
            classes in the test set.

        !!! tip
            Use atom's [classes][self-classes] attribute for an overview
            of the target class distribution per data set.

        """
        columns = kwargs.pop("columns", None)
        balancer = Balancer(
            strategy=strategy,
            **self._prepare_kwargs(kwargs, signature(Balancer).parameters),
        )

        # Add target column mapping for cleaner printing
        balancer.mapping = self.mapping.get(self.target, {})

        self._add_transformer(balancer, columns=columns, train_only=True)

        # Attach the estimator attribute to atom's branch
        setattr(self.branch, strategy.lower(), getattr(balancer, strategy.lower()))

    @composed(crash, method_to_log, typechecked)
    def clean(
        self,
        *,
        drop_types: Optional[Union[str, SEQUENCE_TYPES]] = None,
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
        - Strip categorical features from white spaces.
        - Drop duplicate rows.
        - Drop rows with missing values in the target column.
        - Encode the target column (can't be True for regression tasks).

        See the [Cleaner][] class for a description of the parameters.

        """
        columns = kwargs.pop("columns", None)
        cleaner = Cleaner(
            drop_types=drop_types,
            strip_categorical=strip_categorical,
            drop_duplicates=drop_duplicates,
            drop_missing_target=drop_missing_target,
            encode_target=encode_target if self.goal == "class" else False,
            **self._prepare_kwargs(kwargs, signature(Cleaner).parameters),
        )

        # Pass atom's missing values to the cleaner before transforming
        cleaner.missing = self.missing

        self._add_transformer(cleaner, columns=columns)

        if cleaner.mapping:
            self.mapping.insert(-1, self.target, cleaner.mapping)

    @composed(crash, method_to_log)
    def discretize(
        self,
        strategy: str = "quantile",
        *,
        bins: Union[INT, SEQUENCE_TYPES, dict] = 5,
        labels: Optional[Union[SEQUENCE_TYPES, dict]] = None,
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
            **self._prepare_kwargs(kwargs, signature(Discretizer).parameters),
        )

        self._add_transformer(discretizer, columns=columns)

    @composed(crash, method_to_log, typechecked)
    def encode(
        self,
        strategy: str = "LeaveOneOut",
        *,
        max_onehot: Optional[INT] = 10,
        ordinal: Optional[Dict[Union[INT, str], SEQUENCE_TYPES]] = None,
        rare_to_value: Optional[SCALAR] = None,
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
            rare_to_value=rare_to_value,
            value=value,
            **self._prepare_kwargs(kwargs, signature(Encoder).parameters),
        )

        self._add_transformer(encoder, columns=columns)

        # Add mapping of the encoded columns and reorder because of target col
        self.mapping.update(encoder.mapping)
        self.mapping = self.mapping[[c for c in self.columns if c in self.mapping]]

    @composed(crash, method_to_log, typechecked)
    def impute(
        self,
        strat_num: Union[SCALAR, str] = "drop",
        strat_cat: str = "drop",
        *,
        max_nan_rows: Optional[SCALAR] = None,
        max_nan_cols: Optional[SCALAR] = None,
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
            **self._prepare_kwargs(kwargs, signature(Imputer).parameters),
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
            **self._prepare_kwargs(kwargs, signature(Normalizer).parameters),
        )

        self._add_transformer(normalizer, columns=columns)

        # Attach the estimator attribute to atom's branch
        for attr in ("yeojohnson", "boxcox", "quantile"):
            if hasattr(normalizer, attr):
                setattr(self.branch, attr, getattr(normalizer, attr))

    @composed(crash, method_to_log, typechecked)
    def prune(
        self,
        strategy: Union[str, SEQUENCE_TYPES] = "zscore",
        *,
        method: Union[SCALAR, str] = "drop",
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
            **self._prepare_kwargs(kwargs, signature(Pruner).parameters),
        )

        self._add_transformer(pruner, columns=columns, train_only=True)

        # Attach the estimator attribute to atom's branch
        for strat in lst(strategy):
            if strat.lower() != "zscore":
                setattr(self.branch, strat.lower(), getattr(pruner, strat.lower()))

    @composed(crash, method_to_log)
    def scale(self, strategy: str = "standard", **kwargs):
        """Scale the data.

        Apply one of sklearn's scalers. Ignores categorical columns.

        See the [Scaler][] class for a description of the parameters.

        !!! tip
            Use the [scaled][self-scaled] attribute to check whether
            the dataset is scaled.

        """
        columns = kwargs.pop("columns", None)
        scaler = Scaler(
            strategy=strategy,
            **self._prepare_kwargs(kwargs, signature(Scaler).parameters),
        )

        self._add_transformer(scaler, columns=columns)

        # Attach the estimator attribute to atom's branch
        setattr(self.branch, strategy.lower(), getattr(scaler, strategy.lower()))

    # NLP transformers ============================================= >>

    @composed(crash, method_to_log, typechecked)
    def textclean(
        self,
        *,
        decode: bool = True,
        lower_case: bool = True,
        drop_email: bool = True,
        regex_email: Optional[str] = None,
        drop_url: bool = True,
        regex_url: Optional[str] = None,
        drop_html: bool = True,
        regex_html: Optional[str] = None,
        drop_emoji: bool = True,
        regex_emoji: Optional[str] = None,
        drop_number: bool = True,
        regex_number: Optional[str] = None,
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
            **self._prepare_kwargs(kwargs, signature(TextCleaner).parameters),
        )

        self._add_transformer(textcleaner, columns=columns)

        setattr(self.branch, "drops", getattr(textcleaner, "drops"))

    @composed(crash, method_to_log, typechecked)
    def textnormalize(
        self,
        *,
        stopwords: Union[bool, str] = True,
        custom_stopwords: Optional[SEQUENCE_TYPES] = None,
        stem: Union[bool, str] = False,
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
            **self._prepare_kwargs(kwargs, signature(TextNormalizer).parameters),
        )

        self._add_transformer(normalizer, columns=columns)

    @composed(crash, method_to_log, typechecked)
    def tokenize(
        self,
        bigram_freq: Optional[SCALAR] = None,
        trigram_freq: Optional[SCALAR] = None,
        quadgram_freq: Optional[SCALAR] = None,
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
            **self._prepare_kwargs(kwargs, signature(Tokenizer).parameters),
        )

        self._add_transformer(tokenizer, columns=columns)

        self.branch.bigrams = tokenizer.bigrams
        self.branch.trigrams = tokenizer.trigrams
        self.branch.quadgrams = tokenizer.quadgrams

    @composed(crash, method_to_log, typechecked)
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
            **self._prepare_kwargs(kwargs, signature(Vectorizer).parameters),
        )

        self._add_transformer(vectorizer, columns=columns)

        # Attach the estimator attribute to atom's branch
        for attr in ("bow", "tfidf", "hashing"):
            if hasattr(vectorizer, attr):
                setattr(self.branch, attr, getattr(vectorizer, attr))

    # Feature engineering transformers ============================= >>

    @composed(crash, method_to_log, typechecked)
    def feature_extraction(
        self,
        features: Union[str, SEQUENCE_TYPES] = ["day", "month", "year"],
        fmt: Optional[Union[str, SEQUENCE_TYPES]] = None,
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
            **self._prepare_kwargs(kwargs, signature(FeatureExtractor).parameters),
        )

        self._add_transformer(feature_extractor, columns=columns)

    @composed(crash, method_to_log, typechecked)
    def feature_generation(
        self,
        strategy: str = "dfs",
        *,
        n_features: Optional[INT] = None,
        operators: Optional[Union[str, SEQUENCE_TYPES]] = None,
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
            **self._prepare_kwargs(kwargs, signature(FeatureGenerator).parameters),
        )

        self._add_transformer(feature_generator, columns=columns)

        # Attach the genetic attributes to atom's branch
        if strategy.lower() == "gfg":
            self.branch.gfg = feature_generator.gfg
            self.branch.genetic_features = feature_generator.genetic_features

    @composed(crash, method_to_log, typechecked)
    def feature_grouping(
        self,
        group: Union[str, SEQUENCE_TYPES],
        name: Optional[Union[str, SEQUENCE_TYPES]] = None,
        *,
        operators: Optional[Union[str, SEQUENCE_TYPES]] = None,
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
            name=name,
            operators=operators,
            drop_columns=drop_columns,
            **self._prepare_kwargs(kwargs, signature(FeatureGrouper).parameters),
        )

        self._add_transformer(feature_grouper, columns=columns)
        self.branch.groups = feature_grouper.groups

    @composed(crash, method_to_log, typechecked)
    def feature_selection(
        self,
        strategy: Optional[str] = None,
        *,
        solver: Optional[Union[str, callable]] = None,
        n_features: Optional[SCALAR] = None,
        min_repeated: Optional[SCALAR] = 2,
        max_repeated: Optional[SCALAR] = 1.0,
        max_correlation: Optional[float] = 1.0,
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
            **self._prepare_kwargs(kwargs, signature(FeatureSelector).parameters),
        )

        self._add_transformer(feature_selector, columns=columns)

        # Attach used attributes to atom's branch
        for attr in ("collinear", "feature_importance", str(strategy).lower()):
            if getattr(feature_selector, attr, None) is not None:
                setattr(self.branch, attr, getattr(feature_selector, attr))

    # Training methods ============================================= >>

    def _check(self, metric: Union[str, callable, Scorer, SEQUENCE_TYPES]) -> CustomDict:
        """Check whether the provided metric is valid.

        Parameters
        ----------
        metric: str, func, callable or sequence
            Metric provided for the run.

        Returns
        -------
        CustomDict
            Metric for the run. Should be the same as previous run.

        """
        if self._metric:
            # If the metric is empty, assign the existing one
            if metric is None:
                metric = self._metric
            else:
                # If there's a metric, it should be the same as previous run
                new_metric = CustomDict(
                    {(s := get_custom_scorer(m)).name: s for m in lst(metric)}
                )

                if list(new_metric) != list(self._metric):
                    raise ValueError(
                        "Invalid value for the metric parameter! The metric "
                        "should be the same as previous run. Expected "
                        f"{self.metric}, got {flt(list(new_metric))}."
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
        try:
            trainer._tracking_params = self._tracking_params
            trainer._current = self._current
            trainer._branches = self._branches
            trainer.scaled = self.scaled
            trainer.run()
        finally:
            # Catch errors and pass them to atom's attribute
            for model, error in trainer.errors.items():
                self._errors[model] = error
                self._models.pop(model)

        # Overwrite models with same name as new ones
        for model in trainer._models:
            if model in self._models:
                self._models.pop(model)
                self.log(
                    f"Consecutive runs of model {model}. The former "
                    " model has been overwritten.", 1, severity="warning"
                )

        self._models.update(trainer._models)
        self._metric = trainer._metric

        for model in self._models.values():
            self._errors.pop(model.name)  # Remove model from errors (if there)
            model.T = self  # Change the model's parent class from trainer to atom

    @composed(crash, method_to_log, typechecked)
    def run(
        self,
        models: Optional[Union[str, callable, Predictor, SEQUENCE_TYPES]] = None,
        metric: Optional[Union[str, callable, Scorer, SEQUENCE_TYPES]] = None,
        *,
        est_params: Optional[dict] = None,
        n_trials: Union[INT, dict, SEQUENCE_TYPES] = 0,
        ht_params: Optional[dict] = None,
        n_bootstrap: Union[INT, SEQUENCE_TYPES] = 0,
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
        4. Train the model on various bootstrapped samples of the
           training set and evaluate again on the test set (optional).

        See the [DirectClassifier][] or [DirectRegressor][] class for a
        description of the parameters.

        """
        if self.goal == "class":
            trainer = DirectClassifier
        else:
            trainer = DirectRegressor

        self._run(
            trainer(
                models=models,
                metric=self._check(metric),
                est_params=est_params,
                n_trials=n_trials,
                ht_params=ht_params,
                n_bootstrap=n_bootstrap,
                **self._prepare_kwargs(kwargs),
            )
        )

    @composed(crash, method_to_log, typechecked)
    def successive_halving(
        self,
        models: Union[str, Predictor, SEQUENCE_TYPES],
        metric: Optional[Union[str, callable, Scorer, SEQUENCE_TYPES]] = None,
        *,
        skip_runs: INT = 0,
        est_params: Optional[Union[dict, SEQUENCE_TYPES]] = None,
        n_trials: Union[INT, dict, SEQUENCE_TYPES] = 0,
        ht_params: Optional[dict] = None,
        n_bootstrap: Union[INT, dict, SEQUENCE_TYPES] = 0,
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
        4. Train the model on various bootstrapped samples of the
           training set and evaluate again on the test set (optional).

        See the [SuccessiveHalvingClassifier][] or [SuccessiveHalvingRegressor][]
        class for a description of the parameters.

        """
        if self.goal == "class":
            trainer = SuccessiveHalvingClassifier
        else:
            trainer = SuccessiveHalvingRegressor

        self._run(
            trainer(
                models=models,
                metric=self._check(metric),
                skip_runs=skip_runs,
                est_params=est_params,
                n_trials=n_trials,
                ht_params=ht_params,
                n_bootstrap=n_bootstrap,
                **self._prepare_kwargs(kwargs),
            )
        )

    @composed(crash, method_to_log, typechecked)
    def train_sizing(
        self,
        models: Union[str, Predictor, SEQUENCE_TYPES],
        metric: Optional[Union[str, callable, Scorer, SEQUENCE_TYPES]] = None,
        *,
        train_sizes: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[Union[dict, SEQUENCE_TYPES]] = None,
        n_trials: Union[INT, dict, SEQUENCE_TYPES] = 0,
        ht_params: Optional[dict] = None,
        n_bootstrap: Union[INT, dict, SEQUENCE_TYPES] = 0,
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
        4. Train the model on various bootstrapped samples of the
           training set and evaluate again on the test set (optional).

        See the [TrainSizingClassifier][] or [TrainSizingRegressor][]
        class for a description of the parameters.

        """
        if self.goal == "class":
            trainer = TrainSizingClassifier
        else:
            trainer = TrainSizingRegressor

        self._run(
            trainer(
                models=models,
                metric=self._check(metric),
                train_sizes=train_sizes,
                est_params=est_params,
                n_trials=n_trials,
                ht_params=ht_params,
                n_bootstrap=n_bootstrap,
                **self._prepare_kwargs(kwargs),
            )
        )

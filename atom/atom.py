# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the ATOM class.

"""

import tempfile
from copy import deepcopy
from inspect import signature
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from joblib.memory import Memory
from scipy import stats
from typeguard import typechecked

from atom.basepredictor import BasePredictor
from atom.basetrainer import BaseTrainer
from atom.basetransformer import BaseTransformer
from atom.branch import Branch
from atom.data_cleaning import (
    Balancer, Cleaner, Discretizer, DropTransformer, Encoder, FuncTransformer,
    Gauss, Imputer, Pruner, Scaler,
)
from atom.feature_engineering import (
    FeatureExtractor, FeatureGenerator, FeatureSelector,
)
from atom.models import MODELS_ENSEMBLES, CustomModel
from atom.nlp import Normalizer, TextCleaner, Tokenizer, Vectorizer
from atom.pipeline import Pipeline
from atom.plots import ATOMPlotter
from atom.training import (
    DirectClassifier, DirectRegressor, SuccessiveHalvingClassifier,
    SuccessiveHalvingRegressor, TrainSizingClassifier, TrainSizingRegressor,
)
from atom.utils import (
    INT, SCALAR, SEQUENCE_TYPES, X_TYPES, Y_TYPES, CustomDict, Table,
    check_dim, check_is_fitted, check_scaling, composed, crash,
    custom_transform, delete, divide, fit_one, flt, get_pl_name, infer_task,
    is_multidim, is_sparse, lst, method_to_log, names_from_estimator,
    variable_return,
)


class ATOM(BasePredictor, ATOMPlotter):
    """ATOM base class.

    The ATOM class is a convenient wrapper for all data cleaning,
    feature engineering and trainer estimators in this package.
    Provide the dataset to the class, and apply all transformations
    and model management from here.

    Warning: This class should not be called directly. Use descendant
    classes ATOMClassifier or ATOMRegressor instead.

    """

    @composed(crash, method_to_log)
    def __init__(
        self,
        arrays,
        y=-1,
        index=False,
        shuffle=True,
        stratify=True,
        n_rows=1,
        test_size=0.2,
        holdout_size=None,
    ):
        self.index = index
        self.shuffle = shuffle
        self.stratify = stratify
        self.n_rows = n_rows
        self.test_size = test_size
        self.holdout_size = holdout_size
        self.missing = ["", "?", "NA", "nan", "NaN", "None", "inf"]

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
        if self.gpu:
            self.log("GPU training enabled.", 1)

        self.log("", 1)  # Add empty rows around stats for cleaner look
        self.stats(1)
        self.log("", 1)

    def __repr__(self):
        out = f"{self.__class__.__name__}"
        out += "\n --> Branches:"
        if len(self._branches.min("og")) == 1:
            out += f" {self._current}"
        else:
            for branch in self._branches.min("og"):
                out += f"\n   >>> {branch}{' !' if branch == self._current else ''}"
        out += f"\n --> Models: {', '.join(lst(self.models)) if self.models else None}"
        out += f"\n --> Metric: {', '.join(lst(self.metric)) if self.metric else None}"
        out += f"\n --> Errors: {len(self.errors)}"

        return out

    def __iter__(self):
        yield from self.pipeline.values

    # Utility properties =========================================== >>

    @BasePredictor.branch.setter
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
                new, old = name.split("_from_")
            else:
                new, old = name, self._current

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
                            f"is the acronym of the {model.fullname} model."
                        )

            # Check if the parent branch exists
            if old not in self._branches:
                raise ValueError(
                    "The selected branch to split from does not exist! Use "
                    "atom.status() for an overview of the available branches."
                )

            self._branches[new] = Branch(self, new, parent=self._branches[old])
            self._current = new
            self.log(f"New branch {self._current} successfully created.", 1)

    @property
    def scaled(self):
        """Whether the feature set is scaled."""
        if not is_multidim(self.X) and not is_sparse(self.X):
            est_names = [est.__class__.__name__.lower() for est in self.pipeline]
            return check_scaling(self.X) or any("scaler" in name for name in est_names)

    @property
    def duplicates(self):
        """Number of duplicate rows in the dataset."""
        if not is_multidim(self.X):
            return self.dataset.duplicated().sum()

    @property
    def nans(self):
        """Columns with the number of missing values in them."""
        if not is_multidim(self.X) and not is_sparse(self.X):
            nans = self.dataset.replace(self.missing + [np.inf, -np.inf], np.NaN)
            nans = nans.isna().sum()
            return nans[nans > 0]

    @property
    def n_nans(self):
        """Number of samples containing missing values."""
        if not is_multidim(self.X) and not is_sparse(self.X):
            nans = self.dataset.replace(self.missing + [np.inf, -np.inf], np.NaN)
            nans = nans.isna().sum(axis=1)
            return len(nans[nans > 0])

    @property
    def numerical(self):
        """Names of the numerical features in the dataset."""
        if not is_multidim(self.X):
            return self.X.select_dtypes(include=["number"]).columns

    @property
    def n_numerical(self):
        """Number of numerical features in the dataset."""
        if not is_multidim(self.X):
            return len(self.numerical)

    @property
    def categorical(self):
        """Names of the categorical features in the dataset."""
        if not is_multidim(self.X):
            return self.X.select_dtypes(include=["object", "category"]).columns

    @property
    def n_categorical(self):
        """Number of categorical features in the dataset."""
        if not is_multidim(self.X):
            return len(self.categorical)

    @property
    def outliers(self):
        """Columns in training set with amount of outlier values."""
        if not is_multidim(self.X) and not is_sparse(self.X):
            num_and_target = self.dataset.select_dtypes(include=["number"]).columns
            z_scores = stats.zscore(self.train[num_and_target], nan_policy="propagate")
            srs = pd.Series((np.abs(z_scores) > 3).sum(axis=0), index=num_and_target)
            return srs[srs > 0]

    @property
    def n_outliers(self):
        """Number of samples in the training set containing outliers."""
        if not is_multidim(self.X) and not is_sparse(self.X):
            num_and_target = self.dataset.select_dtypes(include=["number"]).columns
            z_scores = stats.zscore(self.train[num_and_target], nan_policy="propagate")
            return len(np.where((np.abs(z_scores) > 3).any(axis=1))[0])

    @property
    def classes(self):
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
    def n_classes(self):
        """Number of classes in the target column."""
        return self.y.nunique(dropna=False)

    # Utility methods =============================================== >>

    def automl(self, **kwargs):
        """Search for an optimized pipeline in an automated fashion.

        Uses the TPOT package to perform an automated search of
        transformers and a final estimator that maximizes a metric
        on the dataset. The resulting transformations and estimator
        are merged with atom's pipeline. The tpot instance can be
        accessed through the `tpot` attribute.

        Parameters
        ----------
        **kwargs
            Keyword arguments for tpot's classifier/regressor.

        """
        from tpot import TPOTClassifier, TPOTRegressor

        check_dim(self, "automl")

        # Define the scoring parameter
        if self._metric and not kwargs.get("scoring"):
            kwargs["scoring"] = self._metric[0]
        elif kwargs.get("scoring"):
            metric_ = BaseTrainer._prepare_metric([kwargs["scoring"]])
            if not self._metric:
                self._metric = metric_  # Update the pipeline's metric
            elif metric_[0].name != self.metric[0]:
                raise ValueError(
                    "Invalid value for the scoring parameter! The scoring "
                    "should be equal to the primary metric in the pipeline. "
                    f"Expected {self.metric[0]}, got {metric_[0].name}."
                )

        kwargs = dict(
            n_jobs=kwargs.pop("n_jobs", self.n_jobs),
            verbosity=kwargs.pop("verbosity", self.verbose),
            random_state=kwargs.pop("random_state", self.random_state),
            **kwargs,
        )
        if self.goal == "class":
            self.branch.tpot = TPOTClassifier(**kwargs)
        else:
            self.branch.tpot = TPOTRegressor(**kwargs)

        self.log("Fitting automl algorithm...", 1)

        self.tpot.fit(self.X_train, self.y_train)

        self.log("\nMerging automl results with atom...", 1)

        # A pipeline could consist of just a single estimator
        if len(self.tpot.fitted_pipeline_) > 1:
            for name, est in self.tpot.fitted_pipeline_[:-1].named_steps.items():
                self._add_transformer(est)

        # Add the final estimator as a model to atom
        est = self.tpot.fitted_pipeline_[-1]
        est.acronym, est.fullname = names_from_estimator(self, est)
        model = CustomModel(self, estimator=est)
        model.estimator = model.est

        # Save metric scores on train and test set
        for metric in self._metric.values():
            model._calculate_score(metric, "train")
            model._calculate_score(metric, "test")

        self._models.update({model.name: model})
        self.log(f"Adding model {model.fullname} ({model.name}) to the pipeline...", 1)

    @composed(crash, typechecked)
    def distribution(
        self,
        distributions: Optional[Union[str, SEQUENCE_TYPES]] = None,
        columns: Optional[Union[INT, str, slice, SEQUENCE_TYPES]] = None,
    ):
        """Get statistics on column distributions.

        Compute the Kolmogorov-Smirnov test for various distributions
        against columns in the dataset. Only for numerical columns.
        Missing values are ignored.

        Parameters
        ----------
        distributions: str, sequence or None, optional (default=None)
            Names of the distributions in `scipy.stats` to get the
            statistics on. If None, a selection of the most common
            ones is used.

        columns: int, str, slice, sequence or None, optional (default=None)
            Names, indices or dtypes of the columns in the dataset to
            perform the test on. If None, select all numerical columns.

        Returns
        -------
        pd.DataFrame
            Statistic results with multiindex levels:
                - dist: Name of the distribution.
                - stat: Statistic results:
                    - score: KS-test score.
                    - p_value: Corresponding p-value.

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
            X = self[col].replace(self.missing + [np.inf, -np.inf], np.NaN).dropna()

            for dist in distributions:
                # Get KS-statistic with fitted distribution parameters
                param = getattr(stats, dist).fit(X)
                stat = stats.kstest(X, dist, args=param)

                # Add as column to the dataframe
                df.loc[(dist, "score"), col] = round(stat[0], 4)
                df.loc[(dist, "p_value"), col] = round(stat[1], 4)

        return df

    @composed(crash, typechecked)
    def export_pipeline(
        self,
        model: Optional[str] = None,
        memory: Optional[Union[bool, str, Memory]] = None,
        verbose: Optional[INT] = None,
    ):
        """Export atom's pipeline to a sklearn-like Pipeline object.

        Optionally, you can add a model as final estimator. The
        returned pipeline is already fitted on the training set.

        Parameters
        ----------
        model: str or None, optional (default=None)
            Name of the model to add as a final estimator to the
            pipeline. If the model used feature scaling, the Scaler
            is added before the model. If None, only the
            transformers are added.

        memory: bool, str, Memory or None, optional (default=None)
            Used to cache the fitted transformers of the pipeline.
                - If None or False: No caching is performed.
                - If True: A default temp directory is used.
                - If str: Path to the caching directory.
                - If Memory: Object with the joblib.Memory interface.

        verbose: int or None, optional (default=None)
            Verbosity level of the transformers in the pipeline. If
            None, it leaves them to their original verbosity. Note
            that this is not the pipeline's own verbose parameter.
            To change that, use the `set_params` method.

        Returns
        -------
        Pipeline
            Current branch as a sklearn-like Pipeline object.

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
            model = getattr(self, self._get_model_name(model)[0])

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

    @composed(crash, typechecked)
    def report(
        self,
        dataset: str = "dataset",
        n_rows: Optional[SCALAR] = None,
        filename: Optional[str] = None,
        **kwargs,
    ):
        """Create an extensive profile analysis report of the data.

        The profile report is rendered in HTML5 and CSS3. Note that
        this method can be slow for n_rows>10k.

        Parameters
        ----------
        dataset: str, optional (default="dataset")
            Data set to get the report from.

        n_rows: int or None, optional (default=None)
            Number of (randomly picked) rows to process. None to use
            all rows.

        filename: str or None, optional (default=None)
            Name to save the file with (as .html). None to not save
            anything.

        **kwargs
            Additional keyword arguments for the ProfileReport instance.

        Returns
        -------
        ProfileReport
            Created report object.

        """
        from pandas_profiling import ProfileReport

        self.log("Creating profile report...", 1)

        n_rows = getattr(self, dataset).shape[0] if n_rows is None else int(n_rows)
        profile = ProfileReport(getattr(self, dataset).sample(n_rows), **kwargs)

        if filename:
            if not filename.endswith(".html"):
                filename += ".html"
            profile.to_file(filename)
            self.log("Report successfully saved.", 1)

        return profile

    @composed(crash, method_to_log)
    def reset(self):
        """Reset the instance to it's initial state.

        Deletes all branches and models. The dataset is also reset
        to its form after initialization.

        """
        # Delete all models in the instance
        delete(self, self._get_models())

        # Recreate the master branch from original (and drop rest)
        self._current = "master"
        self._branches = CustomDict({self._current: self._get_og_branches()[0]})

        self.log(f"{self.__class__.__name__} successfully reset.", 1)

    @composed(crash, method_to_log, typechecked)
    def save_data(self, filename: str = "auto", dataset: str = "dataset"):
        """Save the data in the current branch to a csv file.

        Parameters
        ----------
        filename: str, optional (default="auto")
            Name of the file. Use "auto" for automatic naming.

        dataset: str, optional (default="dataset")
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

        Partially from: https://github.com/fastai/fastai/blob/master/
        fastai/tabular/core.py

        Parameters
        ----------
        obj2cat: bool, optional (default=True)
            Whether to convert `object` to `category`. Only if the
            number of categories would be less than 30% of the length
            of the column.

        int2uint: bool, optional (default=False)
            Whether to convert `int` to `uint` (unsigned integer). Only if
            the values in the column are strictly positive.

        dense2sparse: bool, optional (default=False)
            Whether to convert all features to sparse format. The value
            that is compressed is the most frequent value in the column.

        columns: int, str, slice, sequence or None, optional (default=None)
            Names, indices or dtypes of the columns in the dataset to
            shrink. If None, transform all columns.

        """
        check_dim(self, "shrink")
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
    def stats(self, _vb: INT = -2):
        """Print basic information about the dataset.

        Parameters
        ----------
        _vb: int, optional (default=-2)
            Internal parameter to always print if called by user.

        """
        self.log("Dataset stats " + "=" * 20 + " >>", _vb)
        self.log(f"Shape: {self.shape}", _vb)

        memory = self.dataset.memory_usage(deep=True).sum()
        if memory < 1e6:
            self.log(f"Memory: {memory / 1e3:.2f} kB", _vb)
        else:
            self.log(f"Memory: {memory / 1e6:.2f} MB", _vb)

        if not is_multidim(self.X):
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
        """Get an overview of atom's status."""
        self.log(str(self))

    @composed(crash, method_to_log, typechecked)
    def transform(
        self,
        X: X_TYPES,
        y: Optional[Y_TYPES] = None,
        verbose: Optional[INT] = None,
    ):
        """Transform new data through the branch.

        Transformers that are only applied on the training set are
        skipped.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored in the transformers.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        verbose: int or None, optional (default=None)
            Verbosity level for the transformers. If None, it uses the
            estimator's own verbosity.

        Returns
        -------
        pd.DataFrame
            Transformed feature set.

        pd.Series
            Transformed target column. Only returned if provided.

        """
        for transformer in self.pipeline:
            if not transformer._train_only:
                X, y = custom_transform(transformer, self.branch, (X, y), verbose)

        return variable_return(X, y)

    # Base transformers ============================================ >>

    def _prepare_kwargs(self, kwargs, params=None):
        """Return kwargs with atom's values if not specified."""
        for attr in BaseTransformer.attrs:
            if (not params or attr in params) and attr not in kwargs:
                kwargs[attr] = getattr(self, attr)

        return kwargs

    def _add_transformer(self, estimator, columns=None, train_only=False, **fit_params):
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
        estimator: transformer
            Estimator to add. Should implement a `transform` method.

        columns: int, str, slice, sequence or None, optional (default=None)
            Names or indices of the columns in the dataset to transform.
            If None, transform all columns.

        train_only: bool, optional (default=False)
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

        if not hasattr(estimator, "transform"):
            raise AttributeError("Added transformers should have a transform method!")

        # Add BaseTransformer params to the estimator if left to default
        if all(hasattr(estimator, attr) for attr in ("get_params", "set_params")):
            sign = signature(estimator.__init__).parameters
            for p in ("n_jobs", "random_state"):
                if p in sign and estimator.get_params()[p] == sign[p]._default:
                    estimator.set_params(**{p: getattr(self, p)})

        # Transformers remember the train_only and cols parameters
        estimator._train_only = train_only
        if columns is not None:
            inc, exc = self._get_columns(columns, return_inc_exc=True)
            estimator._cols = [
                [c for c in inc if c != self.target],  # Included cols
                [c for c in exc if c != self.target],  # Excluded cols
            ]

        if hasattr(estimator, "fit") and not check_is_fitted(estimator, False):
            if not estimator.__module__.startswith("atom"):
                self.log(f"Fitting {estimator.__class__.__name__}...", 1)

            fit_one(estimator, self.X_train, self.y_train, **fit_params)

        # Create an og branch before transforming (if it doesn't exist already)
        if self._get_og_branches() == [self.branch]:
            self._branches.insert(0, "og", Branch(self, "og", parent=self.branch))

        custom_transform(estimator, self.branch, verbose=self.verbose)

        # Add the estimator to the pipeline
        self.branch.pipeline = pd.concat(
            [self.pipeline, pd.Series([estimator], name=self._current, dtype="object")],
            ignore_index=True,
        )

    @composed(crash, method_to_log, typechecked)
    def add(
        self,
        transformer: Any,
        columns: Optional[Union[INT, str, slice, SEQUENCE_TYPES]] = None,
        train_only: bool = False,
        **fit_params,
    ):
        """Add an estimator to the current branch.

        If the estimator is not fitted, it is fitted on the complete
        training set. Afterwards, the data set is transformed and the
        estimator is added to atom's pipeline. If the estimator is
        a sklearn Pipeline, every estimator is merged independently
        with atom.

        If the estimator has a `n_jobs` and/or `random_state` parameter
        that is left to its default value, it adopts atom's value.

        Parameters
        ----------
        transformer: estimator
            Transformer to add to the pipeline. Should implement a
            `transform` method.

        columns: int, str, slice, sequence or None, optional (default=None)
            Names or indices of the columns in the dataset to transform.
            If None, transform all columns.

        train_only: bool, optional (default=False)
            Whether to apply the estimator only on the training set or
            on the complete dataset. Note that if True, the transformation
            is skipped when making predictions on unseen data.

        **fit_params
            Additional keyword arguments for the estimator's fit method.

        """
        check_dim(self, "add")
        if transformer.__class__.__name__ == "Pipeline":
            # Recursively add all transformers to the pipeline
            for name, est in transformer.named_steps.items():
                self._add_transformer(est, columns, train_only, **fit_params)

        else:
            self._add_transformer(transformer, columns, train_only, **fit_params)

    @composed(crash, method_to_log, typechecked)
    def apply(self, func: callable, columns: Union[INT, str], args=(), **kwargs):
        """Apply a function to the dataset.

        Transform one column in the dataset using a function (can
        be a lambda). If the provided column is present in the dataset,
        that same column is transformed. If it's not a column in the
        dataset, a new column with that name is created. The first
        parameter of the function is the complete dataset.

        This approach is preferred over changing the dataset directly
        through the property's `@setter` since the transformation
        is saved to atom's pipeline.

        Parameters
        ----------
        func: function
            Logic to apply to the dataset.

        columns: int or str
            Name or index of the column in the dataset to create
            or transform.

        args: tuple, optional (default=())
            Positional arguments for the function (after the dataset).

        **kwargs
            Additional keyword arguments for the function.

        """
        check_dim(self, "apply")
        if not callable(func):
            raise TypeError(
                "Invalid value for the func parameter. Argument is not callable!"
            )

        # If index, get existing name from dataset
        if isinstance(columns, int):
            columns = self._get_columns(columns)[0]

        kwargs = self._prepare_kwargs(kwargs, ["verbose", "logger"])
        self._add_transformer(FuncTransformer(func, columns, args, **kwargs))

    @composed(crash, method_to_log, typechecked)
    def drop(self, columns: Union[INT, str, slice, SEQUENCE_TYPES], **kwargs):
        """Drop columns from the dataset.

        This approach is preferred over dropping columns from the
        dataset directly through the property's `@setter` since
        the transformation is saved to atom's pipeline.

        Parameters
        ----------
        columns: int, str, slice or sequence
            Names or indices of the columns to drop.

        """
        check_dim(self, "drop")

        columns = self._get_columns(columns, include_target=False)
        kwargs = self._prepare_kwargs(kwargs, ["verbose", "logger"])
        self._add_transformer(DropTransformer(columns=columns, **kwargs))

    # Data cleaning transformers =================================== >>

    @composed(crash, method_to_log)
    def scale(self, strategy: str = "standard", **kwargs):
        """Scale the data.

        Apply one of sklearn's scalers. Categorical columns are ignored.
        The estimator created by the class is attached to atom.

        See data_cleaning.py for a description of the parameters.

        """
        check_dim(self, "scale")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Scaler().get_params())
        scaler = Scaler(strategy=strategy, **kwargs)

        self._add_transformer(scaler, columns=columns)

        # Attach the estimator attribute to atom's branch
        setattr(self.branch, strategy.lower(), getattr(scaler, strategy.lower()))

    @composed(crash, method_to_log)
    def gauss(self, strategy: str = "yeojohnson", **kwargs):
        """Transform the data to follow a Gaussian distribution.

        This transformation is useful for modeling issues related
        to heteroscedasticity (non-constant variance), or other
        situations where normality is desired. Missing values are
        disregarded in fit and maintained in transform. Categorical
        columns are ignored. The estimator created by the class is
        attached to atom.

        See data_cleaning.py for a description of the parameters.

        """
        check_dim(self, "gauss")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Gauss().get_params())
        gauss = Gauss(strategy=strategy, **kwargs)

        self._add_transformer(gauss, columns=columns)

        # Attach the estimator attribute to atom's branch
        for attr in ("yeojohnson", "boxcox", "quantile"):
            if hasattr(gauss, attr):
                setattr(self.branch, attr, getattr(gauss, attr))

    @composed(crash, method_to_log, typechecked)
    def clean(
        self,
        drop_types: Optional[Union[str, SEQUENCE_TYPES]] = None,
        strip_categorical: bool = True,
        drop_max_cardinality: bool = True,
        drop_min_cardinality: bool = True,
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
            - Drop categorical columns with maximal cardinality.
            - Drop columns with minimum cardinality.
            - Drop duplicate rows.
            - Drop rows with missing values in the target column.
            - Encode the target column (only for classification tasks).

        See data_cleaning.py for a description of the parameters.

        """
        check_dim(self, "clean")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Cleaner().get_params())
        cleaner = Cleaner(
            drop_types=drop_types,
            strip_categorical=strip_categorical,
            drop_max_cardinality=drop_max_cardinality,
            drop_min_cardinality=drop_min_cardinality,
            drop_duplicates=drop_duplicates,
            drop_missing_target=drop_missing_target,
            encode_target=encode_target if self.goal == "class" else False,
            **kwargs,
        )
        # Pass atom's missing values to the cleaner before transforming
        cleaner.missing = self.missing

        self._add_transformer(cleaner, columns=columns)

        if cleaner.mapping:
            self.mapping.insert(-1, self.target, cleaner.mapping)

    @composed(crash, method_to_log, typechecked)
    def impute(
        self,
        strat_num: Union[SCALAR, str] = "drop",
        strat_cat: str = "drop",
        max_nan_rows: Optional[SCALAR] = None,
        max_nan_cols: Optional[SCALAR] = None,
        **kwargs,
    ):
        """Handle missing values in the dataset.

        Impute or remove missing values according to the selected strategy.
        Also removes rows and columns with too many missing values. Use
        the `missing` attribute to customize what are considered "missing
        values".

        See data_cleaning.py for a description of the parameters.

        """
        check_dim(self, "impute")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Imputer().get_params())
        imputer = Imputer(
            strat_num=strat_num,
            strat_cat=strat_cat,
            max_nan_rows=max_nan_rows,
            max_nan_cols=max_nan_cols,
            **kwargs,
        )
        # Pass atom's missing values to the imputer before transforming
        imputer.missing = self.missing

        self._add_transformer(imputer, columns=columns)

    @composed(crash, method_to_log)
    def discretize(
        self,
        strategy: str = "quantile",
        bins: Union[INT, SEQUENCE_TYPES, dict] = 5,
        labels: Optional[Union[SEQUENCE_TYPES, dict]] = None,
        **kwargs,
    ):
        """Bin continuous data into intervals.

        For each feature, the bin edges are computed during fit
        and, together with the number of bins, they will define the
        intervals. Ignores numerical columns.

        See data_cleaning.py for a description of the parameters.

        """
        check_dim(self, "discretize")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Discretizer().get_params())
        discretizer = Discretizer(strategy=strategy, bins=bins, labels=labels, **kwargs)

        self._add_transformer(discretizer, columns=columns)

    @composed(crash, method_to_log, typechecked)
    def encode(
        self,
        strategy: str = "LeaveOneOut",
        max_onehot: Optional[INT] = 10,
        ordinal: Optional[Dict[Union[INT, str], SEQUENCE_TYPES]] = None,
        frac_to_other: Optional[SCALAR] = None,
        **kwargs,
    ):
        """Perform encoding of categorical features.

        The encoding type depends on the number of classes in the
        column:
            - If n_classes=2 or ordinal feature, use Ordinal-encoding.
            - If 2 < n_classes <= `max_onehot`, use OneHot-encoding.
            - If n_classes > `max_onehot`, use `strategy`-encoding.

        Missing values are propagated to the output column. Unknown
        classes encountered during transforming are converted to
        `np.NaN`. The class is also capable of replacing classes with
        low occurrences with the value `other` in order to prevent
        too high cardinality.

        See data_cleaning.py for a description of the parameters.

        """
        check_dim(self, "encode")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Encoder().get_params())
        encoder = Encoder(
            strategy=strategy,
            max_onehot=max_onehot,
            ordinal=ordinal,
            frac_to_other=frac_to_other,
            **kwargs,
        )

        self._add_transformer(encoder, columns=columns)

        # Add mapping of the encoded columns to the branch
        self.mapping.update(encoder.mapping)
        self.mapping = self.mapping[[c for c in self.columns if c in self.mapping]]

    @composed(crash, method_to_log, typechecked)
    def prune(
        self,
        strategy: Union[str, SEQUENCE_TYPES] = "zscore",
        method: Union[SCALAR, str] = "drop",
        max_sigma: SCALAR = 3,
        include_target: bool = False,
        **kwargs,
    ):
        """Prune outliers from the training set.

        Replace or remove outliers. The definition of outlier depends
        on the selected strategy and can greatly differ from one
        another. Ignores categorical columns. The estimators
        created by the class are attached to atom.

        This transformation is only applied to the training set in
        order to maintain the original distribution of samples in
        the test set.

        See data_cleaning.py for a description of the parameters.

        """
        check_dim(self, "prune")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Pruner().get_params())
        pruner = Pruner(
            strategy=strategy,
            method=method,
            max_sigma=max_sigma,
            include_target=include_target,
            **kwargs,
        )

        self._add_transformer(pruner, columns=columns, train_only=True)

        # Attach the estimator attribute to atom's branch
        for strat in lst(strategy):
            if strat.lower() != "zscore":
                setattr(self.branch, strat.lower(), getattr(pruner, strat.lower()))

    @composed(crash, method_to_log, typechecked)
    def balance(self, strategy: str = "adasyn", **kwargs):
        """Balance the number of rows per class in the target column.

        Use only for classification tasks. The estimator created by
        the class is attached to atom.

        This transformation is only applied to the training set in
        order to maintain the original distribution of target classes
        in the test set.

        See data_cleaning.py for a description of the parameters.

        """
        check_dim(self, "balance")
        if self.goal != "class":
            raise PermissionError(
                "The balance method is only available for classification tasks!"
            )

        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Balancer().get_params())
        balancer = Balancer(strategy=strategy, **kwargs)

        # Add target column mapping for cleaner printing
        balancer.mapping = self.mapping.get(self.target, {})

        self._add_transformer(balancer, columns=columns, train_only=True)

        # Attach the estimator attribute to atom's branch
        setattr(self.branch, strategy.lower(), getattr(balancer, strategy.lower()))

    # NLP transformers ============================================= >>

    @composed(crash, method_to_log, typechecked)
    def textclean(
        self,
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

        See nlp.py for a description of the parameters.

        """
        check_dim(self, "nlpclean")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, TextCleaner().get_params())
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
            **kwargs,
        )

        self._add_transformer(textcleaner, columns=columns)

        setattr(self.branch, "drops", getattr(textcleaner, "drops"))

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

        See nlp.py for a description of the parameters.

        """
        check_dim(self, "tokenize")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Tokenizer().get_params())
        tokenizer = Tokenizer(
            bigram_freq=bigram_freq,
            trigram_freq=trigram_freq,
            quadgram_freq=quadgram_freq,
            **kwargs,
        )

        self._add_transformer(tokenizer, columns=columns)

        self.branch.bigrams = tokenizer.bigrams
        self.branch.trigrams = tokenizer.trigrams
        self.branch.quadgrams = tokenizer.quadgrams

    @composed(crash, method_to_log, typechecked)
    def normalize(
        self,
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

        See nlp.py for a description of the parameters.

        """
        check_dim(self, "normalize")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Normalizer().get_params())
        normalizer = Normalizer(
            stopwords=stopwords,
            custom_stopwords=custom_stopwords,
            stem=stem,
            lemmatize=lemmatize,
            **kwargs,
        )

        self._add_transformer(normalizer, columns=columns)

    @composed(crash, method_to_log, typechecked)
    def vectorize(self, strategy: str = "bow", return_sparse: bool = True, **kwargs):
        """Vectorize the corpus.

        Transform the corpus into meaningful vectors of numbers. The
        transformation is applied on the column named `corpus`. If
        there is no column with that name, an exception is raised. The
        transformed columns are named after the word they are embedding
        (if the column is already present in the provided dataset,
        `_[strategy]` is added behind the name).

        See nlp.py for a description of the parameters.

        """
        check_dim(self, "normalize")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Vectorizer().get_params())
        vectorizer = Vectorizer(
            strategy=strategy,
            return_sparse=return_sparse,
            **kwargs,
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
        fmt: Optional[Union[str, SEQUENCE_TYPES]] = None,
        features: Union[str, SEQUENCE_TYPES] = ["day", "month", "year"],
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

        See feature_engineering.py for a description of the parameters.

        """
        check_dim(self, "feature_extraction")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, FeatureExtractor().get_params())
        feature_extractor = FeatureExtractor(
            features=features,
            fmt=fmt,
            encoding_type=encoding_type,
            drop_columns=drop_columns,
            **kwargs,
        )

        self._add_transformer(feature_extractor, columns=columns)

    @composed(crash, method_to_log, typechecked)
    def feature_generation(
        self,
        strategy: str = "dfs",
        n_features: Optional[INT] = None,
        operators: Optional[Union[str, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Apply automated feature engineering.

        Create new combinations of existing features to capture the
        non-linear relations between the original features. Attributes
        created by the class are attached to atom.

        See feature_engineering.py for a description of the parameters.

        """
        check_dim(self, "feature_generation")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, FeatureGenerator().get_params())
        feature_generator = FeatureGenerator(
            strategy=strategy,
            n_features=n_features,
            operators=operators,
            **kwargs,
        )

        self._add_transformer(feature_generator, columns=columns)

        # Attach the genetic attributes to atom's branch
        if strategy.lower() == "gfg":
            self.branch.gfg = feature_generator.gfg
            self.branch.genetic_features = feature_generator.genetic_features

    @composed(crash, method_to_log, typechecked)
    def feature_selection(
        self,
        strategy: Optional[str] = None,
        solver: Optional[Union[str, callable]] = None,
        n_features: Optional[SCALAR] = None,
        max_frac_repeated: Optional[SCALAR] = 1.0,
        max_correlation: Optional[float] = 1.0,
        **kwargs,
    ):
        """Apply feature selection techniques.

        Remove features according to the selected strategy. Ties
        between features with equal scores are broken in an unspecified
        way. Additionally, remove multicollinear and low variance
        features.

        See feature_engineering.py for a description of the parameters.

        """
        check_dim(self, "feature_selection")
        if isinstance(strategy, str):
            if strategy.lower() == "univariate" and solver is None:
                solver = "f_classif" if self.goal == "class" else "f_regression"
            elif strategy.lower() not in ("univariate", "pca"):
                if solver is None and self.winner:
                    solver = self.winner.estimator
                elif isinstance(solver, str):
                    # In case the user already filled the task...
                    if not solver.endswith("_class") and not solver.endswith("_reg"):
                        solver += f"_{self.goal}"

            # If the run method was called before, use the main metric
            if strategy.lower() not in ("univariate", "pca", "sfm", "rfe"):
                if self._metric and "scoring" not in kwargs:
                    kwargs["scoring"] = self._metric[0]

        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, FeatureSelector().get_params())
        feature_selector = FeatureSelector(
            strategy=strategy,
            solver=solver,
            n_features=n_features,
            max_frac_repeated=max_frac_repeated,
            max_correlation=max_correlation,
            **kwargs,
        )

        self._add_transformer(feature_selector, columns=columns)

        # Attach used attributes to atom's branch
        for attr in ("collinear", "feature_importance", str(strategy).lower()):
            if getattr(feature_selector, attr, None) is not None:
                setattr(self.branch, attr, getattr(feature_selector, attr))

    # Training methods ============================================= >>

    def _check(self, metric, gib, needs_proba, needs_threshold):
        """Check whether the provided metric is valid.

        Parameters
        ----------
        metric: str, sequence or callable
            Metric provided for the run.

        gib: bool or sequence
            Whether the metric is a score or a loss function.

        needs_proba: bool or sequence
            Whether the metric function requires probability estimates
            out of a classifier.

        needs_threshold: bool or sequence
            Whether the metric function takes a continuous decision
            certainty.

        Returns
        -------
        str, function, scorer or sequence
            Metric for the run. Should be the same as previous run.

        """
        if self._metric:
            # If the metric is empty, assign the existing one
            if metric is None:
                metric = self._metric
            else:
                # If there's a metric, it should be the same as previous run
                new_metric = BaseTrainer._prepare_metric(
                    metric=lst(metric),
                    greater_is_better=gib,
                    needs_proba=needs_proba,
                    needs_threshold=needs_threshold,
                )

                if list(new_metric) != list(self._metric):
                    raise ValueError(
                        "Invalid value for the metric parameter! The metric "
                        "should be the same as previous run. Expected "
                        f"{self.metric}, got {flt(list(new_metric))}."
                    )

        return metric

    def _run(self, trainer):
        """Run the trainer.

        If all models failed, catch the errors and pass them to the
        atom before raising the exception. If successful run, update
        all relevant attributes and methods.

        Parameters
        ----------
        trainer: class
            Initialized trainer to run.

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
                self._models.pop(model, None)

        # Update attributes
        self._models.update(trainer._models)
        self._metric = trainer._metric

        for model in self._models.values():
            self._errors.pop(model.name, None)  # Remove model from errors (if there)
            model.T = self  # Change the model's parent class from trainer to atom

    @composed(crash, method_to_log, typechecked)
    def run(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        greater_is_better: Union[bool, SEQUENCE_TYPES] = True,
        needs_proba: Union[bool, SEQUENCE_TYPES] = False,
        needs_threshold: Union[bool, SEQUENCE_TYPES] = False,
        n_calls: Union[INT, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[INT, SEQUENCE_TYPES] = 0,
        **kwargs,
    ):
        """Fit the models in a direct fashion.

        Fit and evaluate over the models. Contrary to SuccessiveHalving
        and TrainSizing, the direct approach only iterates once over the
        models, using the full dataset.

        See the basetrainer.py module for a description of the parameters.

        """
        metric = self._check(metric, greater_is_better, needs_proba, needs_threshold)

        params = (
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
        )

        kwargs = self._prepare_kwargs(kwargs)
        if self.goal == "class":
            trainer = DirectClassifier(*params, **kwargs)
        else:
            trainer = DirectRegressor(*params, **kwargs)

        self._run(trainer)

    @composed(crash, method_to_log, typechecked)
    def successive_halving(
        self,
        models: Union[str, callable, SEQUENCE_TYPES],
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        greater_is_better: Union[bool, SEQUENCE_TYPES] = True,
        needs_proba: Union[bool, SEQUENCE_TYPES] = False,
        needs_threshold: Union[bool, SEQUENCE_TYPES] = False,
        skip_runs: INT = 0,
        n_calls: Union[INT, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[INT, SEQUENCE_TYPES] = 0,
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

        See the basetrainer.py module for a description of the parameters.

        """
        metric = self._check(metric, greater_is_better, needs_proba, needs_threshold)

        params = (
            models, metric, greater_is_better, needs_proba, needs_threshold,
            skip_runs, n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
        )

        kwargs = self._prepare_kwargs(kwargs)
        if self.goal == "class":
            trainer = SuccessiveHalvingClassifier(*params, **kwargs)
        else:
            trainer = SuccessiveHalvingRegressor(*params, **kwargs)

        self._run(trainer)

    @composed(crash, method_to_log, typechecked)
    def train_sizing(
        self,
        models: Union[str, callable, SEQUENCE_TYPES],
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        greater_is_better: Union[bool, SEQUENCE_TYPES] = True,
        needs_proba: Union[bool, SEQUENCE_TYPES] = False,
        needs_threshold: Union[bool, SEQUENCE_TYPES] = False,
        train_sizes: Union[INT, SEQUENCE_TYPES] = 5,
        n_calls: Union[INT, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[INT, SEQUENCE_TYPES] = 0,
        **kwargs,
    ):
        """Fit the models in a train sizing fashion.

        When training models, there is usually a trade-off between
        model performance and computation time, that is regulated by
        the number of samples in the training set. This method can be
        used to create insights in this trade-off, and help determine
        the optimal size of the training set. The models are fitted
        multiple times, ever-increasing the number of samples in the
        training set.

        See the basetrainer.py module for a description of the parameters.

        """
        metric = self._check(metric, greater_is_better, needs_proba, needs_threshold)

        params = (
            models, metric, greater_is_better, needs_proba, needs_threshold,
            train_sizes, n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
        )

        kwargs = self._prepare_kwargs(kwargs)
        if self.goal == "class":
            trainer = TrainSizingClassifier(*params, **kwargs)
        else:
            trainer = TrainSizingRegressor(*params, **kwargs)

        self._run(trainer)

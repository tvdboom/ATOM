# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the ATOM class.

"""

# Standard packages
import os
import contextlib
import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy
from inspect import signature
from typeguard import typechecked
from typing import Union, Optional, Any, Dict

# Own modules
from .branch import Branch
from .basepredictor import BasePredictor
from .basetrainer import BaseTrainer
from .basetransformer import BaseTransformer
from .nlp import TextCleaner, Tokenizer, Normalizer, Vectorizer
from .pipeline import Pipeline
from .data_cleaning import (
    DropTransformer,
    FuncTransformer,
    Cleaner,
    Gauss,
    Scaler,
    Imputer,
    Encoder,
    Pruner,
    Balancer,
)
from .feature_engineering import (
    FeatureExtractor,
    FeatureGenerator,
    FeatureSelector,
)
from .training import (
    DirectClassifier,
    DirectRegressor,
    SuccessiveHalvingClassifier,
    SuccessiveHalvingRegressor,
    TrainSizingClassifier,
    TrainSizingRegressor,
)
from .models import CustomModel, MODEL_LIST
from .plots import ATOMPlotter
from .utils import (
    SCALAR, SEQUENCE_TYPES, X_TYPES, Y_TYPES, DISTRIBUTIONS, flt, lst,
    divide, infer_task, check_method, check_scaling, check_multidim,
    get_pl_name, names_from_estimator, get_columns, check_is_fitted,
    variable_return, fit_one, delete, custom_transform, method_to_log,
    composed, crash, Table, CustomDict,
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
    def __init__(self, arrays, y, shuffle, n_rows, test_size):
        self.shuffle = shuffle
        self.n_rows = n_rows
        self.test_size = test_size
        self.missing = ["", "?", "NA", "nan", "NaN", "None", "inf"]

        # Branching attributes
        self._branches = CustomDict(
            og=Branch(self, "og"),  # Original branch saves provided dataset
            master=Branch(self, "master"),  # Main branch
        )
        self._current = "master"  # Keeps track of the current branch

        # Training attributes
        self._models = CustomDict()
        self._metric = CustomDict()
        self._errors = CustomDict()

        self.log("<< ================== ATOM ================== >>", 1)

        # Prepare the provided data
        self.branch.data, self.branch.idx = self._get_data_and_idx(arrays, y=y)

        # Attach the data to the original branch
        self.og.data = self.branch.data.copy(deep=True)
        self.og.idx = self.branch.idx.copy()

        # Save the test_size fraction for use during training
        self._test_size = self.branch.idx[1] / len(self.dataset)

        self.task = infer_task(self.y, goal=self.goal)
        self.log(f"Algorithm task: {self.task}.", 1)
        if self.n_jobs > 1:
            self.log(f"Parallel processing with {self.n_jobs} cores.", 1)

        # Assign mapping
        try:  # Can fail if str and NaN in target column
            classes = sorted(self.y.unique())
        except TypeError:
            classes = self.y.unique()
        self.branch.mapping = {str(value): value for value in classes}

        self.log("", 1)  # Add empty rows around stats for cleaner look
        self.stats(1)
        self.log("", 1)

    def __repr__(self):
        out = f"{self.__class__.__name__}"
        out += f"\n --> Branches:"
        if len(self._branches) - 1 == 1:
            out += f" {self._current}"
        else:
            for branch in [b for b in self._branches.keys() if b != "og"]:
                out += f"\n   >>> {branch}{' !' if branch == self._current else ''}"
        out += f"\n --> Models: {', '.join(lst(self.models)) if self.models else None}"
        out += f"\n --> Metric: {', '.join(lst(self.metric)) if self.metric else None}"
        out += f"\n --> Errors: {len(self.errors)}"

        return out

    def __len__(self):
        return len(self.pipeline)

    def __iter__(self):
        yield from self.pipeline.values

    def __contains__(self, item):
        return item in self.dataset

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.pipeline.iloc[item]  # Get estimator from pipeline
        elif isinstance(item, str):
            if item in self._branches:
                return self._branches[item]  # Get branch
            elif item in self.dataset:
                return self.dataset[item]  # Get column from dataset
            else:
                raise ValueError(
                    f"{self.__class__.__name__}'s dataset "
                    f"has no branch nor column called {item}."
                )
        else:
            raise TypeError(
                f"{self.__class__.__name__} is only "
                "subscriptable with types int or str."
            )

    # Utility properties =========================================== >>

    @BasePredictor.branch.setter
    @typechecked
    def branch(self, name: str):
        if name in [b for b in self._branches.keys() if b != "og"]:
            self._current = self._branches[name].name
            self.log(f"Switched to branch {self._current}.", 1)
        else:
            # Branch can be created from current or another
            if "_from_" in name:
                new_branch, from_branch = name.split("_from_")
            else:
                new_branch, from_branch = name, self._current

            # Check if the new name is valid
            if not new_branch:
                raise ValueError("A branch can't have an empty name!")
            elif new_branch.lower() in map(str.lower, MODEL_LIST.keys()):
                raise ValueError(
                    f"Invalid name for the branch. {new_branch} is the "
                    f"acronym of model {MODEL_LIST[new_branch].fullname}. "
                )
            elif new_branch.lower() in ("og", "vote", "stack"):
                raise ValueError(
                    "This name is reserved for internal purposes. "
                    "Choose a different name for the branch."
                )
            elif new_branch in self._branches:
                raise ValueError(
                    f"Branch {self._branches[new_branch].name} already exists!"
                )

            # Check if the parent branch exists
            if from_branch not in self._branches:
                raise ValueError(
                    "The selected branch to split from does not exist! Use "
                    "atom.status() for an overview of the available branches."
                )

            self._branches[new_branch] = Branch(self, new_branch, parent=from_branch)
            self._current = new_branch
            self.log(f"New branch {self._current} successfully created!", 1)

    @property
    def scaled(self):
        """Whether the feature set is scaled."""
        if not check_multidim(self.X):
            est_names = [est.__class__.__name__.lower() for est in self.pipeline]
            return check_scaling(self.X) or any("scaler" in name for name in est_names)

    @property
    def duplicates(self):
        """Number of duplicate rows in the dataset."""
        if not check_multidim(self.X):
            return self.dataset.duplicated().sum()

    @property
    def nans(self):
        """Columns with the number of missing values in them."""
        if not check_multidim(self.X):
            nans = self.dataset.replace(self.missing + [np.inf, -np.inf], np.NaN)
            nans = nans.isna().sum()
            return nans[nans > 0]

    @property
    def n_nans(self):
        """Number of samples containing missing values."""
        if not check_multidim(self.X):
            nans = self.dataset.replace(self.missing + [np.inf, -np.inf], np.NaN)
            nans = nans.isna().sum(axis=1)
            return len(nans[nans > 0])

    @property
    def numerical(self):
        """Names of the numerical features in the dataset."""
        if not check_multidim(self.X):
            return list(self.X.select_dtypes(include=["number"]).columns)

    @property
    def n_numerical(self):
        """Number of numerical features in the dataset."""
        if not check_multidim(self.X):
            return len(self.numerical)

    @property
    def categorical(self):
        """Names of the categorical features in the dataset."""
        if not check_multidim(self.X):
            return list(self.X.select_dtypes(include=["object", "category"]).columns)

    @property
    def n_categorical(self):
        """Number of categorical features in the dataset."""
        if not check_multidim(self.X):
            return len(self.categorical)

    @property
    def outliers(self):
        """Columns in training set with amount of outlier values."""
        if not check_multidim(self.X):
            num_and_target = self.dataset.select_dtypes(include=["number"]).columns
            z_scores = stats.zscore(self.train[num_and_target], nan_policy="propagate")
            srs = pd.Series((np.abs(z_scores) > 3).sum(axis=0), index=num_and_target)
            return srs[srs > 0]

    @property
    def n_outliers(self):
        """Number of samples in the training set containing outliers."""
        if not check_multidim(self.X):
            num_and_target = self.dataset.select_dtypes(include=["number"]).columns
            z_scores = stats.zscore(self.train[num_and_target], nan_policy="propagate")
            return len(np.where((np.abs(z_scores) > 3).any(axis=1))[0])

    @property
    def classes(self):
        """Distribution of target classes per data set."""
        return (
            pd.DataFrame(
                {
                    "dataset": self.y.value_counts(sort=False, dropna=False),
                    "train": self.y_train.value_counts(sort=False, dropna=False),
                    "test": self.y_test.value_counts(sort=False, dropna=False),
                },
                index=self.mapping.values(),
            )
            .fillna(0)  # If 0 counts, it doesnt return the row (gets a NaN)
            .astype(int)
        )

    @property
    def n_classes(self):
        """Number of classes in the target column."""
        return len(self.y.unique())

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

        check_method(self, "automl")

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

        # Save metric scores on complete training and test set
        model.metric_train = flt(
            [
                metric(model.estimator, self.X_train, self.y_train)
                for metric in self._metric
            ]
        )
        model.metric_test = flt(
            [
                metric(model.estimator, self.X_test, self.y_test)
                for metric in self._metric
            ]
        )

        self._models.update({model.name: model})
        self.log(f"Adding model {model.fullname} ({model.name}) to the pipeline...", 1)

    @composed(crash, typechecked)
    def distribution(self, columns: Union[int, str] = 0):
        """Get statistics on a column's distribution.

        Compute the KS-statistic for various distributions against
        a column in the dataset.

        Parameters
        ----------
        columns: int or str, optional (default=0)
            Index or name of the column to get the statistics from.
            Only numerical columns are accepted.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the statistic results.

        """
        if isinstance(columns, int):
            columns = self.columns[columns]

        if columns in self.categorical:
            raise ValueError(
                "Invalid value for the columns parameter. Column should "
                f"be numerical, got categorical column {columns}."
            )

        # Drop missing values from the column before fitting
        X = self[columns].replace(self.missing + [np.inf, -np.inf], np.NaN).dropna()

        df = pd.DataFrame(columns=["ks", "p_value"])
        for dist in DISTRIBUTIONS:
            # Get KS-statistic with fitted distribution parameters
            param = getattr(stats, dist).fit(X)
            stat = stats.kstest(X, dist, args=param)

            # Add as row to the dataframe
            df.loc[dist] = {"ks": round(stat[0], 4), "p_value": round(stat[1], 4)}

        return df.sort_values("ks")

    @composed(crash, typechecked)
    def export_pipeline(
        self,
        model: Optional[str] = None,
        verbose: Optional[int] = None,
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

        verbose: int or None, optional (default=None)
            Verbosity level of the transformers in the pipeline. If
            None, it leaves them to their original verbosity.

        Returns
        -------
        pipeline: Pipeline
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
            if model.scaler:
                steps.append(("scaler", deepcopy(model.scaler)))

            # Redirect stdout to avoid annoying prints
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                steps.append((model.name, deepcopy(model.estimator)))

        return Pipeline(steps)  # ATOM's pipeline, not sklearn

    @composed(crash, method_to_log, typechecked)
    def merge(self, atom: Any, suffix: str = "2"):
        """Merge another atom instance into this one.

        Branches, models, metrics and attributes of the other atom
        instance are merged into this one. If there are branches
        and/or models with the same name, they are merged adding
        the `suffix` parameter to their name. The errors and missing
        attributes are extended with those of the other instance. It's
        only possible to merge two instances if they are initialized
        using the same dataset and trained using the same metric.

        Parameters
        ----------
        atom: ATOMClassifier or ATOMRegressor
            Other atom instance with which to merge.

        suffix: str, optional (default="2")
            Conflicting branches and models are merged adding `suffix`
            to the end of their names.

        """
        if self.goal == "class":
            if not getattr(atom, "goal", "") == "class":
                raise TypeError(
                    "Invalid type for the atom parameter. The provided object should "
                    f"be an ATOMClassifier instance, got {atom.__class__.__name__}."
                )
        elif self.goal == "reg":
            if not getattr(atom, "goal", "") == "reg":
                raise TypeError(
                    "Invalid type for the atom parameter. The provided object should "
                    f"be an ATOMRegressor instance, got {atom.__class__.__name__}."
                )

        # Check that both instances have the same original dataset
        if self["og"].data.shape != atom["og"].data.shape:
            raise ValueError(
                "Invalid value for the atom parameter. The provided atom instance "
                f"was initialized with a different dataset (shape={atom['og'].shape}) "
                f"than this one (shape={self['og'].shape})."
            )

        # Check that both instances have the same metric
        if not self._metric:
            self._metric = atom._metric
        elif atom.metric and self.metric != atom.metric:
            raise ValueError(
                "Invalid value for the atom parameter. The provided atom "
                f"instance uses a different metric ({atom.metric}) than "
                f"this one ({self.metric})."
            )

        self.log("Merging atom...", 1)
        for name, branch in atom._branches.items():
            if name != "og":  # Original dataset is the same
                self.log(f" --> Merging branch {name}.", 1)
                if name in self._branches:
                    name = f"{name}{suffix}"
                branch.name = name
                self._branches[name] = branch

        for name, model in atom._models.items():
            self.log(f" --> Merging model {name}.", 1)
            if name in self._models:
                name = f"{name}{suffix}"
            model.name = name
            self._models[name] = model

        self.log(" --> Merging attributes.", 1)
        self.missing.extend([x for x in atom.missing if x not in self.missing])
        for name, error in atom._errors.items():
            if name in self._errors:
                name = f"{name}{suffix}"
            self._errors[name] = error

    @composed(crash, typechecked)
    def report(
        self,
        dataset: str = "dataset",
        n_rows: Optional[SCALAR] = None,
        filename: Optional[str] = None,
    ):
        """Create an extensive profile analysis report of the data.

        The profile report is rendered in HTML5 and CSS3. Note that
        this method can be slow for n_rows>10k.

        Parameters
        ----------
        dataset: str, optional (default="dataset")
            Data set to get the report from.

        n_rows: int or None, optional (default=None)
            Number of (randomly picked) rows in to process. None for
            all rows.

        filename: str or None, optional (default=None)
            Name to save the file with (as .html). None to not save
            anything.

        Returns
        -------
        profile: ProfileReport
            Created report object.

        """
        from pandas_profiling import ProfileReport

        self.log("Creating profile report...", 1)

        n_rows = getattr(self, dataset).shape[0] if n_rows is None else int(n_rows)
        profile = ProfileReport(getattr(self, dataset).sample(n_rows))

        if filename:
            if not filename.endswith(".html"):
                filename = filename + ".html"
            profile.to_file(filename)
            self.log("Report saved successfully!", 1)

        return profile

    @composed(crash, method_to_log)
    def reset(self):
        """Reset the instance to it's initial state.

        Deletes all branches and models. The dataset is also reset
        to its form after initialization.

        """
        # Delete all models and branches
        delete(self, self._get_models())
        for name in [b for b in self._branches.keys() if b != "og"]:
            self._branches.pop(name)

        # Re-create the master branch from original
        self._branches["master"] = Branch(self, "master", parent="og")
        self._current = "master"

        self.log("The instance is successfully reset!", 1)

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
        self.log("Data set saved successfully!", 1)

    @composed(crash, method_to_log, typechecked)
    def shrink(
        self,
        columns: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        obj2cat: bool = True,
        int2uint: bool = False
    ):
        """Converts the dataset's columns to the smallest possible dtype.

        Use this method for memory optimization. Note that applying
        transformers to the data may alter the dtypes again.
        From: https://github.com/fastai/fastai/blob/master/fastai/tabular/core.py

        Parameters
        ----------
        columns: int, str, slice, sequence or None, optional (default=None)
            Names, indices or dtypes of the columns in the dataset to
            shrink. If None, transform all columns.

        obj2cat: bool, optional (default=True)
            Whether to convert `object` to `category`. Only if the
            number of categories would be less than 30% of the length
            of the column.

        int2uint: bool, optional (default=False)
            Whether to convert `int` to `uint`. Only if the values are
            strictly positive.

        """
        check_method(self, "shrink")
        columns = get_columns(self.dataset, columns)
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
        for c, old_t in {k: v for k, v in self.dtypes.items() if k in columns}.items():
            if old_t.name in exclude_types:
                continue

            t = next((v for k, v in type_map.items() if old_t.name.startswith(k)))
            if isinstance(t, list):
                # Use uint if values are strictly positive
                if int2uint and t == type_map["int"] and self[c].min() >= 0:
                    t = type_map["uint"]

                # Find the smallest type that fits
                new_t = next(
                    (r[0] for r in t if r[1] <= self[c].min() and r[2] >= self[c].max())
                )
                if new_t and new_t == old_t:
                    new_t = None  # Keep as is
            else:
                # Convert to category if number of categories less than 30% of column
                new_t = t if self[c].nunique() <= int(len(self[c]) * 0.3) else "object"

            if new_t:
                new_dtypes[c] = new_t

        self.branch.data = self.branch.data.astype(new_dtypes)
        self.log("The column dtypes are converted successfully!", 1)

    @composed(crash, method_to_log)
    def stats(self, _vb: int = -2):
        """Print basic information about the dataset.

        Parameters
        ----------
        _vb: int, optional (default=-2)
            Internal parameter to always print if called by user.

        """
        self.log("Dataset stats " + "=" * 20 + " >>", _vb)
        self.log(f"Shape: {self.shape}", _vb)

        if not check_multidim(self.X):
            nans = self.nans.sum()
            n_categorical = self.n_categorical
            outliers = self.outliers.sum()
            duplicates = self.dataset.duplicated().sum()

            self.log(f"Scaled: {self.scaled}", _vb)
            if self.nans.sum():
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
        self.log("-" * 37, _vb)

        # Print count and balance of classes
        if self.task != "regression":
            cls = self.classes
            spaces = (2, *[len(str(max(cls["dataset"]))) + 8] * 3)
            func = lambda i, col: f"{i} ({divide(i, min(cls[col])):.1f})"

            table = Table([("", "left"), *cls.columns], spaces)
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
    def transform(self, X: X_TYPES, y: Y_TYPES = None, verbose: Optional[int] = None):
        """Transform new data through the branch.

        Transformers that are only applied on the training set are
        skipped.

        Parameters
        ----------
        X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored in the transformers.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

            Feature set with shape=(n_samples, n_features).

        verbose: int or None, optional (default=None)
            Verbosity level for the transformers. If None, it uses the
            estimator's own verbosity.

        Returns
        -------
        X: pd.DataFrame
            Transformed feature set.

        y: pd.Series
            Transformed target column. Only returned if provided.

        """
        for est in [est for est in self.pipeline if not est._train_only]:
            X, y = custom_transform(self, est, self.branch, (X, y), verbose)

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
            Additional keyword arguments passed to the transformer's fit
            method.

        """
        if not hasattr(estimator, "transform"):
            raise ValueError("Added transformers should have a transform method!")

        # Add BaseTransformer params to the estimator if left to default
        if all(hasattr(estimator, attr) for attr in ("get_params", "set_params")):
            sign = signature(estimator.__init__).parameters
            for p in ("n_jobs", "random_state"):
                if p in sign and estimator.get_params()[p] == sign[p]._default:
                    estimator.set_params(**{p: getattr(self, p)})

        # Transformers remember the train_only and columns parameters
        estimator._train_only = train_only
        estimator._cols = [
            col for col in get_columns(self.dataset, columns) if col != self.target
        ]

        if hasattr(estimator, "fit") and not check_is_fitted(estimator, False):
            if not estimator.__module__.startswith("atom"):
                self.log(f"Fitting {estimator.__class__.__name__}...", 1)

            fit_one(estimator, self.X_train, self.y_train, **fit_params)

        custom_transform(self, estimator, self.branch, verbose=self.verbose)

        # Add the estimator to the pipeline
        self.branch.pipeline = self.pipeline.append(
            pd.Series([estimator], name=self._current), ignore_index=True
        )

    @composed(crash, method_to_log, typechecked)
    def add(
        self,
        transformer: Any,
        columns: Optional[Union[int, str, slice, SEQUENCE_TYPES]] = None,
        train_only: bool = False,
        **fit_params,
    ):
        """Add a estimator to the current branch.

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
            Additional keyword arguments passed to the estimator's fit
            method.

        """
        check_method(self, "add")
        if transformer.__class__.__name__ == "Pipeline":
            # Recursively add all transformers to the pipeline
            for name, est in transformer.named_steps.items():
                self._add_transformer(est, columns, train_only, **fit_params)

        else:
            self._add_transformer(transformer, columns, train_only, **fit_params)

    @composed(crash, method_to_log, typechecked)
    def apply(self, func: callable, columns: Union[int, str], args=(), **kwargs):
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
            Function to apply to the dataset.

        columns: int or str
            Name or index of the column in the dataset to create
            or transform.

        args: tuple, optional (default=())
            Positional arguments passed to func after the dataset.

        **kwargs
            Additional keyword arguments passed to func.

        """
        check_method(self, "apply")
        if not callable(func):
            raise TypeError(
                "Invalid value for the func parameter. Argument is not callable!"
            )

        kwargs = self._prepare_kwargs(kwargs, ["verbose", "logger"])
        self._add_transformer(FuncTransformer(func, columns, args, **kwargs))

    @composed(crash, method_to_log, typechecked)
    def drop(self, columns: Union[int, str, slice, SEQUENCE_TYPES], **kwargs):
        """Drop columns from the dataset.

        This approach is preferred over dropping columns from the
        dataset directly through the property's `@setter` since
        the transformation is saved to atom's pipeline.

        Parameters
        ----------
        columns: int, str, slice or sequence
            Names or indices of the columns to drop.

        """
        check_method(self, "drop")
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
        check_method(self, "scale")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Scaler().get_params())
        scaler = Scaler(strategy=strategy, **kwargs)

        self._add_transformer(scaler, columns=columns)

        # Attach the estimator attribute to atom's branch
        setattr(self.branch, strategy.lower(), getattr(scaler, strategy.lower()))

    @composed(crash, method_to_log)
    def gauss(self, strategy: str = "yeo-johnson", **kwargs):
        """Transform the data to follow a Gaussian distribution.

        This transformation is useful for modeling issues related
        to heteroscedasticity (non-constant variance), or other
        situations where normality is desired. Missing values are
        disregarded in fit and maintained in transform. Categorical
        columns are ignored. The estimator created by the class is
        attached to atom.

        See data_cleaning.py for a description of the parameters.

        """
        check_method(self, "gauss")
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
        check_method(self, "clean")
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

        # Assign mapping (if it changed)
        if cleaner.mapping:
            self.branch.mapping = cleaner.mapping

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
        check_method(self, "impute")
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

    @composed(crash, method_to_log, typechecked)
    def encode(
        self,
        strategy: str = "LeaveOneOut",
        max_onehot: Optional[int] = 10,
        ordinal: Optional[Dict[Union[int, str], SEQUENCE_TYPES]] = None,
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
        check_method(self, "encode")
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

    @composed(crash, method_to_log, typechecked)
    def prune(
        self,
        strategy: Union[str, SEQUENCE_TYPES] = "z-score",
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
        check_method(self, "prune")
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
            if strat.lower() != "z-score":
                setattr(self.branch, strat.lower(), getattr(pruner, strat.lower()))

    @composed(crash, method_to_log, typechecked)
    def balance(self, strategy: str = "ADASYN", **kwargs):
        """Balance the number of rows per class in the target column.

        Use only for classification tasks. The estimator created by
        the class is attached to atom.

        This transformation is only applied to the training set in
        order to maintain the original distribution of target classes
        in the test set.

        See data_cleaning.py for a description of the parameters.

        """
        check_method(self, "balance")
        if self.goal != "class":
            raise PermissionError(
                "The balance method is only available for classification tasks!"
            )

        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, Balancer().get_params())
        balancer = Balancer(strategy=strategy, **kwargs)

        # Add mapping from atom to balancer for cleaner printing
        balancer.mapping = self.mapping

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
        transformations are applied on the column named `Corpus`, in
        the same order the parameters are presented. If there is no
        column with that name, an exception is raised.

        See nlp.py for a description of the parameters.

        """
        check_method(self, "nlpclean")
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

        self._add_transformer(textcleaner)

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
        transformations are applied on the column named `Corpus`. If
        there is no column with that name, an exception is raised.

        See nlp.py for a description of the parameters.

        """
        check_method(self, "tokenize")
        kwargs = self._prepare_kwargs(kwargs, Tokenizer().get_params())
        tokenizer = Tokenizer(
            bigram_freq=bigram_freq,
            trigram_freq=trigram_freq,
            quadgram_freq=quadgram_freq,
            **kwargs,
        )

        self._add_transformer(tokenizer)

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
        are applied on the column named `Corpus`, in the same order the
        parameters are presented. If there is no column with that name,
        an exception is raised.

        See nlp.py for a description of the parameters.

        """
        check_method(self, "normalize")
        kwargs = self._prepare_kwargs(kwargs, Normalizer().get_params())
        normalizer = Normalizer(
            stopwords=stopwords,
            custom_stopwords=custom_stopwords,
            stem=stem,
            lemmatize=lemmatize,
            **kwargs,
        )

        self._add_transformer(normalizer)

    @composed(crash, method_to_log, typechecked)
    def vectorize(self, strategy: str = "BOW", **kwargs):
        """Vectorize the corpus.

        Transform the corpus into meaningful vectors of numbers. The
        transformation is applied on the column named `Corpus`. If
        there is no column with that name, an exception is raised.

        See nlp.py for a description of the parameters.

        """
        check_method(self, "normalize")
        kwargs = self._prepare_kwargs(kwargs, Vectorizer().get_params())
        vectorizer = Vectorizer(strategy=strategy, **kwargs)

        self._add_transformer(vectorizer)

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
        check_method(self, "feature_extraction")
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
        strategy: str = "DFS",
        n_features: Optional[int] = None,
        generations: int = 20,
        population: int = 500,
        operators: Optional[Union[str, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Apply automated feature engineering.

        Use Deep feature Synthesis or a genetic algorithm to create
        new combinations of existing features to capture the non-linear
        relations between the original features. Attributes created by
        the class are attached to atom.

        See feature_engineering.py for a description of the parameters.

        """
        check_method(self, "feature_generation")
        columns = kwargs.pop("columns", None)
        kwargs = self._prepare_kwargs(kwargs, FeatureGenerator().get_params())
        feature_generator = FeatureGenerator(
            strategy=strategy,
            n_features=n_features,
            generations=generations,
            population=population,
            operators=operators,
            **kwargs,
        )

        self._add_transformer(feature_generator, columns=columns)

        # Attach the genetic attributes to atom's branch
        if strategy.lower() in ("gfg", "genetic"):
            self.branch.symbolic_transformer = feature_generator.symbolic_transformer
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
        between features with equal scores are broken in an
        unspecified way. Additionally, removes features with too low
        variance and finds pairs of collinear features based on the
        Pearson correlation coefficient. For each pair above the
        specified limit (in terms of absolute value), it removes one
        of the two. Plotting methods and attributes created by the
        class are attached to atom.

        See feature_engineering.py for a description of the parameters.

        """
        check_method(self, "feature_selection")
        if isinstance(strategy, str):
            if strategy.lower() == "univariate" and solver is None:
                solver = "f_classif" if self.goal == "class" else "f_regression"
            elif strategy.lower() in ("sfm", "rfe", "rfecv", "sfs"):
                if solver is None and self.winner:
                    solver = self.winner.estimator
                elif isinstance(solver, str):
                    # In case the user already filled the task...
                    if not solver.endswith("_class") and not solver.endswith("_reg"):
                        solver += f"_{self.goal}"

            # If the run method was called before, use the main metric
            if strategy.lower() in ("rfecv", "sfs"):
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
        metric: str, function, scorer or sequence
            Metric for the run. Should be the same as previous run.

        """
        if self._metric:
            # If the metric is empty, assign the existing one
            if metric is None:
                metric = self._metric
            else:
                # If there's a metric, it should be the same as previous run
                _metric = BaseTrainer._prepare_metric(
                    metric=lst(metric),
                    greater_is_better=gib,
                    needs_proba=needs_proba,
                    needs_threshold=needs_threshold,
                )

                if list(_metric.keys()) != list(self._metric.keys()):
                    raise ValueError(
                        "Invalid value for the metric parameter! The metric "
                        "should be the same as previous run. Expected "
                        f"{self.metric}, got {flt([m.name for m in _metric])}."
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
            Trainer instance to run.

        """
        try:
            trainer._tracking_params = self._tracking_params
            trainer._branches = {self._current: self.branch}
            trainer._current = self._current
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

        for model in self._models:
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
        n_calls: Union[int, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[int, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[int, SEQUENCE_TYPES] = 0,
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
        skip_runs: int = 0,
        n_calls: Union[int, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[int, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[int, SEQUENCE_TYPES] = 0,
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
        train_sizes: Union[int, SEQUENCE_TYPES] = 5,
        n_calls: Union[int, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[int, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[int, SEQUENCE_TYPES] = 0,
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

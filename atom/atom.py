# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the main ATOM class.

"""

# Standard packages
import numpy as np
import pandas as pd
from inspect import signature
from typeguard import typechecked
from typing import Union, Optional, Sequence
from pandas_profiling import ProfileReport

# Own modules
from .basepredictor import BasePredictor
from .basetransformer import BaseTransformer
from .basetrainer import BaseTrainer
from .data_cleaning import (
    Cleaner,
    Scaler,
    Imputer,
    Encoder,
    Outliers,
    Balancer,
)
from .feature_engineering import FeatureGenerator, FeatureSelector
from .training import (
    TrainerClassifier,
    TrainerRegressor,
    SuccessiveHalvingClassifier,
    SuccessiveHalvingRegressor,
    TrainSizingClassifier,
    TrainSizingRegressor,
)
from .plots import ATOMPlotter
from .utils import (
    CAL, X_TYPES, Y_TYPES, TRAIN_TYPES, flt, lst, merge, check_property, infer_task,
    check_scaling, transform, clear, method_to_log, composed, crash,
)


class ATOM(BasePredictor, ATOMPlotter):
    """ATOM base class.

    The ATOM class is a convenient wrapper for all data_cleaning, feature_engineering
    and training methods in this package. Provide the dataset to the class, and apply
    all transformations and model management from here.

    """

    @composed(crash, method_to_log)
    def __init__(self, arrays, n_rows, test_size):
        """Prepare input, run Cleaner and split data in train/test sets."""
        self.n_rows = n_rows
        self.test_size = test_size
        self.pipeline = pd.Series([], name="pipeline", dtype="object")

        # Training attributes
        self.trainer = None
        self.models = []
        self.metric_ = []
        self.errors = {}
        self._results = pd.DataFrame(
            columns=[
                "metric_bo",
                "time_bo",
                "metric_train",
                "metric_test",
                "time_fit",
                "mean_bagging",
                "std_bagging",
                "time_bagging",
                "time",
            ]
        )

        self.log("<< ================== ATOM ================== >>", 1)

        # Prepare the provided data
        self._data, self._idx = self._get_data_and_idx(arrays)

        # Save the test_size fraction for later use
        self._test_size = self._idx[1] / len(self.dataset)

        # Assign the algorithm's task
        self.task = infer_task(self.y, goal=self.goal)
        self.log(f"Algorithm task: {self.task}.", 1)
        if self.task.startswith('multi'):
            self.log(f" --> Number of classes: {self.n_classes}.", 2)
        if self.n_jobs > 1:
            self.log(f"Parallel processing with {self.n_jobs} cores.", 1)

        # Assign mapping
        try:  # Can fail if str and NaN in target column
            classes = sorted(self.y.unique())
        except TypeError:
            classes = self.y.unique()
        self.mapping = {str(value): value for value in classes}

        self.stats(1)  # Print data stats

    def __repr__(self):
        repr_ = f"{self.__class__.__name__}"
        for est in self.pipeline:
            repr_ += f"\n --> {est.__class__.__name__}"
            for param in signature(est.__init__).parameters:
                if param not in BaseTransformer.attrs + ["self"]:
                    repr_ += f"\n   >>> {param}: {str(flt(getattr(est, param)))}"

        return repr_

    # Utility properties ==================================================== >>

    @property
    def missing(self):
        """Returns columns with missing + inf values."""
        missing = self.dataset.replace([-np.inf, np.inf], np.NaN).isna().sum()
        return missing[missing > 0]

    @property
    def n_missing(self):
        """Returns the total number of missing values in the dataset."""
        return self.missing.sum()

    @property
    def categorical(self):
        """Returns the names of categorical columns in the dataset."""
        return list(self.X.select_dtypes(include=["category", "object"]).columns)

    @property
    def n_categorical(self):
        """Returns the number of categorical columns in the dataset."""
        return len(self.categorical)

    @property
    def scaled(self):
        """Returns whether the dataset is scaled."""
        return check_scaling(self.dataset)

    # Utility methods ======================================================= >>

    @composed(crash, method_to_log)
    def stats(self, _vb: int = -2):
        """Print some information about the dataset.

        Parameters
        ----------
        _vb: int, optional (default=-2)
            Internal parameter to always print if the user calls this method.

        """
        self.log("\nDataset stats ================== >>", _vb)
        self.log(f"Shape: {self.dataset.shape}", _vb)

        if self.n_missing:
            self.log(f"Missing values: {self.n_missing}", _vb)
        if self.n_categorical:
            self.log(f"Categorical columns: {self.n_categorical}", _vb)

        self.log(f"Scaled: {self.scaled}", _vb)
        self.log("-----------------------------------", _vb)
        self.log(f"Train set size: {len(self.train)}", _vb)
        self.log(f"Test set size: {len(self.test)}", _vb)

        # Print count and balance of classes for classification tasks
        if self.task != "regression":
            balance = {}
            keys = ":".join(self.mapping.keys())
            for set_ in ("train", "test"):
                balance[set_] = ":".join(
                    round(self.classes[set_] / min(self.classes[set_]), 1).astype(str)
                )
            self.log("-----------------------------------", _vb + 1)
            if balance["train"] == balance["test"]:
                self.log(f"Dataset balance: {keys} <==> {balance['train']}", _vb + 1)
            else:
                self.log(f"Train set balance: {keys} <==> {balance['train']}", _vb + 1)
                self.log(f"Test set balance: {keys} <==> {balance['test']}", _vb + 1)
            self.log("-----------------------------------", _vb + 1)
            self.log(f"Distribution of classes:", _vb + 1)
            self.log(self.classes.to_markdown(), _vb + 1)

    @composed(crash, method_to_log, typechecked)
    def report(
        self,
        dataset: str = "dataset",
        n_rows: Optional[Union[int, float]] = None,  # float for 1e3...
        filename: Optional[str] = None,
    ):
        """Create an extensive profile analysis of the data.

        The profile report is rendered in HTML5 and CSS3. Note that this
        method can be slow for rows>10k.

        Parameters
        ----------
        dataset: str, optional(default="dataset")
            Name of the data set to get the report from.

        n_rows: int or None, optional(default=None)
            Number of (randomly picked) rows to process. None for all rows.

        filename: str or None, optional (default=None)
            Name of the file when saved (as .html). None to not save anything.

        """
        # If rows=None, select all rows in the dataframe
        n_rows = getattr(self, dataset).shape[0] if n_rows is None else int(n_rows)

        self.log("Creating profile report...", 1)

        profile = ProfileReport(getattr(self, dataset).sample(n_rows))
        try:  # Render if possible (for notebooks)
            from IPython.display import display

            display(profile)
        except ModuleNotFoundError:
            pass

        if filename:
            if not filename.endswith(".html"):
                filename = filename + ".html"
            profile.to_file(filename)
            self.log("Report saved successfully!", 1)

    @composed(crash, method_to_log, typechecked)
    def transform(
        self, X: X_TYPES, y: Y_TYPES = None, verbose: Optional[int] = None, **kwargs
    ):
        """Transform new data through all the pre-processing steps in the pipeline.

        By default, all transformers are included except outliers and balance since
        they should only be applied on the training set.

        When using the pipeline parameter to include/exclude transformers, remember
        that the first transformer (index 0) in `atom`"s pipeline is always the
        Cleaner called during initialization.

        Parameters
        ----------
        X: dict, list, tuple,  np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, list, tuple,  np.array, pd.Series or None, optional (default=None)
            - If None: y is ignored in the transformers.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        verbose: int or None, optional (default=None)
            Verbosity level of the transformers. If None, it uses the `training`"s verbosity.

        **kwargs
            Additional keyword arguments to customize which transformers to apply.
            You can either select them including their index in the `pipeline`
            parameter, e.g. `pipeline=[0, 1, 4]` or include/exclude them individually
            using their methods, e.g. `impute=True` or `feature_selection=False`.

        Returns
        -------
        X: pd.DataFrame
            Transformed dataset.

        y: pd.Series
            Transformed target column. Only returned if provided.

        """
        if verbose is None:
            verbose = self.verbose
        return transform(self.pipeline, X, y, verbose, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def save_data(self, filename: str = None, dataset: str = "dataset"):
        """Save data to a csv file.

        Parameters
        ----------
        filename: str or None, optional (default=None)
            Name of the saved file. None to use default name.

        dataset: str, optional (default="dataset")
            Data set to save.

        """
        if not filename:
            filename = f"{self.__class__.__name__}_{dataset}.csv"
        getattr(self, dataset).to_csv(filename, index=False)

    # Data cleaning methods ================================================= >>

    def _prepare_kwargs(self, kwargs, params=None):
        """Return kwargs with ATOM values if not specified."""
        for attr in ["n_jobs", "verbose", "warnings", "logger", "random_state"]:
            if (not params or attr in params) and attr not in kwargs:
                kwargs[attr] = getattr(self, attr)

        return kwargs

    @composed(crash, method_to_log)
    def scale(self, **kwargs):
        """Scale the features.

        Scale the features in the dataset to mean=0 and std=1. The scaler is
        fitted only on the training set to avoid data leakage.

        """
        kwargs = self._prepare_kwargs(kwargs, Scaler().get_params())
        scaler = Scaler(**kwargs).fit(self.X_train)

        self.X = scaler.transform(self.X)
        self.pipeline = self.pipeline.append(pd.Series([scaler]), ignore_index=True)

    @composed(crash, method_to_log, typechecked)
    def clean(
        self,
        prohibited_types: Optional[Union[str, Sequence[str]]] = None,
        strip_categorical: bool = True,
        maximum_cardinality: bool = True,
        minimum_cardinality: bool = True,
        missing_target: bool = True,
        map_target: Optional[bool] = None,
        **kwargs,
    ):
        """Applies standard data cleaning steps on the dataset.

         These steps can include:
            - Strip categorical features from white spaces.
            - Removing columns with prohibited data types.
            - Removing categorical columns with maximal cardinality.
            - Removing columns with minimum cardinality.
            - Removing rows with missing values in the target column.
            - Encode the target column.

        See the data_cleaning.py module for a description of the parameters.

        """
        kwargs = self._prepare_kwargs(kwargs, Imputer().get_params())
        cleaner = Cleaner(
            prohibited_types=prohibited_types,
            strip_categorical=strip_categorical,
            maximum_cardinality=maximum_cardinality,
            minimum_cardinality=minimum_cardinality,
            missing_target=missing_target,
            map_target=map_target,
            **kwargs,
        )
        X, y = cleaner.transform(self.X, self.y)
        self.dataset = merge(X, y).reset_index(drop=True)
        self.pipeline = self.pipeline.append(pd.Series([cleaner]), ignore_index=True)

        # Assign mapping (if it changed)
        if cleaner.mapping:
            self.mapping = cleaner.mapping

        # Since Cleaner can remove from train and test set, reset indices
        self._idx[1] = int(self._test_size * len(self.dataset))
        self._idx[0] = len(self.dataset) - self._idx[1]

    @composed(crash, method_to_log, typechecked)
    def impute(
        self,
        strat_num: Union[int, float, str] = "drop",
        strat_cat: str = "drop",
        min_frac_rows: float = 0.5,
        min_frac_cols: float = 0.5,
        missing: Optional[Union[int, float, str, list]] = None,
        **kwargs,
    ):
        """Handle missing values in the dataset.

        Impute or remove missing values according to the selected strategy.
        Also removes rows and columns with too many missing values. The imputer
        is fitted only on the training set to avoid data leakage.

        See the data_cleaning.py module for a description of the parameters.

        """
        kwargs = self._prepare_kwargs(kwargs, Imputer().get_params())
        imputer = Imputer(
            strat_num=strat_num,
            strat_cat=strat_cat,
            min_frac_rows=min_frac_rows,
            min_frac_cols=min_frac_cols,
            missing=missing,
            **kwargs,
        ).fit(self.X_train, self.y_train)

        X, y = imputer.transform(self.X, self.y)
        self.dataset = merge(X, y).reset_index(drop=True)
        self.pipeline = self.pipeline.append(pd.Series([imputer]), ignore_index=True)

        # Since Imputer can remove from train and test set, reset indices
        self._idx[1] = int(self._test_size * len(self.dataset))
        self._idx[0] = len(self.dataset) - self._idx[1]

    @composed(crash, method_to_log, typechecked)
    def encode(
        self,
        strategy: str = "LeaveOneOut",
        max_onehot: Optional[int] = 10,
        frac_to_other: Optional[float] = None,
        **kwargs,
    ):
        """Perform encoding of categorical features.

        The encoding type depends on the number of unique values in the column:
            - If n_unique=2, use Label-encoding.
            - If 2 < n_unique <= max_onehot, use OneHot-encoding.
            - If n_unique > max_onehot, use "strategy"-encoding.

        Also replaces classes with low occurrences with the value "other" in
        order to prevent too high cardinality. Categorical features are defined as
        all columns whose dtype.kind not in "ifu". Will raise an error if it
        encounters missing values or unknown classes when transforming. The
        encoder is fitted only on the training set to avoid data leakage.

        See the data_cleaning.py module for a description of the parameters.

        """
        kwargs = self._prepare_kwargs(kwargs, Encoder().get_params())
        encoder = Encoder(
            strategy=strategy,
            max_onehot=max_onehot,
            frac_to_other=frac_to_other,
            **kwargs,
        ).fit(self.X_train, self.y_train)

        self.X = encoder.transform(self.X)
        self.pipeline = self.pipeline.append(pd.Series([encoder]), ignore_index=True)

    @composed(crash, method_to_log, typechecked)
    def outliers(
        self,
        strategy: Union[int, float, str] = "drop",
        max_sigma: Union[int, float] = 3,
        include_target: bool = False,
        **kwargs,
    ):
        """Remove or replace outliers in the training set.

        Outliers are defined as values that lie further than
        `max_sigma` * standard_deviation away from the mean of the column. Only
        outliers from the training set are removed to maintain an original
        sample of target values in the test set. Ignores categorical columns.

        See the data_cleaning.py module for a description of the parameters.

        """
        kwargs = self._prepare_kwargs(kwargs, Outliers().get_params())
        outliers = Outliers(
            strategy=strategy,
            max_sigma=max_sigma,
            include_target=include_target,
            **kwargs,
        )

        X_train, y_train = outliers.transform(self.X_train, self.y_train)
        self.train = merge(X_train, y_train)
        self.dataset.reset_index(drop=True, inplace=True)
        self.pipeline = self.pipeline.append(pd.Series([outliers]), ignore_index=True)

    @composed(crash, method_to_log, typechecked)
    def balance(self, strategy: str = "ADASYN", **kwargs):
        """Balance the target classes in the training set.

        Balance the number of instances per target category in the training set.
        Only the training set is balanced in order to maintain the original
        distribution of target classes in the test set. Use only for
        classification tasks.

        See the data_cleaning.py module for a description of the parameters.

        """
        if not self.goal.startswith("class"):
            raise PermissionError(
                "The balance method is only available for classification tasks!"
            )

        kwargs = self._prepare_kwargs(kwargs, Balancer().get_params())
        balancer = Balancer(strategy=strategy, **kwargs)

        # Add mapping from ATOM to balancer for cleaner printing
        balancer.mapping = self.mapping

        X_train, y_train = balancer.transform(self.X_train, self.y_train)

        # Attach the estimator attribute to ATOM
        setattr(self, strategy.lower(), getattr(balancer, strategy.lower()))

        self.train = merge(X_train, y_train)
        self.dataset.reset_index(drop=True, inplace=True)
        self.pipeline = self.pipeline.append(pd.Series([balancer]), ignore_index=True)

    # Feature engineering methods =========================================== >>

    @composed(crash, method_to_log, typechecked)
    def feature_generation(
        self,
        strategy: str = "DFS",
        n_features: Optional[int] = None,
        generations: int = 20,
        population: int = 500,
        operators: Optional[Union[str, Sequence[str]]] = None,
        **kwargs,
    ):
        """Create new non-linear features.

        Use Deep feature Synthesis or a genetic algorithm to create new combinations
        of existing features to capture the non-linear relations between the original
        features.

        See the feature_engineering.py module for a description of the parameters.

        """
        kwargs = self._prepare_kwargs(kwargs, FeatureGenerator().get_params())
        feature_generator = FeatureGenerator(
            strategy=strategy,
            n_features=n_features,
            generations=generations,
            population=population,
            operators=operators,
            **kwargs,
        ).fit(self.X_train, self.y_train)

        self.X = feature_generator.transform(self.X)
        self.pipeline = self.pipeline.append(
            pd.Series([feature_generator]), ignore_index=True
        )

        # Attach used attributes to ATOM
        if strategy.lower() in ("gfg", "genetic"):
            for attr in ["symbolic_transformer", "genetic_features"]:
                setattr(self, attr, getattr(feature_generator, attr))

    @composed(crash, method_to_log, typechecked)
    def feature_selection(
        self,
        strategy: Optional[str] = None,
        solver: Optional[Union[str, callable]] = None,
        n_features: Optional[Union[int, float]] = None,
        max_frac_repeated: Optional[Union[int, float]] = 1.0,
        max_correlation: Optional[float] = 1.0,
        **kwargs,
    ):
        """Apply feature selection techniques.

        Remove features according to the selected strategy. Ties between
        features with equal scores will be broken in an unspecified way.
        Also removes features with too low variance and finds pairs of
        collinear features based on the Pearson correlation coefficient. For
        each pair above the specified limit (in terms of absolute value), it
        removes one of the two.

        Note that the RFE and RFECV strategies don't work when the solver is a
        CatBoost model due to incompatibility of the APIs. If the run method has
        already been called before running RFECV, the scoring parameter will be set
        to the selected metric (if not explicitly provided).

        After running the method, the created attributes and methods are attached
        to `the instance.

        See the feature_engineering.py module for a description of the parameters.

        """
        if isinstance(strategy, str):
            if strategy.lower() == "univariate" and solver is None:
                if self.goal.startswith("reg"):
                    solver = "f_regression"
                else:
                    solver = "f_classif"
            elif strategy.lower() in ["sfm", "rfe", "rfecv"]:
                if solver is None and self.winner:
                    solver = self.winner.estimator
                elif isinstance(solver, str):
                    # In case the user already filled the task...
                    if not solver.endswith("_class") and not solver.endswith("_reg"):
                        solver += "_reg" if self.task.startswith("reg") else "_class"

            # If the run method was called before, use the main metric_ for RFECV
            if strategy.lower() == "rfecv":
                if self.metric_ and "scoring" not in kwargs:
                    kwargs["scoring"] = self.metric_[0]

        kwargs = self._prepare_kwargs(kwargs, FeatureSelector().get_params())
        feature_selector = FeatureSelector(
            strategy=strategy,
            solver=solver,
            n_features=n_features,
            max_frac_repeated=max_frac_repeated,
            max_correlation=max_correlation,
            **kwargs,
        ).fit(self.X_train, self.y_train)

        self.X = feature_selector.transform(self.X)
        self.pipeline = self.pipeline.append(
            pd.Series([feature_selector]), ignore_index=True
        )

        # Attach used attributes to ATOM
        attributes = (
            "feature_importance",
            "univariate",
            "collinear",
            "pca",
            "sfm",
            "rfe",
            "rfecv",
        )
        for attr in attributes:
            if getattr(feature_selector, attr) is not None:
                setattr(self, attr, getattr(feature_selector, attr))

    # Training methods ====================================================== >>

    def _prepare_run(self, approach, metric, gib, proba, threshold):
        """Prepare the pipeline for a run.

        Clear the pipeline from previous approaches. In case of a rerun,
        check if the provided metric is the same as the one in the pipeline.

        """
        # If this is the first run of this approach, clear all previous results
        if not type(self.trainer).__name__.startswith(approach):
            clear(self, self.models[:])

            # Define the results" index
            if approach == "Trainer":
                self._results.index = pd.Index([], name="model")
            else:
                col_name = "n_models" if approach.startswith("Successive") else "frac"
                self._results.index = pd.MultiIndex(
                    levels=[[], []], codes=[[], []], names=(col_name, "model")
                )

        else:
            if metric is None:  # Assign the existing metric
                metric = self.metric_
            else:  # Check that the selected metric is the same as previous run
                metric_ = BaseTrainer._prepare_metric(
                    lst(metric), gib, proba, threshold
                )

                metric_name = flt([m.name for m in metric_])
                if metric_name != self.metric:
                    raise ValueError(
                        "Invalid value for the metric parameter! Value "
                        "should be the same as previous run. Expected "
                        f"{self.metric}, got {metric_name}."
                    )

        return metric

    def _run(self):
        """Run the trainer.

        If all models failed, catch the errors and pass them to the ATOM instance
        before raising the exception. If successful run, update all relevant
        attributes and methods.

        """
        try:
            self.trainer._data = self._data
            self.trainer._idx = self._idx
            self.trainer.run()
        except ValueError:
            raise
        else:
            # Update attributes
            self.models += [m for m in self.trainer.models if m not in self.models]
            self.metric_ = self.trainer.metric_
            self.trainer.mapping = self.mapping  # Attach mapping for plots

            # Update the results attribute

            for idx, row in self.trainer._results.iterrows():
                self._results.loc[idx, :] = row

            for model in self.trainer.models:
                self.errors.pop(model, None)  # Remove model from errors if there

                # Attach model subclasses to ATOM
                setattr(self, model, getattr(self.trainer, model))
                setattr(self, model.lower(), getattr(self.trainer, model.lower()))

                # Add pipeline to model subclasses for transform method
                setattr(getattr(self, model), "pipeline", self.pipeline)

            # Remove different approaches from the pipeline
            to_drop = []
            for approach in ["Trainer", "SuccessiveHalving", "TrainSizing"]:
                if not self.trainer.__class__.__name__.startswith(approach):
                    to_drop.append(approach)

            self.pipeline = self.pipeline[
                ~self.pipeline.astype(str).str.startswith(tuple(to_drop))
            ]

            # Add the trainer to the pipeline
            self.pipeline = self.pipeline.append(
                pd.Series([self.trainer]), ignore_index=True
            )

        finally:  # Catch errors and pass them to ATOM's attribute
            for model, error in self.trainer.errors.items():
                self.errors[model] = error

    @composed(crash, method_to_log, typechecked)
    def run(
        self,
        models: Union[CAL, Sequence[CAL]],
        metric: Optional[Union[CAL, Sequence[CAL]]] = None,
        greater_is_better: Union[bool, Sequence[bool]] = True,
        needs_proba: Union[bool, Sequence[bool]] = False,
        needs_threshold: Union[bool, Sequence[bool]] = False,
        n_calls: Union[int, Sequence[int]] = 0,
        n_initial_points: Union[int, Sequence[int]] = 5,
        est_params: dict = {},
        bo_params: dict = {},
        bagging: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ):
        """Fit the models to the dataset in a direct fashion.

        See the basetrainer.py module for a description of the parameters.

        """
        metric = self._prepare_run(
            approach="Trainer",
            metric=metric,
            gib=greater_is_better,
            proba=needs_proba,
            threshold=needs_threshold,
        )

        params = (
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, bagging
        )

        kwargs = self._prepare_kwargs(kwargs)
        if self.goal.startswith("class"):
            self.trainer = TrainerClassifier(*params, **kwargs)
        else:
            self.trainer = TrainerRegressor(*params, **kwargs)

        self._run()

    @composed(crash, method_to_log, typechecked)
    def successive_halving(
        self,
        models: Union[CAL, Sequence[CAL]],
        metric: Optional[Union[CAL, Sequence[CAL]]] = None,
        greater_is_better: Union[bool, Sequence[bool]] = True,
        needs_proba: Union[bool, Sequence[bool]] = False,
        needs_threshold: Union[bool, Sequence[bool]] = False,
        skip_runs: int = 0,
        n_calls: Union[int, Sequence[int]] = 0,
        n_initial_points: Union[int, Sequence[int]] = 5,
        est_params: dict = {},
        bo_params: dict = {},
        bagging: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ):
        """Fit the models to the dataset in a successive halving fashion.

        If you want to compare similar models, you can choose to use a successive
        halving approach to run the pipeline. This technique is a bandit-based
        algorithm that fits N models to 1/N of the data. The best half are selected
        to go to the next iteration where the process is repeated. This continues
        until only one model remains, which is fitted on the complete dataset.
        Beware that a model's performance can depend greatly on the amount of data
        on which it is trained. For this reason, we recommend only to use this
        technique with similar models, e.g. only using tree-based models.

        See the basetrainer.py module for a description of the parameters.

        """
        metric = self._prepare_run(
            approach="SuccessiveHalving",
            metric=metric,
            gib=greater_is_better,
            proba=needs_proba,
            threshold=needs_threshold,
        )

        params = (
            models, metric, greater_is_better, needs_proba, needs_threshold,
            skip_runs, n_calls, n_initial_points, est_params, bo_params, bagging
        )

        kwargs = self._prepare_kwargs(kwargs)
        if self.goal.startswith("class"):
            self.trainer = SuccessiveHalvingClassifier(*params, **kwargs)
        else:
            self.trainer = SuccessiveHalvingRegressor(*params, **kwargs)

        self._run()

    @composed(crash, method_to_log, typechecked)
    def train_sizing(
        self,
        models: Union[CAL, Sequence[CAL]],
        metric: Optional[Union[CAL, Sequence[CAL]]] = None,
        greater_is_better: Union[bool, Sequence[bool]] = True,
        needs_proba: Union[bool, Sequence[bool]] = False,
        needs_threshold: Union[bool, Sequence[bool]] = False,
        train_sizes: TRAIN_TYPES = np.linspace(0.2, 1.0, 5),
        n_calls: Union[int, Sequence[int]] = 0,
        n_initial_points: Union[int, Sequence[int]] = 5,
        est_params: dict = {},
        bo_params: dict = {},
        bagging: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ):
        """Fit the models to the dataset in a training sizing fashion.

        If you want to compare how different models perform when training on
        varying dataset sizes, you can choose to use a train_sizing approach
        to run the pipeline. This technique fits the models on increasingly
        large training sets.

        See the basetrainer.py module for a description of the parameters.

        """
        metric = self._prepare_run(
            approach="TrainSizing",
            metric=metric,
            gib=greater_is_better,
            proba=needs_proba,
            threshold=needs_threshold,
        )

        params = (
            models, metric, greater_is_better, needs_proba, needs_threshold,
            train_sizes, n_calls, n_initial_points, est_params, bo_params, bagging
        )

        kwargs = self._prepare_kwargs(kwargs)
        if self.goal.startswith("class"):
            self.trainer = TrainSizingClassifier(*params, **kwargs)
        else:
            self.trainer = TrainSizingRegressor(*params, **kwargs)

        self._run()

    # data attributes ======================================================= >>

    def _update_trainer(self):
        """Update the trainer's data when changing data attributes from ATOM."""
        if self.trainer is not None:
            self.trainer._data = self._data
            self.trainer._idx = self._idx

    @BasePredictor.dataset.setter
    @typechecked
    def dataset(self, dataset: Optional[X_TYPES]):
        # None is also possible because of the save method
        self._data = None if dataset is None else check_property(dataset, "dataset")
        self._update_trainer()

    @BasePredictor.train.setter
    @typechecked
    def train(self, train: X_TYPES):
        df = check_property(train, "train", under=self.test, under_name="test")
        self._data = pd.concat([df, self.test])
        self._idx[0] = len(df)
        self._update_trainer()

    @BasePredictor.test.setter
    @typechecked
    def test(self, test: X_TYPES):
        df = check_property(test, "test", under=self.train, under_name="train")
        self._data = pd.concat([self.train, df])
        self._idx[1] = len(df)
        self._update_trainer()

    @BasePredictor.X.setter
    @typechecked
    def X(self, X: X_TYPES):
        df = check_property(X, "X", side=self.y, side_name="y")
        self._data = merge(df, self.y)
        self._update_trainer()

    @BasePredictor.y.setter
    @typechecked
    def y(self, y: Union[list, tuple, dict, np.ndarray, pd.Series]):
        series = check_property(y, "y", side=self.X, side_name="X")
        self._data = merge(self._data.drop(self.target, axis=1), series)
        self._update_trainer()

    @BasePredictor.X_train.setter
    @typechecked
    def X_train(self, X_train: X_TYPES):
        df = check_property(
            value=X_train,
            value_name="X_train",
            side=self.y_train,
            side_name="y_train",
            under=self.X_test,
            under_name="X_test",
        )
        self._data = pd.concat([merge(df, self.train[self.target]), self.test])
        self._update_trainer()

    @BasePredictor.X_test.setter
    @typechecked
    def X_test(self, X_test: X_TYPES):
        df = check_property(
            value=X_test,
            value_name="X_test",
            side=self.y_test,
            side_name="y_test",
            under=self.X_train,
            under_name="X_train",
        )
        self._data = pd.concat([self.train, merge(df, self.test[self.target])])
        self._update_trainer()

    @BasePredictor.y_train.setter
    @typechecked
    def y_train(self, y_train: Union[list, tuple, dict, np.ndarray, pd.Series]):
        series = check_property(
            value=y_train,
            value_name="y_train",
            side=self.X_train,
            side_name="X_train",
            under=self.y_test,
            under_name="y_test",
        )
        self._data = pd.concat([merge(self.X_train, series), self.test])
        self._update_trainer()

    @BasePredictor.y_test.setter
    @typechecked
    def y_test(self, y_test: Union[list, tuple, dict, np.ndarray, pd.Series]):
        series = check_property(
            value=y_test,
            value_name="y_test",
            side=self.X_test,
            side_name="X_test",
            under=self.y_train,
            under_name="y_train",
        )
        self._data = pd.concat([self.train, merge(self.X_test, series)])
        self._update_trainer()

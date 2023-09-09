# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the BaseModel class.

"""

from __future__ import annotations

import re
from copy import copy, deepcopy
from datetime import datetime as dt
from functools import cached_property, lru_cache
from importlib import import_module
from logging import Logger
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import dill as pickle
import mlflow
import numpy as np
import pandas as pd
import ray
from joblib.memory import Memory
from mlflow.data import from_pandas
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from optuna import TrialPruned, create_study
from optuna.samplers import NSGAIISampler, TPESampler
from optuna.study import Study
from optuna.terminator import report_cross_validation_scores
from optuna.trial import FrozenTrial, Trial, TrialState
from ray import serve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve
from sklearn.model_selection import (
    ShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit,
)
from sklearn.model_selection._validation import cross_validate
from sklearn.multioutput import (
    ClassifierChain, MultiOutputClassifier, MultiOutputRegressor,
)
from sklearn.utils import resample
from sklearn.utils.metaestimators import available_if
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import SingleWindowSplitter
from sktime.proba.normal import Normal
from starlette.requests import Request

from atom.basetracker import BaseTracker
from atom.basetransformer import BaseTransformer
from atom.branch import Branch, BranchManager
from atom.data_cleaning import Scaler
from atom.pipeline import Pipeline
from atom.plots import RunnerPlot
from atom.utils.constants import DF_ATTRS
from atom.utils.types import (
    SLICE, Backend, Bool, DataFrame, DataFrameTypes, Engine, Features, Float,
    FloatTypes, Index, Int, IntTypes, MethodSelector, MetricSelector, Pandas,
    Predictor, Scalar, Scorer, Sequence, Stages, Target, Warnings,
)
from atom.utils.utils import (
    ClassMap, CustomDict, DataConfig, PlotCallback, ShapExplanation,
    TrialsCallback, bk, check_dependency, check_empty, check_scaling, composed,
    crash, estimator_has_attr, fit_and_score, flt, get_cols, get_custom_scorer,
    get_feature_importance, has_task, infer_task, is_binary, is_multioutput,
    it, lst, merge, method_to_log, rnd, sign, time_to_str, to_df, to_pandas,
    to_series,
)
from joblib import Parallel, delayed


class BaseModel(BaseTransformer, BaseTracker, RunnerPlot):
    """Base class for all models.

    Parameters
    ----------
    name: str or None, default=None
        Name for the model. If None, the name is the same as the
        model's acronym.

    goal: str, default="class"
        Model's goal. Choose from: "class", "fc", "reg".

    config: DataConfig or None, default=None
        Data configuration. If None, use the default config values.

    branches: BranchManager or None, default=None
        BranchManager.

    metric: ClassMap or None, default=None
        Metric on which to fit the model.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: dict, default={"data": "numpy", "estimator": "sklearn"}
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be a
        dictionary with keys `data` and/or `estimator`, with their
        corresponding choice as values. Choose from:

        - "data":

            - "numpy"
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn"
            - "sklearnex"
            - "cuml"

    backend: str, default="loky"
        Parallelization backend. Read more in the
        [user guide][parallel-execution]. Choose from:

        - "loky": Single-node, process-based parallelism.
        - "multiprocessing": Legacy single-node, process-based
          parallelism. Less robust than `loky`.
        - "threading": Single-node, thread-based parallelism.
        - "ray": Multi-node, process-based parallelism.

    memory: bool, str, Path or Memory, default=True
        Enables caching for memory optimization. Read more in the
        [user guide][memory-considerations].

        - If False: No caching is performed.
        - If True: A default temp directory is used.
        - If str: Path to the caching directory.
        - If Path: A [pathlib.Path][] to the caching directory.
        - If Memory: Object with the [joblib.Memory][] interface.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWarnings` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic name.
        - Else: Python `logging.Logger` instance.

    experiment: str or None, default=None
        Name of the [mlflow experiment][experiment] to use for tracking.
        If None, no mlflow tracking is performed.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    """

    def __init__(
        self,
        name: str | None = None,
        goal: Literal["class", "reg", "fc"] = "class",
        config: DataConfig | None = None,
        branches: BranchManager | None = None,
        metric: ClassMap | None = None,
        n_jobs: Int = 1,
        device: str = "cpu",
        engine: Engine = {"data": "numpy", "estimator": "sklearn"},
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = True,
        verbose: Literal[0, 1, 2] = 0,
        warnings: Bool | Warnings = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: Int | None = None,
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

        self._name = name or self.acronym
        self.goal = goal

        self._config = config or DataConfig()
        self._metric = metric

        self.scaler = None
        self.app = None
        self.dashboard = None

        self._run = None  # mlflow run (if experiment is active)
        if self.experiment:
            self._run = mlflow.start_run(run_name=self.name)
            mlflow.end_run()

        self._group = self._name  # sh and ts models belong to the same group
        self._evals = CustomDict()
        self._shap_explanation = None

        # Parameter attributes
        self._est_params = {}
        self._est_params_fit = {}

        # Hyperparameter tuning attributes
        self._ht = {"distributions": {}, "cv": 1, "plot": False, "tags": {}}
        self._study = None
        self._trials = None
        self._best_trial = None

        self._estimator = None
        self._time_fit = 0

        self._bootstrap = None
        self._time_bootstrap = 0

        # Skip this part if not called for the estimator
        if branches:
            self._branches = branches
            self._train_idx = len(self.branch._data.train_idx)  # Can change for sh and ts

            self.task = infer_task(self.y, goal=self.goal)

            if self.needs_scaling and not check_scaling(self.X, pipeline=self.pipeline):
                self.scaler = Scaler().fit(self.X_train)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __getattr__(self, item: str) -> Any:
        if item in dir(self.__dict__.get("branch")) and not item.startswith("_"):
            return getattr(self.branch, item)  # Get attr from branch
        elif item in self.__dict__.get("branch").columns:
            return self.branch.dataset[item]  # Get column
        elif item in DF_ATTRS:
            return getattr(self.branch.dataset, item)  # Get attr from dataset
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'."
            )

    def __contains__(self, item: str) -> Bool:
        return item in self.dataset

    def __getitem__(self, item: Int | str | list) -> Pandas:
        if isinstance(item, IntTypes):
            return self.dataset[self.columns[item]]
        else:
            return self.dataset[item]  # Get a subset of the dataset

    @property
    def _fullname(self) -> str:
        """Return the model's class name."""
        return self.__class__.__name__

    @property
    def _est_class(self) -> Predictor:
        """Return the estimator's class (not instance)."""
        try:
            engine = self.engine.get("estimator", "sklearn")
            module = import_module(f"{engine}.{self._module}")
            cls = self._estimators.get(self.goal, self._estimators.get("reg"))
        except (ModuleNotFoundError, AttributeError):
            if "sklearn" in self.supports_engines:
                module = import_module(f"sklearn.{self._module}")
            else:
                module = import_module(self._module)
            cls = self._estimators.get(self.goal, self._estimators.get("reg"))

        return getattr(module, cls)

    @property
    def _shap(self) -> ShapExplanation:
        """Return the ShapExplanation instance for this model."""
        if not self._shap_explanation:
            self._shap_explanation = ShapExplanation(
                estimator=self.estimator,
                task=self.task,
                branch=self.branch,
                random_state=self.random_state,
            )

        return self._shap_explanation

    @property
    def og(self) -> Branch:
        """Original branch."""
        return self._branches.og

    @property
    def branch(self) -> Branch:
        """Current active branch."""
        return self._branches.current

    def _check_est_params(self):
        """Check that the parameters are valid for the estimator.

        A parameter is always accepted if the method accepts kwargs.

        """
        for param in self._est_params:
            if all(p not in sign(self._est_class) for p in (param, "kwargs")):
                raise ValueError(
                    "Invalid value for the est_params parameter. Got unknown "
                    f"parameter {param} for estimator {self._est_class.__name__}."
                )

        for param in self._est_params_fit:
            if all(p not in sign(self._est_class.fit) for p in (param, "kwargs")):
                raise ValueError(
                    f"Invalid value for the est_params parameter. Got "
                    f"unknown parameter {param} for the fit method of "
                    f"estimator {self._est_class.__name__}."
                )

    def _get_param(self, name: str, params: CustomDict) -> Any:
        """Get a parameter from est_params or the objective func.

        Parameters
        ----------
        name: str
            Name of the parameter.

        params: CustomDict
            Parameters in the current trial.

        Returns
        -------
        Any
            Parameter value.

        """
        return self._est_params.get(name) or params.get(name)

    def _get_parameters(self, trial: Trial) -> CustomDict:
        """Get the trial's hyperparameters.

        This method fetches the suggestions from the trial and rounds
        floats to the 4th digit.

        Parameters
        ----------
        trial: [Trial][]
            Current trial.

        Returns
        -------
        CustomDict
            Trial's hyperparameters.

        """
        return CustomDict(
            {k: rnd(trial._suggest(k, v)) for k, v in self._ht["distributions"].items()}
        )

    @staticmethod
    def _trial_to_est(params: CustomDict) -> CustomDict:
        """Convert trial's parameters to parameters for the estimator.

        Some models, such as MLP, use different hyperparameters for the
        study as for the estimator (this is the case when the estimator's
        parameter can not be modeled according to an integer, float or
        categorical distribution). This method converts the parameters
        from the trial to those that can be ingested by the estimator.
        This method is overriden by implementations in the child classes.
        The base method just returns the parameters as is.

        Parameters
        ----------
        params: CustomDict
            Trial's hyperparameters.

        Returns
        -------
        CustomDict
            Estimator's hyperparameters.

        """
        return deepcopy(params)

    def _get_est(self, **params) -> Predictor:
        """Get the estimator instance.

        Use the multioutput meta-estimator if the estimator has
        no native support for multioutput.

        Use sktime's [make_reduction][] function for regressors
        in forecast tasks.

        Parameters
        ----------
        **params
            Unpacked hyperparameters for the estimator.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        # Separate the parameters for the estimator from those in sub-estimators
        base_params, sub_params = {}, {}
        for name, value in params.items():
            if "__" not in name:
                base_params[name] = value
            else:
                sub_params[name] = value

        estimator = self._inherit(self._est_class(**base_params))
        estimator.set_params(**sub_params)

        if self.task.startswith("multilabel") and not self.native_multilabel:
            estimator = ClassifierChain(estimator)
        elif "multioutput" in self.task and not self.native_multioutput:
            if self.goal.startswith("class"):
                estimator = MultiOutputClassifier(estimator)
            else:
                estimator = MultiOutputRegressor(estimator)
        elif hasattr(self, "_estimators") and self.goal not in self._estimators:
            # Forecasting task with a regressor
            estimator = make_reduction(estimator)

        return self._inherit(estimator)

    def _fit_estimator(
        self,
        estimator: Predictor,
        data: tuple[DataFrame, Pandas],
        est_params_fit: dict,
        validation: tuple[DataFrame, Pandas] | None = None,
        trial: Trial | None = None,
    ) -> Predictor:
        """Fit the estimator and perform in-training validation.

        In-training evaluation is performed on models with the
        `partial_fit` method. After every partial fit, the estimator
        is evaluated (using only the main metric) on the validation
        data and, optionally, pruning is performed.

        Parameters
        ----------
        estimator: Predictor
            Instance to fit.

        data: tuple
            Training data of the form (X, y).

        validation: tuple or None
            Validation data of the form (X, y). If None, no validation
            is performed.

        est_params_fit: dict
            Additional parameters for the estimator's fit method.

        trial: [Trial][] or None
            Active trial (during hyperparameter tuning).

        Returns
        -------
        Predictor
            Fitted instance.

        """
        if self.has_validation and hasattr(estimator, "partial_fit") and validation:
            if not trial:
                # In-training validation is skipped during hyperparameter tuning
                self._evals = CustomDict(
                    {
                        f"{self._metric[0].name}_train": [],
                        f"{self._metric[0].name}_test": [],
                    }
                )

            # Loop over first parameter in estimator
            try:
                steps = estimator.get_params()[self.has_validation]
            except KeyError:
                # For meta-estimators like multioutput
                steps = estimator.get_params()[f"estimator__{self.has_validation}"]

            for step in range(steps):
                kwargs = {}
                if self.goal.startswith("class"):
                    if is_multioutput(self.task):
                        if self.native_multilabel:
                            kwargs["classes"] = list(range(self.y.shape[1]))
                        else:
                            kwargs["classes"] = [np.unique(y) for y in get_cols(self.y)]
                    else:
                        kwargs["classes"] = np.unique(self.y)

                estimator.partial_fit(*data, **est_params_fit, **kwargs)

                if not trial:
                    # Store train and validation scores on main metric in evals attribute
                    self._evals[0].append(
                        self._score_from_est(self._metric[0], estimator, *data)
                    )
                    self._evals[1].append(
                        self._score_from_est(self._metric[0], estimator, *validation)
                    )

                # Multi-objective optimization doesn't support pruning
                if trial and len(self._metric) == 1:
                    trial.report(
                        self._score_from_est(self._metric[0], estimator, *validation),
                        step=step,
                    )

                    if trial.should_prune():
                        # Hacky solution to add the pruned step to the output
                        if self.has_validation in trial.params:
                            trial.params[self.has_validation] = f"{step}/{steps}"

                        trial.set_user_attr("estimator", estimator)
                        raise TrialPruned()

        else:
            # Add forecasting horizon to sktime estimators
            if self.goal == "fc":
                if "fh" in sign(estimator.fit):
                    if estimator.get_tag("requires-fh-in-fit"):
                        est_params_fit["fh"] = est_params_fit.get("fh", self.test.index)

                estimator.fit(data[1], X=check_empty(data[0]), **est_params_fit)

            else:
                estimator.fit(*data, **est_params_fit)

        return estimator

    def _final_output(self) -> str:
        """Returns the model's final output as a string.

        If [bootstrapping][] was used, use the format: mean +- std.

        Returns
        -------
        str
            Final score representation.

        """
        try:
            if self.bootstrap is None:
                out = "   ".join(
                    [
                        f"{name}: {rnd(lst(self.score_test)[i])}"
                        for i, name in enumerate(self._metric.keys())
                    ]
                )
            else:
                out = "   ".join(
                    [
                        f"{name}: {rnd(self.bootstrap[name].mean())} "
                        f"\u00B1 {rnd(self.bootstrap[name].std())}"
                        for name in self._metric.keys()
                    ]
                )

            if self.goal != "fc":
                # Annotate if model overfitted when train 20% > test on main metric
                score_train = lst(self.score_train)[0]
                score_test = lst(self.score_test)[0]
                if (1.2 if score_train < 0 else 0.8) * score_train > score_test:
                    out += " ~"

        except TypeError:  # Fails when model failed but errors="keep"
            out = "FAIL"

        return out

    def _get_pred(
        self,
        dataset: Literal["train", "test", "holdout"],
        target: str | None = None,
        attr: MethodSelector = "predict",
    ) -> tuple[Pandas, Pandas]:
        """Get the true and predicted values for a column.

        Predictions are made using the `decision_function` or
        `predict_proba` attributes whenever available, checked in
        that order.

        Parameters
        ----------
        dataset: str
            Data set for which to get the predictions.

        target: str or None, default=None
            Target column to look at. Only for [multioutput tasks][].
            If None, all columns are returned.

        attr: str, default="predict"
            Attribute used to get predictions. Choose from:

            - "predict": Use the `predict` method.
            - "predict_proba": Use the `predict_proba` method.
            - "decision_function": Use the `decision_function` method.
            - "thresh": Use `decision_function` or `predict_proba`.

        Returns
        -------
        series or dataframe
            True values.

        series or dataframe
            Predicted values.

        """
        # Select method to use for predictions
        if attr == "thresh":
            for attribute in ("decision_function", "predict_proba", "predict"):
                if hasattr(self.estimator, attribute):
                    attr = attribute
                    break

        y_true = getattr(self, f"y_{dataset}")
        y_pred = getattr(self, f"{attr}_{dataset}")

        if is_multioutput(self.task):
            if target is not None:
                return y_true.loc[:, target], y_pred.loc[:, target]
        elif self.task.startswith("bin") and y_pred.ndim > 1:
            return y_true, y_pred.iloc[:, 1]

        return y_true, y_pred

    def _score_from_est(
        self,
        scorer: Scorer,
        estimator: Predictor,
        X: DataFrame,
        y: Pandas,
        **kwargs,
    ) -> Float:
        """Calculate the metric score from an estimator.

        Parameters
        ----------
        scorer: Scorer
            Metric to calculate.

        estimator: Predictor
            Estimator instance to get the score from.

        X: dataframe
            Feature set.

        y: series or dataframe
            Target column corresponding to X.

        **kwargs
            Additional keyword arguments for the score function.

        Returns
        -------
        float
            Calculated score.

        """
        if self.goal == "fc":
            # Sktime uses signature estimator.predict(fh, X)
            y_pred = to_series(estimator.predict(y.index, check_empty(X)), index=y.index)
            return self._score_from_pred(scorer, y, y_pred, **kwargs)
        elif self.task.startswith("multiclass-multioutput"):
            # Calculate predictions with shape=(n_samples, n_targets)
            y_pred = to_df(estimator.predict(X), index=y.index, columns=y.columns)
            return self._score_from_pred(scorer, y, y_pred, **kwargs)
        else:
            return scorer(estimator, X, y, **kwargs)

    def _score_from_pred(
        self,
        scorer: Scorer,
        y_true: Pandas,
        y_pred: Pandas,
        **kwargs,
    ) -> Scalar:
        """Calculate the metric score from predicted values.

        Since sklearn metrics don't support multiclass-multioutput
        tasks, it calculates the mean of the scores over the target
        columns for such tasks.

        Parameters
        ----------
        scorer: Scorer
            Metric to calculate.

        y_true: series or dataframe
            True values in the target column(s).

        y_pred: series or dataframe
            Predicted values corresponding to y_true.

        **kwargs
            Additional keyword arguments for the score function.

        Returns
        -------
        int or float
            Calculated score.

        """
        func = lambda x, y: scorer._score_func(x, y, **scorer._kwargs, **kwargs)

        # Forecasting models can have first prediction NaN
        if self.goal == "fc" and all(x.isna()[0] for x in get_cols(y_pred)):
            y_true, y_pred, = y_true.iloc[1:], y_pred.iloc[1:]

        if self.task.startswith("multiclass-multioutput"):
            # Get mean of scores over target columns
            return np.mean(
                [scorer._sign * func(y_true[c], y_pred[c]) for c in y_pred], axis=0
            )
        else:
            return scorer._sign * func(y_true, y_pred)

    @lru_cache
    def _get_score(
        self,
        scorer: Scorer,
        dataset: str,
        threshold: tuple[Float] | None = None,
        sample_weight: tuple | None = None,
    ) -> Scalar:
        """Calculate a metric score using the prediction attributes.

        The method results are cached to avoid recalculation of the
        same metrics. The cache can be cleared using the clear method.

        Parameters
        ----------
        scorer: Scorer
            Metrics to calculate. If None, a selection of the most
            common metrics per task are used.

        dataset: str
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

        threshold: tuple or None, default=None
            Thresholds between 0 and 1 to convert predicted probabilities
            to class labels for every target column. Only used when:

            - The parameter is not None.
            - The task is binary or multilabel classification.
            - The model has a `predict_proba` method.
            - The metric evaluates predicted target values.

        sample_weight: tuple or None, default=None
            Sample weights corresponding to y in `dataset`. Is tuple to
            allow hashing.

        Returns
        -------
        int or float
            Metric score on the selected data set.

        """
        if dataset == "holdout" and self.holdout is None:
            raise ValueError("No holdout data set available.")

        if scorer.__class__.__name__ == "_ThresholdScorer":
            y_true, y_pred = self._get_pred(dataset, attr="thresh")
        elif scorer.__class__.__name__ == "_ProbaScorer":
            y_true, y_pred = self._get_pred(dataset, attr="predict_proba")
        else:
            if threshold and is_binary(self.task) and hasattr(self, "predict_proba"):
                y_true, y_pred = self._get_pred(dataset, attr="predict_proba")
                if isinstance(y_pred, DataFrameTypes):
                    # Update every target column with its corresponding threshold
                    for i, value in enumerate(threshold):
                        y_pred.iloc[:, i] = (y_pred.iloc[:, i] > value).astype("int")
                else:
                    y_pred = (y_pred > threshold[0]).astype("int")
            else:
                y_true, y_pred = self._get_pred(dataset, attr="predict")

        kwargs = {}
        if "sample_weight" in sign(scorer._score_func):
            kwargs["sample_weight"] = sample_weight

        result = rnd(self._score_from_pred(scorer, y_true, y_pred, **kwargs))

        if self._run:  # Log metric to mlflow run
            MlflowClient().log_metric(
                run_id=self._run.info.run_id,
                key=f"{scorer.name}_{dataset}",
                value=it(result),
            )

        return result

    @composed(crash, method_to_log)
    def hyperparameter_tuning(self, n_trials: Int, reset: Bool = False):
        """Run the hyperparameter tuning algorithm.

        Search for the best combination of hyperparameters. The function
        to optimize is evaluated either with a K-fold cross-validation
        on the training set or using a random train and validation split
        every trial. Use this method to continue the optimization.

        Parameters
        ----------
        n_trials: int
            Number of trials for the hyperparameter tuning.

        reset: bool, default=False
            Whether to start a new study or continue the existing one.

        """

        def objective(trial: Trial) -> list[Float]:
            """Objective function for hyperparameter tuning.

            Parameters
            ----------
            trial: optuna.trial.Trial
               Model's hyperparameters used in this call of the BO.

            Returns
            -------
            list of float
                Scores of the estimator in this trial.

            """

            def fit_model(
                estimator: Predictor,
                train_idx: np.ndarray,
                val_idx: np.ndarray,
            ) -> tuple[Predictor, list[Float]]:
                """Fit the model. Function for parallelization.

                Divide the training set in a (sub) train and validation
                set for this fit. The sets are created from the original
                dataset to avoid data leakage since the training set is
                transformed using the pipeline fitted on the same set.
                Fit the model on custom_fit if exists, else normally.
                Return the score on the validation set.

                Parameters
                ----------
                estimator: Predictor
                    Model's estimator to fit.

                train_idx: np.array
                    Indices for the subtrain set.

                val_idx: np.array
                    Indices for the validation set.

                Returns
                -------
                Predictor
                    Fitted estimator.

                list of float
                    Scores of the estimator on the validation set.

                """
                X_sub = self.og.X_train.iloc[train_idx]
                y_sub = self.og.y_train.iloc[train_idx]
                X_val = self.og.X_train.iloc[val_idx]
                y_val = self.og.y_train.iloc[val_idx]

                # Transform subsets if there is a pipeline
                if len(pl := self.pipeline) > 0:
                    X_sub, y_sub = pl.fit_transform(X_sub, y_sub, verbose=0)
                    X_val, y_val = pl.transform(X_val, y_val)

                # Match the sample_weight with the length of the subtrain set
                # Make copy of est_params to not alter the mutable variable
                if "sample_weight" in (est_copy := self._est_params_fit.copy()):
                    est_copy["sample_weight"] = [
                        self._est_params_fit["sample_weight"][i] for i in train_idx
                    ]

                estimator = self._fit_estimator(
                    estimator=estimator,
                    data=(X_sub, y_sub),
                    est_params_fit=est_copy,
                    validation=(X_val, y_val),
                    trial=trial,
                )

                scores = [
                    self._score_from_est(metric, estimator, X_val, y_val)
                    for metric in self._metric
                ]

                return estimator, scores

            # Start trial ========================================== >>

            params = self._get_parameters(trial)

            # Since the suggested values are not the exact same values used in
            # the estimator (often changed by _get_parameters in models.py),
            # we implement this hacky method to overwrite the params in storage
            trial._cached_frozen_trial.params = params
            frozen_trial = self.study._storage._get_trial(trial.number)
            frozen_trial.params = params
            frozen_trial.distributions = self._ht["distributions"]
            self.study._storage._set_trial(trial.number, frozen_trial)

            # Store user defined tags
            for key, value in self._ht["tags"].items():
                trial.set_user_attr(key, value)

            # Create estimator instance with trial specific hyperparameters
            estimator = self._get_est(
                **{**self._est_params, **self._trial_to_est(params)}
            )

            # Skip if the eval function has already been evaluated at this point
            if dict(params) not in self.trials["params"].tolist():
                # Follow same stratification strategy as atom
                cols = self._get_stratify_columns(self.og.train, self.og.y_train)

                if isinstance(cv := self._ht["cv"], IntTypes):
                    if self.goal == "fc":
                        if cv == 1:
                            splitter = SingleWindowSplitter(range(1, len(self.og.test)))
                        else:
                            splitter = TimeSeriesSplit(n_splits=cv)
                    elif isinstance(self._ht["cv"], Int):
                        # We use ShuffleSplit instead of Kfold because it
                        # works with n_splits=1 and multioutput stratification
                        if cols is None:
                            splitter = ShuffleSplit
                        else:
                            splitter = StratifiedShuffleSplit

                        splitter = splitter(
                            n_splits=self._ht["cv"],
                            test_size=self._config.test_size,
                            random_state=trial.number + (self.random_state or 0),
                        )
                else:  # Custom cross-validation generator
                    splitter = self._inherit(cv)

                args = [self.og.X_train]
                if "y" in sign(splitter.split) and cols is not None:
                    args.append(cols)

                # Parallel loop over fit_model
                results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                    delayed(fit_model)(estimator, i, j) for i, j in splitter.split(*args)
                )

                estimator = results[0][0]
                score = list(np.mean(scores := [r[1] for r in results], axis=0))

                if len(results) > 1:
                    # Report cv scores for termination judgement
                    report_cross_validation_scores(trial, scores)
            else:
                # Get same estimator and score as previous evaluation
                idx = self.trials.index[self.trials["params"] == params][0]
                estimator = self.trials.at[idx, "estimator"]
                score = lst(self.trials.at[idx, "score"])

            trial.set_user_attr("estimator", estimator)

            return score

        # Running hyperparameter tuning ============================ >>

        self._log(f"Running hyperparameter tuning for {self._fullname}...", 1)

        # Check validity of provided parameters
        self._check_est_params()

        # Assign custom distributions or use predefined
        dist = self._get_distributions() if hasattr(self, "_get_distributions") else {}
        if self._ht.get("distributions"):
            # Select default distributions
            inc, exc = [], []
            for name in [k for k, v in self._ht["distributions"].items() if v is None]:
                # If it's a name, use the predefined dimension
                if name.startswith("!"):
                    exc.append(n := name[1:])
                else:
                    inc.append(n := name)

                if n not in dist:
                    raise ValueError(
                        "Invalid value for the distributions parameter. "
                        f"Parameter {n} is not a predefined hyperparameter "
                        f"of the {self._fullname} model. See the model's "
                        "documentation for an overview of the available "
                        "hyperparameters and their distributions."
                    )

            if inc and exc:
                raise ValueError(
                    "Invalid value for the distributions parameter. You can either "
                    "include or exclude hyperparameters, not combinations of these."
                )
            elif exc:
                # If distributions were excluded with `!`, select all but those
                self._ht["distributions"] = {
                    k: v for k, v in dist.items() if k not in exc
                }
            elif inc:
                self._ht["distributions"] = {
                    k: v for k, v in dist.items() if k in inc
                }
        else:
            self._ht["distributions"] = dist

        # Drop hyperparameter if already defined in est_params
        self._ht["distributions"] = {
            k: v for k, v in self._ht["distributions"].items()
            if k not in self._est_params
        }

        # If no hyperparameters to optimize, skip BO
        if not self._ht["distributions"]:
            self._log(" --> Skipping study. No hyperparameters to optimize.", 2)
            return

        if not self._study or reset:
            kw = {k: v for k, v in self._ht.items() if k in sign(create_study)}
            if len(self._metric) == 1:
                kw["direction"] = "maximize"
                kw["sampler"] = kw.pop("sampler", TPESampler(seed=self.random_state))
            else:
                kw["directions"] = ["maximize"] * len(self._metric)
                kw["sampler"] = kw.pop("sampler", NSGAIISampler(seed=self.random_state))

            self._trials = pd.DataFrame(
                columns=[
                    "params",
                    "estimator",
                    "score",
                    "time_trial",
                    "time_ht",
                    "state",
                ]
            )
            self._trials.index.name = "trial"
            self._study = create_study(study_name=self.name, **kw)

        kw = {k: v for k, v in self._ht.items() if k in sign(Study.optimize)}
        n_jobs = kw.pop("n_jobs", 1)

        # Initialize live study plot
        if self._ht.get("plot", False) and n_jobs == 1:
            plot_callback = PlotCallback(
                name=self._fullname,
                metric=self._metric.keys(),
                aesthetics=self.aesthetics,
            )
        else:
            plot_callback = None

        callbacks = kw.pop("callbacks", []) + [TrialsCallback(self, n_jobs)]
        callbacks += [plot_callback] if plot_callback else []

        self._study.optimize(
            func=objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=callbacks,
            show_progress_bar=kw.pop("show_progress_bar", self.verbose == 1),
            **kw,
        )

        if len(self.study.get_trials(states=[TrialState.COMPLETE])) == 0:
            self._log(
                "The study didn't complete any trial successfully. "
                "Skipping hyperparameter tuning.", 1, severity="warning"
            )
            return

        if len(self._metric) == 1:
            self._best_trial = self.study.best_trial
        else:
            # Sort trials by best score on main metric
            self._best_trial = sorted(
                self.study.best_trials, key=lambda x: x.values[0]
            )[0]

        self._log(f"Hyperparameter tuning {'-' * 27}", 1)
        self._log(f"Best trial --> {self.best_trial.number}", 1)
        self._log("Best parameters:", 1)
        self._log("\n".join([f" --> {k}: {v}" for k, v in self.best_params.items()]), 1)
        out = [
            f"{m.name}: {rnd(lst(self.score_ht)[i])}"
            for i, m in enumerate(self._metric)
        ]
        self._log(f"Best evaluation --> {'   '.join(out)}", 1)
        self._log(f"Time elapsed: {time_to_str(self.time_ht)}", 1)

    @composed(crash, method_to_log)
    def fit(self, X: DataFrame | None = None, y: Pandas | None = None):
        """Fit and validate the model.

        The estimator is fitted using the best hyperparameters found
        during hyperparameter tuning. Afterwards, the estimator is
        evaluated on the test set. Only use this method to re-fit the
        model after having continued the study.

        Parameters
        ----------
        X: dataframe or None
            Feature set with shape=(n_samples, n_features). If None,
            `self.X_train` is used.

        y: series, dataframe or None
            Target column corresponding to X. If None, `self.y_train`
            is used.

        """
        t_init = dt.now()

        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        self.clear()  # Reset model's state

        if self.trials is None:
            self._log(f"Results for {self._fullname}:", 1)
        self._log(f"Fit {'-' * 45}", 1)

        # Assign estimator if not done already
        if not self._estimator:
            self._check_est_params()
            self._estimator = self._get_est(**{**self._est_params, **self.best_params})

        self._estimator = self._fit_estimator(
            estimator=self.estimator,
            data=(X, y),
            est_params_fit=self._est_params_fit,
            validation=(self.X_test, self.y_test),
        )

        for ds in ("train", "test"):
            out = [
                f"{metric.name}: {self._get_score(metric, ds)}"
                for metric in self._metric
            ]
            self._log(f"T{ds[1:]} evaluation --> {'   '.join(out)}", 1)

        # Get duration and print to log
        self._time_fit += (dt.now() - t_init).total_seconds()
        self._log(f"Time elapsed: {time_to_str(self.time_fit)}", 1)

        # Track results in mlflow ================================== >>

        # Log parameters, metrics, model and data to mlflow
        if self.experiment:
            with mlflow.start_run(run_id=self._run.info.run_id):
                mlflow.set_tags(
                    {
                        "name": self.name,
                        "model": self._fullname,
                        "branch": self.branch.name,
                        **self._ht["tags"],
                    }
                )

                # Mlflow only accepts params with char length <250
                mlflow.log_params(
                    {
                        k: v for k, v in self.estimator.get_params().items()
                        if len(str(v)) <= 250
                    }
                )

                # Save evals for models with in-training validation
                if self.evals:
                    for key, value in self.evals.items():
                        for step in range(len(value)):
                            mlflow.log_metric(f"evals_{key}", value[step], step=step)

                # Rest of metrics are tracked when calling _get_score
                mlflow.log_metric("time_fit", self.time_fit)

                if self.log_model:
                    mlflow.sklearn.log_model(
                        sk_model=self.estimator,
                        artifact_path=self._est_class.__name__,
                        signature=infer_signature(
                            model_input=pd.DataFrame(self.X),
                            model_output=self.predict_test.to_numpy(),
                        ),
                        input_example=pd.DataFrame(self.X.iloc[[0], :]),
                    )

                if self.log_data:
                    for ds in ("train", "test"):
                        mlflow.log_input(
                            dataset=from_pandas(pd.DataFrame(getattr(self, ds))),
                            context=ds,
                        )

                if self.log_pipeline:
                    mlflow.sklearn.log_model(
                        sk_model=self.export_pipeline(),
                        artifact_path=f"{self.name}_pipeline",
                        signature=infer_signature(
                            model_input=pd.DataFrame(self.X),
                            model_output=self.predict_test.to_numpy(),
                        ),
                        input_example=pd.DataFrame(self.X.iloc[[0], :]),
                    )

    @composed(crash, method_to_log)
    def bootstrapping(self, n_bootstrap: Int, reset: Bool = False):
        """Apply a bootstrap algorithm.

        Take bootstrapped samples from the training set and test them
        on the test set to get a distribution of the model's results.

        Parameters
        ----------
        n_bootstrap: int
           Number of bootstrapped samples to fit on.

        reset: bool, default=False
            Whether to start a new run or continue the existing one.

        """
        t_init = dt.now()

        if self._bootstrap is None or reset:
            self._bootstrap = pd.DataFrame(columns=self._metric.keys())
            self._bootstrap.index.name = "sample"

        for i in range(n_bootstrap):
            # Create stratified samples with replacement
            sample_x, sample_y = resample(
                self.X_train,
                self.y_train,
                replace=True,
                random_state=i + (self.random_state or 0),
                stratify=self.y_train,
            )

            # Fit on bootstrapped set
            estimator = self._fit_estimator(
                estimator=self.estimator,
                data=(sample_x, sample_y),
                est_params_fit=self._est_params_fit,
            )

            # Get scores on the test set
            scores = pd.DataFrame(
                {
                    m.name: [self._score_from_est(m, estimator, self.X_test, self.y_test)]
                    for m in self._metric
                }
            )

            self._bootstrap = pd.concat([self._bootstrap, scores], ignore_index=True)

        self._log(f"Bootstrap {'-' * 39}", 1)
        out = [
            f"{m.name}: {rnd(self.bootstrap.mean()[i])}"
            f" \u00B1 {rnd(self.bootstrap.std()[i])}"
            for i, m in enumerate(self._metric)
        ]
        self._log(f"Evaluation --> {'   '.join(out)}", 1)

        self._time_bootstrap += (dt.now() - t_init).total_seconds()
        self._log(f"Time elapsed: {time_to_str(self.time_bootstrap)}", 1)

    # Utility properties =========================================== >>

    @property
    def name(self) -> str:
        """Name of the model.

        Use the property's `@setter` to change the model's name. The
        acronym always stays at the beginning of the model's name. If
        the model is being tracked by [mlflow][tracking], the name of
        the corresponding run also changes.

        """
        return self._name

    @name.setter
    def name(self, value: str):
        """Change the model's name."""
        # Drop the acronym if provided by the user
        if re.match(f"{self.acronym}_", value, re.I):
            value = value[len(self.acronym) + 1:]

        # Add the acronym in front (with right capitalization)
        self._name = f"{self.acronym}{f'_{value}' if value else ''}"

        if self._run:  # Change name in mlflow's run
            MlflowClient().set_tag(self._run.info.run_id, "mlflow.runName", self.name)

        self._log(f"Model {self.name} successfully renamed to {self._name}.", 1)

    @property
    def study(self) -> Study | None:
        """Optuna study used for [hyperparameter tuning][]."""
        return self._study

    @property
    def trials(self) -> pd.DataFrame | None:
        """Overview of the trials' results.

        All durations are in seconds. Columns include:

        - **params:** Parameters used for this trial.
        - **estimator:** Estimator used for this trial.
        - **score:** Objective score(s) of the trial.
        - **time_trial:** Duration of the trial.
        - **time_ht:** Duration of the hyperparameter tuning.
        - **state:** Trial's state (COMPLETE, PRUNED, FAIL).

        """
        if self._trials is not None:
            return self._trials.sort_index()  # Can be in disorder for n_jobs>1

    @property
    def best_trial(self) -> FrozenTrial | None:
        """Trial that returned the highest score.

        For [multi-metric runs][], the best trial is the trial that
        performed best on the main metric. Use the property's `@setter`
        to change the best trial. See [here][example-hyperparameter-tuning]
        an example.

        """
        return self._best_trial

    @best_trial.setter
    def best_trial(self, value: Int):
        """Change the selected best trial."""
        if value not in self.trials.index:
            raise ValueError(
                "Invalid value for the best_trial. The "
                f"value should be a trial number, got {value}."
            )
        self._best_trial = self.study.trials[value]

    @property
    def best_params(self) -> dict:
        """Hyperparameters used by the [best trial][self-best_trial]."""
        if self.best_trial:
            return self.trials.at[self.best_trial.number, "params"]
        else:
            return {}

    @property
    def score_ht(self) -> Scalar | list[Scalar] | None:
        """Metric score obtained by the [best trial][self-best_trial]."""
        if self.best_trial:
            return self.trials.at[self.best_trial.number, "score"]

    @property
    def time_ht(self) -> Float | None:
        """Duration of the hyperparameter tuning (in seconds)."""
        if self.trials is not None:
            return self.trials.iat[-1, -2]

    @property
    def estimator(self) -> Predictor:
        """Estimator fitted on the training set."""
        return self._estimator

    @property
    def evals(self) -> CustomDict:
        """Scores obtained per iteration of the training.

        Only the scores of the [main metric][metric] are tracked.
        Included keys are: train and test. Read more in the
        [user guide][in-training-validation].

        """
        return self._evals

    @property
    def score_train(self) -> Scalar | list[Scalar]:
        """Metric score on the training set."""
        return flt([self._get_score(m, "train") for m in self._metric])

    @property
    def score_test(self) -> Scalar | list[Scalar]:
        """Metric score on the test set."""
        return flt([self._get_score(m, "test") for m in self._metric])

    @property
    def score_holdout(self) -> Scalar | list[Scalar]:
        """Metric score on the holdout set."""
        return flt([self._get_score(m, "holdout") for m in self._metric])

    @property
    def time_fit(self) -> Float:
        """Duration of the model fitting on the train set (in seconds)."""
        return self._time_fit

    @property
    def bootstrap(self) -> pd.DataFrame | None:
        """Overview of the bootstrapping scores.

        The dataframe has shape=(n_bootstrap, metric) and shows the
        score obtained by every bootstrapped sample for every metric.
        Using `atom.bootstrap.mean()` yields the same values as
        [score_bootstrap][self-score_bootstrap].

        """
        return self._bootstrap

    @property
    def score_bootstrap(self) -> Scalar | list[Scalar] | None:
        """Mean metric score on the bootstrapped samples."""
        if self.bootstrap is not None:
            return flt(self.bootstrap.mean().tolist())

    @property
    def time_bootstrap(self) -> Float | None:
        """Duration of the bootstrapping (in seconds)."""
        if self._time_bootstrap:
            return self._time_bootstrap

    @property
    def time(self) -> Float:
        """Total duration of the run (in seconds)."""
        return (self.time_ht or 0) + self._time_fit + self._time_bootstrap

    @property
    def feature_importance(self) -> pd.Series | None:
        """Normalized feature importance scores.

        The sum of importances for all features is 1. The scores are
        extracted from the estimator's `scores_`, `coef_` or
        `feature_importances_` attribute, checked in that order.
        Returns None for estimators without any of those attributes.

        """
        if (data := get_feature_importance(self.estimator)) is not None:
            return pd.Series(
                data=data / data.sum(),
                index=self.features,
                name="feature_importance",
                dtype=float,
            ).sort_values(ascending=False)

    @property
    def results(self) -> pd.Series:
        """Overview of the training results.

        All durations are in seconds. Values include:

        - **score_ht:** Score obtained by the hyperparameter tuning.
        - **time_ht:** Duration of the hyperparameter tuning.
        - **score_train:** Metric score on the train set.
        - **score_test:** Metric score on the test set.
        - **time_fit:** Duration of the model fitting on the train set.
        - **score_bootstrap:** Mean score on the bootstrapped samples.
        - **time_bootstrap:** Duration of the bootstrapping.
        - **time:** Total duration of the run.

        """
        return pd.Series(
            {
                "score_ht": self.score_ht,
                "time_ht": self.time_ht,
                "score_train": self.score_train,
                "score_test": self.score_test,
                "time_fit": self.time_fit,
                "score_bootstrap": self.score_bootstrap,
                "time_bootstrap": self.time_bootstrap,
                "time": self.time,
            },
            name=self.name,
        )

    # Data Properties ============================================== >>

    @property
    def pipeline(self) -> pd.Series:
        """Transformers fitted on the data.

        Models that used [automated feature scaling][] have the scaler
        added. Use this attribute only to access the individual
        instances. To visualize the pipeline, use the [plot_pipeline][]
        method.

         """
        if self.scaler:
            return pd.concat(
                [self.branch.pipeline, pd.Series(self.scaler, dtype="object")],
                ignore_index=True,
            )
        else:
            return self.branch.pipeline

    @property
    def dataset(self) -> DataFrame:
        """Complete data set."""
        return merge(self.X, self.y)

    @property
    def train(self) -> DataFrame:
        """Training set."""
        return merge(self.X_train, self.y_train)

    @property
    def test(self) -> DataFrame:
        """Test set."""
        return merge(self.X_test, self.y_test)

    @property
    def holdout(self) -> DataFrame | None:
        """Holdout set."""
        if (holdout := self.branch.holdout) is not None:
            if self.scaler:
                return merge(
                    self.scaler.transform(holdout.iloc[:, :-self.branch._data.n_cols]),
                    holdout.iloc[:, -self.branch._data.n_cols:],
                )
            else:
                return holdout

    @property
    def X(self) -> DataFrame:
        """Feature set."""
        return bk.concat([self.X_train, self.X_test])

    @property
    def y(self) -> Pandas:
        """Target column."""
        return bk.concat([self.y_train, self.y_test])

    @property
    def X_train(self) -> DataFrame:
        """Features of the training set."""
        if self.scaler:
            return self.scaler.transform(self.branch.X_train[-self._train_idx:])
        else:
            return self.branch.X_train[-self._train_idx:]

    @property
    def y_train(self) -> Pandas:
        """Target column of the training set."""
        return self.branch.y_train[-self._train_idx:]

    @property
    def X_test(self) -> DataFrame:
        """Features of the test set."""
        if self.scaler:
            return self.scaler.transform(self.branch.X_test)
        else:
            return self.branch.X_test

    @property
    def X_holdout(self) -> DataFrame | None:
        """Features of the holdout set."""
        if self.branch.holdout is not None:
            return self.holdout.iloc[:, :-self.branch._data.n_cols]

    @property
    def y_holdout(self) -> Pandas | None:
        """Target column of the holdout set."""
        if self.branch.holdout is not None:
            return self.holdout[self.branch.target]

    # Utility methods ============================================== >>

    @available_if(has_task("class"))
    @composed(crash, method_to_log)
    def calibrate(self, **kwargs):
        """Calibrate the model.

        Applies probability calibration on the model. The estimator
        is trained via cross-validation on a subset of the training
        data, using the rest to fit the calibrator. The new classifier
        will replace the `estimator` attribute. If there is an active
        mlflow experiment, a new run is started using the name
        `[model_name]_calibrate`. Since the estimator changed, the
        model is cleared. Only for classifiers.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for sklearn's CCV. Using
            cv="prefit" will use the trained model and fit the
            calibrator on the test set. Use this only if you have
            another, independent set for testing.

        """
        calibrator = CalibratedClassifierCV(
            estimator=self.estimator,
            n_jobs=kwargs.pop("n_jobs", self.n_jobs),
            **kwargs,
        )

        if kwargs.get("cv") != "prefit":
            self._estimator = calibrator.fit(self.X_train, self.y_train)
        else:
            self._estimator = calibrator.fit(self.X_test, self.y_test)

        # Assign a mlflow run to the new estimator
        if self._run:
            self._run = mlflow.start_run(run_name=f"{self.name}_calibrate")
            mlflow.end_run()

        self.fit()

    @composed(crash, method_to_log)
    def clear(self):
        """Reset attributes and clear cache from the model.

        Reset certain model attributes to their initial state, deleting
        potentially large data arrays. Use this method to free some
        memory before [saving][atomclassifier-save] the instance. The
        affected attributes are:

        - [In-training validation][] scores
        - [Shap values][shap]
        - [App instance][self-create_app]
        - [Dashboard instance][self-create_dashboard]
        - Cached [prediction attributes][]
        - Cached [metric scores][metric]
        - Cached [holdout data sets][data-sets]

        """
        # Reset attributes
        self._evals = CustomDict()
        self._shap_explanation = None
        self.app = None
        self.dashboard = None

        # Clear caching
        for method in self._prediction_methods:
            for ds in ("train", "test", "holdout"):
                self.__dict__.pop(f"{method}_{ds}", None)
        self._get_score.cache_clear()
        self.branch.__dict__.pop("holdout", None)

    @composed(crash, method_to_log)
    def create_app(self, **kwargs):
        """Create an interactive app to test model predictions.

        Demo your machine learning model with a friendly web interface.
        This app launches directly in the notebook or on an external
        browser page. The created [Interface][] instance can be accessed
        through the `app` attribute.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the [Interface][] instance
            or the [Interface.launch][launch] method.

        """

        def inference(*X) -> Scalar | str | list[Scalar | str]:
            """Apply inference on the row provided by the app.

            Parameters
            ----------
            *X
                Features provided by the user in the app.

            Returns
            -------
            int, float, str or list
                Original label or list of labels for multioutput tasks.

            """
            conv = lambda elem: elem.item() if hasattr(elem, "item") else elem

            y_pred = self.inverse_transform(y=self.predict([X], verbose=0), verbose=0)
            if isinstance(y_pred, DataFrameTypes):
                return [conv(elem) for elem in y_pred.iloc[0, :]]
            else:
                return conv(y_pred[0])

        check_dependency("gradio")
        from gradio import Interface
        from gradio.components import Dropdown, Textbox

        self._log("Launching app...", 1)

        inputs = []
        for name, column in self.og.X.items():
            if column.dtype.kind in "ifu":
                inputs.append(Textbox(label=name))
            else:
                inputs.append(Dropdown(list(column.unique()), label=name))

        self.app = Interface(
            fn=inference,
            inputs=inputs,
            outputs=["label"] * self.branch._data.n_cols,
            allow_flagging=kwargs.pop("allow_flagging", "never"),
            **{k: v for k, v in kwargs.items() if k in sign(Interface)},
        )

        self.app.launch(
            **{k: v for k, v in kwargs.items() if k in sign(Interface.launch)}
        )

    @available_if(has_task("multioutput", inverse=True))
    @composed(crash, method_to_log)
    def create_dashboard(
        self,
        dataset: str | Sequence = "test",
        *,
        filename: str | None = None,
        **kwargs,
    ):
        """Create an interactive dashboard to analyze the model.

        ATOM uses the [explainerdashboard][explainerdashboard_package]
        package to provide a quick and easy way to analyze and explain
        the predictions and workings of the model. The dashboard allows
        you to investigate SHAP values, permutation importances,
        interaction effects, partial dependence plots, all kinds of
        performance plots, and even individual decision trees.

        By default, the dashboard renders in a new tab in your default
        browser, but if preferable, you can render it inside the
        notebook using the `mode="inline"` parameter. The created
        [ExplainerDashboard][] instance can be accessed through the
        `dashboard` attribute. This method is not available for
        [multioutput tasks][].

        !!! note
            Plots displayed by the dashboard are not created by ATOM and
            can differ from those retrieved through this package.

        Parameters
        ----------
        dataset: str, default="test"
            Data set to get the report from. Use a sequence or add `+`
            between options to combine more than one dataset. Choose
            from: "train" or "test".

        filename: str or None, default=None
            Name to save the file with (as .html). None to not save
            anything.

        **kwargs
            Additional keyword arguments for the [ExplainerDashboard][]
            instance.

        """
        check_dependency("explainerdashboard")
        from explainerdashboard import (
            ClassifierExplainer, ExplainerDashboard, RegressionExplainer,
        )

        self._log("Creating dashboard...", 1)

        dataset = self._get_set(dataset, max_one=False, allow_holdout=False)

        X = bk.concat([getattr(self, f"X_{ds}") for ds in dataset])
        y = bk.concat([getattr(self, f"y_{ds}") for ds in dataset])

        # Get shap values from the internal ShapExplanation object
        exp = self._shap.get_explanation(X, target=(0,))

        # Explainerdashboard requires all the target classes
        if self.goal.startswith("class"):
            if self.task.startswith("bin"):
                if exp.values.shape[-1] != 2:
                    exp.base_values = [np.array(1 - exp.base_values), exp.base_values]
                    exp.values = [np.array(1 - exp.values), exp.values]
            else:
                # Explainer expects a list of np.array with shap values for each class
                exp.values = list(np.moveaxis(exp.values, -1, 0))

        params = dict(permutation_metric=self._metric, n_jobs=self.n_jobs)
        if self.goal == "class":
            explainer = ClassifierExplainer(self.estimator, X, y, **params)
        else:
            explainer = RegressionExplainer(self.estimator, X, y, **params)

        explainer.set_shap_values(exp.base_values, exp.values)

        self.dashboard = ExplainerDashboard(
            explainer=explainer,
            mode=kwargs.pop("mode", "external"),
            **kwargs,
        )
        self.dashboard.run()

        if filename:
            if not filename.endswith(".html"):
                filename += ".html"
            self.dashboard.save_html(filename)
            self._log("Dashboard successfully saved.", 1)

    @composed(crash, method_to_log)
    def cross_validate(self, **kwargs) -> pd.DataFrame:
        """Evaluate the model using cross-validation.

        This method cross-validates the whole pipeline on the complete
        dataset. Use it to assess the robustness of the solution's
        performance.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for sklearn's cross_validate
            function. If the scoring method is not specified, it uses
            atom's metric.

        Returns
        -------
        pd.DataFrame
            Overview of the results.

        """
        # Assign scoring from atom if not specified
        if kwargs.get("scoring"):
            scoring = get_custom_scorer(kwargs.pop("scoring"))
            scoring = {scoring.name: scoring}

        else:
            scoring = dict(self._metric)

        self._log("Applying cross-validation...", 1)

        # Monkey patch sklearn's _fit_and_score function to allow
        # for pipelines that drop samples during transformation
        with patch("sklearn.model_selection._validation._fit_and_score", fit_and_score):
            self.cv = cross_validate(
                estimator=self.export_pipeline(verbose=0),
                X=self.og.X,
                y=self.og.y,
                scoring=scoring,
                return_train_score=kwargs.pop("return_train_score", True),
                error_score=kwargs.pop("error_score", "raise"),
                n_jobs=kwargs.pop("n_jobs", self.n_jobs),
                verbose=kwargs.pop("verbose", 0),
                **kwargs,
            )

        df = pd.DataFrame()
        for m in scoring:
            if f"train_{m}" in self.cv:
                df[f"train_{m}"] = self.cv[f"train_{m}"]
            df[f"test_{m}"] = self.cv[f"test_{m}"]
        df["time (s)"] = self.cv["fit_time"]
        df.loc["mean"] = df.mean()
        df.loc["std"] = df.std()

        return df

    @crash
    def evaluate(
        self,
        metric: MetricSelector = None,
        dataset: Literal["train", "test", "holdout"] = "test",
        *,
        threshold: Float | Sequence = 0.5,
        sample_weight: Sequence | None = None,
    ) -> pd.Series:
        """Get the model's scores for the provided metrics.

        !!! tip
            Use the [self-get_best_threshold][] or [plot_threshold][]
            method to determine a suitable value for the `threshold`
            parameter.

        Parameters
        ----------
        metric: str, func, scorer, sequence or None, default=None
            Metrics to calculate. If None, a selection of the most
            common metrics per task are used.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

        threshold: float or sequence, default=0.5
            Threshold between 0 and 1 to convert predicted probabilities
            to class labels. Only used when:

            - The task is binary or [multilabel][] classification.
            - The model has a `predict_proba` method.
            - The metric evaluates predicted probabilities.

            For multilabel classification tasks, it's possible to provide
            a sequence of thresholds (one per target column, as returned
            by the [get_best_threshold][self-get_best_threshold] method).
            If float, the same threshold is applied to all target columns.

        sample_weight: sequence or None, default=None
            Sample weights corresponding to y in `dataset`.

        Returns
        -------
        pd.Series
            Scores of the model.

        """
        if isinstance(threshold, FloatTypes):
            threshold = [threshold] * self.branch._data.n_cols  # Length=n_targets
        elif len(threshold) != self.branch._data.n_cols:
            raise ValueError(
                "Invalid value for the threshold parameter. The length of the list "
                f"list should be equal to the number of target columns, got len(target)"
                f"={self.branch._data.n_cols} and len(threshold)={len(threshold)}."
            )

        if any(not 0 < t < 1 for t in threshold):
            raise ValueError(
                "Invalid value for the threshold parameter. Value "
                f"should lie between 0 and 1, got {threshold}."
            )

        # Predefined metrics to show
        if metric is None:
            if self.goal.startswith("class"):
                if self.task.startswith("bin"):
                    metric = [
                        "accuracy",
                        "ap",
                        "ba",
                        "f1",
                        "jaccard",
                        "mcc",
                        "precision",
                        "recall",
                        "auc",
                    ]
                elif self.task.startswith("multiclass"):
                    metric = [
                        "ba",
                        "f1_weighted",
                        "jaccard_weighted",
                        "mcc",
                        "precision_weighted",
                        "recall_weighted",
                    ]
                elif self.task.startswith("multilabel"):
                    metric = [
                        "accuracy",
                        "ap",
                        "f1_weighted",
                        "jaccard_weighted",
                        "precision_weighted",
                        "recall_weighted",
                        "auc",
                    ]
            else:
                # No msle since it fails for negative values
                metric = ["mae", "mape", "mse", "r2", "rmse"]

        scores = pd.Series(name=self.name, dtype=float)
        for met in lst(metric):
            scorer = get_custom_scorer(met)
            scores[scorer.name] = self._get_score(
                scorer=scorer,
                dataset=dataset,
                threshold=tuple(threshold),
                sample_weight=None if sample_weight is None else tuple(sample_weight),
            )

        return scores

    @crash
    def export_pipeline(self) -> Pipeline:
        """Export the model's pipeline to a sklearn-like object.

        The returned pipeline is already fitted on the training set.
        Note that if the model used [automated feature scaling][],
        the [Scaler][] is added to the pipeline.

        !!! info
            The returned pipeline behaves similarly to sklearn's
            [Pipeline][], and additionally:

            - Accepts transformers that drop rows.
            - Accepts transformers that only are fitted on a subset
              of the provided dataset.
            - Accepts transformers that apply only on the target column.
            - Uses transformers that apply only on the training set to
              fit the pipeline, not to make predictions on new data.

        Returns
        -------
        Pipeline
            Current branch as a sklearn-like Pipeline object.

        """
        pipeline = copy(self.pipeline)
        pipeline.steps.append((self._est_class.__name__, deepcopy(self.estimator)))
        return pipeline

    @composed(crash, method_to_log)
    def full_train(self, *, include_holdout: Bool = False):
        """Train the estimator on the complete dataset.

        In some cases it might be desirable to use all available data
        to train a final model. Note that doing this means that the
        estimator can no longer be evaluated on the test set. The newly
        retrained estimator will replace the `estimator` attribute. If
        there is an active mlflow experiment, a new run is started
        with the name `[model_name]_full_train`. Since the estimator
        changed, the model is cleared.

        !!! warning
            Although the model is trained on the complete dataset, the
            pipeline is not. To get a fully trained pipeline, use:
            `pipeline = atom.export_pipeline().fit(atom.X, atom.y)`.

        Parameters
        ----------
        include_holdout: bool, default=False
            Whether to include the holdout set (if available) in the
            training of the estimator. It's discouraged to use this
            option since it means the model can no longer be evaluated
            on any set.

        """
        if include_holdout and self.holdout is None:
            raise ValueError("No holdout data set available.")

        if include_holdout and self.holdout is not None:
            X = bk.concat([self.X, self.X_holdout])
            y = bk.concat([self.y, self.y_holdout])
        else:
            X, y = self.X, self.y

        # Assign a mlflow run to the new estimator
        if self._run:
            self._run = mlflow.start_run(run_name=f"{self.name}_full_train")
            mlflow.end_run()

        self.fit(X, y)

    @available_if(has_task(["binary", "multilabel"]))
    @available_if(estimator_has_attr("predict_proba"))
    @crash
    def get_best_threshold(
        self,
        dataset: Literal["train", "test", "holdout"] = "train",
    ) -> Float | list[Float]:
        """Get the threshold that maximizes the [ROC][] curve.

        Only available for models with a `predict_proba` method in a
        binary or [multilabel][] classification task.

        Parameters
        ----------
        dataset: str, default="train"
            Data set on which to calculate the threshold. Choose from:
            "train", "test" or "holdout".

        Returns
        -------
        float or list
            Best threshold or list of thresholds for multilabel tasks.

        """
        results = []
        for target in lst(self.target):
            y_true, y_pred = self._get_pred(dataset, target, attr="predict_proba")
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            results.append(thresholds[np.argmax(tpr - fpr)])

        return flt(results)

    @composed(crash, method_to_log)
    def inverse_transform(
        self,
        X: Features | None = None,
        y: Target | None = None,
        *,
        verbose: Literal[0, 1, 2] | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Inversely transform new data through the pipeline.

        Transformers that are only applied on the training set are
        skipped. The rest should all implement a `inverse_transform`
        method. If only `X` or only `y` is provided, it ignores
        transformers that require the other parameter. This can be
        of use to, for example, inversely transform only the target
        column. If called from a model that used automated feature
        scaling, the scaling is inverted as well.

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
            Verbosity level for the transformers. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        dataframe
            Original feature set. Only returned if provided.

        series or dataframe
            Original target column. Only returned if provided.

        """
        return self.pipeline.inverse_transform(X, y, verbose=verbose)

    @composed(crash, method_to_log)
    def register(
        self,
        name: str | None = None,
        stage: Stages = "None",
        archive_existing_versions: Bool = False,
    ):
        """Register the model in [mlflow's model registry][registry].

        This method is only available when model [tracking][] is
        enabled using one of the following URI schemes: databricks,
        http, https, postgresql, mysql, sqlite, mssql.

        Parameters
        ----------
        name: str or None, default=None
            Name for the registered model. If None, the model's full name
            is used. If the name of the model already exists, a new model
            version is created.

        stage: str, default="None"
            New desired stage for the model.

        archive_existing_versions: bool, default=False
            Whether all existing model versions in the `stage` will be
            moved to the "Archived" stage. Only valid when `stage` is
            "Staging" or "Production" otherwise an error will be raised.

        """
        if not self._run:
            raise PermissionError(
                "The register method is only available when "
                "there is a mlflow experiment active."
            )

        model = mlflow.register_model(
            model_uri=f"runs:/{self._run.info.run_id}/{name or self._fullname}",
            name=name or self._fullname,
            tags=self._ht["tags"] or None,
        )

        MlflowClient().transition_model_version_stage(
            name=model.name,
            version=model.version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )

    @composed(crash, method_to_log)
    def save_estimator(self, filename: str = "auto"):
        """Save the estimator to a pickle file.

        Parameters
        ----------
        filename: str, default="auto"
            Name of the file. Use "auto" for automatic naming.

        """
        if filename.endswith("auto"):
            filename = filename.replace("auto", self.estimator.__class__.__name__)

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self.estimator, f)

        self._log(f"{self._fullname} estimator successfully saved.", 1)

    @composed(crash, method_to_log)
    def serve(self, method: str = "predict", host: str = "127.0.0.1", port: Int = 8000):
        """Serve the model as rest API endpoint for inference.

        The complete pipeline is served with the model. The inference
        data must be supplied as json to the HTTP request, e.g.
        `requests.get("http://127.0.0.1:8000/", json=X.to_json())`.
        The deployment is done on a ray cluster. The default `host`
        and `port` parameters deploy to localhost.

        !!! tip
            Use `import ray; ray.serve.shutdown()` to close the
            endpoint after finishing.

        Parameters
        ----------
        method: str, default="predict"
            Estimator's method to do inference on.

        host: str, default="127.0.0.1"
            Host for HTTP servers to listen on. To expose serve
            publicly, you probably want to set this to "0.0.0.0".

        port: int, default=8000
            Port for HTTP server.

        """

        @serve.deployment
        class ServeModel:
            """Model deployment class.

            Parameters
            ----------
            model: Pipeline
                Transformers + estimator to make inference on.

            method: str, default="predict"
                Estimator's method to do inference on.

            """

            def __init__(self, model: Pipeline, method: str = "predict"):
                self.model = model
                self.method = method

            async def __call__(self, request: Request) -> str:
                """Inference call.

                Parameters
                ----------
                request: Request.
                    HTTP request. Should contain the rows to predict
                    in a json body.

                Returns
                -------
                str
                    Model predictions as string.

                """
                payload = await request.json()
                return getattr(self.model, self.method)(pd.read_json(payload))

        if not ray.is_initialized():
            ray.init(log_to_driver=False)

        server = ServeModel.bind(model=self.export_pipeline(verbose=0), method=method)
        serve.run(server, host=host, port=port)

        self._log(f"Serving model {self._fullname} on {host}:{port}...", 1)

    @composed(crash, method_to_log)
    def transform(
        self,
        X: Features | None = None,
        y: Target | None = None,
        *,
        verbose: Literal[0, 1, 2] | None = None,
    ) -> Pandas | tuple[DataFrame, Pandas]:
        """Transform new data through the pipeline.

        Transformers that are only applied on the training set are
        skipped. If only `X` or only `y` is provided, it ignores
        transformers that require the other parameter. This can be
        of use to, for example, transform only the target column. If
        called from a model that used automated feature scaling, the
        data is scaled as well.

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
            Verbosity level for the transformers. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        dataframe
            Transformed feature set. Only returned if provided.

        series or dataframe
            Transformed target column. Only returned if provided.

        """
        return self.pipeline.transform(X, y, verbose=0)


class ClassRegModel(BaseModel):
    """Classification and regression models."""

    _prediction_methods = (
        "decision_function",
        "predict",
        "predict_log_proba",
        "predict_proba",
    )

    # Prediction properties ======================================== >>

    def _assign_prediction_indices(self, index: Index) -> Index:
        """Assign index names for the prediction methods.

        Create a multi-index object for multioutput tasks, where the
        first level of the index is the target classes and the second
        the original row indices from the data set.

        Parameters
        ----------
        index: index
            Original row indices of the data set.

        Returns
        -------
        index
            Indices for the dataframe.

        """
        return bk.MultiIndex.from_tuples(
            [(col, idx) for col in np.unique(self.y) for idx in index]
        )

    def _assign_prediction_columns(self) -> list[str]:
        """Assign column names for the prediction methods.

        Returns
        -------
        list of str
            Columns for the dataframe.

        """
        if is_multioutput(self.task):
            return self.target  # When multioutput, target is list of str
        else:
            return self.mapping.get(self.target, np.unique(self.y).astype(str))

    @cached_property
    def decision_function_train(self) -> Pandas:
        """Predicted confidence scores on the training set.

        The shape of the output depends on the task:

        - (n_samples,) for binary classification.
        - (n_samples, n_classes) for multiclass classification.
        - (n_samples, n_targets) for [multilabel][] classification.

        """
        return to_pandas(
            data=self.estimator.decision_function(self.X_train),
            index=self.X_train.index,
            name=self.target,
            columns=self._assign_prediction_columns(),
        )

    @cached_property
    def decision_function_test(self) -> Pandas:
        """Predicted confidence scores on the test set.

        The shape of the output depends on the task:

        - (n_samples,) for binary classification.
        - (n_samples, n_classes) for multiclass classification.
        - (n_samples, n_targets) for [multilabel][] classification.

        """
        return to_pandas(
            data=self.estimator.decision_function(self.X_test),
            index=self.X_test.index,
            name=self.target,
            columns=self._assign_prediction_columns(),
        )

    @cached_property
    def decision_function_holdout(self) -> Pandas | None:
        """Predicted confidence scores on the holdout set.

        The shape of the output depends on the task:

        - (n_samples,) for binary classification.
        - (n_samples, n_classes) for multiclass classification.
        - (n_samples, n_targets) for [multilabel][] classification.

        """
        if self.holdout is not None:
            return to_pandas(
                data=self.estimator.decision_function(self.X_holdout),
                index=self.X_holdout.index,
                name=self.target,
                columns=self._assign_prediction_columns(),
            )

    @cached_property
    def predict_train(self) -> Pandas:
        """Predictions on the training set.

        The shape of the output depends on the task:

        - (n_samples,) for non-multioutput tasks.
        - (n_samples, n_targets) for [multioutput tasks][].

        """
        return to_pandas(
            data=self.estimator.predict(self.X_train),
            index=self.X_train.index,
            name=self.target,
            columns=self._assign_prediction_columns(),
        )

    @cached_property
    def predict_test(self) -> Pandas:
        """Predictions on the test set.

        The shape of the output depends on the task:

        - (n_samples,) for non-multioutput tasks.
        - (n_samples, n_targets) for [multioutput tasks][].

        """
        return to_pandas(
            data=self.estimator.predict(self.X_test),
            index=self.X_test.index,
            name=self.target,
            columns=self._assign_prediction_columns(),
        )

    @cached_property
    def predict_holdout(self) -> Pandas | None:
        """Predictions on the holdout set.

        The shape of the output depends on the task:

        - (n_samples,) for non-multioutput tasks.
        - (n_samples, n_targets) for [multioutput tasks][].

        """
        if self.holdout is not None:
            return to_pandas(
                data=self.estimator.predict(self.X_holdout),
                index=self.X_holdout.index,
                name=self.target,
                columns=self._assign_prediction_columns(),
            )

    @cached_property
    def predict_log_proba_train(self) -> DataFrame:
        """Class log-probability predictions on the training set.

        The shape of the output depends on the task:

        - (n_samples, n_classes) for binary and multiclass.
        - (n_samples, n_targets) for [multilabel][].
        - (n_samples * n_classes, n_targets) for [multiclass-multioutput][].

        """
        data = np.array(self.estimator.predict_log_proba(self.X_train))
        if data.ndim < 3:
            data = bk.DataFrame(
                data=data,
                index=self.X_train.index,
                columns=self._assign_prediction_columns(),
            )
        elif self.task.startswith("multilabel"):
            # Convert to (n_samples, n_targets)
            data = bk.DataFrame(
                data=np.array([d[:, 1] for d in data]).T,
                index=self.X_train.index,
                columns=self._assign_prediction_columns(),
            )
        else:
            data = bk.DataFrame(
                data=data.reshape(-1, data.shape[2]),
                index=self._assign_prediction_indices(self.X_train.index),
                columns=self._assign_prediction_columns(),
            )

        return data

    @cached_property
    def predict_log_proba_test(self) -> DataFrame:
        """Class log-probability predictions on the test set.

        The shape of the output depends on the task:

        - (n_samples, n_classes) for binary and multiclass.
        - (n_samples, n_targets) for [multilabel][].
        - (n_samples * n_classes, n_targets) for [multiclass-multioutput][].

        """
        data = np.array(self.estimator.predict_log_proba(self.X_test))
        if data.ndim < 3:
            data = bk.DataFrame(
                data=data,
                index=self.X_test.index,
                columns=self._assign_prediction_columns(),
            )
        elif self.task.startswith("multilabel"):
            # Convert to (n_samples, n_targets)
            data = bk.DataFrame(
                data=np.array([d[:, 1] for d in data]).T,
                index=self.X_test.index,
                columns=self._assign_prediction_columns(),
            )
        else:
            data = bk.DataFrame(
                data=data.reshape(-1, data.shape[2]),
                index=self._assign_prediction_indices(self.X_test.index),
                columns=self._assign_prediction_columns(),
            )

        return data

    @cached_property
    def predict_log_proba_holdout(self) -> DataFrame | None:
        """Class log-probability predictions on the holdout set.

        The shape of the output depends on the task:

        - (n_samples, n_classes) for binary and multiclass.
        - (n_samples, n_targets) for [multilabel][].
        - (n_samples * n_classes, n_targets) for [multiclass-multioutput][].

        """
        if self.holdout is not None:
            data = np.array(self.estimator.predict_log_proba(self.X_holdout))
            if data.ndim < 3:
                data = bk.DataFrame(
                    data=data,
                    index=self.X_holdout.index,
                    columns=self._assign_prediction_columns(),
                )
            elif self.task.startswith("multilabel"):
                # Convert to (n_samples, n_targets)
                data = bk.DataFrame(
                    data=np.array([d[:, 1] for d in data]).T,
                    index=self.X_holdout.index,
                    columns=self._assign_prediction_columns(),
                )
            else:
                data = bk.DataFrame(
                    data=data.reshape(-1, data.shape[2]),
                    index=self._assign_prediction_indices(self.X_holdout.index),
                    columns=self._assign_prediction_columns(),
                )

            return data

    @cached_property
    def predict_proba_train(self) -> DataFrame:
        """Class probability predictions on the training set.

        The shape of the output depends on the task:

        - (n_samples, n_classes) for binary and multiclass.
        - (n_samples, n_targets) for [multilabel][].
        - (n_samples * n_classes, n_targets) for [multiclass-multioutput][].

        """
        data = np.array(self.estimator.predict_proba(self.X_train))
        if data.ndim < 3:
            data = bk.DataFrame(
                data=data,
                index=self.X_train.index,
                columns=self._assign_prediction_columns(),
            )
        elif self.task.startswith("multilabel"):
            # Convert to (n_samples, n_targets)
            data = bk.DataFrame(
                data=np.array([d[:, 1] for d in data]).T,
                index=self.X_train.index,
                columns=self._assign_prediction_columns(),
            )
        else:
            data = bk.DataFrame(
                data=data.reshape(-1, data.shape[2]),
                index=self._assign_prediction_indices(self.X_train.index),
                columns=self._assign_prediction_columns(),
            )

        return data

    @cached_property
    def predict_proba_test(self) -> DataFrame:
        """Class probability predictions on the test set.

        The shape of the output depends on the task:

        - (n_samples, n_classes) for binary and multiclass.
        - (n_samples, n_targets) for [multilabel][].
        - (n_samples * n_classes, n_targets) for [multiclass-multioutput][].

        """
        data = np.array(self.estimator.predict_proba(self.X_test))
        if data.ndim < 3:
            data = bk.DataFrame(
                data=data,
                index=self.X_test.index,
                columns=self._assign_prediction_columns(),
            )
        elif self.task.startswith("multilabel"):
            # Convert to (n_samples, n_targets)
            data = bk.DataFrame(
                data=np.array([d[:, 1] for d in data]).T,
                index=self.X_test.index,
                columns=self._assign_prediction_columns(),
            )
        else:
            data = bk.DataFrame(
                data=data.reshape(-1, data.shape[2]),
                index=self._assign_prediction_indices(self.X_test.index),
                columns=self._assign_prediction_columns(),
            )

        return data

    @cached_property
    def predict_proba_holdout(self) -> DataFrame | None:
        """Class probability predictions on the holdout set.

        The shape of the output depends on the task:

        - (n_samples, n_classes) for binary and multiclass.
        - (n_samples, n_targets) for [multilabel][].
        - (n_samples * n_classes, n_targets) for [multiclass-multioutput][].

        """
        if self.holdout is not None:
            data = np.array(self.estimator.predict_proba(self.X_holdout))
            if data.ndim < 3:
                data = bk.DataFrame(
                    data=data,
                    index=self.X_holdout.index,
                    columns=self._assign_prediction_columns(),
                )
            elif self.task.startswith("multilabel"):
                # Convert to (n_samples, n_targets)
                data = bk.DataFrame(
                    data=np.array([d[:, 1] for d in data]).T,
                    index=self.X_holdout.index,
                    columns=self._assign_prediction_columns(),
                )
            else:
                data = bk.DataFrame(
                    data=data.reshape(-1, data.shape[2]),
                    index=self._assign_prediction_indices(self.X_holdout.index),
                    columns=self._assign_prediction_columns(),
                )

            return data

    # Prediction methods =========================================== >>

    def _prediction(
        self,
        X: slice | Features | Target,
        y: Target | None = None,
        metric: MetricSelector = None,
        sample_weight: Sequence | None = None,
        verbose: Int | None = None,
        method: str = "predict",
    ) -> Float | Pandas:
        """Get predictions on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The model should implement the provided method.

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Index names or positions of rows in the dataset, or new
            feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence, dataframe or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        metric: str, func, scorer or None, default=None
            Metric to calculate. Choose from any of sklearn's scorers,
            a function with signature metric(y_true, y_pred) or a scorer
            object. If None, it returns mean accuracy for classification
            tasks and r2 for regression tasks. Only for method="score".

        sample_weight: sequence or None, default=None
            Sample weights for the score method.

        verbose: int or None, default=None
            Verbosity level for the transformers. If None, it uses the
            estimator's own verbosity.

        method: str, default="predict"
            Prediction method to be applied to the estimator.

        Returns
        -------
        float, series or dataframe
            Calculated predictions. The return type depends on the method
            called.

        """
        # Two options: select from existing predictions (X has to be able
        # to get rows from dataset) or calculate predictions from new data
        try:
            # Duck type _get_rows -> raises an error if X can't select indices
            rows = self.branch._get_rows(X, return_test=True)
        except (ValueError, TypeError, IndexError, KeyError):
            rows = None
            X, y = self.transform(X, y)  # TODO: Fix return only X

        if method != "score":
            if rows:
                # Concatenate the predictions for all sets and retrieve indices
                sets = ("train", "test", "holdout")
                pred = bk.concat([getattr(self, f"{method}_{ds}") for ds in sets])
                return pred.loc[rows]
            else:
                if (data := np.array(getattr(self.estimator, method)(X))).ndim < 3:
                    return to_pandas(
                        data=data,
                        index=X.index,
                        name=self.target,
                        columns=self._assign_prediction_columns(),
                    )
                else:
                    return bk.DataFrame(
                        data=data.reshape(-1, data.shape[2]),
                        index=bk.MultiIndex.from_tuples(
                            [(col, i) for col in self.target for i in X.index]
                        ),
                        columns=np.unique(self.y),
                    )
        else:
            if metric is None:
                metric = self._metric[0]
            else:
                metric = get_custom_scorer(metric)

            if rows:
                # Define X and y for the score method
                data = bk.concat([self.dataset, self.holdout])
                X, y = data.loc[rows, self.features], data.loc[rows, self.target]

            return metric(self.estimator, X, y, sample_weight)

    @available_if(estimator_has_attr("decision_function"))
    @composed(crash, method_to_log)
    def decision_function(
        self,
        X: SLICE | Features,
        verbose: Int | None = None,
    ) -> Pandas:
        """Get confidence scores on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `decision_function` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Names or positions of rows in the dataset, or new feature
            set with shape=(n_samples, n_features).

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        series or dataframe
            Predicted confidence scores with shape=(n_samples,) for
            binary classification tasks or shape=(n_samples, n_classes)
            for multiclass classification tasks.

        """
        return self._prediction(X, verbose=verbose, method="decision_function")

    @available_if(estimator_has_attr("predict"))
    @composed(crash, method_to_log)
    def predict(
        self,
        X: SLICE | Features,
        verbose: Int | None = None,
    ) -> Pandas:
        """Get predictions on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `predict` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Names or indices of rows in the dataset, or new
            feature set with shape=(n_samples, n_features).

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        series or dataframe
            Predictions with shape=(n_samples,) or shape=(n_samples,
            n_targets) for [multioutput tasks][].

        """
        return self._prediction(X, verbose=verbose, method="predict")

    @available_if(estimator_has_attr("predict_log_proba"))
    @composed(crash, method_to_log)
    def predict_log_proba(
        self,
        X: SLICE | Features,
        verbose: Int | None = None,
    ) -> DataFrame:
        """Get class log-probabilities on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `predict_log_proba` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Names or positions of rows in the dataset, or new feature
            set with shape=(n_samples, n_features).

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        dataframe
            Class log-probability predictions with shape=(n_samples,
            n_classes).

        """
        return self._prediction(X, verbose=verbose, method="predict_log_proba")

    @available_if(estimator_has_attr("predict_proba"))
    @composed(crash, method_to_log)
    def predict_proba(
        self,
        X: SLICE | Features,
        verbose: Int | None = None,
    ) -> DataFrame:
        """Get class probabilities on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `predict_proba` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Names or indices of rows in the dataset, or new
            feature set with shape=(n_samples, n_features).

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        dataframe
            Class probability predictions with shape=(n_samples, n_classes)
            or (n_targets * n_samples, n_classes) with a multiindex format
            for [multioutput tasks][].

        """
        return self._prediction(X, verbose=verbose, method="predict_proba")

    @available_if(estimator_has_attr("score"))
    @composed(crash, method_to_log)
    def score(
        self,
        X: SLICE | Features,
        y: Target | None = None,
        metric: MetricSelector = None,
        sample_weight: Sequence | None = None,
        verbose: Int | None = None,
    ) -> Float:
        """Get a metric score on new data.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped.

        Read more in the [user guide][predicting].

        !!! info
            If the `metric` parameter is left to its default value, the
            method returns atom's metric score, not the metric returned
            by sklearn's score method for estimators.

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Names or positions of rows in the dataset, or new feature
            set with shape=(n_samples, n_features).

        y: int, str, dict, sequence, dataframe or None, default=None
            Target column corresponding to X.

            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - If dict: Name of the target column and sequence of values.
            - If sequence: Target column with shape=(n_samples,) or
              sequence of column names or positions for multioutput
              tasks.
            - If dataframe: Target columns for multioutput tasks.

        metric: str, func, scorer or None, default=None
            Metric to calculate. Choose from any of sklearn's scorers,
            a function with signature `metric(y_true, y_pred) -> score`
            or a scorer object. If None, it uses atom's metric (the main
            metric for [multi-metric runs][]).

        sample_weight: sequence or None, default=None
            Sample weights corresponding to y.

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        float
            Metric score of X with respect to y.

        """
        return self._prediction(
            X=X,
            y=y,
            metric=metric,
            sample_weight=sample_weight,
            verbose=verbose,
            method="score",
        )


class ForecastModel(BaseModel):
    """Forecasting models."""

    _prediction_methods = (
        "predict",
        "predict_interval",
        "predict_proba",
        "predict_quantiles",
        "predict_var",
    )

    # Prediction properties ======================================== >>

    @cached_property
    def predict_train(self) -> Pandas:
        """Predictions on the training set.

        The shape of the output depends on the task:

        - (n_samples,) for univariate tasks.
        - (n_samples, n_targets) for [multivariate][] tasks.

        """
        return self.estimator.predict(
            fh=self.train.index,
            X=check_empty(self.X_train),
        )

    @cached_property
    def predict_test(self) -> Pandas:
        """Predictions on the test set.

        The shape of the output depends on the task:

        - (n_samples,) for univariate tasks.
        - (n_samples, n_targets) for [multivariate][] tasks.

        """
        return self.estimator.predict(
            fh=self.test.index,
            X=check_empty(self.X_test),
        )

    @cached_property
    def predict_holdout(self) -> Pandas | None:
        """Predictions on the holdout set.

        The shape of the output depends on the task:

        - (n_samples,) for univariate tasks.
        - (n_samples, n_targets) for [multivariate][] tasks.

        """
        if self.holdout is not None:
            return self.estimator.predict(
                fh=self.holdout.index,
                X=check_empty(self.X_holdout),
            )

    @cached_property
    def predict_interval_train(self) -> DataFrame:
        """Prediction interval on the training set.

        The shape of the output depends on the task:

        - (n_samples, 2) for univariate tasks.
        - (n_samples, 2 * n_targets) for [multivariate][] tasks.

        """
        return self.estimator.predict_interval(
            fh=self.train.index,
            X=check_empty(self.X_train),
        )

    @cached_property
    def predict_interval_test(self) -> DataFrame:
        """Prediction interval on the test set.

        The shape of the output depends on the task:

        - (n_samples, 2) for univariate tasks.
        - (n_samples, 2 * n_targets) for [multivariate][] tasks.

        """
        return self.estimator.predict_interval(
            fh=self.test.index,
            X=check_empty(self.X_test),
        )

    @cached_property
    def predict_interval_holdout(self) -> DataFrame | None:
        """Prediction interval on the holdout set.

        The shape of the output depends on the task:

        - (n_samples, 2) for univariate tasks.
        - (n_samples, 2 * n_targets) for [multivariate][] tasks.

        """
        if self.holdout:
            return self.estimator.predict_interval(
                fh=self.train.index,
                X=check_empty(self.X_train),
            )

    @cached_property
    def predict_proba_train(self) -> Normal:
        """Probabilistic forecast on the training set."""
        return self.estimator.predict_proba(
            fh=self.train.index,
            X=check_empty(self.X_train),
        )

    @cached_property
    def predict_proba_test(self) -> Normal:
        """Probabilistic forecast on the test set."""
        return self.estimator.predict_proba(
            fh=self.test.index,
            X=check_empty(self.X_test),
        )

    @cached_property
    def predict_proba_holdout(self) -> Normal | None:
        """Probabilistic forecast on the holdout set."""
        if self.holdout:
            return self.estimator.predict_proba(
                fh=self.holdout.index,
                X=check_empty(self.X_holdout),
            )

    @cached_property
    def predict_quantiles_train(self) -> DataFrame:
        """Quantile forecast on the training set.

        The shape of the output depends on the task:

        - (n_samples, 2) for univariate tasks.
        - (n_samples, 2 * n_targets) for [multivariate][] tasks.

        """
        return self.estimator.predict_quantiles(
            fh=self.train.index,
            X=check_empty(self.X_train),
        )

    @cached_property
    def predict_quantiles_test(self) -> DataFrame:
        """Quantile forecast on the test set.

        The shape of the output depends on the task:

        - (n_samples, 2) for univariate tasks.
        - (n_samples, 2 * n_targets) for [multivariate][] tasks.

        """
        return self.estimator.predict_quantiles(
            fh=self.test.index,
            X=check_empty(self.X_test),
        )

    @cached_property
    def predict_quantiles_holdout(self) -> DataFrame | None:
        """Quantile forecast on the holdout set.

        The shape of the output depends on the task:

        - (n_samples, 2) for univariate tasks.
        - (n_samples, 2 * n_targets) for [multivariate][] tasks.

        """
        if self.holdout:
            return self.estimator.predict_quantiles(
                fh=self.holdout.index,
                X=check_empty(self.X_holdout),
            )

    @cached_property
    def predict_residuals_train(self) -> Pandas:
        """Residuals forecast on the training set.

        The shape of the output depends on the task:

        - (n_samples,) for univariate tasks.
        - (n_samples, n_targets) for [multivariate][] tasks.

        """
        return self.estimator.predict_residuals(
            y=self.y_train,
            X=check_empty(self.X_train),
        )

    @cached_property
    def predict_residuals_test(self) -> Pandas:
        """Residuals forecast on the test set.

        The shape of the output depends on the task:

        - (n_samples,) for univariate tasks.
        - (n_samples, n_targets) for [multivariate][] tasks.

        """
        return self.estimator.predict_residuals(
            y=self.y_test,
            X=check_empty(self.X_test),
        )

    @cached_property
    def predict_residuals_holdout(self) -> Pandas | None:
        """Residuals forecast on the holdout set.

        The shape of the output depends on the task:

        - (n_samples,) for univariate tasks.
        - (n_samples, n_targets) for [multivariate][] tasks.

        """
        if self.holdout:
            return self.estimator.predict_residuals(
                y=self.y_holdout,
                X=check_empty(self.X_holdout),
            )

    @cached_property
    def predict_var_train(self) -> DataFrame:
        """Variance forecast on the training set.

        The shape of the output depends on the task:

        - (n_samples,) for univariate tasks.
        - (n_samples, n_targets) for [multivariate][] tasks.

        """
        return self.estimator.predict_var(
            fh=self.y_train.index,
            X=check_empty(self.X_train),
        )

    @cached_property
    def predict_var_test(self) -> DataFrame:
        """Variance forecast on the test set.

        The shape of the output depends on the task:

        - (n_samples,) for univariate tasks.
        - (n_samples, n_targets) for [multivariate][] tasks.

        """
        return self.estimator.predict_var(
            fh=self.y_test.index,
            X=check_empty(self.X_test),
        )

    @cached_property
    def predict_var_holdout(self) -> DataFrame | None:
        """Variance forecast on the holdout set.

        The shape of the output depends on the task:

        - (n_samples,) for univariate tasks.
        - (n_samples, n_targets) for [multivariate][] tasks.

        """
        if self.holdout:
            return self.estimator.predict_var(
                fh=self.y_holdout.index,
                X=check_empty(self.holdout),
            )

    # Prediction methods =========================================== >>

    def _prediction(
        self,
        metric: MetricSelector = None,
        verbose: Int | None = None,
        method: str = "predict",
        **kwargs,
    ) -> Float | Pandas:
        """Get predictions on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The model should implement the provided method.

        Parameters
        ----------
        metric: str, func, scorer or None, default=None
            Metric to calculate. Choose from any of sklearn's scorers,
            a function with signature metric(y_true, y_pred) or a scorer
            object. If None, it returns mean accuracy for classification
            tasks and r2 for regression tasks. Only for method="score".

        verbose: int or None, default=None
            Verbosity level for the transformers. If None, it uses the
            estimator's own verbosity.

        method: str, default="predict"
            Prediction method to be applied to the estimator.

        **kwargs
            Additional keyword arguments for the method.

        Returns
        -------
        float, series or dataframe
            Calculated predictions. The return type depends on the method
            called.

        """
        if (X := kwargs.get("X")) is not None and (y := kwargs.get("y")) is not None:
            X, y = self.transform(X, y)

        if method != "score":
            return getattr(self.estimator, method)(**kwargs)
        else:
            if metric is None:
                metric = self._metric[0]
            else:
                metric = get_custom_scorer(metric)

            return self._score_from_est(metric, self.estimator, X, y, **kwargs)

    @available_if(estimator_has_attr("predict"))
    @composed(crash, method_to_log)
    def predict(
        self,
        fh: int | range | Sequence | ForecastingHorizon,
        X: Features | None = None,
        verbose: Int | None = None,
    ) -> Pandas:
        """Get predictions on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `predict` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        fh: int, range, sequence or [ForecastingHorizon][]
            The forecasting horizon encoding the time stamps to
            forecast at.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to fh.

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        series or dataframe
            Predictions with shape=(n_samples,) or shape=(n_samples,
            n_targets) for [multivariate][] tasks.

        """
        return self._prediction(fh=fh, X=X, verbose=verbose, method="predict")

    @available_if(estimator_has_attr("predict_interval"))
    @composed(crash, method_to_log)
    def predict_interval(
        self,
        fh: int | range | Sequence | ForecastingHorizon,
        X: Features | None = None,
        coverage: Float | Sequence = 0.9,
        verbose: Int | None = None,
    ) -> DataFrame:
        """Get prediction intervals on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `predict_interval` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        fh: int, range, sequence or [ForecastingHorizon][]
            The forecasting horizon encoding the time stamps to
            forecast at.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to fh.

        coverage: float or sequence, default=0.9
            Nominal coverage(s) of predictive interval(s).

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        dataframe
            Predictions with shape=(n_samples, 2) or shape=(n_samples,
            2 * n_targets) for [multivariate][] tasks.

        """
        return self._prediction(
            fh=fh,
            X=X,
            coverage=coverage,
            verbose=verbose,
            method="predict_interval",
        )

    @available_if(estimator_has_attr("predict_proba"))
    @composed(crash, method_to_log)
    def predict_proba(
        self,
        fh: int | range | Sequence | ForecastingHorizon,
        X: Features | None = None,
        marginal: Bool = True,
        verbose: Int | None = None,
    ) -> Normal:
        """Get probabilistic forecasts on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `predict_proba` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        fh: int, range, sequence or [ForecastingHorizon][]
            The forecasting horizon encoding the time stamps to
            forecast at.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to fh.

        marginal: bool, default=True
            Whether returned distribution is marginal by time index.

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        sktime.proba.[Normal][]
            Predicted distribution.

        """
        return self._prediction(
            fh=fh,
            X=X,
            marginal=marginal,
            verbose=verbose,
            method="predict_proba",
        )

    @available_if(estimator_has_attr("predict_quantiles"))
    @composed(crash, method_to_log)
    def predict_quantiles(
        self,
        fh: int | range | Sequence | ForecastingHorizon,
        X: Features | None = None,
        alpha: Float | list[Float] = [0.05, 0.95],
        verbose: Int | None = None,
    ) -> DataFrame:
        """Get probabilistic forecasts on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `predict_quantiles` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        fh: int, range, sequence or [ForecastingHorizon][]
            The forecasting horizon encoding the time stamps to
            forecast at.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to fh.

        alpha: float or list of float, default=[0.05, 0.95]
            A probability or list of, at which quantile forecasts are
            computed.

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        dataframe
            Predictions with shape=(n_samples, len(alpha)) or
            shape=(n_samples, len(alpha) * n_targets) for [multivariate][]
            tasks.

        """
        return self._prediction(
            fh=fh,
            X=X,
            alpha=alpha,
            verbose=verbose,
            method="predict_quantiles",
        )

    @available_if(estimator_has_attr("predict_residuals"))
    @composed(crash, method_to_log)
    def predict_residuals(
        self,
        y: Sequence | DataFrame,
        X: Features | None = None,
        verbose: Int | None = None,
    ) -> DataFrame:
        """Get residuals of forecasts on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `predict_residuals` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        y: sequence or dataframe-like
            Ground truth observations to compute residuals to.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to fh.

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        series or dataframe
            Predictions with shape=(n_samples,) or shape=(n_samples,
            n_targets) for [multivariate][] tasks.

        """
        return self._prediction(y=y, X=X, verbose=verbose, method="predict_residuals")

    @available_if(estimator_has_attr("predict_var"))
    @composed(crash, method_to_log)
    def predict_var(
        self,
        fh: int | range | Sequence | ForecastingHorizon,
        X: Features | None = None,
        cov: Bool = False,
        verbose: Int | None = None,
    ) -> DataFrame:
        """Get probabilistic forecasts on new data or existing rows.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `predict_var` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        fh: int, range, sequence or [ForecastingHorizon][]
            The forecasting horizon encoding the time stamps to
            forecast at.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to fh.

        cov: bool, default=False
            Whether to computes covariance matrix forecast or marginal
            variance forecasts.

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        dataframe
            Predictions with shape=(n_samples,) or shape=(n_samples,
            n_targets) for [multivariate][] tasks.

        """
        return self._prediction(
            fh=fh,
            X=X,
            cov=cov,
            verbose=verbose,
            method="predict_var",
        )

    @available_if(estimator_has_attr("score"))
    @composed(crash, method_to_log)
    def score(
        self,
        y: Sequence | DataFrame,
        X: DataFrame | None = None,
        fh: int | Sequence | ForecastingHorizon | None = None,
        metric: MetricSelector = None,
        verbose: Int | None = None,
    ) -> Float:
        """Get a metric score on new data.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped.

        Read more in the [user guide][predicting].

        !!! info
            If the `metric` parameter is left to its default value, the
            method returns atom's metric score, not the metric returned
            by sktime's score method for estimators.

        Parameters
        ----------
        y: sequence or dataframe-like
            Ground truth observations.

        X: dataframe-like or None, default=None
            Exogenous time series corresponding to fh.

        fh: int, sequence or [ForecastingHorizon][] or None, default=None
            The forecasting horizon encoding the time stamps to
            forecast at.

        metric: str, func, scorer or None, default=None
            Metric to calculate. Choose from any of sklearn's scorers,
            a function with signature `metric(y_true, y_pred) -> score`
            or a scorer object. If None, it uses atom's metric (the main
            metric for [multi-metric runs][]).

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformers' own verbosity.

        Returns
        -------
        float
            Metric score of y with respect to a ground truth.

        """
        return self._prediction(
            y=y,
            X=X,
            fh=fh,
            metric=metric,
            verbose=verbose,
            method="score",
        )

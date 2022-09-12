# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseModel class.

"""

import os
import tempfile
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime as dt
from importlib import import_module
from inspect import signature
from typing import Any, Callable, List, Optional, Tuple, Union
from unittest.mock import patch

import dill as pickle
import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.memory import Memory
from mlflow.tracking import MlflowClient
from optuna import TrialPruned, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import NSGAIISampler, TPESampler
from optuna.study import Study
from optuna.trial import Trial
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit,
)
from sklearn.model_selection._validation import _score, cross_validate
from sklearn.utils import resample
from sklearn.utils.metaestimators import available_if
from typeguard import typechecked

from atom.data_cleaning import Scaler
from atom.pipeline import Pipeline
from atom.plots import ModelPlot, ShapPlot
from atom.utils import (
    DF_ATTRS, FLOAT, INT, PANDAS_TYPES, SEQUENCE, SEQUENCE_TYPES, X_TYPES,
    Y_TYPES, CustomDict, PlotCallback, Predictor, Scorer, ShapExplanation,
    TrialsCallback, composed, crash, custom_transform, estimator_has_attr, flt,
    get_best_score, get_custom_scorer, get_feature_importance, get_pl_name,
    has_task, it, lst, merge, method_to_log, rnd, score, time_to_str,
    variable_return,
)


class BaseModel(ModelPlot, ShapPlot):
    """Base class for all models."""

    def __init__(self, *args, **kwargs):
        self.T = args[0]  # Parent class

        self.scaler = None
        self.app = None
        self.dashboard = None

        self._run = None  # mlflow run (if experiment is active)
        self._name = self.acronym if len(args) == 1 else args[1]
        self._group = self.name  # sh and ts models belong to the same group

        self._evals = CustomDict()
        self._pred = [None] * 12
        self._scores = CustomDict(
            train=CustomDict(),
            test=CustomDict(),
            holdout=CustomDict(),
        )
        self._shap = ShapExplanation(self)

        # Parameter attributes
        self._est_params = {}
        self._est_params_fit = {}

        # Hyperparameter tuning attributes
        self._ht = {}
        self._study = None
        self._trials = pd.DataFrame(
            columns=["params", "estimator", "score", "time_trial", "time_ht"]
        )
        self._trials.index.name = "trial"
        self._best_trial = None

        self._estimator = None
        self._time_fit = 0

        self._bootstrap = None
        self._time_bootstrap = 0

        # Skip this (slower) part if not called for the estimator
        if not kwargs.get("fast_init"):
            self.branch = self.T.branch
            self._train_idx = len(self.branch._idx[0])  # Can change for sh and ts
            if getattr(self, "needs_scaling", None) and self.T.scaled is False:
                self.scaler = Scaler().fit(self.X_train)

    def __repr__(self) -> str:
        out_1 = f"{self.fullname}\n --> Estimator: {self.estimator.__class__.__name__}"
        out_2 = [
            f"{m.name}: {rnd(get_best_score(self, i))}"
            for i, m in enumerate(self.T._metric.values())
        ]
        return out_1 + f"\n --> Evaluation: {'   '.join(out_2)}"

    def __getattr__(self, item: str) -> Any:
        if item in self.__dict__.get("branch")._get_attrs():
            return getattr(self.branch, item)  # Get attr from branch
        elif item in self.__dict__.get("branch").columns:
            return self.branch.dataset[item]  # Get column
        elif item in DF_ATTRS:
            return getattr(self.branch.dataset, item)  # Get attr from dataset
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'."
            )

    def __contains__(self, item: str) -> bool:
        return item in self.dataset

    def __getitem__(self, item: Union[INT, str, list]) -> PANDAS_TYPES:
        if isinstance(item, int):
            return self.dataset[self.columns[item]]
        elif isinstance(item, (str, list)):
            return self.dataset[item]  # Get a subset of the dataset
        else:
            raise TypeError(
                f"'{self.__class__.__name__}' object is only "
                "subscriptable with types int, str or list."
            )

    @property
    def _est_class(self) -> Predictor:
        """Return the estimator's class (not the instance)."""
        try:
            module = import_module(f"{self.T.engine}.{self._module}")
        except (ModuleNotFoundError, AttributeError):
            module = import_module(self._module)

        # Goal can fail when model initialized with fast_init=True
        return getattr(module, self._estimators.get(self.T.goal, self._estimators[0]))

    @property
    def _gpu(self) -> bool:
        """Return whether the model uses a GPU implementation."""
        return "gpu" in self.T.device.lower()

    @staticmethod
    def _sign(obj: Callable) -> OrderedDict:
        """Get the parameters of a class or method."""
        return signature(obj).parameters

    @staticmethod
    def _get_distributions() -> dict:
        """Get the predefined hyperparameter distributions."""
        return {}

    def _check_est_params(self):
        """Check that the parameters are valid for the estimator."""
        # The parameter is always accepted if the estimator accepts kwargs
        for param in self._est_params:
            init_sign = self._sign(self._est_class)
            if param not in init_sign and "kwargs" not in init_sign:
                raise ValueError(
                    "Invalid value for the est_params parameter. Got unknown "
                    f"parameter {param} for estimator {self._est_class.__name__}."
                )

        for param in self._est_params_fit:
            fit_sign = self._sign(self._est_class.fit)
            if param not in fit_sign and "kwargs" not in fit_sign:
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

    def _get_est(self, **params) -> Predictor:
        """Get the estimator instance.

        Add the parent's `n_jobs` and `random_state` parameters to
        the instance if available in the constructor.

        Parameters
        ----------
        **params
            Unpacked hyperparameters for the estimator.

        Returns
        -------
        Predictor
            Estimator instance.

        """
        for param in ("n_jobs", "random_state"):
            if param in self._sign(self._est_class):
                params[param] = params.pop(param, getattr(self.T, param))

        return self._est_class(**params)

    def _fit_estimator(
        self,
        estimator: Predictor,
        data: Tuple[pd.DataFrame, pd.Series],
        est_params_fit: dict = None,
        validation: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        trial: Optional[Trial] = None,
    ):
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
        if self.has_validation:
            self._evals = CustomDict(metric=self.T._metric[0].name, train=[], test=[])

            # Loop over first parameter in estimator
            for step in range(steps := estimator.get_params()[self.has_validation]):
                estimator.partial_fit(*data, classes=self.y.unique(), **est_params_fit)

                self._evals["train"].append(self.T._metric[0](estimator, *data))
                self._evals["test"].append(self.T._metric[0](estimator, *validation))

                # Multi-objective optimization doesn't support pruning
                if trial and len(self.T._metric) == 1:
                    trial.report(self.evals["test"][-1], step)

                    if trial.should_prune():
                        # Hacky solution to add the pruned step to the output
                        p = trial.storage.get_trial_user_attrs(trial.number)["params"]
                        p[self.has_validation] = f"{step}/{steps}"

                        trial.set_user_attr("estimator", estimator)
                        raise TrialPruned()

        else:
            estimator.fit(*data, **est_params_fit)

        return estimator

    def _final_output(self) -> str:
        """Returns the model's final output as a string."""
        # If bootstrap was used, use a different format
        if self.bootstrap is None:
            out = "   ".join(
                [
                    f"{m.name}: {rnd(lst(self.score_test)[i])}"
                    for i, m in enumerate(self.T._metric.values())
                ]
            )
        else:
            out = "   ".join(
                [
                    f"{m.name}: {rnd(self.bootstrap.mean()[i])} "
                    f"\u00B1 {rnd(self.bootstrap.std()[i])}"
                    for i, m in enumerate(self.T._metric.values())
                ]
            )

        # Annotate if model overfitted when train 20% > test
        score_train = self._get_score(self.T._metric[0], "train")
        score_test = self._get_score(self.T._metric[0], "test")
        if score_train - 0.2 * score_train > score_test:
            out += " ~"

        return out

    def _get_score(
        self,
        scorer: Scorer,
        dataset: str,
        threshold: FLOAT = 0.5,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
    ) -> FLOAT:
        """Calculate a metric score using the prediction attributes.

        Instead of using the scorer to make new predictions and
        recalculate the same metrics, use the model's prediction
        attributes and store calculated metrics in the `_scores`
        attribute. Return directly if already calculated.

        Parameters
        ----------
        scorer: Scorer
            Metrics to calculate. If None, a selection of the most
            common metrics per task are used.

        dataset: str
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

        threshold: float, default=0.5
            Threshold between 0 and 1 to convert predicted probabilities
            to class labels. Only used when:

            - The task is binary classification.
            - The model has a `predict_proba` method.
            - The metric evaluates predicted target values.

        sample_weight: sequence or None, default=None
            Sample weights corresponding to y in `dataset`.

        Returns
        -------
        float
            Metric score on the selected data set.

        """
        if dataset == "holdout" and self.T.holdout is None:
            raise ValueError("No holdout data set available.")

        # Only re-calculate if not default parameters
        if (
            scorer.name not in self._scores[dataset]
            or threshold != 0.5
            or sample_weight is not None
        ):
            has_pred_proba = hasattr(self.estimator, "predict_proba")
            has_dec_func = hasattr(self.estimator, "decision_function")

            # Select method to use for predictions
            if scorer.__class__.__name__ == "_ThresholdScorer":
                attr = "decision_function" if has_dec_func else "predict_proba"
            elif scorer.__class__.__name__ == "_ProbaScorer":
                attr = "predict_proba" if has_pred_proba else "decision_function"
            elif self.T.task.startswith("bin") and has_pred_proba:
                attr = "predict_proba"  # Needed to use threshold parameter
            else:
                attr = "predict"

            y_pred = getattr(self, f"{attr}_{dataset}")
            if self.T.task.startswith("bin") and attr == "predict_proba":
                y_pred = y_pred.iloc[:, 1]

                # Exclude metrics that use probability estimates (e.g. ap, auc)
                if scorer.__class__.__name__ == "_PredictScorer":
                    y_pred = (y_pred > threshold).astype("int")

            if "sample_weight" in self._sign(scorer._score_func):
                score = scorer._score_func(
                    getattr(self, f"y_{dataset}"),
                    y_pred,
                    sample_weight=sample_weight,
                    **scorer._kwargs,
                )
            else:
                score = scorer._score_func(
                    getattr(self, f"y_{dataset}"), y_pred, **scorer._kwargs
                )

            score = rnd(scorer._sign * float(score))

            if threshold == 0.5 and sample_weight is None:
                self._scores[dataset][scorer.name] = score

            if self._run:  # Log metric to mlflow run
                MlflowClient().log_metric(
                    run_id=self._run.info.run_id,
                    key=f"{scorer.name}_{dataset}",
                    value=it(score),
                )

            return score

        else:
            return self._scores[dataset][scorer.name]

    @composed(crash, method_to_log, typechecked)
    def hyperparameter_tuning(
        self,
        n_trials: int,
        ht_params: Optional[dict] = None,
        reset: bool = False,
    ):
        """Run the hyperparameter tuning algorithm.

        Search for the best combination of hyperparameters. The function
        to optimize is evaluated either with a K-fold cross-validation
        on the training set or using a random train and validation split
        every trial. Use this method to continue the optimization.

        Parameters
        ----------
        n_trials: int
            Number of trials for the hyperparameter tuning.

        ht_params: dict or None
            Additional parameters for the hyperparameter tuning. If
            None, it uses the same parameters as the first run. Can
            include:

            - **cv: int, default=1**<br>
              Number of folds for the cross-validation. If 1, the
              training set is randomly split in a subtrain and
              validation set.
            - **plot: bool, default=False**<br>
              Whether to plot the BO's progress as it runs. Creates a
              canvas with two plots: the first plot shows the score of
              every trial and the second shows the distance between the
              last consecutive steps. See the [plot_trials][] method.
            - **\*\*kwargs**<br>
              Additional Keyword arguments for the constructor of the
              [study][] class or the [optimize][] method.

        reset: bool, default=False
            Whether to start a new study or continue the existing one.

        """

        def objective(trial: Trial) -> FLOAT:
            """Objective function for hyperparameter tuning.

            Parameters
            ----------
            trial: optuna.trial.Trial
               Model's hyperparameters used in this call of the BO.

            Returns
            -------
            float
                Score achieved by the model.

            """

            def fit_model(train_idx: list, val_idx: list) -> FLOAT:
                """Fit the model. Function for parallelization.

                Divide the training set in a (sub) train and validation
                set for this fit. The sets are created from the original
                dataset to avoid data leakage since the training set is
                transformed using the pipeline fitted on the same set.
                Fit the model on custom_fit if exists, else normally.
                Return the score on the validation set.

                Parameters
                ----------
                train_idx: list
                    Indices for the subtrain set.

                val_idx: list
                    Indices for the validation set.

                Returns
                -------
                float
                    Score of the fitted model on the validation set.

                """
                nonlocal estimator

                X_subtrain = og.dataset.iloc[train_idx, :-1]
                y_subtrain = og.dataset.iloc[train_idx, -1]
                X_val = og.dataset.iloc[val_idx, :-1]
                y_val = og.dataset.iloc[val_idx, -1]

                # Transform subsets if there is a pipeline
                if not self.T.pipeline.empty:
                    pl = self.export_pipeline(verbose=0)[:-1]  # Drop the estimator
                    X_subtrain, y_subtrain = pl.fit_transform(X_subtrain, y_subtrain)
                    X_val, y_val = pl.transform(X_val, y_val)

                # Match the sample_weight with the length of the subtrain set
                # Make copy of est_params to not alter the mutable variable
                est_copy = self._est_params_fit.copy()
                if "sample_weight" in est_copy:
                    est_copy["sample_weight"] = [
                        self._est_params_fit["sample_weight"][i] for i in train_idx
                    ]

                estimator = self._fit_estimator(
                    estimator=estimator,
                    data=(X_subtrain, y_subtrain),
                    est_params_fit=est_copy,
                    validation=(X_val, y_val),
                    trial=trial,
                )

                # Calculate metrics on the validation set
                return [m(estimator, X_val, y_val) for m in self.T._metric.values()]

            # Start trial ========================================== >>

            # Get parameter suggestions and store rounded values in user_attrs
            params = self._get_parameters(trial)
            trial.set_user_attr("params", params)

            # Create estimator instance with trial specific hyperparameters
            estimator = self._get_est(**{**self._est_params, **params})

            # Get original branch to define subsets
            og = self.T._get_og_branches()[0]

            # Skip if the eval function has already been evaluated at this point
            if params not in self.trials["params"].values:
                if ht_params.get("cv", 1) == 1:
                    if self.T.goal == "class":
                        split = StratifiedShuffleSplit  # Keep % of samples per class
                    else:
                        split = ShuffleSplit

                    # Get the ShuffleSplit cross-validator object
                    fold = split(
                        n_splits=1,
                        test_size=len(og.test) / og.shape[0],
                        random_state=trial.number + (self.T.random_state or 0),
                    )

                    # Fit model just on the one fold
                    score = fit_model(*next(fold.split(og.X_train, og.y_train)))

                else:  # Use cross validation to get the score
                    if self.T.goal == "class":
                        k_fold = StratifiedKFold  # Keep % of samples per class
                    else:
                        k_fold = KFold

                    # Get the K-fold cross-validator object
                    fold = k_fold(
                        n_splits=ht_params.get("cv", 1),
                        shuffle=True,
                        random_state=trial.number + (self.T.random_state or 0),
                    )

                    # Parallel loop over fit_model (threading fixes PickleError)
                    parallel = Parallel(n_jobs=self.T.n_jobs, backend="threading")
                    scores = parallel(
                        delayed(fit_model)(i, j)
                        for i, j in fold.split(og.X_train, og.y_train)
                    )
                    score = list(np.mean(scores, axis=0))
            else:
                # Get same score as previous evaluation
                mask = self.trials["params"] == params
                score = lst(self.trials.loc[mask, "score"][0])

            trial.set_user_attr("estimator", estimator)

            return score

        # Running hyperparameter tuning ============================ >>

        # Use provided parameters or those from the first run
        ht_params = self._ht = ht_params or self._ht

        self.T.log(f"\n\nRunning hyperparameter tuning for {self.fullname}...", 1)

        # Assign custom distributions or use predefined
        dist = self._get_distributions()
        if ht_params.get("distributions"):
            # If dict, use the user provided values
            if isinstance(ht_params["distributions"], SEQUENCE):
                inc, exc = [], []
                for name in ht_params["distributions"]:
                    # If it's a name, use the predefined dimension
                    if name.startswith("!"):
                        exc.append(n := name[1:])
                    else:
                        inc.append(n := name)

                    if n not in dist:
                        raise ValueError(
                            "Invalid value for the distributions parameter. "
                            f"Parameter {n} is not a predefined hyperparameter "
                            f"of the {self.fullname} model. See the model's "
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
            self.T.log(" --> Skipping BO. No hyperparameters to optimize.", 2)
            return

        if not self._study or reset:
            kwargs = {
                k: v for k, v in ht_params.items() if k in self._sign(create_study)
            }

            if len(self.T._metric) == 1:
                kwargs["direction"] = "maximize"
                kwargs["sampler"] = kwargs.pop(
                    "sampler", TPESampler(seed=self.T.random_state)
                )
            else:
                kwargs["directions"] = ["maximize"] * len(self.T._metric)
                kwargs["sampler"] = kwargs.pop(
                    "sampler", NSGAIISampler(seed=self.T.random_state)
                )

            self._study = create_study(**kwargs)

        kwargs = {k: v for k, v in ht_params.items() if k in self._sign(Study.optimize)}
        if ht_params.get("plot", False):
            plot_callback = PlotCallback(self.T, len(self._study.trials) + n_trials)
            kwargs["callbacks"] = kw.get("callbacks", []) + plot_callback
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [TrialsCallback(self)]

        self._study.optimize(
            func=objective,
            n_trials=n_trials,
            n_jobs=kwargs.pop("n_jobs", self.T.n_jobs),
            show_progress_bar=kwargs.pop("show_progress_bar", self.T.verbose == 1),
            **kwargs,
        )

        if len(self.T._metric) == 1:
            self._best_trial = self.study.best_trial
        else:
            # Sort trials by best score on main metric
            self._best_trial = sorted(
                self.study.best_trials, key=lambda x: x.values[0]
            )[0]

        self.T.log(f"Hyperparameter tuning {'-' * 27}", 1)
        self.T.log(f"Best trial --> {self.best_trial.number}", 1)
        self.T.log("Best parameters:", 1)
        self.T.log("\n".join([f" --> {k}: {v}" for k, v in self.best_params.items()]), 1)
        out = [
            f"{m.name}: {rnd(lst(self.score_ht)[i])}"
            for i, m in enumerate(self.T._metric.values())
        ]
        self.T.log(f"Best evaluation --> {'   '.join(out)}", 1)
        self.T.log(f"Time elapsed: {time_to_str(self.time_ht)}", 1)

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None):
        """Fit and validate the model.

        The estimator is fitted using the best hyperparameters found
        during hyperparameter tuning. Afterwards, the estimator is
        evaluated on the test set. Only use this method to re-fit the
        model after having continued the study.

        Parameters
        ----------
        X: pd.DataFrame or None
            Feature set with shape=(n_samples, n_features). If None,
            `self.X_train` is used.

        y: pd.Series or None
            Target column corresponding to X. If None, `self.y_train`
            is used.

        """
        t_init = dt.now()

        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        self.clear()

        if self.trials.empty:
            self.T.log(f"\n\nResults for {self.fullname}:", 1)
        self.T.log(f"Fit {'-' * 45}", 1)

        # Assign estimator if not done already
        if not self._estimator:
            self._estimator = self._get_est(**{**self._est_params, **self.best_params})

        self._estimator = self._fit_estimator(
            estimator=self.estimator,
            data=(X, y),
            est_params_fit=self._est_params_fit,
            validation=(self.X_test, self.y_test),
        )

        # Track results to mlflow ================================== >>

        # Log parameters, metrics, model and data to mlflow
        if self._run:
            mlflow.set_tags(
                {
                    "name": self.name,
                    "fullname": self.fullname,
                    "branch": self.branch.name,
                }
            )

            # Mlflow only accepts params with char length <250
            mlflow.log_params(
                {
                    k: v for k, v in self.estimator.get_params().items()
                    if len(str(v)) <= 250}
            )

            # Save evals for models with in-training validation
            if self.evals:
                name = self.evals["metric"]
                zipper = zip(self.evals["train"], self.evals["test"])
                for step, (train, test) in enumerate(zipper):
                    mlflow.log_metric(f"evals_{name}_train", train, step=step)
                    mlflow.log_metric(f"evals_{name}_test", test, step=step)

            # Rest of metrics are tracked when calling _get_score
            mlflow.log_metric("time_fit", self.time_fit)

            if self.T.log_model:
                mlflow.sklearn.log_model(self.estimator, self._est_class.__name__)

            if self.T.log_data:
                for set_ in ("train", "test"):
                    getattr(self, set_).to_csv(f"{set_}.csv")
                    mlflow.log_artifact(f"{set_}.csv")
                    os.remove(f"{set_}.csv")

            if self.T.log_pipeline:
                mlflow.sklearn.log_model(self.export_pipeline(), f"pl_{self.name}")

            mlflow.end_run()

        # Print and log results ==================================== >>

        for set_ in ("train", "test"):
            out = [
                f"{metric.name}: {self._get_score(metric, set_)}"
                for metric in self.T._metric.values()
            ]
            self.T.log(f"T{set_[1:]} evaluation --> {'   '.join(out)}", 1)

        # Get duration and print to log
        self._time_fit += (dt.now() - t_init).total_seconds()
        self.T.log(f"Time elapsed: {time_to_str(self.time_fit)}", 1)

    @composed(crash, method_to_log, typechecked)
    def bootstrapping(self, n_bootstrap: int, reset: bool = False):
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
            self._bootstrap = pd.DataFrame(columns=self.T._metric)
            self._bootstrap.index.name = "sample"

        for i in range(n_bootstrap):
            # Create stratified samples with replacement
            sample_x, sample_y = resample(
                self.X_train,
                self.y_train,
                replace=True,
                random_state=i + (self.T.random_state or 0),
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
                    metric.name: [metric(estimator, self.X_test, self.y_test)]
                    for metric in self.T._metric.values()
                }
            )

            self._bootstrap = pd.concat([self._bootstrap, scores], ignore_index=True)

        self.T.log(f"Bootstrap {'-' * 39}", 1)
        out = [
            f"{m.name}: {rnd(self.bootstrap.mean()[i])}"
            f" \u00B1 {rnd(self.bootstrap.std()[i])}"
            for i, m in enumerate(self.T._metric.values())
        ]
        self.T.log(f"Evaluation --> {'   '.join(out)}", 1)

        self._time_bootstrap += (dt.now() - t_init).total_seconds()
        self.T.log(f"Time elapsed: {time_to_str(self.time_bootstrap)}", 1)

    # Utility properties =========================================== >>

    @property
    def name(self) -> str:
        """Name of the model."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Change the model's tag.

        The acronym always stays at the beginning of the model's name.
        If the model is being tracked by mlflow, the name of the
        corresponding run is also changed.

        """
        if not value:
            value = self.acronym  # Back to default acronym
        else:
            # Drop the acronym if not provided by the user
            if value.lower().startswith(self.acronym.lower()):
                value = value[len(self.acronym):]

            # Add the acronym (with right capitalization)
            value = self.acronym + value

        # Check if the name is available
        if value.lower() in map(str.lower, self.T._models):
            raise PermissionError(f"There already exists a model named {value}!")

        # Replace the model in the _models attribute
        self.T._models.replace_key(self.name, value)
        self.T.log(f"Model {self.name} successfully renamed to {value}.", 1)
        self.name = value

        if self._run:  # Change name in mlflow's run
            MlflowClient().set_tag(self._run.info.run_id, "mlflow.runName", self.name)

    @property
    def study(self) -> Study:
        """Optuna study used for hyperparameter tuning."""
        return self._study

    @property
    def trials(self) -> pd.DataFrame:
        """Overview of the trials. Columns include:

        - **params:** Parameters used for this trial.
        - **estimator:** Estimator used for this trial.
        - **score:** Objective score(s) of the trial.
        - **time_trial:** Duration of the trial.
        - **time_ht:** Duration of the hyperparameter tuning.

        """
        return self._trials

    @property
    def best_trial(self) -> Trial:
        """Best trial found during hyperparameter tuning.

        For [multi-metric runs][], the best trial is the trial that
        preformed best on the main metric.

        """
        return self._best_trial

    @best_trial.setter
    @typechecked
    def best_trial(self, value: int):
        if value not in self.trials.index:
            raise ValueError(
                "Invalid value for the best_trial. The "
                f"value should be a trial number, got {value}."
            )
        self._best_trial = value

    @property
    def best_params(self) -> dict:
        """Best hyperparameters found during hyperparameter tuning."""
        if not self.trials.empty:
            return self.trials.at[self.best_trial.number, "params"]
        else:
            return {}

    @property
    def score_ht(self) -> Optional[Union[FLOAT, List[FLOAT]]]:
        """Metric score obtained during hyperparameter tuning."""
        if not self.trials.empty:
            return self.trials.at[self.best_trial.number, "score"]

    @property
    def time_ht(self) -> int:
        """Duration of the hyperparameter tuning."""
        return 0 if self.trials.empty else self.trials.iat[-1, -1]

    @property
    def estimator(self) -> Predictor:
        """Estimator fitted on the training set."""
        return self._estimator

    @property
    def evals(self) -> CustomDict:
        """Main metric score per step during [in-training validation][]."""
        return self._evals

    @property
    def score_train(self) -> Union[FLOAT, List[FLOAT]]:
        """Metric score on the training set."""
        return flt([self._get_score(m, "train") for m in self.T._metric.values()])

    @property
    def score_test(self) -> Union[FLOAT, List[FLOAT]]:
        """Metric score on the test set."""
        return flt([self._get_score(m, "test") for m in self.T._metric.values()])

    @property
    def score_holdout(self) -> Union[FLOAT, List[FLOAT]]:
        """Metric score on the holdout set."""
        return flt([self._get_score(m, "holdout") for m in self.T._metric.values()])

    @property
    def time_fit(self) -> int:
        """Duration of the model fitting on the train set."""
        return self._time_fit

    @property
    def bootstrap(self) -> Optional[pd.DataFrame]:
        """Overview of the bootstrapping scores."""
        return self._bootstrap

    @property
    def score_bootstrap(self) -> Union[FLOAT, List[FLOAT]]:
        """Mean metric score on the bootstrapped samples."""
        if self.bootstrap is not None:
            return flt(self.bootstrap.mean().tolist())

    @property
    def time_bootstrap(self) -> int:
        """Duration of the bootstrapping."""
        return self._time_bootstrap

    @property
    def time(self) -> int:
        """Total duration of the model run."""
        return self.time_ht + self.time_fit + self.time_bootstrap

    @property
    def feature_importance(self) -> Optional[pd.Series]:
        """Normalized feature importance scores.

        The scores are extracted from the estimator's `coef_` or
        `feature_importances_` attribute, checked in that order.
        Returns None for estimators without those attributes.

        """
        if data := get_feature_importance(self.estimator):
            return pd.Series(
                data=data / max(data),
                index=self.features,
                name="feature_importance",
                dtype="float",
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
        - **time:** Total duration of the model run.

        """
        return pd.Series(
            {
                "score_ht": getattr(self, "score_ht", None),
                "time_ht": getattr(self, "time_ht", None),
                "score_train": getattr(self, "score_train", None),
                "score_test": getattr(self, "score_test", None),
                "time_fit": getattr(self, "time_fit", None),
                "score_bootstrap": getattr(self, "score_bootstrap", None),
                "time_bootstrap": getattr(self, "time_bootstrap", None),
                "time": getattr(self, "time", None),
            },
            name=self.name,
        )

    # Data Properties ============================================== >>

    @property
    def dataset(self) -> pd.DataFrame:
        """Complete data set."""
        return merge(self.X, self.y)

    @property
    def train(self) -> pd.DataFrame:
        """Training set."""
        return merge(self.X_train, self.y_train)

    @property
    def test(self) -> pd.DataFrame:
        """Test set."""
        return merge(self.X_test, self.y_test)

    @property
    def holdout(self) -> Optional[pd.DataFrame]:
        """Holdout set."""
        if self.branch.holdout is not None:
            if self.scaler:
                return merge(
                    self.scaler.transform(self.branch.holdout.iloc[:, :-1]),
                    self.branch.holdout.iloc[:, -1],
                )
            else:
                return self.branch.holdout

    @property
    def X(self) -> pd.DataFrame:
        """Feature set."""
        return pd.concat([self.X_train, self.X_test])

    @property
    def y(self) -> pd.Series:
        """Target column."""
        return pd.concat([self.y_train, self.y_test])

    @property
    def X_train(self) -> pd.DataFrame:
        """Features of the training set."""
        if self.scaler:
            return self.scaler.transform(self.branch.X_train[:self._train_idx])
        else:
            return self.branch.X_train[:self._train_idx]

    @property
    def y_train(self) -> pd.Series:
        """Target column of the training set."""
        return self.branch.y_train[:self._train_idx]

    @property
    def X_test(self) -> pd.DataFrame:
        """Features of the test set."""
        if self.scaler:
            return self.scaler.transform(self.branch.X_test)
        else:
            return self.branch.X_test

    @property
    def X_holdout(self) -> Optional[pd.DataFrame]:
        """Features of the holdout set."""
        if self.branch.holdout is not None:
            return self.holdout.iloc[:, :-1]

    @property
    def y_holdout(self) -> Optional[pd.Series]:
        """Target column of the holdout set."""
        if self.branch.holdout is not None:
            return self.holdout.iloc[:, -1]

    # Prediction properties ======================================== >>

    @property
    def decision_function_train(self) -> PANDAS_TYPES:
        """Confidence scores on the training set."""
        if self._pred[0] is None:
            data = self.estimator.decision_function(self.X_train)
            if data.ndim == 1:
                self._pred[0] = pd.Series(
                    data=data,
                    index=self.X_train.index,
                    name="decision_function_train",
                )
            else:
                self._pred[0] = pd.DataFrame(
                    data=data,
                    index=self.X_train.index,
                    columns=self.mapping.get(self.target),
                )

        return self._pred[0]

    @property
    def decision_function_test(self) -> PANDAS_TYPES:
        """Confidence scores on the test set."""
        if self._pred[1] is None:
            data = self.estimator.decision_function(self.X_test)
            if data.ndim == 1:
                self._pred[1] = pd.Series(
                    data=data,
                    index=self.X_test.index,
                    name="decision_function_test",
                )
            else:
                self._pred[1] = pd.DataFrame(
                    data=data,
                    index=self.X_test.index,
                    columns=self.mapping.get(self.target),
                )

        return self._pred[1]

    @property
    def decision_function_holdout(self) -> Optional[PANDAS_TYPES]:
        """Confidence scores on the holdout set."""
        if self.T.holdout is not None and self._pred[2] is None:
            data = self.estimator.decision_function(self.X_holdout)
            if data.ndim == 1:
                self._pred[2] = pd.Series(
                    data=data,
                    index=self.X_holdout.index,
                    name="decision_function_holdout",
                )
            else:
                self._pred[2] = pd.DataFrame(
                    data=data,
                    index=self.X_holdout.index,
                    columns=self.mapping.get(self.target),
                )

        return self._pred[2]

    @property
    def predict_train(self) -> pd.Series:
        """Class predictions on the training set."""
        if self._pred[3] is None:
            self._pred[3] = pd.Series(
                data=self.estimator.predict(self.X_train),
                index=self.X_train.index,
                name="predict_train",
            )

        return self._pred[3]

    @property
    def predict_test(self) -> pd.Series:
        """Class predictions on the test set."""
        if self._pred[4] is None:
            self._pred[4] = pd.Series(
                data=self.estimator.predict(self.X_test),
                index=self.X_test.index,
                name="predict_test",
            )

        return self._pred[4]

    @property
    def predict_holdout(self) -> Optional[pd.Series]:
        """Class predictions on the holdout set."""
        if self.T.holdout is not None and self._pred[5] is None:
            self._pred[5] = pd.Series(
                data=self.estimator.predict(self.X_holdout),
                index=self.X_holdout.index,
                name="predict_holdout",
            )

        return self._pred[5]

    @property
    def predict_log_proba_train(self) -> pd.DataFrame:
        """Class log-probabilities predictions on the training set."""
        if self._pred[6] is None:
            self._pred[6] = pd.DataFrame(
                data=self.estimator.predict_log_proba(self.X_train),
                index=self.X_train.index,
                columns=self.mapping.get(self.target),
            )

        return self._pred[6]

    @property
    def predict_log_proba_test(self) -> pd.DataFrame:
        """Class log-probabilities predictions on the test set."""
        if self._pred[7] is None:
            self._pred[7] = pd.DataFrame(
                data=self.estimator.predict_log_proba(self.X_test),
                index=self.X_test.index,
                columns=self.mapping.get(self.target),
            )

        return self._pred[7]

    @property
    def predict_log_proba_holdout(self) -> Optional[pd.DataFrame]:
        """Class log-probabilities predictions on the holdout set."""
        if self.T.holdout is not None and self._pred[8] is None:
            self._pred[8] = pd.DataFrame(
                data=self.estimator.predict_log_proba(self.X_holdout),
                index=self.X_holdout.index,
                columns=self.mapping.get(self.target),
            )

        return self._pred[8]

    @property
    def predict_proba_train(self) -> pd.DataFrame:
        """Class probabilities predictions on the training set."""
        if self._pred[9] is None:
            self._pred[9] = pd.DataFrame(
                data=self.estimator.predict_proba(self.X_train),
                index=self.X_train.index,
                columns=self.mapping.get(self.target),
            )

        return self._pred[9]

    @property
    def predict_proba_test(self) -> pd.DataFrame:
        """Class probabilities predictions on the test set."""
        if self._pred[10] is None:
            self._pred[10] = pd.DataFrame(
                data=self.estimator.predict_proba(self.X_test),
                index=self.X_test.index,
                columns=self.mapping.get(self.target),
            )

        return self._pred[10]

    @property
    def predict_proba_holdout(self) -> Optional[pd.DataFrame]:
        """Class probabilities predictions on the holdout set."""
        if self.T.holdout is not None and self._pred[11] is None:
            self._pred[11] = pd.DataFrame(
                data=self.estimator.predict_proba(self.X_holdout),
                index=self.X_holdout.index,
                columns=self.mapping.get(self.target),
            )

        return self._pred[11]

    # Prediction methods =========================================== >>

    def _prediction(
        self,
        X: Union[slice, Y_TYPES, X_TYPES],
        y: Optional[Y_TYPES] = None,
        metric: Optional[Union[str, callable]] = None,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        verbose: Optional[INT] = None,
        method: str = "predict",
    ) -> Union[FLOAT, PANDAS_TYPES]:
        """Get predictions on new data or rows in the dataset.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The model should implement the provided method.

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Index names or positions of rows in the dataset, or new
            feature set with shape=(n_samples, n_features).

        y: int, str, dict, sequence or None, default=None
            Target column corresponding to X.

            - If None: y is ignored.
            - If int: Position of the target column in X.
            - If str: Name of the target column in X.
            - Else: Array with shape=(n_samples,) to use as target.

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
        float, pd.Series or pd.DataFrame
            Calculated predictions. The return type depends on the method
            called.

        """
        # Two options: select from existing predictions (X has to be able
        # to get rows from dataset) or calculate predictions from new data
        try:
            # Raises ValueError if X can't select indices
            rows = self.T._get_rows(X, branch=self.branch)
        except ValueError:
            rows = None

            # When there is a pipeline, apply transformations first
            X, y = self.T._prepare_input(X, y)
            X = self.T._set_index(X)
            if y is not None:
                y.index = X.index

            for transformer in self.pipeline:
                if not transformer._train_only:
                    X, y = custom_transform(transformer, self.branch, (X, y), verbose)

            # Scale the data if needed
            if self.scaler:
                X, y = custom_transform(self.scaler, self.branch, (X, y), verbose)

        if method != "score":
            if rows:
                # Concatenate the predictions for all sets and retrieve indices
                predictions = pd.concat(
                    [
                        getattr(self, f"{method}_{set_}")
                        for set_ in ("train", "test", "holdout")
                    ]
                )
                return predictions.loc[rows]
            else:
                predictions = getattr(self.estimator, method)(X)

                if predictions.ndim == 1:
                    return pd.Series(data=predictions, index=X.index, name=method)
                else:
                    return pd.DataFrame(
                        data=predictions,
                        index=X.index,
                        columns=self.mapping.get(self.target),
                    )
        else:
            if metric is None:
                metric = self.T._metric[0]
            else:
                metric = get_custom_scorer(metric)

            if rows:
                # Define X and y for the score method
                if self.holdout is None:
                    data = self.dataset
                else:
                    data = pd.concat([self.dataset, self.holdout], axis=0)

                X, y = data.loc[rows, self.features], data.loc[rows, self.target]

            return metric(self.estimator, X, y, sample_weight)

    @available_if(estimator_has_attr("decision_function"))
    @composed(crash, method_to_log, typechecked)
    def decision_function(
        self,
        X: Union[INT, str, slice, SEQUENCE_TYPES, X_TYPES],
        /,
        *,
        verbose: Optional[INT] = None,
    ) -> PANDAS_TYPES:
        """Get confidence scores on new data or rows in the dataset.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `decision_function` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Names or indices of rows in the dataset, or new feature
            set with shape=(n_samples, n_features).

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformer's own verbosity.

        Returns
        -------
        pd.Series or pd.DataFrame
            Predicted confidence scores, with shape=(n_samples,) for
            binary classification tasks and (n_samples, n_classes) for
            multiclass classification tasks.

        """
        return self._prediction(X, verbose=verbose, method="decision_function")

    @composed(crash, method_to_log, typechecked)
    def predict(
        self,
        X: Union[INT, str, slice, SEQUENCE_TYPES, X_TYPES],
        /,
        *,
        verbose: Optional[INT] = None,
    ) -> pd.Series:
        """Get class predictions on new data or rows in the dataset.

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
            transformer's own verbosity.

        Returns
        -------
        pd.Series
            Predicted classes with shape=(n_samples,).

        """
        return self._prediction(X, verbose=verbose, method="predict")

    @available_if(estimator_has_attr("predict_log_proba"))
    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(
        self,
        X: Union[INT, str, slice, SEQUENCE_TYPES, X_TYPES],
        /,
        *,
        verbose: Optional[INT] = None,
    ) -> pd.DataFrame:
        """Get class log-probabilities on new data or rows in the dataset.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The estimator must have a `predict_log_proba` method.

        Read more in the [user guide][predicting].

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Names or indices of rows in the dataset, or new feature
            set with shape=(n_samples, n_features).

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformer's own verbosity.

        Returns
        -------
        pd.DataFrame
            Predicted class log-probabilities with shape=(n_samples,
            n_classes).

        """
        return self._prediction(X, verbose=verbose, method="predict_log_proba")

    @available_if(estimator_has_attr("predict_proba"))
    @composed(crash, method_to_log, typechecked)
    def predict_proba(
        self,
        X: Union[INT, str, slice, SEQUENCE_TYPES, X_TYPES],
        /,
        *,
        verbose: Optional[INT] = None,
    ) -> pd.DataFrame:
        """Get class probabilities on new data or rows in the dataset.

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
            transformer's own verbosity.

        Returns
        -------
        pd.DataFrame
            Predicted class probabilities with shape=(n_samples,
            n_classes).

        """
        return self._prediction(X, verbose=verbose, method="predict_proba")

    @available_if(estimator_has_attr("score"))
    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: Union[INT, str, slice, SEQUENCE_TYPES, X_TYPES],
        /,
        y: Optional[Y_TYPES] = None,
        metric: Optional[Union[str, callable]] = None,
        *,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        verbose: Optional[INT] = None,
    ) -> FLOAT:
        """Get a metric score on new data.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. If called from atom, the best model (under the `winner`
        attribute) is used. If called from a model, that model is used.

        Read more in the [user guide][predicting].

        !!! info
            If the `metric` parameter is left to its default value, the
            method returns atom's metric score, not the metric returned
            by sklearn's score method for estimators.

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Names or indices of rows in the dataset, or new feature
            set with shape=(n_samples, n_features).

        y: int, str or sequence
            Target column corresponding to X.
                - If int: Position of the target column in X.
                - If str: Name of the target column in X.
                - Else: Array with shape=(n_samples,) to use as target.

        metric: str, func, scorer or None, default=None
            Metric to calculate. Choose from any of sklearn's scorers,
            a function with signature `metric(y_true, y_pred) -> score`
            or a scorer object. If None, it uses atom's metric (the main
            metric for [multi-metric runs][]).

        sample_weight: sequence or None, default=None
            Sample weights corresponding to y.

        verbose: int or None, default=None
            Verbosity level of the output. If None, it uses the
            transformer's own verbosity.

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
        calibrator = CalibratedClassifierCV(self.estimator, **kwargs)
        if kwargs.get("cv") != "prefit":
            self._estimator = calibrator.fit(self.X_train, self.y_train)
        else:
            self._estimator = calibrator.fit(self.X_test, self.y_test)

        # Start a new mlflow run for the new estimator
        if self._run:
            self._run = mlflow.start_run(run_name=f"{self.name}_calibrate")

    @composed(crash, method_to_log)
    def clear(self):
        """Clear attributes from the model.

        Reset the model attributes to their initial state, deleting
        potentially large data arrays. Use this method to free some
        memory before saving the instance. The cleared attributes are:

        - [In-training validation][] scores.
        - [Metric scores][metric]
        - [Prediction attributes][]
        - [Shap values][shap]
        - [App instance][self-create_app]
        - [Dashboard instance][self-create_dashboard]

        """
        self._evals = CustomDict()
        self._scores = CustomDict(
            train=CustomDict(),
            test=CustomDict(),
            holdout=CustomDict(),
        )
        self._pred = [None] * 12
        self._shap = ShapExplanation(self)
        self.app = None
        self.dashboard = None

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
        from gradio import Interface
        from gradio.components import Dropdown, Textbox

        self.T.log("Launching app...", 1)

        inputs = []
        og_branch = self.T._get_og_branches()[0].name
        for name, column in self.T._branches[og_branch].X.items():
            if column.dtype.kind in "ifu":
                inputs.append(Textbox(label=name))
            else:
                inputs.append(Dropdown(list(column.unique()), label=name))

        self.app = Interface(
            fn=lambda *x: self.inverse_transform(
                y=self.predict(pd.DataFrame([x], columns=self.features))
            )[0],
            inputs=inputs,
            outputs="label",
            allow_flagging=kwargs.pop("allow_flagging", "never"),
            **{k: v for k, v in kwargs.items() if k in self._sign(Interface)},
        )

        self.app.launch(
            **{k: v for k, v in kwargs.items() if k in self._sign(Interface.launch)}
        )

    @composed(crash, typechecked, method_to_log)
    def create_dashboard(
        self,
        dataset: str = "test",
        filename: Optional[str] = None,
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
        browser, but if preferable, you can render it inside the notebook
        using the `mode="inline"` parameter. The created
        [ExplainerDashboard][] instance can be accessed through the
        `dashboard` attribute.

        !!! note
            Plots displayed by the dashboard are not created by ATOM and
            can differ from those retrieved through this package.

        Parameters
        ----------
        dataset: str, default="test"
            Data set to get the report from. Choose from: "train", "test",
            "both" (train and test) or "holdout".

        filename: str or None, default=None
            Name to save the file with (as .html). None to not save
            anything.

        **kwargs
            Additional keyword arguments for the [ExplainerDashboard][]
            instance.

        """
        from explainerdashboard import (
            ClassifierExplainer, ExplainerDashboard, RegressionExplainer,
        )

        self.T.log("Creating dashboard...", 1)

        dataset = dataset.lower()
        if dataset == "both":
            X, y = self.X, self.y
        elif dataset in ("train", "test"):
            X, y = getattr(self, f"X_{dataset}"), getattr(self, f"y_{dataset}")
        elif dataset == "holdout":
            if self.holdout is None:
                raise ValueError(
                    "Invalid value for the dataset parameter. No holdout "
                    "data set was specified when initializing atom."
                )
            X, y = self.holdout.iloc[:, :-1], self.holdout.iloc[:, -1]
        else:
            raise ValueError(
                "Invalid value for the dataset parameter, got "
                f"{dataset}. Choose from: train, test, both or holdout."
            )

        params = dict(permutation_metric=self.T._metric.values(), n_jobs=self.T.n_jobs)
        if self.T.goal == "class":
            explainer = ClassifierExplainer(self.estimator, X, y, **params)
        else:
            explainer = RegressionExplainer(self.estimator, X, y, **params)

        # Add shap values from the internal ShapExplanation object
        explainer.set_shap_values(
            base_value=self._shap.get_expected_value(return_one=False),
            shap_values=self._shap.get_shap_values(X, return_all_classes=True),
        )

        # Some explainers (like Linear) don't have interaction values
        if hasattr(self._shap.explainer, "shap_interaction_values"):
            explainer.set_shap_interaction_values(self._shap.get_interaction_values(X))

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
            self.T.log("Dashboard successfully saved.", 1)

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
            scoring = dict(self.T._metric)

        self.T.log("Applying cross-validation...", 1)

        # Monkey patch the _score function to allow for
        # pipelines that drop samples during transformation
        with patch("sklearn.model_selection._validation._score", score(_score)):
            branch = self.T._get_og_branches()[0]
            self.cv = cross_validate(
                estimator=self.export_pipeline(verbose=0),
                X=branch.X,
                y=branch.y,
                scoring=scoring,
                return_train_score=kwargs.pop("return_train_score", True),
                n_jobs=kwargs.pop("n_jobs", self.T.n_jobs),
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

    @composed(crash, method_to_log)
    def delete(self):
        """Delete the model.

        If it's the last model in atom, the metric is reset. Use this
        method to drop unwanted models from the pipeline or to free
        some memory before saving. The model is not removed from any
        active mlflow experiment.

        """
        self.T.delete(self.name)

    @composed(crash, typechecked)
    def evaluate(
        self,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        threshold: FLOAT = 0.5,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
    ) -> pd.Series:
        """Get the model's scores for the provided metrics.

        Parameters
        ----------
        metric: str, func, scorer, sequence or None, default=None
            Metrics to calculate. If None, a selection of the most
            common metrics per task are used.

        dataset: str, default="test"
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

        threshold: float, default=0.5
            Threshold between 0 and 1 to convert predicted probabilities
            to class labels. Only used when:

            - The task is binary classification.
            - The model has a `predict_proba` method.
            - The metric evaluates predicted target values.

        sample_weight: sequence or None, default=None
            Sample weights corresponding to y in `dataset`.

        Returns
        -------
        pd.Series
            Scores of the model.

        """
        if not 0 < threshold < 1:
            raise ValueError(
                "Invalid value for the threshold parameter. Value "
                f"should lie between 0 and 1, got {threshold}."
            )

        if dataset.lower() not in ("train", "test", "holdout"):
            raise ValueError(
                "Unknown value for the dataset parameter. "
                "Choose from: train, test or holdout."
            )

        # Predefined metrics to show
        if metric is None:
            if self.T.task.startswith("bin"):
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
            elif self.T.task.startswith("multi"):
                metric = [
                    "ba",
                    "f1_weighted",
                    "jaccard_weighted",
                    "mcc",
                    "precision_weighted",
                    "recall_weighted",
                ]
            else:
                metric = ["mae", "mape", "me", "mse", "msle", "r2", "rmse"]

        scores = pd.Series(name=self.name, dtype=float)
        for met in lst(metric):
            scorer = get_custom_scorer(met)
            scores[scorer.name] = self._get_score(
                scorer=scorer,
                dataset=dataset.lower(),
                threshold=threshold,
                sample_weight=sample_weight,
            )

        return scores

    @composed(crash, typechecked)
    def export_pipeline(
        self,
        memory: Optional[Union[bool, str, Memory]] = None,
        verbose: Optional[INT] = None,
    ) -> Pipeline:
        """Export the model's pipeline to a sklearn-like object.

        If the model used feature scaling, the Scaler is added
        before the model. The returned pipeline is already fitted
        on the training set.

        Parameters
        ----------
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
            Current branch as a sklearn-like Pipeline object.

        """
        steps = []
        for transformer in self.pipeline:
            est = deepcopy(transformer)  # Not clone to keep fitted

            # Set the new verbosity (if possible)
            if verbose is not None and hasattr(est, "verbose"):
                est.verbose = verbose

            steps.append((get_pl_name(est.__class__.__name__, steps), est))

        if self.scaler:
            steps.append(("scaler", deepcopy(self.scaler)))

        steps.append((self.name, deepcopy(self.estimator)))

        if not memory:  # None or False
            memory = None
        elif memory is True:
            memory = tempfile.gettempdir()

        return Pipeline(steps, memory=memory)  # ATOM's pipeline, not sklearn

    @composed(crash, method_to_log, typechecked)
    def full_train(self, include_holdout: bool = False):
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
        if include_holdout and self.T.holdout is None:
            raise ValueError("No holdout data set available.")

        if include_holdout and self.T.holdout is not None:
            X = pd.concat([self.X, self.X_holdout])
            y = pd.concat([self.y, self.y_holdout])
        else:
            X, y = self.X, self.y

        # Start a new mlflow run for the new estimator
        if self._run:
            self._run = mlflow.start_run(run_name=f"{self.name}_full_train")

        self.fit(X, y)

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
        of use to, for example, inversely transform only the target
        column. If called from a model that used automated feature
        scaling, the scaling is inversed as well.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Transformed feature set with shape=(n_samples, n_features).
            If None, X is ignored in the transformers.

        y: int, str, dict, sequence or None, default=None
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
        X, y = self.T._prepare_input(X, y)

        # Inversely scale the data if needed
        if self.scaler:
            X, y = custom_transform(self.scaler, self.branch, (X, y), verbose)

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

    @composed(crash, method_to_log, typechecked)
    def save_estimator(self, filename: str = "auto"):
        """Save the estimator to a pickle file.

        Parameters
        ----------
        filename: str, default=None
            Name of the file. Use "auto" for automatic naming.

        """
        if filename.endswith("auto"):
            filename = filename.replace("auto", self.estimator.__class__.__name__)

        with open(filename, "wb") as f:
            pickle.dump(self.estimator, f)

        self.T.log(f"{self.fullname} estimator successfully saved.", 1)

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
        of use to, for example, transform only the target column. If
        called from a model that used automated feature scaling, the
        data is scaled as well.

        Parameters
        ----------
        X: dataframe-like or None, default=None
            Feature set with shape=(n_samples, n_features). If None,
            X is ignored. If None,
            X is ignored in the transformers.

        y: int, str, dict, sequence or None, default=None
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

        y: pd.Series
            Transformed target column. Only returned if provided.

        """
        X, y = self.T._prepare_input(X, y)

        for transformer in self.pipeline:
            if not transformer._train_only:
                X, y = custom_transform(transformer, self.branch, (X, y), verbose)

        # Scale the data if needed
        if self.scaler:
            X, y = custom_transform(self.scaler, self.branch, (X, y), verbose)

        return variable_return(X, y)

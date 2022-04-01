# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseModel class.

"""

import os
import tempfile
from copy import deepcopy
from datetime import datetime
from inspect import Parameter, signature
from pickle import PickleError
from typing import Optional, Union
from unittest.mock import patch

import dill
import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.memory import Memory
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit,
)
from sklearn.model_selection._validation import _score, cross_validate
from sklearn.utils import resample
from skopt.optimizer import (
    base_minimize, forest_minimize, gbrt_minimize, gp_minimize,
)
from skopt.space.space import Categorical, check_dimension
from skopt.space.transformers import LabelEncoder
from tqdm import tqdm
from typeguard import typechecked

from atom.data_cleaning import Scaler
from atom.patches import fit, inverse_transform, score, transform
from atom.pipeline import Pipeline
from atom.plots import BaseModelPlotter
from atom.utils import (
    DF_ATTRS, FLOAT, INT, SEQUENCE_TYPES, X_TYPES, Y_TYPES, CustomDict,
    ShapExplanation, Table, arr, composed, crash, custom_transform, flt,
    get_best_score, get_custom_scorer, get_pl_name, it, lst, merge,
    method_to_log, time_to_str, variable_return,
)


class BaseModel(BaseModelPlotter):
    """Base class for all models."""

    def __init__(self, *args, **kwargs):
        self.T = args[0]  # Trainer instance
        self.name = self.acronym if len(args) == 1 else args[1]
        self.scaler = None
        self.estimator = None

        self._run = None  # mlflow run (if experiment is active)
        self._group = self.name  # sh and ts models belong to the same group
        self._pred = [None] * 15
        self._scores = CustomDict(
            train=CustomDict(),
            test=CustomDict(),
            holdout=CustomDict(),
        )
        self._shap = ShapExplanation(self)

        # BO attributes
        self._iter = 0
        self._stopped = ("---", "---")
        self._early_stopping = None
        self._dimensions = []
        self.bo = pd.DataFrame(
            columns=["call", "params", "estimator", "score", "time", "total_time"],
        )

        # Parameter attributes
        self._n_calls = 0
        self._n_initial_points = 5
        self._est_params = {}
        self._est_params_fit = {}
        self._n_bootstrap = 0

        # Results attributes
        self.best_call = None
        self.best_params = None
        self.metric_bo = None
        self.time_bo = None
        self.time_fit = None
        self.metric_bootstrap = None
        self.mean_bootstrap = None
        self.std_bootstrap = None
        self.time_bootstrap = None
        self.time = None

        # Skip this (slower) part if not called for the estimator
        if not kwargs.get("fast_init"):
            self.branch = self.T.branch
            self._train_idx = len(self.branch._idx[0])  # Can change for sh and ts
            if getattr(self, "needs_scaling", None) and self.T.scaled is False:
                self.scaler = Scaler().fit(self.X_train)

    def __repr__(self):
        out_1 = f"{self.fullname}\n --> Estimator: {self.estimator.__class__.__name__}"
        out_2 = [
            f"{m.name}: {round(get_best_score(self, i), 4)}"
            for i, m in enumerate(self.T._metric.values())
        ]
        return out_1 + f"\n --> Evaluation: {'   '.join(out_2)}"

    def __getattr__(self, item):
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

    def __contains__(self, item):
        return item in self.dataset

    def __getitem__(self, item):
        if isinstance(item, (str, list)):
            return self.dataset[item]  # Get a subset of the dataset
        else:
            raise TypeError(
                f"'{self.__class__.__name__}' object is "
                "only subscriptable with types str or list."
            )

    @property
    def _gpu(self):
        """Return if the model uses the GPU implementation."""
        return "sklearn" not in self.est_class.__module__

    @property
    def _dims(self):
        """Get the names of the hyperparameter dimension space."""
        return [d.name for d in self._dimensions]

    def _sign(self, method="__init__"):
        """Get the estimator's parameters."""
        return signature(getattr(self.est_class, method)).parameters

    def _check_est_params(self):
        """Make sure the parameters are valid keyword argument for the estimator."""
        # The parameter is always accepted if the estimator accepts kwargs
        for param in self._est_params:
            if param not in self._sign() and "kwargs" not in self._sign():
                raise ValueError(
                    f"Invalid value for the est_params parameter. "
                    f"Got unknown parameter {param} for estimator "
                    f"{self.get_estimator().__class__.__name__}."
                )

        for param in self._est_params_fit:
            if param not in self._sign("fit") and "kwargs" not in self._sign("fit"):
                raise ValueError(
                    f"Invalid value for the est_params parameter. Got "
                    f"unknown parameter {param} for the fit method of "
                    f"estimator {self.get_estimator().__class__.__name__}."
                )

    def _get_default_params(self):
        """Get the estimator's default parameters for the BO dimensions."""
        x0 = CustomDict()
        for dim in self._dimensions:
            # Special case for MLP, layers are different from default
            if dim.name == "hidden_layer_1":
                x0[dim.name] = 100
            elif dim.name.startswith("hidden_layer"):
                x0[dim.name] = 0
            elif dim.name in self._sign():
                if self._sign()[dim.name].default is not Parameter.empty:
                    x0[dim.name] = self._sign()[dim.name].default
            else:
                # Return random value in dimension if it's not possible to
                # extract a default value. This can happen when the value
                # is not present in the parameter list (with kwargs) or when
                # the parameter has no default value
                x0[dim.name] = dim.rvs(1, random_state=self.T.random_state)[0]
                self.T.log(
                    " --> Couldn't find a default value for parameter "
                    f"{dim.name}. Using a random initialization.", 2
                )

        # If default value isn't in dimension space, get a random value
        for (name, value), dimension in zip(x0.items(), self._dimensions):
            try:
                is_valid = value in dimension
            except TypeError:  # Can fail when e.g. checking None in Integer
                is_valid = False
            finally:
                if not is_valid:
                    x0[name] = dimension.rvs(1, random_state=self.T.random_state)[0]
                    self.T.log(
                        f" --> The default value of parameter {name} doesn't lie "
                        "within the dimension space. Using a random initialization", 2
                    )

        return x0

    def _get_param(self, params, parameter):
        """Get the estimator's parameter from est_params or BO."""
        return params.get(parameter) or self._est_params.get(parameter)

    def _get_early_stopping_rounds(self, params, max_iter):
        """Get the number of rounds for early stopping."""
        if "early_stopping_rounds" in params:
            return params.pop("early_stopping_rounds")
        elif not self._early_stopping or self._early_stopping >= 1:  # None or int
            return self._early_stopping
        elif self._early_stopping < 1:
            return int(max_iter * self._early_stopping)

    def get_parameters(self, x):
        """Get a dictionary of the model's hyperparameters."""
        return CustomDict(
            {
                p: round(v, 4) if np.issubdtype(type(v), np.floating) else v
                for p, v in zip([d.name for d in self._dimensions], x)
            }
        )

    def get_estimator(self, **params):
        """Return the model's estimator with unpacked parameters."""
        for param in ("n_jobs", "random_state"):
            if param in self._sign():
                params[param] = params.pop(param, getattr(self.T, param))

        return self.est_class(**params)

    def bayesian_optimization(self):
        """Run the bayesian optimization algorithm.

        Search for the best combination of hyperparameters. The
        function to optimize is evaluated either with a K-fold
        cross-validation on the training set or using a different
        split for train and validation set every iteration.

        """

        def optimize(**params):
            """Optimization function for the BO.

            Parameters
            ----------
            params: dict
               Model's hyperparameters used in this call of the BO.

            Returns
            -------
            float
                Score achieved by the model.

            """

            def fit_model(train_idx, val_idx):
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

                if hasattr(self, "custom_fit"):
                    self.custom_fit(
                        est=est,
                        train=(X_subtrain, y_subtrain),
                        validation=(X_val, y_val),
                        **est_copy,
                    )
                else:
                    est.fit(arr(X_subtrain), y_subtrain, **est_copy)

                # Calculate metrics on the validation set
                return [m(est, arr(X_val), y_val) for m in self.T._metric.values()]

            # Start iteration ====================================== >>

            t_iter = datetime.now()  # Get current time for start of the iteration

            # Start printing the information table
            self._iter += 1
            if self._iter > self._n_initial_points:
                call = f"Iteration {self._iter}"
            else:
                call = f"Initial point {self._iter}"

            if pbar:
                pbar.set_description(call)

            # Get estimator instance with call specific hyperparameters
            est = self.get_estimator(**{**self._est_params, **params})

            # Get original branch to define subsets
            og = self.T._get_og_branches()[0]

            # Skip if the eval function has already been evaluated at this point
            if params not in self.bo["params"].values:
                # Same splits per model, but different per call
                rs = self._iter
                if self.T.random_state is not None:
                    rs += self.T.random_state

                if self.T._bo["cv"] == 1:
                    if self.T.goal == "class":
                        split = StratifiedShuffleSplit  # Keep % of samples per class
                    else:
                        split = ShuffleSplit

                    # Get the ShuffleSplit cross-validator object
                    fold = split(
                        n_splits=1,
                        test_size=len(og.test) / og.shape[0],
                        random_state=rs,
                    )

                    # Fit model just on the one fold
                    score = fit_model(*next(fold.split(og.X_train, og.y_train)))

                else:  # Use cross validation to get the score
                    if self.T.goal == "class":
                        fold = StratifiedKFold  # Keep % of samples per class
                    else:
                        fold = KFold

                    # Get the K-fold cross-validator object
                    k_fold = fold(self.T._bo["cv"], shuffle=True, random_state=rs)

                    try:
                        # Parallel loop over fit_model
                        jobs = Parallel(self.T.n_jobs)(
                            delayed(fit_model)(i, j)
                            for i, j in k_fold.split(og.X_train, og.y_train)
                        )
                        score = list(np.mean(jobs, axis=0))
                    except PickleError:
                        raise PickleError(
                            f"Could not pickle the {self.acronym} model to send "
                            "it to the workers. Try using one of the predefined "
                            "models, or use n_jobs=1 or bo_params={'cv': 1}."
                        )
            else:
                # Get same score as previous evaluation
                score = lst(self.bo.loc[self.bo["params"] == params, "score"].values[0])
                self._stopped = ("---", "---")

            # Add row to the bo attribute
            row = pd.Series(
                {
                    "call": call,
                    "params": params,
                    "estimator": est,
                    "score": flt(score),
                    "time": time_to_str(t_iter),
                    "total_time": time_to_str(init_bo),
                }
            )
            self.bo.loc[self._iter - 1] = row

            # Save BO calls to experiment as nested runs
            if self._run and self.T.log_bo:
                with mlflow.start_run(run_name=f"{self.name} - {call}", nested=True):
                    mlflow.set_tag("time", row["time"])
                    mlflow.log_params(params)
                    for i, m in enumerate(self.T._metric):
                        mlflow.log_metric(m, score[i])

            # Update the progress bar with one step
            if pbar:
                pbar.update(1)

            # Print output of the BO
            sequence = {"call": call, **{k: v for k, v in params.items()}}
            for i, m in enumerate(self.T._metric.values()):
                sequence.update(
                    {
                        m.name: score[i],
                        f"best_{m.name}": max([lst(s)[i] for s in self.bo.score]),
                    }
                )
            if self._early_stopping and self.T._bo["cv"] == 1:
                sequence.update(
                    {"early_stopping": f"{self._stopped[0]}/{self._stopped[1]}"}
                )
            sequence.update({"time": row["time"], "total_time": row["total_time"]})
            self.T.log(table.print(sequence), 2)

            return -score[0]  # Negative since skopt tries to minimize

        # Running optimization ===================================== >>

        if self._n_calls < self._n_initial_points:
            raise ValueError(
                "Invalid value for the n_calls parameter. Value "
                f"should be >n_initial_points, got {self._n_calls}."
            )

        # Check validity of parameters (not in basetrainer to skip if error)
        self._check_est_params()

        init_bo = datetime.now()  # Track the BO's duration

        self.T.log(f"\n\nRunning BO for {self.fullname}...", 1)

        # Assign proper dimensions format or use predefined
        if self._dimensions:
            # Some models (e.g. OLS) don't have predefined dimensions (and
            # thus no get_dimensions method), but can accept user defined ones
            dims = self.get_dimensions() if hasattr(self, "get_dimensions") else []

            inc, exc = [], []
            for dim in self._dimensions:
                if isinstance(dim, str):
                    # If it's a name, use the predefined dimension
                    try:
                        if dim.startswith("!"):
                            exc.append(next(d.name for d in dims if d.name == dim[1:]))
                        else:
                            inc.append(next(d for d in dims if d.name == dim))
                    except StopIteration:
                        raise ValueError(
                            "Invalid value for the dimensions parameter. Dimension "
                            f"{dim} is not a predefined hyperparameter of the "
                            f"{self.fullname} model. See the model's documentation "
                            "for an overview of the available hyperparameters and "
                            "their dimensions."
                        )
                else:
                    inc.append(check_dimension(dim))

            if inc and exc:
                raise ValueError(
                    "Invalid value for the dimensions parameter. You can either "
                    "include or exclude parameters, not combinations of these."
                )
            elif exc:
                # If dimensions were excluded with `!`, select all but those
                self._dimensions = [d for d in dims if d.name not in exc]
            elif inc:
                self._dimensions = inc
        else:
            self._dimensions = self.get_dimensions()

        # Drop hyperparameter if already defined in est_params
        self._dimensions = [
            d for d in self._dimensions if d.name not in self._est_params
        ]

        # If no hyperparameters to optimize, skip BO
        if not self._dimensions:
            self.T.log(" --> Skipping BO. No hyperparameters to optimize.", 2)
            return

        pbar = None
        if self.T.verbose == 1:
            pbar = tqdm(total=self._n_calls, desc="Initial point 1")

        # If only 1 initial point, use the model's default parameters
        x0 = self._get_default_params() if self._n_initial_points == 1 else {}

        # Initialize the BO table
        headers = [("call", "left")] + self._dims
        for m in self.T._metric.values():
            headers.extend([m.name, "best_" + m.name])
        if self._early_stopping and self.T._bo["cv"] == 1:
            headers.append("early_stopping")
        headers.extend(["time", "total_time"])

        # Define the width op every column in the table
        spaces = [len(str(headers[0]))]
        for dim in self._dimensions:
            # If the dimension has categories, take the mean of the widths
            # Else take the max of 7 (a minimum) and the width of the name
            if hasattr(dim, "categories"):
                options = np.mean([len(str(cat)) for cat in dim.categories], dtype=int)
            else:
                options = 0
            spaces.append(max(7, len(dim.name), options))
        spaces.extend([max(7, len(t)) for t in headers[1 + len(self._dimensions):]])

        table = Table(headers, spaces)
        self.T.log(table.print_header(), 2)
        self.T.log(table.print_line(), 2)

        # Prepare keyword arguments for the optimizer
        bo_kwargs = self.T._bo.copy()  # Don't pop params from trainer
        kwargs = dict(
            func=lambda x: optimize(**self.get_parameters(x)),
            dimensions=self._dimensions,
            n_calls=self._n_calls,
            n_initial_points=self._n_initial_points,
            x0=bo_kwargs.pop("x0", list(x0.values()) or None),
            callback=self.T._bo["callback"],
            n_jobs=bo_kwargs.pop("n_jobs", self.T.n_jobs),
            random_state=bo_kwargs.pop("random_state", self.T.random_state),
            **bo_kwargs["kwargs"],
        )

        # Monkey patch skopt objects to fix bug with str and num in Categorical
        with patch.object(Categorical, "inverse_transform", inverse_transform):
            with patch.object(LabelEncoder, "fit", fit):
                with patch.object(LabelEncoder, "transform", transform):
                    base_estimator = self.T._bo["base_estimator"]
                    if isinstance(base_estimator, str):
                        if base_estimator.lower() == "gp":
                            gp_minimize(**kwargs)
                        elif base_estimator.lower() == "et":
                            forest_minimize(base_estimator="ET", **kwargs)
                        elif base_estimator.lower() == "rf":
                            forest_minimize(base_estimator="RF", **kwargs)
                        elif base_estimator.lower() == "gbrt":
                            gbrt_minimize(**kwargs)
                    else:
                        base_minimize(base_estimator=base_estimator, **kwargs)

        if pbar:
            pbar.close()

        # Get optimal row. If duplicate scores, select the shortest training
        best = self.bo.copy()
        best["score"] = self.bo["score"].apply(lambda x: lst(x)[0])
        best_idx = best.sort_values(["score", "time"], ascending=False).index[0]

        # Optimal call, params and score to attrs
        self.best_call = self.bo.loc[best_idx, "call"]
        self.best_params = self.bo.loc[best_idx, "params"]
        self.metric_bo = self.bo.loc[best_idx, "score"]

        # Save best model (not yet fitted)
        self.estimator = self.get_estimator(**{**self._est_params, **self.best_params})

        # Get the BO duration
        self.time_bo = time_to_str(init_bo)

        # Print results
        self.T.log(f"Bayesian Optimization {'-' * 27}", 1)
        self.T.log(f"Best call --> {self.best_call}", 1)
        self.T.log(f"Best parameters --> {self.best_params}", 1)
        out = [
            f"{m.name}: {round(lst(self.metric_bo)[i], 4)}"
            for i, m in enumerate(self.T._metric.values())
        ]
        self.T.log(f"Best evaluation --> {'   '.join(out)}", 1)
        self.T.log(f"Time elapsed: {self.time_bo}", 1)

    def fit(self):
        """Fit and validate the model."""
        t_init = datetime.now()

        if self.bo.empty:
            self.T.log(f"\n\nResults for {self.fullname}:", 1)
        self.T.log(f"Fit {'-' * 45}", 1)

        # In case the bayesian_optimization method wasn't called
        if self.estimator is None:
            self._check_est_params()
            self.estimator = self.get_estimator(**self._est_params)

        # Fit the selected model on the complete training set
        if hasattr(self, "custom_fit"):
            self.custom_fit(
                est=self.estimator,
                train=(self.X_train, self.y_train),
                validation=(self.X_test, self.y_test),
                **self._est_params_fit,
            )
        else:
            self.estimator.fit(arr(self.X_train), self.y_train, **self._est_params_fit)

        # Save metric scores on complete training and test set
        for metric in self.T._metric.values():
            self._calculate_score(metric, "train")
            self._calculate_score(metric, "test")

        # Print and log results ==================================== >>

        if self._stopped[0] < self._stopped[1] and self.T._bo["cv"] == 1:
            self.T.log(
                f"Early stop at iteration {self._stopped[0]} of {self._stopped[1]}.", 1
            )
        for set_ in ("train", "test"):
            out = [f"{m}: {round(self._scores[set_][m], 4)}" for m in self.T._metric]
            self.T.log(f"T{set_[1:]} evaluation --> {'   '.join(out)}", 1)

        # Get duration and print to log
        self.time_fit = time_to_str(t_init)
        self.T.log(f"Time elapsed: {self.time_fit}", 1)

        # Log parameters, metrics, model and data to mlflow
        if self._run:
            mlflow.set_tags({"fullname": self.fullname, "time": self.time_fit})

            # Save evals for models with in-training evaluation
            if hasattr(self, "evals"):
                zipper = zip(self.evals["train"], self.evals["test"])
                for step, (train, test) in enumerate(zipper):
                    mlflow.log_metric(f"{self.evals['metric']}_train", train, step=step)
                    mlflow.log_metric(f"{self.evals['metric']}_test", test, step=step)

            self._log_to_mlflow()  # Track rest of information

    def bootstrap(self):
        """Apply a bootstrap algorithm.

        Take bootstrapped samples from the training set and test them
        on the test set to get a distribution of the model's results.

        """
        t_init = datetime.now()

        self.metric_bootstrap = []
        for i in range(self._n_bootstrap):
            # Same splits per model, but different for every iteration
            rs = i
            if self.T.random_state is not None:
                rs += self.T.random_state

            # Create stratified samples with replacement
            sample_x, sample_y = resample(
                self.X_train,
                self.y_train,
                replace=True,
                random_state=rs,
                stratify=self.y_train,
            )

            # Clone to not overwrite when fitting
            estimator = clone(self.estimator)

            # Fit on bootstrapped set and predict on the independent test set
            if hasattr(self, "custom_fit"):
                self.custom_fit(estimator, (sample_x, sample_y), **self._est_params_fit)
            else:
                estimator.fit(arr(sample_x), sample_y, **self._est_params_fit)

            self.metric_bootstrap.append(
                flt(
                    [
                        metric(estimator, arr(self.X_test), self.y_test)
                        for metric in self.T._metric.values()
                    ]
                )
            )

        # Separate for multi-metric, transform numpy types to python types
        if len(self.T._metric) == 1:
            self.metric_bootstrap = np.array(self.metric_bootstrap)
            self.mean_bootstrap = np.mean(self.metric_bootstrap, axis=0).item()
            self.std_bootstrap = np.std(self.metric_bootstrap, axis=0).item()
        else:
            self.metric_bootstrap = np.array(self.metric_bootstrap).T
            self.mean_bootstrap = np.mean(self.metric_bootstrap, axis=1).tolist()
            self.std_bootstrap = np.std(self.metric_bootstrap, axis=1).tolist()

        self.T.log(f"Bootstrap {'-' * 39}", 1)
        out = [
            f"{m.name}: {round(lst(self.mean_bootstrap)[i], 4)}"
            f" \u00B1 {round(lst(self.std_bootstrap)[i], 4)}"
            for i, m in enumerate(self.T._metric.values())
        ]
        self.T.log(f"Evaluation --> {'   '.join(out)}", 1)

        self.time_bootstrap = time_to_str(t_init)
        self.T.log(f"Time elapsed: {self.time_bootstrap}", 1)

    # Utility properties =========================================== >>

    @property
    def results(self):
        """Overview of the training results."""
        return pd.Series(
            {
                "metric_bo": getattr(self, "metric_bo", None),
                "time_bo": getattr(self, "time_bo", None),
                "metric_train": getattr(self, "metric_train", None),
                "metric_test": getattr(self, "metric_test", None),
                "time_fit": getattr(self, "time_fit", None),
                "mean_bootstrap": getattr(self, "mean_bootstrap", None),
                "std_bootstrap": getattr(self, "std_bootstrap", None),
                "time_bootstrap": getattr(self, "time_bootstrap", None),
                "time": getattr(self, "time", None),
            },
            name=self.name,
        )

    @property
    def metric_train(self):
        """Metric scores on the training set."""
        return flt([self._scores["train"][name] for name in self.T._metric])

    @property
    def metric_test(self):
        """Metric scores on the test set."""
        return flt([self._scores["test"][name] for name in self.T._metric])

    # Prediction methods =========================================== >>

    def _prediction(
        self,
        X,
        y=None,
        metric=None,
        sample_weight=None,
        verbose=None,
        method="predict"
    ):
        """Get predictions on unseen data or rows in the dataset.

        New data is first transformed through the model's pipeline.
        Transformers that are only applied on the training set are
        skipped. The model should have the provided method.

        Parameters
        ----------
        X: int, str, slice, sequence or dataframe-like
            Index names or positions of rows in the dataset, or unseen
            feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        metric: str, func, scorer or None, optional (default=None)
            Metric to calculate. Choose from any of sklearn's SCORERS,
            a function with signature metric(y_true, y_pred) or a scorer
            object. If None, it returns mean accuracy for classification
            tasks and r2 for regression tasks. Only for method="score".

        sample_weight: sequence or None, optional (default=None)
            Sample weights for the score method.

        verbose: int or None, optional (default=None)
            Verbosity level for the transformers. If None, it uses the
            estimator's own verbosity.

        method: str, optional (default="predict")
            Prediction method to be applied to the estimator.

        Returns
        -------
        float or np.array
            Calculated prediction.

        """
        if not hasattr(self.estimator, method):
            raise AttributeError(
                f"{self.estimator.__class__.__name__} doesn't have a {method} method!"
            )

        try:  # Return from prediction attributes
            # Raises ValueError if X doesn't select indices
            rows = self.T._get_rows(X, branch=self.branch)

            indices = list(self.dataset.index)
            pred = np.concatenate(
                [getattr(self, f"{method}_train"), getattr(self, f"{method}_test")]
            )

            # If there is a holdout set, add its indices and predictions
            if self.holdout is not None:
                indices += list(self.holdout.index)
                pred = np.concatenate([pred, getattr(self, f"{method}_holdout")])

            # Since the pred attrs don't have an index, we need
            # to know the index positions of the requested rows
            return flt(pred[[indices.index(i) for i in rows]])

        except ValueError:  # Calculate new predictions
            # When there is a pipeline, apply transformations first
            for transformer in self.pipeline:
                if not transformer._train_only:
                    X, y = custom_transform(transformer, self.branch, (X, y), verbose)

            # Scale the data if needed
            if self.scaler:
                X = self.scaler.transform(X)

            if y is None:
                return getattr(self.estimator, method)(X)
            else:
                if metric is None:
                    if self.T.goal == "class":
                        metric = get_custom_scorer("accuracy")
                    else:
                        metric = get_custom_scorer("r2")
                else:
                    metric = get_custom_scorer(metric)

                return metric(self.estimator, X, y, sample_weight)

    @composed(crash, method_to_log, typechecked)
    def predict(
        self,
        X: Union[slice, X_TYPES, Y_TYPES],
        verbose: Optional[INT] = None,
    ):
        """Get predictions on new data."""
        return self._prediction(X, verbose=verbose, method="predict")

    @composed(crash, method_to_log, typechecked)
    def predict_proba(
        self,
        X: Union[slice, X_TYPES, Y_TYPES],
        verbose: Optional[INT] = None,
    ):
        """Get probability predictions on new data."""
        return self._prediction(X, verbose=verbose, method="predict_proba")

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(
        self,
        X: Union[slice, X_TYPES, Y_TYPES],
        verbose: Optional[INT] = None,
    ):
        """Get log probability predictions on new data."""
        return self._prediction(X, verbose=verbose, method="predict_log_proba")

    @composed(crash, method_to_log, typechecked)
    def decision_function(
        self,
        X: Union[slice, X_TYPES, Y_TYPES],
        verbose: Optional[INT] = None,
    ):
        """Get the decision function on new data."""
        return self._prediction(X, verbose=verbose, method="decision_function")

    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: X_TYPES,
        y: Y_TYPES,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        verbose: Optional[INT] = None,
    ):
        """Get the score function on new data."""
        return self._prediction(
            X=X,
            y=y,
            metric=metric,
            sample_weight=sample_weight,
            verbose=verbose,
            method="score",
        )

    # Prediction properties ======================================== >>

    @property
    def predict_train(self):
        if self._pred[0] is None:
            self._pred[0] = self.estimator.predict(arr(self.X_train))
        return self._pred[0]

    @property
    def predict_test(self):
        if self._pred[1] is None:
            self._pred[1] = self.estimator.predict(arr(self.X_test))
        return self._pred[1]

    @property
    def predict_holdout(self):
        if self.T.holdout is not None and self._pred[2] is None:
            self._pred[2] = self.estimator.predict(arr(self.X_holdout))
        return self._pred[2]

    @property
    def predict_proba_train(self):
        if self._pred[3] is None:
            self._pred[3] = self.estimator.predict_proba(arr(self.X_train))
        return self._pred[3]

    @property
    def predict_proba_test(self):
        if self._pred[4] is None:
            self._pred[4] = self.estimator.predict_proba(arr(self.X_test))
        return self._pred[4]

    @property
    def predict_proba_holdout(self):
        if self.T.holdout is not None and self._pred[5] is None:
            self._pred[5] = self.estimator.predict_proba(arr(self.X_holdout))
        return self._pred[5]

    @property
    def predict_log_proba_train(self):
        if self._pred[6] is None:
            self._pred[6] = self.estimator.predict_log_proba(arr(self.X_train))
        return self._pred[6]

    @property
    def predict_log_proba_test(self):
        if self._pred[7] is None:
            self._pred[7] = self.estimator.predict_log_proba(arr(self.X_test))
        return self._pred[7]

    @property
    def predict_log_proba_holdout(self):
        if self.T.holdout is not None and self._pred[8] is None:
            self._pred[8] = self.estimator.predict_log_proba(arr(self.X_holdout))
        return self._pred[8]

    @property
    def decision_function_train(self):
        if self._pred[9] is None:
            self._pred[9] = self.estimator.decision_function(arr(self.X_train))
        return self._pred[9]

    @property
    def decision_function_test(self):
        if self._pred[10] is None:
            self._pred[10] = self.estimator.decision_function(arr(self.X_test))
        return self._pred[10]

    @property
    def decision_function_holdout(self):
        if self.T.holdout is not None and self._pred[11] is None:
            self._pred[11] = self.estimator.decision_function(arr(self.X_holdout))
        return self._pred[11]

    @property
    def score_train(self):
        if self._pred[12] is None:
            self._pred[12] = self.estimator.score(arr(self.X_train), self.y_train)
        return self._pred[12]

    @property
    def score_test(self):
        if self._pred[13] is None:
            self._pred[13] = self.estimator.score(arr(self.X_test), self.y_test)
        return self._pred[13]

    @property
    def score_holdout(self):
        if self.T.holdout is not None and self._pred[14] is None:
            self._pred[14] = self.estimator.score(arr(self.X_holdout), self.y_holdout)
        return self._pred[14]

    # Data Properties ============================================== >>

    @property
    def dataset(self):
        return merge(self.X, self.y)

    @property
    def train(self):
        return merge(self.X_train, self.y_train)

    @property
    def test(self):
        return merge(self.X_test, self.y_test)

    @property
    def holdout(self):
        if self.branch.holdout is not None:
            if self.scaler:
                return merge(
                    self.scaler.transform(self.branch.holdout.iloc[:, :-1]),
                    self.branch.holdout.iloc[:, -1],
                )
            else:
                return self.branch.holdout

    @property
    def X(self):
        return pd.concat([self.X_train, self.X_test])

    @property
    def y(self):
        return pd.concat([self.y_train, self.y_test])

    @property
    def X_train(self):
        if self.scaler:
            return self.scaler.transform(self.branch.X_train[:self._train_idx])
        else:
            return self.branch.X_train[:self._train_idx]

    @property
    def X_test(self):
        if self.scaler:
            return self.scaler.transform(self.branch.X_test)
        else:
            return self.branch.X_test

    @property
    def X_holdout(self):
        if self.branch.holdout is not None:
            return self.holdout.iloc[:, :-1]

    @property
    def y_train(self):
        return self.branch.y_train[:self._train_idx]

    @property
    def y_holdout(self):
        if self.branch.holdout is not None:
            return self.holdout.iloc[:, -1]

    # Utility methods ============================================== >>

    def _final_output(self):
        """Returns the model's final output as a string."""
        # If bootstrap was used, we use a different format
        if self.mean_bootstrap is None:
            out = "   ".join(
                [
                    f"{m.name}: {round(lst(self.metric_test)[i], 4)}"
                    for i, m in enumerate(self.T._metric.values())
                ]
            )
        else:
            out = "   ".join(
                [
                    f"{m.name}: {round(lst(self.mean_bootstrap)[i], 4)} "
                    f"\u00B1 {round(lst(self.std_bootstrap)[i], 4)}"
                    for i, m in enumerate(self.T._metric.values())
                ]
            )

        # Annotate if model overfitted when train 20% > test
        score_train = self._scores["train"][list(self.T._metric.keys())[0]]
        score_test = self._scores["test"][list(self.T._metric.keys())[0]]
        if score_train - 0.2 * score_train > score_test:
            out += " ~"

        return out

    def _log_to_mlflow(self):
        """Log the model's information to the current mlflow run."""
        mlflow.set_tag("branch", self.branch.name)

        # Only save params for children of BaseEstimator
        if hasattr(self.estimator, "get_params"):
            # Mlflow only accepts params with char length <250
            pars = self.estimator.get_params()
            mlflow.log_params({k: v for k, v in pars.items() if len(str(v)) <= 250})

        if self.T.log_model:
            mlflow.sklearn.log_model(self.estimator, self.estimator.__class__.__name__)

        if self.T.log_data:
            for set_ in ("train", "test"):
                getattr(self, set_).to_csv(f"{set_}.csv")
                mlflow.log_artifact(f"{set_}.csv")
                os.remove(f"{set_}.csv")

        if self.T.log_pipeline:
            pl = self.export_pipeline()
            mlflow.sklearn.log_model(pl, f"pipeline_{self.name}")

    def _calculate_score(self, scorer, dataset, threshold=0.5, sample_weight=None):
        """Calculate a metric score using the prediction attributes."""
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
            y_pred = y_pred[:, 1]

            # Exclude metrics that use probability estimates (e.g. ap, auc)
            if scorer.__class__.__name__ == "_PredictScorer":
                y_pred = (y_pred > threshold).astype("int")

        if "sample_weight" in signature(scorer._score_func).parameters:
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

        self._scores[dataset][scorer.name] = scorer._sign * float(score)

        if self._run:  # Log metric to mlflow run
            MlflowClient().log_metric(
                run_id=self._run.info.run_id,
                key=f"{scorer.name}_{dataset}",
                value=it(self._scores[dataset][scorer.name]),
            )

        return self._scores[dataset][scorer.name]

    @composed(crash, method_to_log)
    def calibrate(self, **kwargs):
        """Calibrate the model.

        Applies probability calibration on the model. The estimator
        is trained via cross-validation on a subset of the training
        data, using the rest to fit the calibrator. The new classifier
        will replace the `estimator` attribute. If there is an active
        mlflow experiment, a new run is started using the name
        `[model_name]_calibrate`. Since the estimator changed, the
        model is cleared. Only if classifier.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for sklearn's CCV. Using
            cv="prefit" will use the trained model and fit the
            calibrator on the test set. Use this only if you have
            another, independent set for testing.

        """
        if self.T.goal != "class":
            raise PermissionError(
                "The calibrate method is only available for classification tasks!"
            )

        calibrator = CalibratedClassifierCV(self.estimator, **kwargs)
        if kwargs.get("cv") != "prefit":
            self.estimator = calibrator.fit(self.X_train, self.y_train)
        else:
            self.estimator = calibrator.fit(self.X_test, self.y_test)

        self.clear()  # Clear model since we have a new estimator

        # Start a new mlflow run for the new estimator
        if self._run:
            self._run = mlflow.start_run(run_name=f"{self.name}_calibrate")
            self._log_to_mlflow()
            mlflow.end_run()

        # Save new metric scores on train and test set
        # Also automatically stores scores to the mlflow run
        for metric in self.T._metric.values():
            self._calculate_score(metric, "train")
            self._calculate_score(metric, "test")

        self.T.log(f"Model {self.name} successfully calibrated.", 1)

    @composed(crash, method_to_log)
    def clear(self):
        """Clear attributes from the model.

        Reset attributes to their initial state, deleting potentially
        large data arrays. Use this method to free some memory before
        saving the class. The cleared attributes per model are:
            - Prediction attributes.
            - Metrics scores.
            - Shap values.

        """
        self._pred = [None] * 15
        self._scores = CustomDict(
            train=CustomDict(),
            test=CustomDict(),
            holdout=CustomDict(),
        )
        self._shap = ShapExplanation(self)

    @composed(crash, method_to_log)
    def cross_validate(self, **kwargs):
        """Evaluate the model using cross-validation.

        This method cross-validates the whole pipeline on the complete
        dataset. Use it to assess the robustness of the solution's
        performance.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for sklearn's cross_validate
            function. If the scoring method is not specified, it uses
            the trainer's metric.

        Returns
        -------
        dict
            Return of sklearn's cross_validate function.

        """
        # Assign scoring from predefined or trainer if not specified
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
            cv = cross_validate(
                estimator=self.export_pipeline(verbose=0),
                X=branch.X,
                y=branch.y,
                scoring=scoring,
                n_jobs=kwargs.pop("n_jobs", self.T.n_jobs),
                verbose=kwargs.pop("verbose", 0),
                **kwargs,
            )

        # Output mean and std of metric
        mean = [np.mean(cv[f"test_{m}"]) for m in scoring]
        std = [np.std(cv[f"test_{m}"]) for m in scoring]

        out = "   ".join(
            [
                f"{name}: {round(mean[i], 4)} " f"\u00B1 {round(std[i], 4)}"
                for i, name in enumerate(scoring)
            ]
        )

        self.T.log(f"{self.fullname} --> {out}", 1)

        return cv

    @composed(crash, typechecked, method_to_log)
    def dashboard(
        self,
        dataset: str = "test",
        filename: Optional[str] = None,
        **kwargs,
    ):
        """Create an interactive dashboard to analyze the model.

        The dashboard allows you to investigate SHAP values, permutation
        importances, interaction effects, partial dependence plots, all
        kinds of performance plots, and even individual decision trees.
        By default, the dashboard opens in an external dash app.

        Parameters
        ----------
        dataset: str, optional (default="test")
            Data set to get the report from. Choose from: "train", "test",
            "both" (train and test) or "holdout".

        filename: str or None, optional (default=None)
            Name to save the file with (as .html). None to not save
            anything.

        **kwargs
            Additional keyword arguments for the ExplainerDashboard
            instance.

        Returns
        -------
        ExplainerDashboard
            Created dashboard object.

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
                    "data set was specified when initializing the trainer."
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

        dashboard = ExplainerDashboard(
            explainer=explainer,
            mode=kwargs.pop("mode", "external"),
            **kwargs,
        )
        dashboard.run()

        if filename:
            if not filename.endswith(".html"):
                filename += ".html"
            dashboard.save_html(filename)
            self.T.log("Dashboard successfully saved.", 1)

        return dashboard

    @composed(crash, method_to_log)
    def delete(self):
        """Delete the model from the trainer.

        If it's the last model in the trainer, the metric is reset.
        Use this method to drop unwanted models from the pipeline or
        to free some memory before saving. The model is not removed
        from any active mlflow experiment.

        """
        self.T.delete(self.name)

    @composed(crash, typechecked)
    def evaluate(
        self,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
        threshold: FLOAT = 0.5,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
    ):
        """Get the model's scores for the provided metrics.

        Parameters
        ----------
        metric: str, func, scorer, sequence or None, optional (default=None)
            Metrics to calculate. If None, a selection of the most
            common metrics per task are used.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

        threshold: float, optional (default=0.5)
            Threshold between 0 and 1 to convert predicted probabilities
            to class labels. Only used when:
                - The task is binary classification.
                - The model has a `predict_proba` method.
                - The metric evaluates predicted target values.

        sample_weight: sequence or None, optional (default=None)
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

        dataset = dataset.lower()
        if dataset not in ("train", "test", "holdout"):
            raise ValueError(
                "Unknown value for the dataset parameter. "
                "Choose from: train, test or holdout."
            )
        if dataset == "holdout" and self.T.holdout is None:
            raise ValueError(
                "Invalid value for the dataset parameter. No holdout "
                "data set was specified when initializing the trainer."
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

            # Skip if the scorer has already been calculated
            if (
                scorer.name in self._scores[dataset]
                and threshold == 0.5
                and sample_weight is None
            ):
                scores[scorer.name] = self._scores[dataset][scorer.name]
            else:
                scores[scorer.name] = self._calculate_score(
                    scorer=scorer,
                    dataset=dataset,
                    threshold=threshold,
                    sample_weight=sample_weight,
                )

        return scores

    @composed(crash, typechecked)
    def export_pipeline(
        self,
        memory: Optional[Union[bool, str, Memory]] = None,
        verbose: Optional[INT] = None,
    ):
        """Export the model's pipeline to a sklearn-like object.

        If the model used feature scaling, the Scaler is added
        before the model. The returned pipeline is already fitted
        on the training set.

        Parameters
        ----------
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

        Parameters
        ----------
        include_holdout: bool, optional (default=False)
            Whether to include the holdout set (if available) in the
            training of the estimator. It's discouraged to use this
            option since it means the model can no longer be evaluated
            on any set.

        """
        if include_holdout and self.T.holdout is not None:
            X = pd.concat([self.X, self.X_holdout])
            y = pd.concat([self.y, self.y_holdout])
        else:
            X, y = self.X, self.y

        if hasattr(self, "custom_fit"):
            self.custom_fit(self.estimator, (X, y), **self._est_params_fit)
        else:
            self.estimator.fit(arr(X), y, **self._est_params_fit)

        self.clear()  # Clear model since we have a new estimator

        # Start a new mlflow run for the new estimator
        if self._run:
            self._run = mlflow.start_run(run_name=f"{self.name}_full_train")
            self._log_to_mlflow()
            mlflow.end_run()

        # Save new metric scores on train and test set
        # Also automatically stores scores to the mlflow run
        for metric in self.T._metric.values():
            self._calculate_score(metric, "train")
            self._calculate_score(metric, "test")

        self.T.log(f"Model {self.name} successfully retrained.", 1)

    @composed(crash, method_to_log, typechecked)
    def rename(self, name: Optional[str] = None):
        """Change the model's tag.

        The acronym always stays at the beginning of the model's name.
        If the model is being tracked by mlflow, the name of the
        corresponding run is also changed.

        Parameters
        ----------
        name: str or None, optional (default=None)
            New tag for the model. If None, the tag is removed.

        """
        if not name:
            name = self.acronym  # Back to default acronym
        else:
            # Drop the acronym if not provided by the user
            if name.lower().startswith(self.acronym.lower()):
                name = name[len(self.acronym):]

            # Add the acronym (with right capitalization)
            name = self.acronym + name

        # Check if the name is available
        if name.lower() in map(str.lower, self.T._models):
            raise PermissionError(f"There already exists a model named {name}!")

        # Replace the model in the _models attribute
        self.T._models.replace_key(self.name, name)
        self.T.log(f"Model {self.name} successfully renamed to {name}.", 1)
        self.name = name

        if self._run:  # Change name in mlflow's run
            MlflowClient().set_tag(self._run.info.run_id, "mlflow.runName", self.name)

    @composed(crash, method_to_log, typechecked)
    def save_estimator(self, filename: str = "auto"):
        """Save the estimator to a pickle file.

        Parameters
        ----------
        filename: str, optional (default=None)
            Name of the file. Use "auto" for automatic naming.

        """
        if filename.endswith("auto"):
            filename = filename.replace("auto", self.estimator.__class__.__name__)

        with open(filename, "wb") as f:
            dill.dump(self.estimator, f)

        self.T.log(f"{self.fullname} estimator successfully saved.", 1)

    @composed(crash, method_to_log, typechecked)
    def transform(
        self,
        X: X_TYPES,
        y: Optional[Y_TYPES] = None,
        verbose: Optional[INT] = None,
    ):
        """Transform new data through the model's branch.

        Transformers that are only applied on the training set are
        skipped. If the model used feature scaling, the data is also
        scaled.

        Parameters
        ----------
        X: dataframe-like
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
        pd.DataFrame
            Transformed feature set.

        y: pd.Series
            Transformed target column. Only returned if provided.

        """
        for transformer in self.pipeline:
            if not transformer._train_only:
                X, y = custom_transform(transformer, self.branch, (X, y), verbose)

        if self.scaler:
            X = self.scaler.transform(X)

        return variable_return(X, y)

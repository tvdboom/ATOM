# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseModel class.

"""

# Standard packages
import os
import dill
import mlflow
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from inspect import signature
from datetime import datetime
from pickle import PickleError
from typeguard import typechecked
from typing import Optional, Union
from joblib import Parallel, delayed
from joblib.memory import Memory
from mlflow.tracking import MlflowClient

# Sklearn
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

# Others
from skopt.utils import use_named_args
from skopt.optimizer import (
    base_minimize,
    gp_minimize,
    forest_minimize,
    gbrt_minimize,
)

# Own modules
from .data_cleaning import Scaler
from .pipeline import Pipeline
from .plots import BaseModelPlotter
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, DF_ATTRS, flt, lst, it, arr,
    merge, time_to_str, get_best_score, get_scorer, get_pl_name,
    variable_return, custom_transform, composed, crash, method_to_log,
    score_decorator, Table, ShapExplanation,
)


class BaseModel(BaseModelPlotter):
    """Base class for all models."""

    def __init__(self, *args):
        self.T = args[0]  # Trainer instance
        self.name = self.acronym if len(args) == 1 else args[1]
        self.scaler = None
        self.estimator = None
        self._run = None  # mlflow run (if experiment is active)
        self._group = self.name  # sh and ts models belong to the same group
        self._holdout = None
        self._pred = [None] * 15
        self._shap = ShapExplanation(self)

        # BO attributes
        self.params = {}
        self._iter = 0
        self._stopped = ("---", "---")
        self._early_stopping = None
        self._dimensions = []
        self.bo = pd.DataFrame(
            columns=["call", "params", "model", "score", "time", "total_time"],
        )

        # Parameter attributes
        self._n_calls = 0
        self._n_initial_points = 5
        self._est_params = {}
        self._est_params_fit = {}
        self._n_bootstrap = 0

        # Results attributes
        self.best_params = None
        self.metric_bo = None
        self.time_bo = None
        self.metric_train = None
        self.metric_test = None
        self.time_fit = None
        self.metric_bootstrap = None
        self.mean_bootstrap = None
        self.std_bootstrap = None
        self.time_bootstrap = None
        self.time = None

        # Skip if called from FeatureSelector
        if hasattr(self.T, "_branches"):
            self.branch = self.T.branch
            self._train_idx = self.branch.idx[0]  # Can change for sh and ts
            if getattr(self, "needs_scaling", None) and not self.T.scaled:
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

    def _check_est_params(self):
        """Make sure the parameters are valid keyword argument for the estimator."""
        signature_init = signature(self.est_class.__init__).parameters
        signature_fit = signature(self.est_class.fit).parameters

        # The parameter is always accepted if the estimator accepts kwargs
        for param in self._est_params:
            if param not in signature_init and "kwargs" not in signature_init:
                raise ValueError(
                    f"Invalid value for the est_params parameter. "
                    f"Got unknown parameter {param} for estimator "
                    f"{self.get_estimator().__class__.__name__}."
                )

        for param in self._est_params_fit:
            if param not in signature_fit and "kwargs" not in signature_fit:
                raise ValueError(
                    f"Invalid value for the est_params parameter. Got "
                    f"unknown parameter {param} for the fit method of "
                    f"estimator {self.get_estimator().__class__.__name__}."
                )

    def _get_early_stopping_rounds(self, params, max_iter):
        """Get the number of rounds for early stopping."""
        if "early_stopping_rounds" in params:
            return params.pop("early_stopping_rounds")
        elif not self._early_stopping or self._early_stopping >= 1:  # None or int
            return self._early_stopping
        elif self._early_stopping < 1:
            return int(max_iter * self._early_stopping)

    def get_params(self, x):
        """Get a dictionary of the model's hyperparameters.

        For numerical parameters, the second arg in value indicates
        if the number should have decimals and how many.

        Parameters
        ----------
        x: list
            Hyperparameters returned by the BO in order of self.params.

        """
        params = {}
        for i, (key, value) in enumerate(self.params.items()):
            params[key] = round(x[i], value[1]) if value[1] else x[i]

        return params

    def get_init_values(self):
        """Return the default values of the model's hyperparameters."""
        return [value[0] for value in self.params.values()]

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
            score: float
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
                score: float
                    Score of the fitted model on the validation set.

                """
                # Define subsets from original dataset
                X_subtrain = self.T.og.dataset.iloc[train_idx, :-1]
                y_subtrain = self.T.og.dataset.iloc[train_idx, -1]
                X_val = self.T.og.dataset.iloc[val_idx, :-1]
                y_val = self.T.og.dataset.iloc[val_idx, -1]

                # Transform subsets if there is a pipeline
                pl = self.export_pipeline(verbose=0)
                if len(pl) > 1:
                    pl = pl[:-1]  # Drop the estimator
                    X_subtrain, y_subtrain = pl.fit_transform(X_subtrain, y_subtrain)
                    X_val, y_val = pl.transform(X_val, y_val)

                # Match the sample_weights with the length of the subtrain set
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
                        params=est_copy,
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

            est = self.get_estimator({**self._est_params, **params})

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

                    # Select test_size from ATOM or use default of 0.2
                    t_size = self.T._test_size if hasattr(self.T, "_test_size") else 0.2

                    # Get the ShuffleSplit cross-validator object
                    fold = split(1, test_size=t_size, random_state=rs)

                    # Fit model just on the one fold
                    score = fit_model(*next(fold.split(self.X_train, self.y_train)))

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
                            for i, j in k_fold.split(self.X_train, self.y_train)
                        )
                        score = list(np.mean(jobs, axis=0))
                    except PickleError:
                        raise PickleError(
                            f"Could not pickle the {self.acronym} model to send "
                            "it to the workers. Try using one of the predefined "
                            "models or use n_jobs=1 or bo_params={'cv': 1}."
                        )
            else:
                # Get same score as previous evaluation
                score = lst(self.bo.loc[self.bo["params"] == params, "score"].values[0])
                self._stopped = ("---", "---")

            # Append row to the bo attribute
            t = time_to_str(t_iter)
            t_tot = time_to_str(init_bo)
            self.bo = self.bo.append(
                {
                    "call": call,
                    "params": params,
                    "estimator": est,
                    "score": flt(score),
                    "time": t,
                    "total_time": t_tot,
                },
                ignore_index=True,
            )

            # Save BO calls to experiment as nested runs
            if self.T.log_bo:
                with mlflow.start_run(run_name=f"{self.name} - {call}", nested=True):
                    mlflow.set_tag("time", t)
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
            sequence.update({"time": t, "total_time": t_tot})
            self.T.log(table.print(sequence), 2)

            return -score[0]  # Negative since skopt tries to minimize

        # Running optimization ===================================== >>

        if self._n_calls < self._n_initial_points:
            raise ValueError(
                "Invalid value for the n_calls parameter. Value "
                f"should be >n_initial_points, got {self._n_calls}."
            )

        self._check_est_params()  # Check validity of parameters

        init_bo = datetime.now()  # Track the BO's duration

        self.T.log(f"\n\nRunning BO for {self.fullname}...", 1)

        pbar = None
        if self.T.verbose == 1:
            pbar = tqdm(total=self._n_calls, desc="Initial point 1")

        # Drop dimensions from BO if already in est_params
        for param in self._est_params:
            if param in self.params:
                self.params.pop(param)

        # Specify model dimensions
        def pre_defined_hyperparameters(x):
            return optimize(**self.get_params(x))

        # Get custom dimensions (if provided)
        if self._dimensions:
            @use_named_args(self._dimensions)
            def custom_hyperparameters(**x):
                return optimize(**x)

            dimensions = self._dimensions
            func = custom_hyperparameters  # Use custom hyperparameters

        else:  # If there were no custom dimensions, use the default
            dimensions = self.get_dimensions()
            func = pre_defined_hyperparameters  # Default optimization func

        # If no hyperparameters left to optimize, skip BO
        if not dimensions:
            self.T.log(" --> Skipping BO. No hyperparameters found to optimize.", 2)
            return

        # Start with the table output
        sequence = [("call", "left")] + [dim.name for dim in dimensions]
        for m in self.T._metric.values():
            sequence.extend([m.name, "best_" + m.name])
        if self._early_stopping and self.T._bo["cv"] == 1:
            sequence.append("early_stopping")
        sequence.extend(["time", "total_time"])
        table = Table(sequence, [max(7, len(str(text))) for text in sequence])
        self.T.log(table.print_header(), 2)
        self.T.log(table.print_line(), 2)

        # If only 1 initial point, use the model's default parameters
        x0 = None
        if self._n_initial_points == 1 and hasattr(self, "get_init_values"):
            x0 = self.get_init_values()

        # Prepare keyword arguments for the optimizer
        bo_kwargs = self.T._bo.copy()  # Don't pop params from trainer
        kwargs = dict(
            func=func,
            dimensions=dimensions,
            n_calls=self._n_calls,
            n_initial_points=self._n_initial_points,
            x0=bo_kwargs.pop("x0", x0),
            callback=self.T._bo["callback"],
            n_jobs=bo_kwargs.pop("n_jobs", self.T.n_jobs),
            random_state=bo_kwargs.pop("random_state", self.T.random_state),
            **bo_kwargs["kwargs"],
        )

        if isinstance(self.T._bo["base_estimator"], str):
            if self.T._bo["base_estimator"].lower() == "gp":
                optimizer = gp_minimize(**kwargs)
            elif self.T._bo["base_estimator"].lower() == "et":
                optimizer = forest_minimize(base_estimator="ET", **kwargs)
            elif self.T._bo["base_estimator"].lower() == "rf":
                optimizer = forest_minimize(base_estimator="RF", **kwargs)
            elif self.T._bo["base_estimator"].lower() == "gbrt":
                optimizer = gbrt_minimize(**kwargs)
        else:
            optimizer = base_minimize(
                base_estimator=self.T._bo["base_estimator"],
                **kwargs,
            )

        if pbar:
            pbar.close()

        # Optimal parameters found by the BO
        # Return from skopt wrapper to get dict of custom hyperparameter space
        if func is pre_defined_hyperparameters:
            self.best_params = self.get_params(optimizer.x)
        else:

            @use_named_args(dimensions)
            def get_custom_params(**x):
                return x

            self.best_params = get_custom_params(optimizer.x)

        # Optimal call and score found by the BO
        # Drop duplicates in case the best value is repeated through calls
        best = self.bo["score"].apply(lambda x: lst(x)[0]).drop_duplicates().idxmax()
        best_call = self.bo.loc[best, "call"]
        self.metric_bo = self.bo.loc[best, "score"]

        # Save best model (not yet fitted)
        self.estimator = self.get_estimator({**self._est_params, **self.best_params})

        # Get the BO duration
        self.time_bo = time_to_str(init_bo)

        # Print results
        self.T.log(f"\nResults for {self.fullname}:{' ':9s}", 1)
        self.T.log(f"Bayesian Optimization {'-' * 27}", 1)
        self.T.log(f"Best call --> {best_call}", 1)
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
            self.T.log(f"Results for {self.fullname}:", 1)
        self.T.log(f"Fit {'-' * 45}", 1)

        # In case the bayesian_optimization method wasn't called
        if self.estimator is None:
            self._check_est_params()
            self.estimator = self.get_estimator(self._est_params)

        # Fit the selected model on the complete training set
        if hasattr(self, "custom_fit"):
            self.custom_fit(
                est=self.estimator,
                train=(self.X_train, self.y_train),
                validation=(self.X_test, self.y_test),
                params=self._est_params_fit,
            )
        else:
            self.estimator.fit(arr(self.X_train), self.y_train, **self._est_params_fit)

        # Save metric scores on complete training and test set
        self.metric_train = flt(
            [
                metric(self.estimator, arr(self.X_train), self.y_train)
                for metric in self.T._metric.values()
            ]
        )
        self.metric_test = flt(
            [
                metric(self.estimator, arr(self.X_test), self.y_test)
                for metric in self.T._metric.values()
            ]
        )

        # Print and log results ==================================== >>

        if self._stopped[0] < self._stopped[1] and self.T._bo["cv"] == 1:
            self.T.log(
                f"Early stop at iteration {self._stopped[0]} of {self._stopped[1]}.", 1
            )
        for set_ in ("train", "test"):
            out = [
                f"{m.name}: {round(lst(getattr(self, f'metric_{set_}'))[i], 4)}"
                for i, m in enumerate(self.T._metric.values())
            ]
            self.T.log(f"T{set_[1:]} evaluation --> {'   '.join(out)}", 1)

        # Get duration and print to log
        self.time_fit = time_to_str(t_init)
        self.T.log(f"Time elapsed: {self.time_fit}", 1)

        # Log parameters, metrics, model and data to mlflow
        if self._run:
            mlflow.set_tags(
                {
                    "fullname": self.fullname,
                    "branch": self.branch.name,
                    "time": self.time_fit,
                }
            )

            # Only save params for children of BaseEstimator
            if hasattr(self, "get_params"):
                # Mlflow only accepts params with char length <250
                pars = self.estimator.get_params()
                mlflow.log_params({k: v for k, v in pars.items() if len(str(v)) <= 250})

            for i, m in enumerate(self.T._metric):
                mlflow.log_metric(f"{m}_train", lst(self.metric_train)[i])
                mlflow.log_metric(f"{m}_test", lst(self.metric_test)[i])

            # Save evals for models with in-training evaluation
            if hasattr(self, "evals"):
                zipper = zip(self.evals["train"], self.evals["test"])
                for step, (train, test) in enumerate(zipper):
                    mlflow.log_metric(f"{self.evals['metric']}_train", train, step=step)
                    mlflow.log_metric(f"{self.evals['metric']}_test", test, step=step)

            if self.T.log_model:
                name = self.estimator.__class__.__name__
                mlflow.sklearn.log_model(self.estimator, name)

            if self.T.log_data:
                for set_ in ("train", "test"):
                    getattr(self, set_).to_csv(f"{set_}.csv")
                    mlflow.log_artifact(f"{set_}.csv")
                    os.remove(f"{set_}.csv")

            if self.T.log_pipeline:
                pl = self.export_pipeline()
                mlflow.sklearn.log_model(pl, f"pipeline_{self.name}")

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
                self.custom_fit(
                    est=estimator,
                    train=(sample_x, sample_y),
                    validation=None,
                    params=self._est_params_fit,
                )
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
        """Return the results as a pd.Series."""
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

    # Prediction methods =========================================== >>

    def _check_method(self, method):
        """Raise an error if the estimator doesn't have a method."""
        if not hasattr(self.estimator, method):
            raise AttributeError(
                f"{self.estimator.__class__.__name__} doesn't have a {method} method!"
            )

    def _prediction(
        self,
        X,
        y=None,
        metric=None,
        sample_weight=None,
        verbose=None,
        method="predict"
    ):
        """Apply prediction methods on new data.

        First transform the new data and then apply the attribute on
        the best model. The model has to have the provided attribute.

        Parameters
        ----------
        X: dict, list, tuple, np.array, sps.matrix or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

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
        pred: np.ndarray
            Return of the attribute.

        """
        self._check_method(method)

        # When there is a pipeline, apply transformations first
        for est in self.pipeline:
            if not est._train_only:
                X, y = custom_transform(self.T, est, self.branch, (X, y), verbose)

        # Scale the data if needed
        if self.scaler:
            X = self.scaler.transform(X)

        if y is None:
            return getattr(self.estimator, method)(X)
        else:
            if metric is None:
                if self.T.goal == "class":
                    metric = get_scorer("accuracy")
                else:
                    metric = get_scorer("r2")
            else:
                metric = get_scorer(metric)

            kwargs = {}
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight

            return metric(self.estimator, X, y, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, verbose: Optional[int] = None):
        """Get predictions on new data."""
        return self._prediction(X, verbose=verbose, method="predict")

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, verbose: Optional[int] = None):
        """Get probability predictions on new data."""
        return self._prediction(X, verbose=verbose, method="predict_proba")

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, verbose: Optional[int] = None):
        """Get log probability predictions on new data."""
        return self._prediction(X, verbose=verbose, method="predict_log_proba")

    @composed(crash, method_to_log, typechecked)
    def decision_function(self, X: X_TYPES, verbose: Optional[int] = None):
        """Get the decision function on new data."""
        return self._prediction(X, verbose=verbose, method="decision_function")

    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: X_TYPES,
        y: Y_TYPES,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        verbose: Optional[int] = None,
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

    @composed(crash, method_to_log)
    def reset_predictions(self):
        """Clear all the prediction attributes."""
        self._pred = [None] * 15

    @property
    def predict_train(self):
        self._check_method("predict")
        if self._pred[0] is None:
            self._pred[0] = self.estimator.predict(arr(self.X_train))
        return self._pred[0]

    @property
    def predict_test(self):
        self._check_method("predict")
        if self._pred[1] is None:
            self._pred[1] = self.estimator.predict(arr(self.X_test))
        return self._pred[1]

    @property
    def predict_holdout(self):
        self._check_method("predict")
        if self.T.holdout is not None and self._pred[2] is None:
            self._pred[2] = self.estimator.predict(arr(self.X_holdout))
        return self._pred[2]

    @property
    def predict_proba_train(self):
        self._check_method("predict_proba")
        if self._pred[3] is None:
            self._pred[3] = self.estimator.predict_proba(arr(self.X_train))
        return self._pred[3]

    @property
    def predict_proba_test(self):
        self._check_method("predict_proba")
        if self._pred[4] is None:
            self._pred[4] = self.estimator.predict_proba(arr(self.X_test))
        return self._pred[4]

    @property
    def predict_proba_holdout(self):
        self._check_method("predict_proba")
        if self.T.holdout is not None and self._pred[5] is None:
            self._pred[5] = self.estimator.predict_proba(arr(self.X_holdout))
        return self._pred[5]

    @property
    def predict_log_proba_train(self):
        self._check_method("predict_log_proba")
        if self._pred[6] is None:
            self._pred[6] = self.estimator.predict_log_proba(arr(self.X_train))
        return self._pred[6]

    @property
    def predict_log_proba_test(self):
        self._check_method("predict_log_proba")
        if self._pred[7] is None:
            self._pred[7] = self.estimator.predict_log_proba(arr(self.X_test))
        return self._pred[7]

    @property
    def predict_log_proba_holdout(self):
        self._check_method("predict_log_proba")
        if self.T.holdout is not None and self._pred[8] is None:
            self._pred[8] = self.estimator.predict_log_proba(arr(self.X_holdout))
        return self._pred[8]

    @property
    def decision_function_train(self):
        self._check_method("decision_function")
        if self._pred[9] is None:
            self._pred[9] = self.estimator.decision_function(arr(self.X_train))
        return self._pred[9]

    @property
    def decision_function_test(self):
        self._check_method("decision_function")
        if self._pred[10] is None:
            self._pred[10] = self.estimator.decision_function(arr(self.X_test))
        return self._pred[10]

    @property
    def decision_function_holdout(self):
        self._check_method("decision_function")
        if self.T.holdout is not None and self._pred[11] is None:
            self._pred[11] = self.estimator.decision_function(arr(self.X_holdout))
        return self._pred[11]

    @property
    def score_train(self):
        self._check_method("score")
        if self._pred[12] is None:
            self._pred[12] = self.estimator.score(arr(self.X_train), self.y_train)
        return self._pred[12]

    @property
    def score_test(self):
        self._check_method("score")
        if self._pred[13] is None:
            self._pred[13] = self.estimator.score(arr(self.X_test), self.y_test)
        return self._pred[13]

    @property
    def score_holdout(self):
        self._check_method("score")
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
        if self.T.holdout is not None:
            if self._holdout is None:
                self._holdout = merge(
                    *self.transform(
                        X=self.T.holdout.iloc[:, :-1],
                        y=self.T.holdout.iloc[:, -1],
                        verbose=0,
                    )
                )

            return self._holdout

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
        if self.T.holdout is not None:
            return self.holdout.iloc[:, :-1]

    @property
    def y_train(self):
        return self.branch.y_train[:self._train_idx]

    @property
    def y_holdout(self):
        if self.T.holdout is not None:
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
        metric_train = lst(self.metric_train)
        metric_test = lst(self.metric_test)
        if metric_train[0] - 0.2 * metric_train[0] > metric_test[0]:
            out += " ~"

        return out

    @composed(crash, method_to_log)
    def calibrate(self, **kwargs):
        """Calibrate the model.

        Applies probability calibration on the model. The estimator
        is trained via cross-validation on a subset of the training
        data, using the rest to fit the calibrator. The new classifier
        will replace the `estimator` attribute and is logged to any
        active mlflow experiment. Since the estimator changed, all the
        model's prediction attributes are reset.

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

        # Log the CCV to the model's mlflow run
        if self._run and self.T.log_model:
            with mlflow.start_run(self._run.info.run_id):
                mlflow.sklearn.log_model(self.estimator, "CalibratedClassifierCV")

        # Reset attrs dependent on estimator
        self._pred = [None] * 15
        self._shap = ShapExplanation(self)

        self.T.log(f"Model {self.name} successfully calibrated.", 1)

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
        score: dict
            Return of sklearn's cross_validate function.

        """
        from sklearn.model_selection import _validation

        # Adopt params from the trainer if not specified
        kwargs["n_jobs"] = kwargs.get("n_jobs", self.T.n_jobs)
        kwargs["verbose"] = kwargs.get("verbose", self.T.verbose)
        if kwargs.get("scoring"):
            scoring = get_scorer(kwargs.pop("scoring"))
            scoring = {scoring.name: scoring}

        else:
            scoring = dict(self.T._metric)

        # Get the complete pipeline
        pl = self.export_pipeline(verbose=0)

        # Use the untransformed dataset (in the og branch)
        og = getattr(self.T, "og", self.T._current)

        self.T.log("Applying cross-validation...", 1)

        # Workaround the _score function to allow for pipelines
        # that drop samples during transformation
        og_score = deepcopy(_validation._score)
        _validation._score = score_decorator(og_score)

        cv = _validation.cross_validate(pl, og.X, og.y, scoring=scoring, **kwargs)
        _validation._score = og_score  # Reset _score to og value

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

    @composed(crash, method_to_log)
    def delete(self):
        """Delete the model from the trainer."""
        self.T.delete(self.name)

    @composed(crash, typechecked)
    def evaluate(
        self,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
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

        Returns
        -------
        score: pd.Series
            Scores of the model.

        """
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
            scorer = get_scorer(met)
            if scorer.__class__.__name__ == "_ThresholdScorer":
                if hasattr(self.estimator, "decision_function"):
                    y_pred = getattr(self, f"decision_function_{dataset}")
                else:
                    y_pred = getattr(self, f"predict_proba_{dataset}")
                    if self.T.task.startswith("bin"):
                        y_pred = y_pred[:, 1]
            elif scorer.__class__.__name__ == "_ProbaScorer":
                if hasattr(self.estimator, "predict_proba"):
                    y_pred = getattr(self, f"predict_proba_{dataset}")
                    if self.T.task.startswith("bin"):
                        y_pred = y_pred[:, 1]
                else:
                    y_pred = getattr(self, f"decision_function_{dataset}")
            else:
                y_pred = getattr(self, f"predict_{dataset}")

            scores[scorer.name] = scorer._sign * float(
                scorer._score_func(
                    getattr(self, f"y_{dataset}"), y_pred, **scorer._kwargs
                )
            )

            if self._run:  # Log metric to mlflow run
                MlflowClient().log_metric(
                    run_id=self._run.info.run_id,
                    key=f"{scorer.name}_{dataset}",
                    value=it(scores[scorer.name]),
                )

        return scores

    @composed(crash, typechecked)
    def export_pipeline(
        self,
        memory: Optional[Union[bool, str, Memory]] = None,
        verbose: Optional[int] = None,
    ):
        """Export the model's pipeline to a sklearn-like object.

        If the model used feature scaling, the Scaler is added
        before the model. The returned pipeline is already fitted
        on the training set.

        Parameters
        ----------
        memory: bool, str, Memory or None, optional (default=None)
            Used to cache the fitted transformers of the pipeline.
                If None or False: No caching is performed.
                If True: A default temp directory is used.
                If str: Path to the caching directory.

        verbose: int or None, optional (default=None)
            Verbosity level of the transformers in the pipeline. If
            None, it leaves them to their original verbosity. Note
            that this is not the pipeline's own verbose parameter.
            To change that, use the `set_params` method.

        Returns
        -------
        pipeline: Pipeline
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
            memory = Memory(tempfile.gettempdir())

        return Pipeline(steps, memory=memory)  # ATOM's pipeline, not sklearn

    @composed(crash, method_to_log, typechecked)
    def full_train(self, include_holdout: bool = False):
        """Train the estimator on the complete dataset.

        In some cases it might be desirable to use all available
        data to train a final model. Note that doing this means
        that the estimator can no longer be evaluated on the test
        set. The newly retrained estimator will replace the
        `estimator` attribute and is logged to any active mlflow
        experiment. Since the estimator changed, all the model's
        prediction attributes are reset.

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
            self.custom_fit(
                est=self.estimator,
                train=(X, y),
                params=self._est_params_fit,
            )
        else:
            self.estimator.fit(arr(X), y, **self._est_params_fit)

        # Log the new estimator to the model's mlflow run
        if self._run and self.T.log_model:
            with mlflow.start_run(self._run.info.run_id):
                name = f"full_train_{self.estimator.__class__.__name__}"
                mlflow.sklearn.log_model(self.estimator, name)

        # Reset attrs dependent on estimator
        self._pred = [None] * 15
        self._shap = ShapExplanation(self)

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
        self.T._models.insert(self.name, name, self)
        self.T._models.pop(self.name)
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
    def transform(self, X: X_TYPES, y: Y_TYPES = None, verbose: Optional[int] = None):
        """Transform new data through the model's branch.

        Transformers that are only applied on the training set are
        skipped. If the model used feature scaling, the data is also
        scaled.

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
        for est in self.pipeline:
            if not est._train_only:
                X, y = custom_transform(self, est, self.branch, (X, y), verbose)

        if self.scaler:
            X = self.scaler.transform(X)

        return variable_return(X, y)

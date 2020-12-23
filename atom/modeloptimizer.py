# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the ModelOptimizer class.

"""

# Standard packages
import pickle
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
from typeguard import typechecked
from typing import Optional
from inspect import signature

# Sklearn
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# Others
from skopt.utils import use_named_args
from skopt.callbacks import DeadlineStopper, DeltaXStopper, DeltaYStopper
from skopt.optimizer import base_minimize, gp_minimize, forest_minimize, gbrt_minimize

# Own modules
from .data_cleaning import Scaler
from .basemodel import BaseModel
from .plots import SuccessiveHalvingPlotter, TrainSizingPlotter
from .utils import (
    flt, lst, arr, check_scaling, time_to_string, composed,
    get_best_score, crash, method_to_log, PlotCallback,
)


class ModelOptimizer(BaseModel, SuccessiveHalvingPlotter, TrainSizingPlotter):
    """Class for model optimization.

    Contains the Bayesian Optimization, fitting and bagging of the
    models as well as some utility attributes not shared with the
    ensemble classes.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Skip if called from FeatureSelector
        if hasattr(self.T, "_branches"):
            self.branch = self.T._branches[self.T._current]
            self._train_idx = self.branch.idx[0]  # Can change for sh and ts
            if self.needs_scaling and not check_scaling(self.branch.X):
                self.scaler = Scaler().fit(self.branch.X_train)

        # BO attributes
        self._iter = 0
        self._init_bo = None
        self._pbar = None
        self._cv = 5  # Number of cross-validation folds
        self._early_stopping = None
        self._stopped = None
        self.bo = pd.DataFrame(
            columns=["params", "model", "score", "time_iteration", "time"]
        )
        self.bo.index.name = "call"

        # Parameter attributes
        self._est_params = {}
        self._est_params_fit = {}

        # BaseModel attributes
        self.best_params = None
        self.time_fit = None
        self.metric_bo = None
        self.time_bo = None
        self.metric_train = None
        self.metric_test = None
        self.metric_bagging = None
        self.mean_bagging = None
        self.std_bagging = None
        self.time_bagging = None

        # Results
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
        self._results.index.name = "model"

    def __repr__(self):
        out = f"{self.fullname}\n --> Estimator: {self.estimator.__class__.__name__}"
        for i, metric in enumerate(self.T.metric_):
            out += f"\n --> {metric.name}: {get_best_score(self, i)}"

        return out

    @property
    def results(self):
        """Return the results dataframe without empty columns."""
        return self._results.dropna(axis=1, how="all")

    def get_params(self, x):
        """Get a dictionary of the model's hyperparameters.

        Parameters
        ----------
        x: list
            Hyperparameters returned by the BO in order of self.params.

        """
        params = {}
        for i, (key, value) in enumerate(self.params.items()):
            if value[1]:  # If it has decimals...
                params[key] = round(x[i], value[1])
            else:
                params[key] = x[i]

        return params

    def get_init_values(self):
        """Return the default values of the model's hyperparameters."""
        return [value[0] for value in self.params.values()]

    @composed(crash, method_to_log, typechecked)
    def bayesian_optimization(
        self,
        n_calls: int = 15,
        n_initial_points: int = 5,
        bo_params: dict = {},
    ):
        """Run the bayesian optimization algorithm.

        Search for the best combination of hyperparameters. The
        function to optimize is evaluated either with a K-fold
        cross-validation on the training set or using a different
        split for train and validation set every iteration.

        Parameters
        ----------
        n_calls: int or sequence, optional (default=15)
            Maximum number of iterations of the BO. It includes the
            random points of `n_initial_points`. If 0, skip the BO
            and fit the model on its default Parameters. If sequence,
            the n-th value will apply to the n-th model.

        n_initial_points: int or sequence, optional (default=5)
            Initial number of random tests of the BO before fitting
            the surrogate function. If equal to `n_calls`, the
            optimizer will technically be performing a random search.
            If sequence, the n-th value will apply to the n-th model.

        bo_params: dict, optional (default={})
            Additional parameters to for the BO. These can include:
                - base_estimator: str, optional (default="GP")
                    Surrogate model to use. Choose from:
                        - "GP" for Gaussian Process
                        - "RF" for Random Forest
                        - "ET" for Extra-Trees
                        - "GBRT" for Gradient Boosted Regression Trees
                - max_time: int, optional (default=np.inf)
                    Stop the optimization after `max_time` seconds.
                - delta_x: int or float, optional (default=0)
                    Stop the optimization when `|x1 - x2| < delta_x`.
                - delta_y: int or float, optional (default=0)
                    Stop the optimization if the 5 minima are within
                    `delta_y` (skopt always minimizes the function).
                - early_stopping: int, float or None, optional (default=None)
                    Training will stop if the model didn't improve in
                    last `early_stopping` rounds. If <1, fraction of
                    rounds from the total. If None, no early stopping
                    is performed. Only available for models that allow
                    in-training evaluation.
                - cv: int, optional (default=5)
                    Number of folds for the cross-validation. If 1, the
                    training set will be randomly split in a (sub)train
                    and validation set.
                - callbacks: callable or sequence, optional (default=None)
                    Callbacks for the BO.
                - dimensions: dict, sequence or None, optional (default=None)
                    Custom hyperparameter space for the BO. Can be an
                    array to share the same dimensions across models
                    or a dictionary with the model names as key. If
                    None, ATOM's predefined dimensions are used.
                - plot_bo: bool, optional (default=False)
                    Whether to plot the BO's progress as it runs.
                    Creates a canvas with two plots: the first plot
                    shows the score of every trial and the second shows
                    the distance between the last consecutive steps.
                    Don't forget to call `%matplotlib` at the start of
                    the cell if you are using an interactive notebook!
                - Additional keyword arguments for skopt's optimizer.

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

                Divide the training set in a (sub)train and validation
                set for this fit. Fit the model on custom_fit if exists,
                else normally. Return the score on the validation set.

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
                X_subtrain = self.X_train.loc[train_idx]
                y_subtrain = self.y_train.loc[train_idx]
                X_val = self.X_train.loc[val_idx]
                y_val = self.y_train.loc[val_idx]

                # Match the sample_weights with the length of the subtrain set
                # Make copy of est_params to not alter the mutable variable
                est_copy = self._est_params_fit.copy()
                if "sample_weight" in est_copy:
                    est_copy["sample_weight"] = [
                        self._est_params_fit["sample_weight"][i] for i in train_idx
                    ]

                if hasattr(self, "custom_fit"):
                    self.custom_fit(
                        estimator=est,
                        train=(X_subtrain, y_subtrain),
                        validation=(X_val, y_val),
                        params=est_copy
                    )

                    # Alert if early stopping was applied (only for cv=1)
                    if self._cv == 1 and self._stopped:
                        self.T.log(
                            f"Early stop at iteration {self._stopped[0]} "
                            f"of {self._stopped[1]}.", 2
                        )
                else:
                    est.fit(arr(X_subtrain), y_subtrain, **est_copy)

                # Calculate metrics on the validation set
                return [metric(est, arr(X_val), y_val) for metric in self.T.metric_]

            t_iter = time()  # Get current time for start of the iteration

            # Print iteration and time
            self._iter += 1
            if self._iter > n_initial_points:
                call = f"Iteration {self._iter}"
            else:
                call = f"Initial point {self._iter}"

            if self._pbar:
                self._pbar.set_description(call)
            len_ = "-" * (48 - len(call))
            self.T.log(f"{call} {len_}", 2)
            self.T.log(f"Parameters --> {params}", 2)

            est = self.get_estimator({**self._est_params, **params})

            # Same splits per model, but different for every iteration of the BO
            rs = self.T.random_state + self._iter if self.T.random_state else None

            if self._cv == 1:
                # Select test_size from ATOM or use default of 0.2
                t_size = self.T._test_size if hasattr(self.T, "_test_size") else 0.2
                kwargs = dict(test_size=t_size, random_state=rs)
                if self.T.goal.startswith("class"):
                    # Folds are made preserving the % of samples for each class
                    split = StratifiedShuffleSplit(1, **kwargs)
                else:
                    split = ShuffleSplit(1, **kwargs)

                scores = fit_model(*next(split.split(self.X_train, self.y_train)))

            else:  # Use cross validation to get the score
                kwargs = dict(n_splits=self._cv, shuffle=True, random_state=rs)
                if self.T.goal.startswith("class"):
                    # Folds are made preserving the % of samples for each class
                    k_fold = StratifiedKFold(**kwargs)
                else:
                    k_fold = KFold(**kwargs)

                # Parallel loop over fit_model
                jobs = Parallel(self.T.n_jobs)(
                    delayed(fit_model)(i, j)
                    for i, j in k_fold.split(self.X_train, self.y_train)
                )
                scores = list(np.mean(jobs, axis=0))

            # Append row to the bo attribute
            t = time_to_string(t_iter)
            t_tot = time_to_string(self._init_bo)
            self.bo.loc[call] = {
                "params": params,
                "estimator": est,
                "score": flt(scores),
                "time_iteration": t,
                "time": t_tot,
            }

            # Update the progress bar
            if self._pbar:
                self._pbar.update(1)

            # Print output of the BO
            out = [
                f"{m.name}: {scores[i]:.4f}  Best {m.name}: "
                f"{max([lst(s)[i] for s in self.bo.score]):.4f}"
                for i, m in enumerate(self.T.metric_)
            ]
            self.T.log(f"Evaluation --> {'   '.join(out)}", 2)
            self.T.log(f"Time iteration: {t}   Total time: {t_tot}", 2)

            return -scores[0]  # Negative since skopt tries to minimize

        # Running optimization ===================================== >>

        # Check parameters
        if n_initial_points < 1:
            raise ValueError(
                "Invalid value for the n_initial_points parameter. "
                f"Value should be >0, got {n_initial_points}."
            )
        if n_calls < n_initial_points:
            raise ValueError(
                "Invalid value for the n_calls parameter. Value "
                f"should be >n_initial_points, got {n_calls}."
            )

        self.T.log(f"\n\nRunning BO for {self.fullname}...", 1)

        self._init_bo = time()
        if self.T.verbose == 1:
            self._pbar = tqdm(total=n_calls, desc="Random start 1")

        # Prepare callbacks
        callbacks = []
        if bo_params.get("callbacks"):
            callbacks = lst(bo_params["callbacks"])
            bo_params.pop("callbacks")

        if bo_params.get("max_time"):
            if bo_params["max_time"] <= 0:
                raise ValueError(
                    "Invalid value for the max_time parameter. "
                    f"Value should be >0, got {bo_params['max_time']}."
                )
            callbacks.append(DeadlineStopper(bo_params["max_time"]))
            bo_params.pop("max_time")

        if bo_params.get("delta_x"):
            if bo_params["delta_x"] < 0:
                raise ValueError(
                    "Invalid value for the delta_x parameter. "
                    f"Value should be >=0, got {bo_params['delta_x']}."
                )
            callbacks.append(DeltaXStopper(bo_params["delta_x"]))
            bo_params.pop("delta_x")

        if bo_params.get("delta_y"):
            if bo_params["delta_y"] < 0:
                raise ValueError(
                    "Invalid value for the delta_y parameter. "
                    f"Value should be >=0, got {bo_params['delta_y']}."
                )
            callbacks.append(DeltaYStopper(bo_params["delta_y"], n_best=5))
            bo_params.pop("delta_y")

        if "plot_bo" in bo_params:
            if bo_params["plot_bo"]:
                callbacks.append(PlotCallback(self))
            bo_params.pop("plot_bo")

        # Prepare additional arguments
        if bo_params.get("cv"):
            if bo_params["cv"] <= 0:
                raise ValueError(
                    "Invalid value for the max_time parameter. "
                    f"Value should be >=0, got {bo_params['cv']}."
                )
            self._cv = bo_params["cv"]
            bo_params.pop("cv")

        if bo_params.get("early_stopping"):
            if bo_params["early_stopping"] <= 0:
                raise ValueError(
                    "Invalid value for the early_stopping parameter. "
                    f"Value should be >=0, got {bo_params['early_stopping']}."
                )
            self._early_stopping = bo_params["early_stopping"]
            bo_params.pop("early_stopping")

        # Drop dimensions from BO if already in est_params
        for param in self._est_params:
            if param not in signature(self.get_estimator().__init__).parameters:
                raise ValueError(
                    f"Invalid value for the est_params parameter. Got {param} "
                    f"for estimator {self.get_estimator().__class__.__name__}."
                )
            elif param in self.params:
                self.params.pop(param)

        # Specify model dimensions
        def pre_defined_hyperparameters(x):
            return optimize(**self.get_params(x))

        # Get custom dimensions (if provided)
        dimensions = None
        if bo_params.get("dimensions"):
            if bo_params["dimensions"].get(self.name):
                dimensions = bo_params.get("dimensions")[self.name]

                @use_named_args(dimensions)
                def custom_hyperparameters(**x):
                    return optimize(**x)

                func = custom_hyperparameters  # Use custom hyperparameters
            bo_params.pop("dimensions")

        # If there were no custom dimensions, use the default
        if not dimensions:
            dimensions = self.get_dimensions()
            func = pre_defined_hyperparameters  # Default optimization func

        # If only 1 initial point, use the model's default parameters
        if n_initial_points == 1 and hasattr(self, "get_init_values"):
            bo_params["x0"] = self.get_init_values()

        # Choose base estimator (GP is chosen as default)
        base = bo_params.pop("base_estimator", "GP")

        # Prepare keyword arguments for the optimizer
        kwargs = dict(
            func=func,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            callback=callbacks,
            n_jobs=self.T.n_jobs,
            random_state=self.T.random_state,
        )
        kwargs.update(**bo_params)

        if isinstance(base, str):
            if base.lower() == "gp":
                optimizer = gp_minimize(**kwargs)
            elif base.lower() == "et":
                optimizer = forest_minimize(base_estimator="ET", **kwargs)
            elif base.lower() == "rf":
                optimizer = forest_minimize(base_estimator="RF", **kwargs)
            elif base.lower() == "gbrt":
                optimizer = gbrt_minimize(**kwargs)
            else:
                raise ValueError(
                    f"Invalid value for the base_estimator parameter, got {base}."
                    " Value should be one of: 'GP', 'ET', 'RF', 'GBRT'."
                )
        else:
            optimizer = base_minimize(base_estimator=base, **kwargs)

        if self._pbar:
            self._pbar.close()

        # Optimal parameters found by the BO
        # Return from skopt wrapper to get dict of custom hyperparameter space
        if func is pre_defined_hyperparameters:
            self.best_params = self.get_params(optimizer.x)
        else:

            @use_named_args(dimensions)
            def get_custom_params(**x):
                return x

            self.best_params = get_custom_params(optimizer.x)

        # Optimal score found by the BO
        self.metric_bo = self.bo.score.max(axis=0)

        # Save best model (not yet fitted)
        self.estimator = self.get_estimator({**self._est_params, **self.best_params})

        # Get the BO duration
        self.time_bo = time_to_string(self._init_bo)

        # Print results
        self.T.log(f"\nResults for {self.fullname}:{' ':9s}", 1)
        self.T.log("Bayesian Optimization ---------------------------", 1)
        self.T.log(f"Best parameters --> {self.best_params}", 1)
        out = [
            f"{m.name}: {lst(self.metric_bo)[i]:.4f}"
            for i, m in enumerate(self.T.metric_)
        ]
        self.T.log(f"Best evaluation --> {'   '.join(out)}", 1)
        self.T.log(f"Time elapsed: {self.time_bo}", 1)

    @composed(crash, method_to_log)
    def fit(self):
        """Fit to the complete training set and get the score on the test set."""
        t_init = time()

        # In case the bayesian_optimization method wasn't called
        if self.estimator is None:
            self.estimator = self.get_estimator(self._est_params)

        # Fit the selected model on the complete training set
        if hasattr(self, "custom_fit"):
            self.custom_fit(
                estimator=self.estimator,
                train=(self.X_train, self.y_train),
                validation=(self.X_test, self.y_test),
                params=self._est_params_fit
            )
        else:
            self.estimator.fit(arr(self.X_train), self.y_train, **self._est_params_fit)

        # Save metric scores on complete training and test set
        self.metric_train = flt([
            metric(self.estimator, arr(self.X_train), self.y_train)
            for metric in self.T.metric_
        ])
        self.metric_test = flt([
            metric(self.estimator, arr(self.X_test), self.y_test)
            for metric in self.T.metric_
        ])

        # Print stats ============================================== >>

        if self.bo.empty:
            self.T.log("\n", 1)  # Print 2 extra lines
            self.T.log(f"Results for {self.fullname}:{' ':9s}", 1)
        self.T.log("Fit ---------------------------------------------", 1)
        if self._stopped:
            out = f"Early stop at iteration {self._stopped[0]} of {self._stopped[1]}."
            self.T.log(out, 1)
        out_train = [
            f"{m.name}: {lst(self.metric_train)[i]:.4f}"
            for i, m in enumerate(self.T.metric_)
        ]
        self.T.log(f"Train evaluation --> {'   '.join(out_train)}", 1)
        out_test = [
            f"{m.name}: {lst(self.metric_test)[i]:.4f}"
            for i, m in enumerate(self.T.metric_)
        ]
        self.T.log(f"Test evaluation --> {'   '.join(out_test)}", 1)

        # Get duration and print to log
        self.time_fit = time_to_string(t_init)
        self.T.log(f"Time elapsed: {self.time_fit}", 1)

    @composed(crash, method_to_log, typechecked)
    def bagging(self, bagging: int = 5):
        """Apply bagging on the model.

        Take bootstrap samples from the training set and test them on
        the test set to get a distribution of the model's results.

        Parameters
        ----------
        bagging: int, optional (default=5)
            Number of data sets (bootstrapped from the training set)
            to use in the bagging algorithm.

        """
        t_init = time()

        if bagging < 1:
            raise ValueError(
                "Invalid value for the bagging parameter."
                f"Value should be >0, got {bagging}."
            )

        self.metric_bagging = []
        for _ in range(bagging):
            # Create samples with replacement
            sample_x, sample_y = resample(self.X_train, self.y_train)

            # Make a clone to not overwrite when fitting
            estimator = clone(self.estimator)

            # Fit on bootstrapped set and predict on the independent test set
            if hasattr(self, "custom_fit"):
                self.custom_fit(
                    estimator=estimator,
                    train=(sample_x, sample_y),
                    validation=None,
                    params=self._est_params_fit
                )
            else:
                estimator.fit(arr(sample_x), sample_y, **self._est_params_fit)

            scores = flt([
                metric(estimator, arr(self.X_test), self.y_test)
                for metric in self.T.metric_
            ])

            # Append metric result to list
            self.metric_bagging.append(scores)

        # Separate for multi-metric to transform numpy types in python types
        if len(self.T.metric_) == 1:
            self.mean_bagging = np.mean(self.metric_bagging, axis=0).item()
            self.std_bagging = np.std(self.metric_bagging, axis=0).item()
        else:
            self.metric_bagging = list(zip(*self.metric_bagging))
            self.mean_bagging = np.mean(self.metric_bagging, axis=1).tolist()
            self.std_bagging = np.std(self.metric_bagging, axis=1).tolist()

        self.T.log("Bagging -----------------------------------------", 1)
        out = [
            f"{m.name}: {lst(self.mean_bagging)[i]:.4f}"
            " \u00B1 "
            f"{lst(self.std_bagging)[i]:.4f}"
            for i, m in enumerate(self.T.metric_)
        ]
        self.T.log(f"Evaluation --> {'   '.join(out)}", 1)

        # Get duration and print to log
        self.time_bagging = time_to_string(t_init)
        self.T.log(f"Time elapsed: {self.time_bagging}", 1)

    # Utility methods ============================================== >>

    def _final_output(self):
        """Returns the model's final output as a string."""
        # If bagging was used, we use a different format
        if self.mean_bagging is None:
            out = "   ".join([
                f"{m.name}: {lst(self.metric_test)[i]:.3f}"
                for i, m, in enumerate(self.T.metric_)
            ])
        else:
            out = "   ".join([
                f"{m.name}: {lst(self.mean_bagging)[i]:.3f}"
                " \u00B1 "
                f"{lst(self.std_bagging)[i]:.3f}"
                for i, m in enumerate(self.T.metric_)
            ])

        # Annotate if model overfitted when train 20% > test
        metric_train = lst(self.metric_train)
        metric_test = lst(self.metric_test)
        if metric_train[0] - 0.2 * metric_train[0] > metric_test[0]:
            out += " ~"

        return out

    @composed(crash, method_to_log)
    def calibrate(self, **kwargs):
        """Calibrate the model.

        Applies probability calibration on the winning model. The
        calibration is done with the CalibratedClassifierCV class from
        sklearn. The estimator will be trained via cross-validation on
        a subset of the training data, using the rest to fit the
        calibrator. The new classifier will replace the `estimator`
        attribute. All prediction attributes will reset.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the CalibratedClassifierCV
            instance. Using cv="prefit" will use the trained model and
            fit the calibrator on the test set. Note that doing this
            will result in data leakage in the test set. Use this only
            if you have another, independent set for testing.

        """
        if self.T.goal.startswith("reg"):
            raise PermissionError(
                "The calibrate method is only available for classification tasks!"
            )

        calibrator = CalibratedClassifierCV(self.estimator, **kwargs)
        if kwargs.get("cv") != "prefit":
            self.estimator = calibrator.fit(self.X_train, self.y_train)
        else:
            self.estimator = calibrator.fit(self.X_test, self.y_test)

        # Reset all prediction attrs since the model's estimator changed
        self._pred_attrs = [None] * 10

    @composed(crash, method_to_log, typechecked)
    def save_estimator(self, filename: Optional[str] = None):
        """Save the estimator to a pickle file.

        Parameters
        ----------
        filename: str, optional (default=None)
            Name of the file. If None or "auto", the estimator's
            __name__ is used.

        """
        if not filename:
            filename = self.estimator.__class__.__name__
        elif filename == "auto" or filename.endswith("/auto"):
            filename = filename.replace("auto", self.estimator.__class__.__name__)

        with open(filename, "wb") as file:
            pickle.dump(self.estimator, file)
        self.T.log(f"{self.fullname} estimator saved successfully!", 1)

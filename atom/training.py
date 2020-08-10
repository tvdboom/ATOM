# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the training classes.

"""

# Standard packages
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union, Sequence
from sklearn.base import BaseEstimator

# Own modules
from .basetrainer import BaseTrainer
from .plots import BaseModelPlotter, SuccessiveHalvingPlotter, TrainSizingPlotter
from .utils import (
    CAL, TRAIN_TYPES, get_best_score, get_default_metric, infer_task,
    composed, method_to_log, crash
    )


# Classes =================================================================== >>

class Trainer(BaseEstimator, BaseTrainer, BaseModelPlotter):
    """Train the models in a direct fashion.

    See basetrainer.py for a description of the parameters.

    """

    def __init__(self, models, metric, greater_is_better, needs_proba,
                 needs_threshold, n_calls, n_random_starts, bo_params,
                 bagging, n_jobs, verbose, warnings, logger, random_state):
        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts,
                         bo_params, bagging, n_jobs, verbose, warnings,
                         logger, random_state)

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Run the trainer.

        Parameters
        ----------
        *arrays: array-like
            Either a train and test set or X_train, X_test, y_train, y_test.

        """
        self._params_to_attr(*arrays)
        self.task = infer_task(self.y_train, goal=self.goal)

        # Assign default metric_ (not in __init__ since we need the task)
        if self.metric_ == [None]:
            self.metric_ = [get_default_metric(self.task)]

        self.log("\nRunning pipeline ============================= >>", 1)
        self.log(f"Models in pipeline: {', '.join(self.models)}", 1)
        self.log(f"Metric: {', '.join([m.name for m in self.metric_])}", 1)

        self._results = self._run()
        self._results.index.name = 'model'


class SuccessiveHalving(BaseEstimator, BaseTrainer, SuccessiveHalvingPlotter):
    """Train the models in a successive halving fashion.

    If you want to compare similar models, you can choose to use a successive
    halving approach to run the pipeline. This technique is a bandit-based
    algorithm that fits N models to 1/N of the data. The best half are selected
    to go to the next iteration where the process is repeated. This continues
    until only one model remains, which is fitted on the complete dataset.
    Beware that a model's performance can depend greatly on the amount of data
    on which it is trained. For this reason, we recommend only to use this
    technique with similar models, e.g. only using tree-based models.

    See basetrainer.py for a description of the remaining parameters.

    Parameters
    ----------
    skip_iter: int, optional (default=0)
        Skip last `skip_iter` iterations of the successive halving.

    """

    def __init__(self, models, metric, greater_is_better, needs_proba,
                 needs_threshold, skip_iter, n_calls, n_random_starts, bo_params,
                 bagging, n_jobs, verbose, warnings, logger, random_state):
        if skip_iter < 0:
            raise ValueError("Invalid value for the skip_iter parameter." +
                             f"Value should be >=0, got {skip_iter}.")
        else:
            self.skip_iter = skip_iter

        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts,
                         bo_params, bagging, n_jobs, verbose, warnings,
                         logger, random_state)

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Run the trainer.

        Parameters
        ----------
        *arrays: array-like
            Either a train and test set or X_train, X_test, y_train, y_test.

        """
        self._params_to_attr(*arrays)
        self.task = infer_task(self.y_train, goal=self.goal)

        # Assign default metric (not in __init__ since we need the task)
        if self.metric_ == [None]:
            self.metric_ = [get_default_metric(self.task)]

        self.log("\nRunning pipeline ============================= >>", 1)
        self.log(f"Metric: {', '.join([m.name for m in self.metric_])}", 1)

        run = 0
        results = []  # List of dataframes returned by self._run
        all_models = self.models[:]
        _train_idx = self._idx[0]  # Save the size of the original training set
        while len(self.models) > 2 ** self.skip_iter - 1:
            # Select 1/N of training set to use for this iteration
            self._idx[0] = int(1./len(self.models) * _train_idx)

            # Print stats for this subset of the data
            p = round(100. / len(self.models))
            self.log(f"\n\nRun {run} ({p}% of set) {'='*(30-len(str(p)))}>>", 1)
            self.log(f"Models in pipeline: {', '.join(self.models)}", 1)
            self.log(f"Size of training set: {len(self.train)}", 1)
            self.log(f"Size of test set: {len(self.test)}", 1)

            # Run iteration and append to the results list
            df = self._run()
            results.append(df)

            # Select best models for halving
            best = df.apply(lambda row: get_best_score(row), axis=1)
            best = best.nlargest(n=int(len(self.models) / 2), keep='first')
            self.models = list(best.index.values)

            run += 1

        # Concatenate all resulting dataframes with multi-index
        self._results = pd.concat(objs=[df for df in results],
                                  keys=range(len(results)),
                                  names=('run', 'model'))

        # Renew self.models and restore the training set
        self.models = all_models
        self._idx[0] = _train_idx


class TrainSizing(BaseEstimator, BaseTrainer, TrainSizingPlotter):
    """Train the models in a train sizing fashion.

    When training models, there is usually a trade-off between model performance
    and computation time that is regulated by the number of samples in the
    training set. The TrainSizing class can be used to create insights in this
    trade-off and help determine the optimal size of the training set.

    See basetrainer.py for a description of the remaining parameters.

    Parameters
    ----------
    train_sizes: sequence, optional (default=np.linspace(0.2, 1.0, 5))
        Relative or absolute numbers of training examples that will be used
        to generate the learning curve. If the value is <=1, it is
        interpreted as a fraction of the maximum size of the training set.
        If the value is > 1, it is interpreted as the total number of samples
        in the set.

    """

    def __init__(self, models, metric, greater_is_better, needs_proba,
                 needs_threshold, train_sizes, n_calls, n_random_starts, bo_params,
                 bagging, n_jobs, verbose, warnings, logger, random_state):
        self.train_sizes = train_sizes
        self._sizes = []  # Number of training samples (attr for plot)

        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts,
                         bo_params, bagging, n_jobs, verbose, warnings,
                         logger, random_state)

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Run the trainer.

        Parameters
        ----------
        *arrays: array-like
            Either a train and test set or X_train, X_test, y_train, y_test.

        """
        self._params_to_attr(*arrays)
        self.task = infer_task(self.y_train, goal=self.goal)

        # Assign default metric_ (not in __init__ since we need the task)
        if self.metric_ == [None]:
            self.metric_ = [get_default_metric(self.task)]

        self.log("\nRunning pipeline ============================= >>", 1)
        self.log(f"Models in pipeline: {', '.join(self.models)}", 1)
        self.log(f"Metric: {', '.join([m.name for m in self.metric_])}", 1)

        results = []  # List of dataframes returned by self._run
        _train_idx = self._idx[0]  # Save the size of the original training set
        for run, n_rows in enumerate(self.train_sizes):
            # Select fraction of data to use for this iteration
            self._idx[0] = int(n_rows * _train_idx if n_rows <= 1 else n_rows)
            self._sizes.append(len(self.train))

            # Print stats for this subset of the data
            p = round(len(self.train) * 100. / _train_idx)
            self.log(f"\n\nRun {run} ({p}% of set) {'='*(30-len(str(p)))}>>", 1)
            self.log(f"Size of training set: {len(self.train)}", 1)
            self.log(f"Size of test set: {len(self.test)}", 1)

            # Run iteration and append to the results list
            results.append(self._run())

        # Concatenate all resulting dataframes with multi-index
        self._results = pd.concat(objs=[df for df in results],
                                  keys=range(len(results)),
                                  names=('run', 'model'))

        self._idx[0] = _train_idx  # Restore original training set


class TrainerClassifier(Trainer):
    """Trainer class for classification tasks."""

    @typechecked
    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                 greater_is_better: Union[bool, Sequence[bool]] = True,
                 needs_proba: Union[bool, Sequence[bool]] = False,
                 needs_threshold: Union[bool, Sequence[bool]] = False,
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_params: dict = {},
                 bagging: Optional[Union[int, Sequence[int]]] = None,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 warnings: Union[bool, str] = True,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        self.goal = 'classification'
        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts, bo_params,
                         bagging, n_jobs, verbose, warnings, logger, random_state)


class TrainerRegressor(Trainer):
    """Trainer class for regression tasks."""

    @typechecked
    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                 greater_is_better: Union[bool, Sequence[bool]] = True,
                 needs_proba: Union[bool, Sequence[bool]] = False,
                 needs_threshold: Union[bool, Sequence[bool]] = False,
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_params: dict = {},
                 bagging: Optional[Union[int, Sequence[int]]] = None,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 warnings: Union[bool, str] = True,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        self.goal = 'regression'
        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts, bo_params,
                         bagging, n_jobs, verbose, warnings, logger, random_state)


class SuccessiveHalvingClassifier(SuccessiveHalving):
    """SuccessiveHalving class for classification tasks."""

    @typechecked
    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                 greater_is_better: Union[bool, Sequence[bool]] = True,
                 needs_proba: Union[bool, Sequence[bool]] = False,
                 needs_threshold: Union[bool, Sequence[bool]] = False,
                 skip_iter: int = 0,
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_params: dict = {},
                 bagging: Optional[Union[int, Sequence[int]]] = None,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 warnings: Union[bool, str] = True,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        self.goal = 'classification'
        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, skip_iter, n_calls, n_random_starts,
                         bo_params, bagging, n_jobs, verbose, warnings, logger,
                         random_state)


class SuccessiveHalvingRegressor(SuccessiveHalving):
    """SuccessiveHalving class for regression tasks."""

    @typechecked
    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                 greater_is_better: Union[bool, Sequence[bool]] = True,
                 needs_proba: Union[bool, Sequence[bool]] = False,
                 needs_threshold: Union[bool, Sequence[bool]] = False,
                 skip_iter: int = 0,
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_params: dict = {},
                 bagging: Optional[Union[int, Sequence[int]]] = None,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 warnings: Union[bool, str] = True,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        self.goal = 'regression'
        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, skip_iter, n_calls, n_random_starts,
                         bo_params, bagging, n_jobs, verbose, warnings, logger,
                         random_state)


class TrainSizingClassifier(TrainSizing):
    """TrainSizing class for classification tasks."""

    @typechecked
    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                 greater_is_better: Union[bool, Sequence[bool]] = True,
                 needs_proba: Union[bool, Sequence[bool]] = False,
                 needs_threshold: Union[bool, Sequence[bool]] = False,
                 train_sizes: TRAIN_TYPES = np.linspace(0.2, 1.0, 5),
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_params: dict = {},
                 bagging: Optional[Union[int, Sequence[int]]] = None,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 warnings: Union[bool, str] = True,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        self.goal = 'classification'
        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, train_sizes, n_calls, n_random_starts,
                         bo_params, bagging, n_jobs, verbose, warnings, logger,
                         random_state)


class TrainSizingRegressor(TrainSizing):
    """TrainSizing class for regression tasks."""

    @typechecked
    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                 greater_is_better: Union[bool, Sequence[bool]] = True,
                 needs_proba: Union[bool, Sequence[bool]] = False,
                 needs_threshold: Union[bool, Sequence[bool]] = False,
                 train_sizes: TRAIN_TYPES = np.linspace(0.2, 1.0, 5),
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_params: dict = {},
                 bagging: Optional[Union[int, Sequence[int]]] = None,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 warnings: Union[bool, str] = True,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        self.goal = 'regression'
        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, train_sizes, n_calls, n_random_starts,
                         bo_params, bagging, n_jobs, verbose, warnings, logger,
                         random_state)

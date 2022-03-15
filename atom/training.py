# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the training classes.

"""

# Standard packages
import numpy as np
import pandas as pd
from copy import copy
from typeguard import typechecked
from typing import Optional, Union
from sklearn.base import BaseEstimator

# Own modules
from .basetrainer import BaseTrainer
from .plots import BaseModelPlotter
from .utils import (
    SEQUENCE_TYPES, lst, get_best_score, infer_task, composed,
    method_to_log, crash, CustomDict,
)


class Direct(BaseEstimator, BaseTrainer, BaseModelPlotter):
    """Direct training approach.

    Fit and evaluate over the models. Contrary to SuccessiveHalving
    and TrainSizing, the direct approach only iterates once over the
    models, using the full dataset.

    See basetrainer.py for a description of the parameters.

    """

    def __init__(
        self, models, metric, greater_is_better, needs_proba, needs_threshold,
        n_calls, n_initial_points, est_params, bo_params, n_bootstrap, n_jobs,
        verbose, warnings, logger, experiment, gpu, random_state,
    ):
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
            n_jobs, verbose, warnings, logger, experiment, gpu, random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Run the trainer.

        Parameters
        ----------
        *arrays: sequence of indexables
            Training set and test set. Allowed formats are:
                - train, test
                - X_train, X_test, y_train, y_test
                - (X_train, y_train), (X_test, y_test)

        """
        self.branch._data, self.branch._idx, self.holdout = self._get_data(arrays)
        self.task = infer_task(self.y_train, goal=self.goal)
        self._prepare_parameters()

        self._core_iteration()


class SuccessiveHalving(BaseEstimator, BaseTrainer, BaseModelPlotter):
    """Successive halving training approach.

    The successive halving technique is a bandit-based algorithm that
    fits N models to 1/N of the data. The best half are selected to
    go to the next iteration where the process is repeated. This
    continues until only one model remains, which is fitted on the
    complete dataset. Beware that a model's performance can depend
    greatly on the amount of data on which it is trained. For this
    reason, it is recommended to only use this technique with similar
    models, e.g. only using tree-based models.

    See basetrainer.py for a description of the remaining parameters.

    Parameters
    ----------
    skip_runs: int, optional (default=0)
        Skip last `skip_runs` runs of the successive halving.

    """

    def __init__(
        self, models, metric, greater_is_better, needs_proba, needs_threshold,
        skip_runs, n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
        n_jobs, verbose, warnings, logger, experiment, gpu, random_state,
    ):
        self.skip_runs = skip_runs
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
            n_jobs, verbose, warnings, logger, experiment, gpu, random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Run the trainer.

        Parameters
        ----------
        *arrays: sequence of indexables
            Training set and test set. Allowed formats are:
                - train, test
                - X_train, X_test, y_train, y_test
                - (X_train, y_train), (X_test, y_test)

        """
        self.branch._data, self.branch._idx, self.holdout = self._get_data(arrays)
        self.task = infer_task(self.y_train, goal=self.goal)
        self._prepare_parameters()

        if self.skip_runs < 0:
            raise ValueError(
                "Invalid value for the skip_runs parameter."
                f"Value should be >=0, got {self.skip_runs}."
            )
        elif self.skip_runs >= len(self._models) // 2 + 1:
            raise ValueError(
                "Invalid value for the skip_runs parameter. Less than "
                f"1 run remaining, got n_runs={len(self._models) // 2 + 1} "
                f"and skip_runs={self.skip_runs}."
            )

        run = 0
        models = CustomDict()
        og_models = {k: copy(v) for k, v in self._models.items()}
        while len(self._models) > 2 ** self.skip_runs - 1:
            # Create the new set of models for the run
            for m in self._models.values():
                m.name += str(len(self._models))
                m._pred = [None] * 15  # Avoid shallow copy
                m._train_idx = len(self.train) // len(self._models)

            # Print stats for this subset of the data
            p = round(100.0 / len(self._models))
            self.log(f"\n\nRun: {run} {'='*32} >>", 1)
            self.log(f"Models: {', '.join(lst(self.models))}", 1)
            self.log(f"Size of training set: {len(self.train)} ({p}%)", 1)
            self.log(f"Size of test set: {len(self.test)}", 1)

            self._core_iteration()
            models.update({m.name: m for m in self._models.values()})

            # Select next models for halving
            best = pd.Series(
                data=[get_best_score(m) for m in self._models.values()],
                index=[m.name for m in self._models.values()],
            ).nlargest(n=len(self._models) // 2, keep="first")
            names = [m.acronym for m in self._models.values() if m.name in best.index]
            self._models = CustomDict(
                {k: copy(v) for k, v in og_models.items() if v.acronym in names}
            )

            run += 1

        self._models = models  # Restore all models


class TrainSizing(BaseEstimator, BaseTrainer, BaseModelPlotter):
    """Train Sizing training approach.

    When training models, there is usually a trade-off between model
    performance and computation time, that is regulated by the number
    of samples in the training set. This class can be used to create
    insights in this trade-off, and help determine the optimal size of
    the training set. The models are fitted multiple times,
    ever-increasing the number of samples in the training set.

    See basetrainer.py for a description of the remaining parameters.

    Parameters
    ----------
    train_sizes: int or sequence, optional (default=5)
        Sequence of training set sizes used to run the trainings.
            - If int: Number of equally distributed splits, i.e. for a
                      value N it's equal to np.linspace(1.0/N, 1.0, N).
            - If sequence: Fraction of the training set when <=1, else
                           total number of samples.

    """

    def __init__(
        self, models, metric, greater_is_better, needs_proba, needs_threshold,
        train_sizes, n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
        n_jobs, verbose, warnings, logger, experiment, gpu, random_state
    ):
        self.train_sizes = train_sizes
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
            n_jobs, verbose, warnings, logger, experiment, gpu, random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Run the trainer.

        Parameters
        ----------
        *arrays: sequence of indexables
            Training set and test set. Allowed formats are:
                - train, test
                - X_train, X_test, y_train, y_test
                - (X_train, y_train), (X_test, y_test)

        """
        self.branch._data, self.branch._idx, self.holdout = self._get_data(arrays)
        self.task = infer_task(self.y_train, goal=self.goal)
        self._prepare_parameters()

        # Convert integer train_sizes to sequence
        if isinstance(self.train_sizes, int):
            self.train_sizes = np.linspace(1 / self.train_sizes, 1.0, self.train_sizes)

        models = CustomDict()
        og_models = {k: copy(v) for k, v in self._models.items()}
        for run, size in enumerate(self.train_sizes):
            # Select fraction of data to use in this run
            if size <= 1:
                frac = round(size, 2)
                train_idx = int(size * len(self.branch.train))
            else:
                frac = round(size / len(self.branch.train), 2)
                train_idx = size

            for m in self._models.values():
                m.name += str(frac).replace(".", "")  # Add frac to the name
                m._pred = [None] * 15  # Avoid shallow copy
                m._train_idx = train_idx

            # Print stats for this subset of the data
            p = round(train_idx * 100.0 / len(self.branch.train))
            self.log(f"\n\nRun: {run} {'='*32} >>", 1)
            self.log(f"Size of training set: {train_idx} ({p}%)", 1)
            self.log(f"Size of test set: {len(self.test)}", 1)

            self._core_iteration()
            models.update({m.name.lower(): m for m in self._models.values()})

            # Create next models for sizing
            self._models = CustomDict({k: copy(v) for k, v in og_models.items()})

        self._models = models  # Restore original models


class DirectClassifier(Direct):
    """Direct trainer for classification tasks."""

    @typechecked
    def __init__(
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
        n_jobs: int = 1,
        verbose: int = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, callable]] = None,
        experiment: Optional[str] = None,
        gpu: Union[bool, str] = False,
        random_state: Optional[int] = None,
    ):
        self.goal = "class"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
            n_jobs, verbose, warnings, logger, experiment, gpu, random_state,
        )


class DirectRegressor(Direct):
    """Direct trainer for regression tasks."""

    @typechecked
    def __init__(
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
        n_jobs: int = 1,
        verbose: int = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, callable]] = None,
        experiment: Optional[str] = None,
        gpu: Union[bool, str] = False,
        random_state: Optional[int] = None,
    ):
        self.goal = "reg"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
            n_jobs, verbose, warnings, logger, experiment, gpu, random_state,
        )


class SuccessiveHalvingClassifier(SuccessiveHalving):
    """SuccessiveHalving trainer for classification tasks."""

    @typechecked
    def __init__(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
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
        n_jobs: int = 1,
        verbose: int = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, callable]] = None,
        experiment: Optional[str] = None,
        gpu: Union[bool, str] = False,
        random_state: Optional[int] = None,
    ):
        self.goal = "class"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            skip_runs, n_calls, n_initial_points, est_params, bo_params,
            n_bootstrap, n_jobs, verbose, warnings, logger, experiment, gpu,
            random_state,
        )


class SuccessiveHalvingRegressor(SuccessiveHalving):
    """SuccessiveHalving trainer for regression tasks."""

    @typechecked
    def __init__(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
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
        n_jobs: int = 1,
        verbose: int = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, callable]] = None,
        experiment: Optional[str] = None,
        gpu: Union[bool, str] = False,
        random_state: Optional[int] = None,
    ):
        self.goal = "reg"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            skip_runs, n_calls, n_initial_points, est_params, bo_params,
            n_bootstrap, n_jobs, verbose, warnings, logger, experiment, gpu,
            random_state,
        )


class TrainSizingClassifier(TrainSizing):
    """TrainSizing trainer for classification tasks."""

    @typechecked
    def __init__(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
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
        n_jobs: int = 1,
        verbose: int = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, callable]] = None,
        experiment: Optional[str] = None,
        gpu: Union[bool, str] = False,
        random_state: Optional[int] = None,
    ):
        self.goal = "class"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            train_sizes, n_calls, n_initial_points, est_params, bo_params,
            n_bootstrap, n_jobs, verbose, warnings, logger, experiment, gpu,
            random_state,
        )


class TrainSizingRegressor(TrainSizing):
    """TrainSizing trainer for regression tasks."""

    @typechecked
    def __init__(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
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
        n_jobs: int = 1,
        verbose: int = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, callable]] = None,
        experiment: Optional[str] = None,
        gpu: Union[bool, str] = False,
        random_state: Optional[int] = None,
    ):
        self.goal = "reg"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            train_sizes, n_calls, n_initial_points, est_params, bo_params,
            n_bootstrap, n_jobs, verbose, warnings, logger, experiment, gpu,
            random_state,
        )

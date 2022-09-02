# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the training classes.

"""

from copy import copy
from logging import Logger
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typeguard import typechecked

from atom.basetrainer import BaseTrainer
from atom.plots import ModelPlot
from atom.utils import (
    INT, SEQUENCE_TYPES, CustomDict, composed, crash, get_best_score,
    infer_task, lst, method_to_log,
)


class Direct(BaseEstimator, BaseTrainer, ModelPlot):
    """Direct training approach.

    Fit and evaluate over the models. Contrary to SuccessiveHalving
    and TrainSizing, the direct approach only iterates once over the
    models, using the full dataset.

    See basetrainer.py for a description of the parameters.

    """

    def __init__(
        self, models, metric, greater_is_better, needs_proba, needs_threshold,
        n_calls, n_initial_points, est_params, bo_params, n_bootstrap, n_jobs,
        device, engine, verbose, warnings, logger, experiment, random_state,
    ):
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
            n_jobs, device, engine, verbose, warnings, logger, experiment,
            random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Train and evaluate the models.

        Parameters
        ----------
        *arrays: sequence of indexables
            Training set and test set. Allowed formats are:
                - train, test
                - X_train, X_test, y_train, y_test
                - (X_train, y_train), (X_test, y_test)

        """
        self.branch._data, self.branch._idx, self.holdout = self._get_data(arrays)
        self.task = infer_task(self.y, goal=self.goal)
        self._prepare_parameters()

        self._core_iteration()


class SuccessiveHalving(BaseEstimator, BaseTrainer, ModelPlot):
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
    skip_runs: int, default=0
        Skip last `skip_runs` runs of the successive halving.

    """

    def __init__(
        self, models, metric, greater_is_better, needs_proba, needs_threshold,
        skip_runs, n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
        n_jobs, device, engine, verbose, warnings, logger, experiment, random_state,
    ):
        self.skip_runs = skip_runs
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
            n_jobs, device, engine, verbose, warnings, logger, experiment,
            random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Train and evaluate the models.

        Parameters
        ----------
        *arrays: sequence of indexables
            Training set and test set. Allowed formats are:
                - train, test
                - X_train, X_test, y_train, y_test
                - (X_train, y_train), (X_test, y_test)

        """
        self.branch._data, self.branch._idx, self.holdout = self._get_data(arrays)
        self.task = infer_task(self.y, goal=self.goal)
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


class TrainSizing(BaseEstimator, BaseTrainer, ModelPlot):
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
    train_sizes: int or sequence, default=5
        Sequence of training set sizes used to run the trainings.
            - If int: Number of equally distributed splits, i.e. for a
                      value N it's equal to np.linspace(1.0/N, 1.0, N).
            - If sequence: Fraction of the training set when <=1, else
                           total number of samples.

    """

    def __init__(
        self, models, metric, greater_is_better, needs_proba, needs_threshold,
        train_sizes, n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
        n_jobs, device, engine, verbose, warnings, logger, experiment, random_state
    ):
        self.train_sizes = train_sizes
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
            n_jobs, device, engine, verbose, warnings, logger, experiment,
            random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Train and evaluate the models.

        Parameters
        ----------
        *arrays: sequence of indexables
            Training set and test set. Allowed formats are:
                - train, test
                - X_train, X_test, y_train, y_test
                - (X_train, y_train), (X_test, y_test)

        """
        self.branch._data, self.branch._idx, self.holdout = self._get_data(arrays)
        self.task = infer_task(self.y, goal=self.goal)
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
    """Direct trainer for classification tasks.

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of ATOM's predefined models, an ATOMModel or a custom
        estimator as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        scorers, a function with signature `metric(y_true, y_pred)`, a
        scorer object or a sequence of these. If multiple metrics are
        selected, only the first is used to optimize the BO. If None, a
        default scorer is selected:
            - "f1" for binary classification
            - "f1_weighted" for multiclass classification
            - "r2" for regression

    greater_is_better: bool or sequence, default=True
        Whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower
        is better. This parameter is ignored if the metric is a
        string or a scorer. If sequence, the n-th value applies to
        the n-th metric.

    needs_proba: bool or sequence, default=False
        Whether the metric function requires probability estimates out
        of a classifier. If True, make sure that every selected model
        has a `predict_proba` method. This parameter is ignored if the
        metric is a string or a scorer. If sequence, the n-th value
        applies to the n-th metric.

    needs_threshold: bool or sequence, default=False
        Whether the metric function takes a continuous decision
        certainty. This only works for estimators that have either a
        `decision_function` or `predict_proba` method. This parameter
        is ignored if the metric is a string or a scorer. If sequence,
        the n-th value applies to the n-th metric.

    n_calls: int or sequence, default=15
        Maximum number of iterations of the BO. It includes the random
        points of `n_initial_points`. If 0, skip the BO and fit the
        model on its default Parameters. If sequence, the n-th value
        applies to the n-th model.

    n_initial_points: int or sequence, default=5
        Initial number of random tests of the BO before fitting the
        surrogate function. If equal to `n_calls`, the optimizer will
        technically be performing a random search. If sequence, the
        n-th value applies to the n-th model.

    est_params: dict or None, default=None
        Additional parameters for the estimators. See the corresponding
        documentation for the available options. For multiple models,
        use the acronyms as key (or 'all' for all models) and a dict
        of the parameters as value. Add _fit to the parameter's name
        to pass it to the fit method instead of the initializer.

    bo_params: dict or None, default=None
        Additional parameters to for the BO. These can include:
            - base_estimator: str, default="GP"
                Surrogate model to use. Choose from:
                    - "GP" for Gaussian Process
                    - "RF" for Random Forest
                    - "ET" for Extra-Trees
                    - "GBRT" for Gradient Boosted Regression Trees
            - max_time: int, default=np.inf
                Stop the optimization after `max_time` seconds.
            - delta_x: int or float, default=0
                Stop the optimization when `|x1 - x2| < delta_x`.
            - delta_y: int or float, default=0
                Stop the optimization if the 5 minima are within
                `delta_y` (skopt always minimizes the function).
            - early_stopping: int, float or None, default=None
                Training will stop if the model didn't improve in
                last `early_stopping` rounds. If <1, fraction of
                rounds from the total. If None, no early stopping
                is performed. Only available for models that allow
                in-training evaluation.
            - cv: int, default=5
                Number of folds for the cross-validation. If 1, the
                training set is randomly split in a (sub)train and
                validation set.
            - callback: callable or list of callables, default=None
                Callbacks for the BO.
            - dimensions: dict, list or None, default=None
                Custom hyperparameter space for the BO. Can be a list
                to share the same dimensions across models or a dict
                with the model names as key (or `all` for all models).
                If None, ATOM's predefined dimensions are used.
            - plot: bool, default=False
                Whether to plot the BO's progress as it runs.
                Creates a canvas with two plots: the first plot
                shows the score of every trial and the second shows
                the distance between the last consecutive steps.
            - Additional keyword arguments for skopt's optimizer.

    n_bootstrap: int or sequence, default=0
        Number of data sets (bootstrapped from the training set) to
        use in the bootstrap algorithm. If 0, no bootstrap is performed.
        If sequence, the n-th value will apply to the n-th model.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 + `n_jobs`.

    verbose: int, default=0
        Verbosity level of the class. Choose from:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    warnings: bool or str, default=True
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of the actions in python's warnings environment.

        Note that changing this parameter will affect the
        `PYTHONWARNINGS` environment.

        Note that ATOM can't manage warnings that go directly
        from C/C++ code to the stdout/stderr.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic name.
        - Else: Python `logging.Logger` instance.

        Note that warnings will not be saved to the logger.

    experiment: str or None, default=None
        Name of the mlflow experiment to use for tracking. If None,
        no mlflow tracking is performed.

    gpu: bool or str, default=False
        Train estimators on GPU. Refer to the
        documentation to check which estimators are supported.
            - If False: Always use CPU implementation.
            - If True: Use GPU implementation if possible.
            - If "force": Force GPU implementation.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    """

    @typechecked
    def __init__(
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
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, Logger]] = None,
        experiment: Optional[str] = None,
        random_state: Optional[INT] = None,
    ):
        self.goal = "class"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
            n_jobs, device, engine, verbose, warnings, logger, experiment,
            random_state,
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
        n_calls: Union[INT, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[INT, SEQUENCE_TYPES] = 0,
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, Logger]] = None,
        experiment: Optional[str] = None,
        random_state: Optional[INT] = None,
    ):
        self.goal = "reg"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            n_calls, n_initial_points, est_params, bo_params, n_bootstrap,
            n_jobs, device, engine, verbose, warnings, logger, experiment,
            random_state,
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
        skip_runs: INT = 0,
        n_calls: Union[INT, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[INT, SEQUENCE_TYPES] = 0,
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, Logger]] = None,
        experiment: Optional[str] = None,
        random_state: Optional[INT] = None,
    ):
        self.goal = "class"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            skip_runs, n_calls, n_initial_points, est_params, bo_params,
            n_bootstrap, n_jobs, device, engine, verbose, warnings, logger,
            experiment, random_state,
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
        skip_runs: INT = 0,
        n_calls: Union[INT, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[INT, SEQUENCE_TYPES] = 0,
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, Logger]] = None,
        experiment: Optional[str] = None,
        random_state: Optional[INT] = None,
    ):
        self.goal = "reg"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            skip_runs, n_calls, n_initial_points, est_params, bo_params,
            n_bootstrap, n_jobs, device, engine, verbose, warnings, logger,
            experiment, random_state,
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
        train_sizes: Union[INT, SEQUENCE_TYPES] = 5,
        n_calls: Union[INT, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[INT, SEQUENCE_TYPES] = 0,
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, Logger]] = None,
        experiment: Optional[str] = None,
        random_state: Optional[INT] = None,
    ):
        self.goal = "class"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            train_sizes, n_calls, n_initial_points, est_params, bo_params,
            n_bootstrap, n_jobs, device, engine, verbose, warnings, logger,
            experiment, random_state,
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
        train_sizes: Union[INT, SEQUENCE_TYPES] = 5,
        n_calls: Union[INT, SEQUENCE_TYPES] = 0,
        n_initial_points: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[dict] = None,
        bo_params: Optional[dict] = None,
        n_bootstrap: Union[INT, SEQUENCE_TYPES] = 0,
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        warnings: Union[bool, str] = True,
        logger: Optional[Union[str, Logger]] = None,
        experiment: Optional[str] = None,
        random_state: Optional[INT] = None,
    ):
        self.goal = "reg"
        super().__init__(
            models, metric, greater_is_better, needs_proba, needs_threshold,
            train_sizes, n_calls, n_initial_points, est_params, bo_params,
            n_bootstrap, n_jobs, device, engine, verbose, warnings, logger,
            experiment, random_state,
        )

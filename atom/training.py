# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the training classes.

"""

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
        self, models, metric, est_params, n_trials, ht_params, n_bootstrap, n_jobs,
        device, engine, verbose, warnings, logger, experiment, random_state,
    ):
        super().__init__(
            models, metric, est_params, n_trials, ht_params, n_bootstrap, n_jobs,
            device, engine, verbose, warnings, logger, experiment, random_state,
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
        self, models, metric, skip_runs, est_params, n_trials, ht_params, n_bootstrap,
        n_jobs, device, engine, verbose, warnings, logger, experiment, random_state,
    ):
        self.skip_runs = skip_runs
        super().__init__(
            models, metric, est_params, n_trials, ht_params, n_bootstrap, n_jobs,
            device, engine, verbose, warnings, logger, experiment, random_state,
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
        og_models = {k: v._new_copy() for k, v in self._models.items()}
        while len(self._models) > 2 ** self.skip_runs - 1:
            # Create the new set of models for the run
            for m in self._models.values():
                m._name += str(len(self._models))
                m._train_idx = len(self.train) // len(self._models)

            # Print stats for this subset of the data
            p = round(100.0 / len(self._models))
            self.log(f"\n\nRun: {run} {'='*32} >>", 1)
            self.log(f"Models: {', '.join(lst(self.models))}", 1)
            self.log(f"Size of training set: {len(self.train)} ({p}%)", 1)
            self.log(f"Size of test set: {len(self.test)}", 1)

            self._core_iteration()
            models.update({m.name: m for m in self._models.values()})

            # Select best models for halving
            best = pd.Series(
                data=[get_best_score(m) for m in self._models.values()],
                index=[m._group for m in self._models.values()],
                dtype="float",
            ).nlargest(n=len(self._models) // 2, keep="first")

            self._models = CustomDict(
                {k: v._new_copy() for k, v in og_models.items() if k in best.index}
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
        self, models, metric, train_sizes, est_params, n_trials, ht_params, n_bootstrap,
        n_jobs, device, engine, verbose, warnings, logger, experiment, random_state
    ):
        self.train_sizes = train_sizes
        super().__init__(
            models, metric, est_params, n_trials, ht_params, n_bootstrap, n_jobs,
            device, engine, verbose, warnings, logger, experiment, random_state,
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
        og_models = {k: v._new_copy() for k, v in self._models.items()}
        for run, size in enumerate(self.train_sizes):
            # Select fraction of data to use in this run
            if size <= 1:
                frac = round(size, 2)
                train_idx = int(size * len(self.branch.train))
            else:
                frac = round(size / len(self.branch.train), 2)
                train_idx = size

            for m in self._models.values():
                m._name += str(frac).replace(".", "")  # Add frac to the name
                m._train_idx = train_idx

            # Print stats for this subset of the data
            p = round(train_idx * 100.0 / len(self.branch.train))
            self.log(f"\n\nRun: {run} {'='*32} >>", 1)
            self.log(f"Size of training set: {train_idx} ({p}%)", 1)
            self.log(f"Size of test set: {len(self.test)}", 1)

            self._core_iteration()
            models.update({m.name.lower(): m for m in self._models.values()})

            # Create next models for sizing
            self._models = CustomDict({k: v._new_copy() for k, v in og_models.items()})

        self._models = models  # Restore original models


class DirectClassifier(Direct):
    """Train and evaluate the models in a direct fashion.

    The following steps are applied to every model:

    1. Apply [hyperparameter tuning][] (optional).
    2. Fit the model on the training set using the best combination
       of hyperparameters found.
    3. Evaluate the model on the test set.
    4. Train the model on various bootstrapped samples of the
       training set and evaluate again on the test set (optional).

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of the [predefined models][], an [ATOMModel][] or a custom
        predictor as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        scorers, a function with signature `function(y_true, y_pred) ->
        score`, a scorer object or a sequence of these. If None, a
        default metric is selected for every task:

        - "f1" for binary classification
        - "f1_weighted" for multiclass classification
        - "r2" for regression

    n_trials: int or sequence, default=0
        Maximum number of iterations for the [hyperparameter tuning][].
        If 0, skip the tuning and fit the model on its default
        parameters. If sequence, the n-th value applies to the n-th
        model.

    est_params: dict or None, default=None
        Additional parameters for the models. See their corresponding
        documentation for the available options. For multiple models,
        use the acronyms as key (or 'all' for all models) and a dict
        of the parameters as value. Add `_fit` to the parameter's name
        to pass it to the estimator's fit method instead of the
        constructor.

    ht_params: dict or None, default=None
        Additional parameters for the hyperparameter tuning. If None,
        it uses the same parameters as the first run. Can include:

        - **cv: int, dict or sequence, default=1**<br>
          Number of folds for the cross-validation. If 1, the training
          set is randomly split in a subtrain and validation set.
        - **plot: bool, dict or sequence, default=False**<br>
          Whether to plot the optimization's progress as it runs.
          Creates a canvas with two plots: the first plot shows the
          score of every trial and the second shows the distance between
          the last consecutive steps. See the [plot_trials][] method.
        - **distributions: dict, sequence or None, default=None**<br>
          Custom hyperparameter distributions for the models. If None,
          it uses ATOM's predefined distributions.
        - **tags: dict, sequence or None, default=None**<br>
          Custom tags for the model's [mlflow run][tracking].
        - **\*\*kwargs**<br>
          Additional Keyword arguments for the constructor of the
          [study][] class or the [optimize][] method.

    n_bootstrap: int or sequence, default=0
        Number of data sets to use for [bootstrapping][]. If 0, no
        bootstrapping is performed. If sequence, the n-th value applies
        to the n-th model.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

    device: str, default="cpu"
        Device on which to train the estimators. Use any string
        that follows the [SYCL_DEVICE_FILTER][] filter selector,
        e.g. `device="gpu"` to use the GPU. Read more in the
        [user guide][accelerating-pipelines].

    engine: str, default="sklearn"
        Execution engine to use for the estimators. Refer to the
        [user guide][accelerating-pipelines] for an explanation
        regarding every choice. Choose from:

        - "sklearn" (only if device="cpu")
        - "sklearnex"
        - "cuml" (only if device="gpu")

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWARNINGS` environment.
        ATOM can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic name.
        - Else: Python `logging.Logger` instance.

    experiment: str or None, default=None
        Name of the [mlflow experiment][experiment] to use for tracking.
        If None, no mlflow tracking is performed.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    See Also
    --------
    atom.api:ATOMClassifier
    atom.training.SuccessiveHalvingClassifier
    atom.training.TrainSizingClassifier

    Examples
    --------

    ```pycon
    >>> from atom.training import DirectClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    >>> bin_train, bin_test = train_test_split(
    ...     X_bin.merge(y_bin.to_frame(), left_index=True, right_index=True),
    ...     test_size=0.3,
    ...     random_state=1,
    ... )

    >>> runner = DirectClassifier(models=["LR", "RF"], metric="auc", verbose=2)
    >>> runner.run(bin_train, bin_test)

    Training ========================= >>
    Models: Tree
    Metric: roc_auc


    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 1.0
    Test evaluation --> roc_auc: 0.9603
    Time elapsed: 0.015s
    -------------------------------------------------
    Total time: 0.015s


    Final results ==================== >>
    Total time: 0.015s
    -------------------------------------
    DecisionTree --> roc_auc: 0.9603

    >>> # Analyze the results
    >>> atom.evaluate()

         accuracy  average_precision  ...    recall   roc_auc
    LR   0.970588           0.995739  ...  0.981308  0.993324
    RF   0.958824           0.982602  ...  0.962617  0.983459
    XGB  0.964706           0.996047  ...  0.971963  0.993473

    [3 rows x 9 columns]

    ```

    """

    @typechecked
    def __init__(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        *,
        est_params: Optional[Union[dict, SEQUENCE_TYPES]] = None,
        n_trials: Union[INT, dict, SEQUENCE_TYPES] = 0,
        ht_params: Optional[dict] = None,
        n_bootstrap: Union[INT, dict, SEQUENCE_TYPES] = 0,
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
            models, metric, est_params, n_trials, ht_params, n_bootstrap, n_jobs,
            device, engine, verbose, warnings, logger, experiment, random_state,
        )


class DirectRegressor(Direct):
    """Direct trainer for regression tasks."""

    @typechecked
    def __init__(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        *,
        est_params: Optional[Union[dict, SEQUENCE_TYPES]] = None,
        n_trials: Union[INT, dict, SEQUENCE_TYPES] = 0,
        ht_params: Optional[dict] = None,
        n_bootstrap: Union[INT, dict, SEQUENCE_TYPES] = 0,
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
            models, metric, est_params, n_trials, ht_params, n_bootstrap, n_jobs,
            device, engine, verbose, warnings, logger, experiment, random_state,
        )


class SuccessiveHalvingClassifier(SuccessiveHalving):
    """SuccessiveHalving trainer for classification tasks."""

    @typechecked
    def __init__(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        *,
        skip_runs: INT = 0,
        est_params: Optional[Union[dict, SEQUENCE_TYPES]] = None,
        n_trials: Union[INT, dict, SEQUENCE_TYPES] = 0,
        ht_params: Optional[dict] = None,
        n_bootstrap: Union[INT, dict, SEQUENCE_TYPES] = 0,
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
            models, metric, skip_runs, est_params, n_trials, ht_params, n_bootstrap,
            n_jobs, device, engine, verbose, warnings, logger, experiment, random_state,
        )


class SuccessiveHalvingRegressor(SuccessiveHalving):
    """SuccessiveHalving trainer for regression tasks."""

    @typechecked
    def __init__(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        *,
        skip_runs: INT = 0,
        est_params: Optional[Union[dict, SEQUENCE_TYPES]] = None,
        n_trials: Union[INT, dict, SEQUENCE_TYPES] = 0,
        ht_params: Optional[dict] = None,
        n_bootstrap: Union[INT, dict, SEQUENCE_TYPES] = 0,
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
            models, metric, skip_runs, est_params, n_trials, ht_params, n_bootstrap,
            n_jobs, device, engine, verbose, warnings, logger, experiment, random_state,
        )


class TrainSizingClassifier(TrainSizing):
    """TrainSizing trainer for classification tasks."""

    @typechecked
    def __init__(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        *,
        train_sizes: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[Union[dict, SEQUENCE_TYPES]] = None,
        n_trials: Union[INT, dict, SEQUENCE_TYPES] = 0,
        ht_params: Optional[dict] = None,
        n_bootstrap: Union[INT, dict, SEQUENCE_TYPES] = 0,
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
            models, metric, train_sizes, est_params, n_trials, ht_params, n_bootstrap,
            n_jobs, device, engine, verbose, warnings, logger, experiment, random_state,
        )


class TrainSizingRegressor(TrainSizing):
    """TrainSizing trainer for regression tasks."""

    @typechecked
    def __init__(
        self,
        models: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        *,
        train_sizes: Union[INT, SEQUENCE_TYPES] = 5,
        est_params: Optional[Union[dict, SEQUENCE_TYPES]] = None,
        n_trials: Union[INT, dict, SEQUENCE_TYPES] = 0,
        ht_params: Optional[dict] = None,
        n_bootstrap: Union[INT, dict, SEQUENCE_TYPES] = 0,
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
            models, metric, train_sizes, est_params, n_trials, ht_params, n_bootstrap,
            n_jobs, device, engine, verbose, warnings, logger, experiment, random_state,
        )

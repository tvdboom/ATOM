# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the training classes.

"""

from __future__ import annotations

from copy import copy
from logging import Logger
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typeguard import typechecked

from atom.basetrainer import BaseTrainer
from atom.utils.types import (
    BOOL, ENGINE, INT, INT_TYPES, METRIC_SELECTOR, PREDICTOR, SEQUENCE,
    WARNINGS,
)
from atom.utils.utils import (
    ClassMap, composed, crash, get_best_score, infer_task, lst, method_to_log,
)


@typechecked
class Direct(BaseEstimator, BaseTrainer):
    """Direct training approach.

    Fit and evaluate over the models. Contrary to SuccessiveHalving
    and TrainSizing, the direct approach only iterates once over the
    models, using the full dataset.

    See basetrainer.py for a description of the parameters.

    """

    def __init__(
        self, models, metric, est_params, n_trials, ht_params, n_bootstrap,
        parallel, errors, n_jobs, device, engine, backend, verbose, warnings,
        logger, experiment, random_state,
    ):
        super().__init__(
            models, metric, est_params, n_trials, ht_params, n_bootstrap,
            parallel, errors, n_jobs, device, engine, backend, verbose,
            warnings, logger, experiment, random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Train and evaluate the models.

        Read more in the [user guide][training].

        Parameters
        ----------
        *arrays: sequence of indexables
            Training set and test set. Allowed formats are:

            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        """
        self.branch._data, self.branch._idx, holdout = self._get_data(arrays)
        self.holdout = self.branch._holdout = holdout

        self.task = infer_task(self.y, goal=self.goal)
        self._prepare_parameters()

        self.log("\nTraining " + "=" * 25 + " >>", 1)
        self.log(f"Models: {', '.join(lst(self.models))}", 1)
        self.log(f"Metric: {', '.join(lst(self.metric))}", 1)

        self._core_iteration()


@typechecked
class SuccessiveHalving(BaseEstimator, BaseTrainer):
    """Train and evaluate the models in a [successive halving][] fashion.

    See [SuccessiveHalvingClassifier][] or [SuccessiveHalvingRegressor][]
    for a description of the remaining parameters.

    """

    def __init__(
        self, models, metric, skip_runs, est_params, n_trials, ht_params,
        n_bootstrap, parallel, errors, n_jobs, device, engine, backend,
        verbose, warnings, logger, experiment, random_state,
    ):
        self.skip_runs = skip_runs
        super().__init__(
            models, metric, est_params, n_trials, ht_params, n_bootstrap,
            parallel, errors, n_jobs, device, engine, backend, verbose,
            warnings, logger, experiment, random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Train and evaluate the models.

        Read more in the [user guide][training].

        Parameters
        ----------
        *arrays: sequence of indexables
            Training set and test set. Allowed formats are:

            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        """
        self.branch._data, self.branch._idx, holdout = self._get_data(arrays)
        self.holdout = self.branch._holdout = holdout

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

        self.log("\nTraining " + "=" * 25 + " >>", 1)
        self.log(f"Metric: {', '.join(lst(self.metric))}", 1)

        run = 0
        models = ClassMap()
        og_models = ClassMap(copy(m) for m in self._models)
        while len(self._models) > 2 ** self.skip_runs - 1:
            # Create the new set of models for the run
            for m in self._models:
                m._name += str(len(self._models))
                m._train_idx = len(self.train) // len(self._models)

            # Print stats for this subset of the data
            p = round(100.0 / len(self._models))
            self.log(f"\n\nRun: {run} {'='*27} >>", 1)
            self.log(f"Models: {', '.join(lst(self.models))}", 1)
            self.log(f"Size of training set: {len(self.train)} ({p}%)", 1)
            self.log(f"Size of test set: {len(self.test)}", 1)

            self._core_iteration()
            models.extend(self._models)

            # Select best models for halving
            best = pd.Series(
                data=[get_best_score(m) for m in self._models],
                index=[m._group for m in self._models],
                dtype=float,
            ).nlargest(n=len(self._models) // 2, keep="first")

            self._models = ClassMap(copy(m) for m in og_models if m.name in best.index)

            run += 1

        self._models = models  # Restore all models


@typechecked
class TrainSizing(BaseEstimator, BaseTrainer):
    """Train and evaluate the models in a [train sizing][] fashion.

    See [TrainSizingClassifier][] or [TrainSizingRegressor][] for a
    description of the remaining parameters.

    """

    def __init__(
        self, models, metric, train_sizes, est_params, n_trials, ht_params,
        n_bootstrap, parallel, errors, n_jobs, device, engine, backend,
        verbose, warnings, logger, experiment, random_state
    ):
        self.train_sizes = train_sizes
        super().__init__(
            models, metric, est_params, n_trials, ht_params, n_bootstrap,
            parallel, errors, n_jobs, device, engine, backend, verbose,
            warnings, logger, experiment, random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Train and evaluate the models.

        Read more in the [user guide][training].

        Parameters
        ----------
        *arrays: sequence of indexables
            Training set and test set. Allowed formats are:

            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        """
        self.branch._data, self.branch._idx, holdout = self._get_data(arrays)
        self.holdout = self.branch._holdout = holdout

        self.task = infer_task(self.y, goal=self.goal)
        self._prepare_parameters()

        self.log("\nTraining " + "=" * 25 + " >>", 1)
        self.log(f"Metric: {', '.join(lst(self.metric))}", 1)

        # Convert integer train_sizes to sequence
        if isinstance(self.train_sizes, INT_TYPES):
            self.train_sizes = np.linspace(1 / self.train_sizes, 1.0, self.train_sizes)

        models = ClassMap()
        og_models = ClassMap(copy(m) for m in self._models)
        for run, size in enumerate(self.train_sizes):
            # Select fraction of data to use in this run
            if size <= 1:
                frac = round(size, 2)
                train_idx = int(size * len(self.train))
            else:
                frac = round(size / len(self.train), 2)
                train_idx = size

            for m in self._models:
                m._name += str(frac).replace(".", "")  # Add frac to the name
                m._train_idx = train_idx

            # Print stats for this subset of the data
            p = round(train_idx * 100.0 / len(self.branch.train))
            self.log(f"\n\nRun: {run} {'='*27} >>", 1)
            self.log(f"Models: {', '.join(lst(self.models))}", 1)
            self.log(f"Size of training set: {train_idx} ({p}%)", 1)
            self.log(f"Size of test set: {len(self.test)}", 1)

            self._core_iteration()
            models.extend(self._models)

            # Create next models for sizing
            self._models = ClassMap(copy(m) for m in og_models)

        self._models = models  # Restore original models


@typechecked
class DirectClassifier(Direct):
    """Train and evaluate the models in a direct fashion.

    The following steps are applied to every model:

    1. Apply [hyperparameter tuning][] (optional).
    2. Fit the model on the training set using the best combination
       of hyperparameters found.
    3. Evaluate the model on the test set.
    4. Train the estimator on various [bootstrapped][bootstrapping]
       samples of the training set and evaluate again on the test set
       (optional).

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of the [predefined models][], an [ATOMModel][] or a custom
        predictor as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        [scorers][], a function with signature `function(y_true, y_pred)
        -> score`, a scorer object or a sequence of these. If None, a
        default metric is selected for every task:

        - "f1" for binary classification
        - "f1_weighted" for multiclass(-multioutput) classification
        - "average_precision" for multilabel classification

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

        - **cv: int, cv-generator, dict or sequence, default=1**<br>
          Cross-validation object or number of splits. If 1, the
          data is randomly split in a subtrain and validation set.
        - **plot: bool, dict or sequence, default=False**<br>
          Whether to plot the optimization's progress as it runs.
          Creates a canvas with two plots: the first plot shows the
          score of every trial and the second shows the distance between
          the last consecutive steps. See the [plot_trials][] method.
        - **distributions: dict, sequence or None, default=None**<br>
          Custom hyperparameter distributions. If None, it uses the
          model's predefined distributions. Read more in the
          [user guide][hyperparameter-tuning].
        - **tags: dict, sequence or None, default=None**<br>
          Custom tags for the model's trial and [mlflow run][tracking].
        - **\*\*kwargs**<br>
          Additional Keyword arguments for the constructor of the
          [study][] class or the [optimize][] method.

    n_bootstrap: int or sequence, default=0
        Number of data sets to use for [bootstrapping][]. If 0, no
        bootstrapping is performed. If sequence, the n-th value applies
        to the n-th model.

    parallel: bool, default=False
        Whether to train the models in a parallel or sequential
        fashion. Using `parallel=True` turns off the verbosity of the
        models during training. Note that many models also have
        build-in parallelizations (often when the estimator has the
        `n_jobs` parameter).

    errors: str, default="skip"
        How to handle exceptions encountered during model [training][].
        Choose from:

        - "raise": Raise any encountered exception.
        - "skip": Skip a failed model. This model is not accessible
          after training.
        - "keep": Keep the model in its state at failure. Note that
          this model can break down many other methods after training.
          This option is useful to be able to rerun hyperparameter
          optimization after failure without losing previous successful
          trials.

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
        - If None: Logging isn't used.
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
    atom.training:SuccessiveHalvingClassifier
    atom.training:TrainSizingClassifier

    Examples
    --------
    ```pycon
    from atom.training import DirectClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    train, test = train_test_split(
        X.merge(y.to_frame(), left_index=True, right_index=True),
        test_size=0.3,
    )

    runner = DirectClassifier(models=["LR", "RF"], verbose=2)
    runner.run(train, test)

    # Analyze the results
    print(runner.results)

    print(runner.evaluate())
    ```

    """

    def __init__(
        self,
        models: str | PREDICTOR | SEQUENCE | None = None,
        metric: METRIC_SELECTOR = None,
        *,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: BOOL = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: ENGINE = {"data": "numpy", "estimator": "sklearn"},
        backend: str = "loky",
        verbose: Literal[0, 1, 2] = 0,
        warnings: BOOL | WARNINGS = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: INT | None = None,
    ):
        self.goal = "class"
        super().__init__(
            models, metric, est_params, n_trials, ht_params, n_bootstrap,
            parallel, errors, n_jobs, device, engine, backend, verbose,
            warnings, logger, experiment, random_state,
        )


@typechecked
class DirectForecaster(Direct):
    """Train and evaluate the models in a direct fashion.

    The following steps are applied to every model:

    1. Apply [hyperparameter tuning][] (optional).
    2. Fit the model on the training set using the best combination
       of hyperparameters found.
    3. Evaluate the model on the test set.
    4. Train the estimator on various [bootstrapped][bootstrapping]
       samples of the training set and evaluate again on the test set
       (optional).

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of the [predefined models][], an [ATOMModel][] or a custom
        predictor as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        [scorers][], a function with signature `function(y_true, y_pred)
        -> score`, a scorer object or a sequence of these. If None, the
        default metric `mean_absolute_percentage_error` is selected.

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

        - **cv: int, cv-generator, dict or sequence, default=1**<br>
          Cross-validation object or number of splits. If 1, the
          data is randomly split in a subtrain and validation set.
        - **plot: bool, dict or sequence, default=False**<br>
          Whether to plot the optimization's progress as it runs.
          Creates a canvas with two plots: the first plot shows the
          score of every trial and the second shows the distance between
          the last consecutive steps. See the [plot_trials][] method.
        - **distributions: dict, sequence or None, default=None**<br>
          Custom hyperparameter distributions. If None, it uses the
          model's predefined distributions. Read more in the
          [user guide][hyperparameter-tuning].
        - **tags: dict, sequence or None, default=None**<br>
          Custom tags for the model's trial and [mlflow run][tracking].
        - **\*\*kwargs**<br>
          Additional Keyword arguments for the constructor of the
          [study][] class or the [optimize][] method.

    n_bootstrap: int or sequence, default=0
        Number of data sets to use for [bootstrapping][]. If 0, no
        bootstrapping is performed. If sequence, the n-th value applies
        to the n-th model.

    parallel: bool, default=False
        Whether to train the models in a parallel or sequential
        fashion. Using `parallel=True` turns off the verbosity of the
        models during training. Note that many models also have
        build-in parallelizations (often when the estimator has the
        `n_jobs` parameter).

    errors: str, default="skip"
        How to handle exceptions encountered during model [training][].
        Choose from:

        - "raise": Raise any encountered exception.
        - "skip": Skip a failed model. This model is not accessible
          after training.
        - "keep": Keep the model in its state at failure. Note that
          this model can break down many other methods after training.
          This option is useful to be able to rerun hyperparameter
          optimization after failure without losing previous successful
          trials.

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
        - If None: Logging isn't used.
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
    atom.api:ATOMForecaster
    atom.training:SuccessiveHalvingForecaster
    atom.training:TrainSizingForecaster

    Examples
    --------
    ```pycon
    from atom.training import DirectForecaster
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split

    y = load_airline()

    train, test = temporal_train_test_split(y, test_size=0.2)

    runner = DirectForecaster(models=["ES", "ETS"], verbose=2)
    runner.run(train, test)

    # Analyze the results
    print(runner.results)

    print(runner.evaluate())
    ```

    """

    def __init__(
        self,
        models: str | PREDICTOR | SEQUENCE | None = None,
        metric: METRIC_SELECTOR = None,
        *,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: BOOL = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: ENGINE = {"data": "numpy", "estimator": "sklearn"},
        backend: str = "loky",
        verbose: Literal[0, 1, 2] = 0,
        warnings: BOOL | WARNINGS = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: INT | None = None,
    ):
        self.goal = "fc"
        super().__init__(
            models, metric, est_params, n_trials, ht_params, n_bootstrap,
            parallel, errors, n_jobs, device, engine, backend, verbose, warnings,
            logger, experiment, random_state,
        )


@typechecked
class DirectRegressor(Direct):
    """Train and evaluate the models in a direct fashion.

    The following steps are applied to every model:

    1. Apply [hyperparameter tuning][] (optional).
    2. Fit the model on the training set using the best combination
       of hyperparameters found.
    3. Evaluate the model on the test set.
    4. Train the estimator on various [bootstrapped][bootstrapping]
       samples of the training set and evaluate again on the test set
       (optional).

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of the [predefined models][], an [ATOMModel][] or a custom
        predictor as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        [scorers][], a function with signature `function(y_true, y_pred)
        -> score`, a scorer object or a sequence of these. If None, the
        default metric `r2` is selected.

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

        - **cv: int, cv-generator, dict or sequence, default=1**<br>
          Cross-validation object or number of splits. If 1, the
          data is randomly split in a subtrain and validation set.
        - **plot: bool, dict or sequence, default=False**<br>
          Whether to plot the optimization's progress as it runs.
          Creates a canvas with two plots: the first plot shows the
          score of every trial and the second shows the distance between
          the last consecutive steps. See the [plot_trials][] method.
        - **distributions: dict, sequence or None, default=None**<br>
          Custom hyperparameter distributions. If None, it uses the
          model's predefined distributions. Read more in the
          [user guide][hyperparameter-tuning].
        - **tags: dict, sequence or None, default=None**<br>
          Custom tags for the model's trial and [mlflow run][tracking].
        - **\*\*kwargs**<br>
          Additional Keyword arguments for the constructor of the
          [study][] class or the [optimize][] method.

    n_bootstrap: int or sequence, default=0
        Number of data sets to use for [bootstrapping][]. If 0, no
        bootstrapping is performed. If sequence, the n-th value applies
        to the n-th model.

    parallel: bool, default=False
        Whether to train the models in a parallel or sequential
        fashion. Using `parallel=True` turns off the verbosity of the
        models during training. Note that many models also have
        build-in parallelizations (often when the estimator has the
        `n_jobs` parameter).

    errors: str, default="skip"
        How to handle exceptions encountered during model [training][].
        Choose from:

        - "raise": Raise any encountered exception.
        - "skip": Skip a failed model. This model is not accessible
          after training.
        - "keep": Keep the model in its state at failure. Note that
          this model can break down many other methods after training.
          This option is useful to be able to rerun hyperparameter
          optimization after failure without losing previous successful
          trials.

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
        - If None: Logging isn't used.
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
    atom.api:ATOMRegressor
    atom.training:SuccessiveHalvingRegressor
    atom.training:TrainSizingRegressor

    Examples
    --------
    ```pycon
    from atom.training import DirectRegressor
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True, as_frame=True)

    train, test = train_test_split(
        X.merge(y.to_frame(), left_index=True, right_index=True),
        test_size=0.3,
    )

    runner = DirectRegressor(models=["OLS", "RF"], verbose=2)
    runner.run(train, test)

    # Analyze the results
    print(runner.results)

    print(runner.evaluate())
    ```

    """

    def __init__(
        self,
        models: str | PREDICTOR | SEQUENCE | None = None,
        metric: METRIC_SELECTOR = None,
        *,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: BOOL = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: ENGINE = {"data": "numpy", "estimator": "sklearn"},
        backend: str = "loky",
        verbose: Literal[0, 1, 2] = 0,
        warnings: BOOL | str = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: INT | None = None,
    ):
        self.goal = "reg"
        super().__init__(
            models, metric, est_params, n_trials, ht_params, n_bootstrap,
            parallel, errors, n_jobs, device, engine, backend, verbose, warnings,
            logger, experiment, random_state,
        )


@typechecked
class SuccessiveHalvingClassifier(SuccessiveHalving):
    """Train and evaluate the models in a [successive halving][] fashion.

    The following steps are applied to every model (per iteration):

    1. Apply [hyperparameter tuning][] (optional).
    2. Fit the model on the training set using the best combination
       of hyperparameters found.
    3. Evaluate the model on the test set.
    4. Train the estimator on various [bootstrapped][bootstrapping]
       samples of the training set and evaluate again on the test set
       (optional).

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of the [predefined models][], an [ATOMModel][] or a custom
        predictor as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        [scorers][], a function with signature `function(y_true, y_pred)
        -> score`, a scorer object or a sequence of these. If None, a
        default metric is selected for every task:

        - "f1" for binary classification
        - "f1_weighted" for multiclass(-multioutput) classification
        - "average_precision" for multilabel classification

    skip_runs: int, default=0
        Skip last `skip_runs` runs of the successive halving.

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

        - **cv: int, cv-generator, dict or sequence, default=1**<br>
          Cross-validation object or number of splits. If 1, the
          data is randomly split in a subtrain and validation set.
        - **plot: bool, dict or sequence, default=False**<br>
          Whether to plot the optimization's progress as it runs.
          Creates a canvas with two plots: the first plot shows the
          score of every trial and the second shows the distance between
          the last consecutive steps. See the [plot_trials][] method.
        - **distributions: dict, sequence or None, default=None**<br>
          Custom hyperparameter distributions. If None, it uses the
          model's predefined distributions. Read more in the
          [user guide][hyperparameter-tuning].
        - **tags: dict, sequence or None, default=None**<br>
          Custom tags for the model's trial and [mlflow run][tracking].
        - **\*\*kwargs**<br>
          Additional Keyword arguments for the constructor of the
          [study][] class or the [optimize][] method.

    n_bootstrap: int or sequence, default=0
        Number of data sets to use for [bootstrapping][]. If 0, no
        bootstrapping is performed. If sequence, the n-th value applies
        to the n-th model.

    parallel: bool, default=False
        Whether to train the models in a parallel or sequential
        fashion. Using `parallel=True` turns off the verbosity of the
        models during training. Note that many models also have
        build-in parallelizations (often when the estimator has the
        `n_jobs` parameter).

    errors: str, default="skip"
        How to handle exceptions encountered during model [training][].
        Choose from:

        - "raise": Raise any encountered exception.
        - "skip": Skip a failed model. This model is not accessible
          after training.
        - "keep": Keep the model in its state at failure. Note that
          this model can break down many other methods after training.
          This option is useful to be able to rerun hyperparameter
          optimization after failure without losing previous successful
          trials.

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
        - If None: Logging isn't used.
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
    atom.training:DirectClassifier
    atom.training:TrainSizingClassifier

    Examples
    --------
    ```pycon
    from atom.training import SuccessiveHalvingClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    train, test = train_test_split(
        X.merge(y.to_frame(), left_index=True, right_index=True),
        test_size=0.3,
    )

    runner = SuccessiveHalvingClassifier(["LR", "RF"], verbose=2)
    runner.run(train, test)

    # Analyze the results
    print(runner.results)

    print(runner.evaluate())
    ```

    """

    def __init__(
        self,
        models: str | PREDICTOR | SEQUENCE | None = None,
        metric: METRIC_SELECTOR = None,
        *,
        skip_runs: INT = 0,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: BOOL = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: ENGINE = {"data": "numpy", "estimator": "sklearn"},
        backend: str = "loky",
        verbose: Literal[0, 1, 2] = 0,
        warnings: BOOL | str = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: INT | None = None,
    ):
        self.goal = "class"
        super().__init__(
            models, metric, skip_runs, est_params, n_trials, ht_params,
            n_bootstrap, parallel, errors, n_jobs, device, engine, backend,
            verbose, warnings, logger, experiment, random_state,
        )


@typechecked
class SuccessiveHalvingForecaster(SuccessiveHalving):
    """Train and evaluate the models in a [successive halving][] fashion.

    The following steps are applied to every model (per iteration):

    1. Apply [hyperparameter tuning][] (optional).
    2. Fit the model on the training set using the best combination
       of hyperparameters found.
    3. Evaluate the model on the test set.
    4. Train the estimator on various [bootstrapped][bootstrapping]
       samples of the training set and evaluate again on the test set
       (optional).

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of the [predefined models][], an [ATOMModel][] or a custom
        predictor as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        [scorers][], a function with signature `function(y_true, y_pred)
        -> score`, a scorer object or a sequence of these. If None, the
        default metric `mean_absolute_percentage_error` is selected.

    skip_runs: int, default=0
        Skip last `skip_runs` runs of the successive halving.

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

        - **cv: int, cv-generator, dict or sequence, default=1**<br>
          Cross-validation object or number of splits. If 1, the
          data is randomly split in a subtrain and validation set.
        - **plot: bool, dict or sequence, default=False**<br>
          Whether to plot the optimization's progress as it runs.
          Creates a canvas with two plots: the first plot shows the
          score of every trial and the second shows the distance between
          the last consecutive steps. See the [plot_trials][] method.
        - **distributions: dict, sequence or None, default=None**<br>
          Custom hyperparameter distributions. If None, it uses the
          model's predefined distributions. Read more in the
          [user guide][hyperparameter-tuning].
        - **tags: dict, sequence or None, default=None**<br>
          Custom tags for the model's trial and [mlflow run][tracking].
        - **\*\*kwargs**<br>
          Additional Keyword arguments for the constructor of the
          [study][] class or the [optimize][] method.

    n_bootstrap: int or sequence, default=0
        Number of data sets to use for [bootstrapping][]. If 0, no
        bootstrapping is performed. If sequence, the n-th value applies
        to the n-th model.

    parallel: bool, default=False
        Whether to train the models in a parallel or sequential
        fashion. Using `parallel=True` turns off the verbosity of the
        models during training. Note that many models also have
        build-in parallelizations (often when the estimator has the
        `n_jobs` parameter).

    errors: str, default="skip"
        How to handle exceptions encountered during model [training][].
        Choose from:

        - "raise": Raise any encountered exception.
        - "skip": Skip a failed model. This model is not accessible
          after training.
        - "keep": Keep the model in its state at failure. Note that
          this model can break down many other methods after training.
          This option is useful to be able to rerun hyperparameter
          optimization after failure without losing previous successful
          trials.

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
        - If None: Logging isn't used.
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
    atom.api:ATOMForecaster
    atom.training:DirectForecaster
    atom.training:TrainSizingForecaster

    Examples
    --------
    ```pycon
    from atom.training import SuccessiveHalvingForecaster
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split

    y = load_airline()

    train, test = temporal_train_test_split(y, test_size=0.2)

    runner = SuccessiveHalvingForecaster(["ETS", "ES"], verbose=2)
    runner.run(train, test)

    # Analyze the results
    print(runner.results)

    print(runner.evaluate())
    ```

    """

    def __init__(
        self,
        models: str | PREDICTOR | SEQUENCE | None = None,
        metric: METRIC_SELECTOR = None,
        *,
        skip_runs: INT = 0,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: ENGINE = {"data": "numpy", "estimator": "sklearn"},
        backend: str = "loky",
        verbose: Literal[0, 1, 2] = 0,
        warnings: bool | str = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: INT | None = None,
    ):
        self.goal = "fc"
        super().__init__(
            models, metric, skip_runs, est_params, n_trials, ht_params,
            n_bootstrap, parallel, errors, n_jobs, device, engine, backend,
            verbose, warnings, logger, experiment, random_state,
        )


@typechecked
class SuccessiveHalvingRegressor(SuccessiveHalving):
    """Train and evaluate the models in a [successive halving][] fashion.

    The following steps are applied to every model (per iteration):

    1. Apply [hyperparameter tuning][] (optional).
    2. Fit the model on the training set using the best combination
       of hyperparameters found.
    3. Evaluate the model on the test set.
    4. Train the estimator on various [bootstrapped][bootstrapping]
       samples of the training set and evaluate again on the test set
       (optional).

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of the [predefined models][], an [ATOMModel][] or a custom
        predictor as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        [scorers][], a function with signature `function(y_true, y_pred)
        -> score`, a scorer object or a sequence of these. If None, the
        default metric `r2` is selected.

    skip_runs: int, default=0
        Skip last `skip_runs` runs of the successive halving.

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

        - **cv: int, cv-generator, dict or sequence, default=1**<br>
          Cross-validation object or number of splits. If 1, the
          data is randomly split in a subtrain and validation set.
        - **plot: bool, dict or sequence, default=False**<br>
          Whether to plot the optimization's progress as it runs.
          Creates a canvas with two plots: the first plot shows the
          score of every trial and the second shows the distance between
          the last consecutive steps. See the [plot_trials][] method.
        - **distributions: dict, sequence or None, default=None**<br>
          Custom hyperparameter distributions. If None, it uses the
          model's predefined distributions. Read more in the
          [user guide][hyperparameter-tuning].
        - **tags: dict, sequence or None, default=None**<br>
          Custom tags for the model's trial and [mlflow run][tracking].
        - **\*\*kwargs**<br>
          Additional Keyword arguments for the constructor of the
          [study][] class or the [optimize][] method.

    n_bootstrap: int or sequence, default=0
        Number of data sets to use for [bootstrapping][]. If 0, no
        bootstrapping is performed. If sequence, the n-th value applies
        to the n-th model.

    parallel: bool, default=False
        Whether to train the models in a parallel or sequential
        fashion. Using `parallel=True` turns off the verbosity of the
        models during training. Note that many models also have
        build-in parallelizations (often when the estimator has the
        `n_jobs` parameter).

    errors: str, default="skip"
        How to handle exceptions encountered during model [training][].
        Choose from:

        - "raise": Raise any encountered exception.
        - "skip": Skip a failed model. This model is not accessible
          after training.
        - "keep": Keep the model in its state at failure. Note that
          this model can break down many other methods after training.
          This option is useful to be able to rerun hyperparameter
          optimization after failure without losing previous successful
          trials.

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
        - If None: Logging isn't used.
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
    atom.api:ATOMRegressor
    atom.training:DirectRegressor
    atom.training:TrainSizingRegressor

    Examples
    --------
    ```pycon
    from atom.training import SuccessiveHalvingRegressor
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True, as_frame=True)

    train, test = train_test_split(
        X.merge(y.to_frame(), left_index=True, right_index=True),
        test_size=0.3,
    )

    runner = SuccessiveHalvingRegressor(["OLS", "RF"], verbose=2)
    runner.run(train, test)

    # Analyze the results
    print(runner.results)

    print(runner.evaluate())
    ```

    """

    def __init__(
        self,
        models: str | PREDICTOR | SEQUENCE | None = None,
        metric: METRIC_SELECTOR = None,
        *,
        skip_runs: INT = 0,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: ENGINE = {"data": "numpy", "estimator": "sklearn"},
        backend: str = "loky",
        verbose: Literal[0, 1, 2] = 0,
        warnings: bool | str = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: INT | None = None,
    ):
        self.goal = "reg"
        super().__init__(
            models, metric, skip_runs, est_params, n_trials, ht_params,
            n_bootstrap, parallel, errors, n_jobs, device, engine, backend,
            verbose, warnings, logger, experiment, random_state,
        )


@typechecked
class TrainSizingClassifier(TrainSizing):
    """Train and evaluate the models in a [train sizing][] fashion.

    The following steps are applied to every model (per iteration):

    1. Apply [hyperparameter tuning][] (optional).
    2. Fit the model on the training set using the best combination
       of hyperparameters found.
    3. Evaluate the model on the test set.
    4. Train the estimator on various [bootstrapped][bootstrapping]
       samples of the training set and evaluate again on the test set
       (optional).

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of the [predefined models][], an [ATOMModel][] or a custom
        predictor as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        [scorers][], a function with signature `function(y_true, y_pred)
        -> score`, a scorer object or a sequence of these. If None, a
        default metric is selected for every task:

        - "f1" for binary classification
        - "f1_weighted" for multiclass(-multioutput) classification
        - "average_precision" for multilabel classification

    train_sizes: int or sequence, default=5
        Sequence of training set sizes used to run the trainings.

        - If int: Number of equally distributed splits, i.e. for a value
          `N`, it's equal to `np.linspace(1.0/N, 1.0, N)`.
        - If sequence: Fraction of the training set when <=1, else total
          number of samples.

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

        - **cv: int, cv-generator, dict or sequence, default=1**<br>
          Cross-validation object or number of splits. If 1, the
          data is randomly split in a subtrain and validation set.
        - **plot: bool, dict or sequence, default=False**<br>
          Whether to plot the optimization's progress as it runs.
          Creates a canvas with two plots: the first plot shows the
          score of every trial and the second shows the distance between
          the last consecutive steps. See the [plot_trials][] method.
        - **distributions: dict, sequence or None, default=None**<br>
          Custom hyperparameter distributions. If None, it uses the
          model's predefined distributions. Read more in the
          [user guide][hyperparameter-tuning].
        - **tags: dict, sequence or None, default=None**<br>
          Custom tags for the model's trial and [mlflow run][tracking].
        - **\*\*kwargs**<br>
          Additional Keyword arguments for the constructor of the
          [study][] class or the [optimize][] method.

    n_bootstrap: int or sequence, default=0
        Number of data sets to use for [bootstrapping][]. If 0, no
        bootstrapping is performed. If sequence, the n-th value applies
        to the n-th model.

    parallel: bool, default=False
        Whether to train the models in a parallel or sequential
        fashion. Using `parallel=True` turns off the verbosity of the
        models during training. Note that many models also have
        build-in parallelizations (often when the estimator has the
        `n_jobs` parameter).

    errors: str, default="skip"
        How to handle exceptions encountered during model [training][].
        Choose from:

        - "raise": Raise any encountered exception.
        - "skip": Skip a failed model. This model is not accessible
          after training.
        - "keep": Keep the model in its state at failure. Note that
          this model can break down many other methods after training.
          This option is useful to be able to rerun hyperparameter
          optimization after failure without losing previous successful
          trials.

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
        - If None: Logging isn't used.
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
    atom.api:ATOMRegressor
    atom.training:DirectRegressor
    atom.training:SuccessiveHalvingRegressor

    Examples
    --------
    ```pycon
    from atom.training import TrainSizingClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    train, test = train_test_split(
        X.merge(y.to_frame(), left_index=True, right_index=True),
        test_size=0.3,
    )

    runner = TrainSizingClassifier(models="LR", verbose=2)
    runner.run(train, test)

    # Analyze the results
    print(runner.results)

    print(runner.evaluate())
    ```

    """

    def __init__(
        self,
        models: str | PREDICTOR | SEQUENCE | None = None,
        metric: METRIC_SELECTOR = None,
        *,
        train_sizes: INT | SEQUENCE = 5,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: ENGINE = {"data": "numpy", "estimator": "sklearn"},
        backend: str = "loky",
        verbose: Literal[0, 1, 2] = 0,
        warnings: bool | str = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: INT | None = None,
    ):
        self.goal = "class"
        super().__init__(
            models, metric, train_sizes, est_params, n_trials, ht_params,
            n_bootstrap, parallel, errors, n_jobs, device, engine, backend,
            verbose, warnings, logger, experiment, random_state,
        )


@typechecked
class TrainSizingForecaster(TrainSizing):
    """Train and evaluate the models in a [train sizing][] fashion.

    The following steps are applied to every model (per iteration):

    1. Apply [hyperparameter tuning][] (optional).
    2. Fit the model on the training set using the best combination
       of hyperparameters found.
    3. Evaluate the model on the test set.
    4. Train the estimator on various [bootstrapped][bootstrapping]
       samples of the training set and evaluate again on the test set
       (optional).

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of the [predefined models][], an [ATOMModel][] or a custom
        predictor as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        [scorers][], a function with signature `function(y_true, y_pred)
        -> score`, a scorer object or a sequence of these. If None, the
        default metric `mean_absolute_percentage_error` is selected.

    train_sizes: int or sequence, default=5
        Sequence of training set sizes used to run the trainings.

        - If int: Number of equally distributed splits, i.e. for a value
          `N`, it's equal to `np.linspace(1.0/N, 1.0, N)`.
        - If sequence: Fraction of the training set when <=1, else total
          number of samples.

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

        - **cv: int, cv-generator, dict or sequence, default=1**<br>
          Cross-validation object or number of splits. If 1, the
          data is randomly split in a subtrain and validation set.
        - **plot: bool, dict or sequence, default=False**<br>
          Whether to plot the optimization's progress as it runs.
          Creates a canvas with two plots: the first plot shows the
          score of every trial and the second shows the distance between
          the last consecutive steps. See the [plot_trials][] method.
        - **distributions: dict, sequence or None, default=None**<br>
          Custom hyperparameter distributions. If None, it uses the
          model's predefined distributions. Read more in the
          [user guide][hyperparameter-tuning].
        - **tags: dict, sequence or None, default=None**<br>
          Custom tags for the model's trial and [mlflow run][tracking].
        - **\*\*kwargs**<br>
          Additional Keyword arguments for the constructor of the
          [study][] class or the [optimize][] method.

    n_bootstrap: int or sequence, default=0
        Number of data sets to use for [bootstrapping][]. If 0, no
        bootstrapping is performed. If sequence, the n-th value applies
        to the n-th model.

    parallel: bool, default=False
        Whether to train the models in a parallel or sequential
        fashion. Using `parallel=True` turns off the verbosity of the
        models during training. Note that many models also have
        build-in parallelizations (often when the estimator has the
        `n_jobs` parameter).

    errors: str, default="skip"
        How to handle exceptions encountered during model [training][].
        Choose from:

        - "raise": Raise any encountered exception.
        - "skip": Skip a failed model. This model is not accessible
          after training.
        - "keep": Keep the model in its state at failure. Note that
          this model can break down many other methods after training.
          This option is useful to be able to rerun hyperparameter
          optimization after failure without losing previous successful
          trials.

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
        - If None: Logging isn't used.
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
    atom.api:ATOMForecaster
    atom.training:DirectForecaster
    atom.training:SuccessiveHalvingForecaster

    Examples
    --------
    ```pycon
    from atom.training import TrainSizingForecaster
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split

    y = load_airline()

    train, test = temporal_train_test_split(y, test_size=0.2)

    runner = TrainSizingForecaster(["ETS", "ES"], verbose=2)
    runner.run(train, test)

    # Analyze the results
    print(runner.results)

    print(runner.evaluate())
    ```

    """

    def __init__(
        self,
        models: str | PREDICTOR | SEQUENCE | None = None,
        metric: METRIC_SELECTOR = None,
        *,
        train_sizes: INT | SEQUENCE = 5,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: ENGINE = {"data": "numpy", "estimator": "sklearn"},
        backend: str = "loky",
        verbose: Literal[0, 1, 2] = 0,
        warnings: bool | str = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: INT | None = None,
    ):
        self.goal = "fc"
        super().__init__(
            models, metric, train_sizes, est_params, n_trials, ht_params,
            n_bootstrap, parallel, errors, n_jobs, device, engine, backend,
            verbose, warnings, logger, experiment, random_state,
        )


@typechecked
class TrainSizingRegressor(TrainSizing):
    """Train and evaluate the models in a [train sizing][] fashion.

    The following steps are applied to every model (per iteration):

    1. Apply [hyperparameter tuning][] (optional).
    2. Fit the model on the training set using the best combination
       of hyperparameters found.
    3. Evaluate the model on the test set.
    4. Train the estimator on various [bootstrapped][bootstrapping]
       samples of the training set and evaluate again on the test set
       (optional).

    Parameters
    ----------
    models: str, estimator or sequence, default=None
        Models to fit to the data. Allowed inputs are: an acronym from
        any of the [predefined models][], an [ATOMModel][] or a custom
        predictor as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, default=None
        Metric on which to fit the models. Choose from any of sklearn's
        [scorers][], a function with signature `function(y_true, y_pred)
        -> score`, a scorer object or a sequence of these. If None, the
        default metric `r2` is selected.

    train_sizes: int or sequence, default=5
        Sequence of training set sizes used to run the trainings.

        - If int: Number of equally distributed splits, i.e. for a value
          `N`, it's equal to `np.linspace(1.0/N, 1.0, N)`.
        - If sequence: Fraction of the training set when <=1, else total
          number of samples.

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

        - **cv: int, cv-generator, dict or sequence, default=1**<br>
          Cross-validation object or number of splits. If 1, the
          data is randomly split in a subtrain and validation set.
        - **plot: bool, dict or sequence, default=False**<br>
          Whether to plot the optimization's progress as it runs.
          Creates a canvas with two plots: the first plot shows the
          score of every trial and the second shows the distance between
          the last consecutive steps. See the [plot_trials][] method.
        - **distributions: dict, sequence or None, default=None**<br>
          Custom hyperparameter distributions. If None, it uses the
          model's predefined distributions. Read more in the
          [user guide][hyperparameter-tuning].
        - **tags: dict, sequence or None, default=None**<br>
          Custom tags for the model's trial and [mlflow run][tracking].
        - **\*\*kwargs**<br>
          Additional Keyword arguments for the constructor of the
          [study][] class or the [optimize][] method.

    n_bootstrap: int or sequence, default=0
        Number of data sets to use for [bootstrapping][]. If 0, no
        bootstrapping is performed. If sequence, the n-th value applies
        to the n-th model.

    parallel: bool, default=False
        Whether to train the models in a parallel or sequential
        fashion. Using `parallel=True` turns off the verbosity of the
        models during training. Note that many models also have
        build-in parallelizations (often when the estimator has the
        `n_jobs` parameter).

    errors: str, default="skip"
        How to handle exceptions encountered during model [training][].
        Choose from:

        - "raise": Raise any encountered exception.
        - "skip": Skip a failed model. This model is not accessible
          after training.
        - "keep": Keep the model in its state at failure. Note that
          this model can break down many other methods after training.
          This option is useful to be able to rerun hyperparameter
          optimization after failure without losing previous successful
          trials.

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
        - If None: Logging isn't used.
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
    atom.api:ATOMRegressor
    atom.training:DirectRegressor
    atom.training:SuccessiveHalvingRegressor

    Examples
    --------
    ```pycon
    from atom.training import TrainSizingRegressor
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True, as_frame=True)

    train, test = train_test_split(
        X.merge(y.to_frame(), left_index=True, right_index=True),
        test_size=0.3,
    )

    runner = TrainSizingRegressor(models="OLS", verbose=2)
    runner.run(train, test)

    # Analyze the results
    print(runner.results)

    print(runner.evaluate())
    ```

    """

    def __init__(
        self,
        models: str | PREDICTOR | SEQUENCE | None = None,
        metric: METRIC_SELECTOR = None,
        *,
        train_sizes: INT | SEQUENCE = 5,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: ENGINE = {"data": "numpy", "estimator": "sklearn"},
        backend: str = "loky",
        verbose: Literal[0, 1, 2] = 0,
        warnings: bool | str = False,
        logger: str | Logger | None = None,
        experiment: str | None = None,
        random_state: INT | None = None,
    ):
        self.goal = "reg"
        super().__init__(
            models, metric, train_sizes, est_params, n_trials, ht_params,
            n_bootstrap, parallel, errors, n_jobs, device, engine, backend,
            verbose, warnings, logger, experiment, random_state,
        )

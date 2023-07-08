# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the training classes.

"""

from __future__ import annotations

from copy import copy
from logging import Logger
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from atom.basetrainer import BaseTrainer
from atom.utils import (
    INT, INT_TYPES, SEQUENCE, ClassMap, Predictor, composed, crash,
    get_best_score, infer_task, lst, method_to_log,
)


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
        - "r2" for regression or multioutput regression

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
          optimization after failure without losing previous succesfull
          trials.

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

    backend: str, default="loky"
        Parallelization backend. Choose from:

        - "loky": Single-node, process-based parallelism.
        - "multiprocessing": Legacy single-node, process-based
          parallelism. Less robust than 'loky'.
        - "threading": Single-node, thread-based parallelism.
        - "ray": Multi-node, process-based parallelism.

        Selecting the ray backend also parallelizes the data using
        [modin][].

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
    atom.training:SuccessiveHalvingClassifier
    atom.training:TrainSizingClassifier

    Examples
    --------
    ```pycon
    >>> from atom.training import DirectClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    >>> train, test = train_test_split(
    ...     X.merge(y.to_frame(), left_index=True, right_index=True),
    ...     test_size=0.3,
    ... )

    >>> runner = DirectClassifier(models=["LR", "RF"], metric="auc", verbose=2)
    >>> runner.run(train, test)

    Training ========================= >>
    Models: LR, RF
    Metric: roc_auc


    Results for LogisticRegression:
    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 0.9925
    Test evaluation --> roc_auc: 0.9871
    Time elapsed: 0.035s
    -------------------------------------------------
    Total time: 0.035s


    Results for RandomForest:
    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 1.0
    Test evaluation --> roc_auc: 0.9807
    Time elapsed: 0.137s
    -------------------------------------------------
    Total time: 0.137s


    Final results ==================== >>
    Total time: 0.173s
    -------------------------------------
    LogisticRegression --> roc_auc: 0.9871 !
    RandomForest       --> roc_auc: 0.9807

    >>> # Analyze the results
    >>> runner.evaluate()

        accuracy  average_precision  ...  precision  recall  roc_auc
    LR    0.9357             0.9923  ...     0.9533  0.9444   0.9325
    RF    0.9532             0.9810  ...     0.9464  0.9815   0.9431

    [2 rows x 9 columns]

    ```

    """

    def __init__(
        self,
        models: str | Predictor | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        backend: str = "loky",
        verbose: INT = 0,
        warnings: bool | str = False,
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
        -> score`, a scorer object or a sequence of these. If None, a
        default metric is selected for every task:

        - "f1" for binary classification
        - "f1_weighted" for multiclass(-multioutput) classification
        - "average_precision" for multilabel classification
        - "r2" for regression or multioutput regression

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
          optimization after failure without losing previous succesfull
          trials.

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
    atom.api:ATOMRegressor
    atom.training:SuccessiveHalvingRegressor
    atom.training:TrainSizingRegressor

    Examples
    --------
    ```pycon
    >>> from atom.training import DirectRegressor
    >>> from sklearn.datasets import load_digits

    >>> X, y = load_digits(return_X_y=True, as_frame=True)
    >>> train, test = train_test_split(
    ...     X.merge(y.to_frame(), left_index=True, right_index=True),
    ...     test_size=0.3,
    ... )

    >>> runner = DirectClassifier(models=["OLS", "RF"], metric="r2", verbose=2)
    >>> runner.run(train, test)

    Models: OLS, RF
    Metric: r2


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.5881
    Test evaluation --> r2: 0.6029
    Time elapsed: 0.022s
    -------------------------------------------------
    Total time: 0.022s


    Results for RandomForest:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.981
    Test evaluation --> r2: 0.8719
    Time elapsed: 0.838s
    -------------------------------------------------
    Total time: 0.838s


    Final results ==================== >>
    Total time: 0.862s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.6029
    RandomForest         --> r2: 0.8719 !

    >>> # Analyze the results
    >>> runner.evaluate()

         neg_mean_absolute_error  ...  neg_root_mean_squared_error
    OLS                  -1.4124  ...                      -1.8109
    RF                   -0.6569  ...                      -1.0692

    [2 rows x 6 columns]

    ```

    """

    def __init__(
        self,
        models: str | Predictor | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        backend: str = "loky",
        verbose: INT = 0,
        warnings: bool | str = False,
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
        -> score`, a scorer object or a sequence of these. If None, a
        default metric is selected for every task:

        - "f1" for binary classification
        - "f1_weighted" for multiclass(-multioutput) classification
        - "average_precision" for multilabel classification
        - "r2" for regression or multioutput regression

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
          optimization after failure without losing previous succesfull
          trials.

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
    atom.api:ATOMRegressor
    atom.training:SuccessiveHalvingRegressor
    atom.training:TrainSizingRegressor

    Examples
    --------
    ```pycon
    >>> from atom.training import DirectRegressor
    >>> from sklearn.datasets import load_digits

    >>> X, y = load_digits(return_X_y=True, as_frame=True)
    >>> train, test = train_test_split(
    ...     X.merge(y.to_frame(), left_index=True, right_index=True),
    ...     test_size=0.3,
    ... )

    >>> runner = DirectClassifier(models=["OLS", "RF"], metric="r2", verbose=2)
    >>> runner.run(train, test)

    Models: OLS, RF
    Metric: r2


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.5881
    Test evaluation --> r2: 0.6029
    Time elapsed: 0.022s
    -------------------------------------------------
    Total time: 0.022s


    Results for RandomForest:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.981
    Test evaluation --> r2: 0.8719
    Time elapsed: 0.838s
    -------------------------------------------------
    Total time: 0.838s


    Final results ==================== >>
    Total time: 0.862s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.6029
    RandomForest         --> r2: 0.8719 !

    >>> # Analyze the results
    >>> runner.evaluate()

         neg_mean_absolute_error  ...  neg_root_mean_squared_error
    OLS                  -1.4124  ...                      -1.8109
    RF                   -0.6569  ...                      -1.0692

    [2 rows x 6 columns]

    ```

    """

    def __init__(
        self,
        models: str | Predictor | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        backend: str = "loky",
        verbose: INT = 0,
        warnings: bool | str = False,
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
        - "r2" for regression or multioutput regression

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

        - **cv: int, dict or sequence, default=1**<br>
          Number of folds for the cross-validation. If 1, the training
          set is randomly split in a subtrain and validation set.
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
          optimization after failure without losing previous succesfull
          trials.

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
    atom.training:DirectClassifier
    atom.training:TrainSizingClassifier

    Examples
    --------
    ```pycon
    >>> from atom.training import SuccessiveHalvingClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    >>> train, test = train_test_split(
    ...     X.merge(y.to_frame(), left_index=True, right_index=True),
    ...     test_size=0.3,
    ... )

    >>> runner = SuccessiveHalvingClassifier(["LR", "RF"], metric="auc", verbose=2)
    >>> runner.run(train, test)

    Training ========================= >>
    Metric: roc_auc

    Run: 0 ================================ >>
    Models: LR2, RF2
    Size of training set: 398 (50%)
    Size of test set: 171


    Results for LogisticRegression:
    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 0.984
    Test evaluation --> roc_auc: 0.9793
    Time elapsed: 0.018s
    -------------------------------------------------
    Total time: 0.018s


    Results for RandomForest:
    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 1.0
    Test evaluation --> roc_auc: 0.9805
    Time elapsed: 0.113s
    -------------------------------------------------
    Total time: 0.113s


    Final results ==================== >>
    Total time: 0.131s
    -------------------------------------
    LogisticRegression --> roc_auc: 0.9793
    RandomForest       --> roc_auc: 0.9805 !


    Run: 1 ================================ >>
    Models: RF1
    Size of training set: 398 (100%)
    Size of test set: 171


    Results for RandomForest:
    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 1.0
    Test evaluation --> roc_auc: 0.9806
    Time elapsed: 0.137s
    -------------------------------------------------
    Total time: 0.137s


    Final results ==================== >>
    Total time: 0.137s
    -------------------------------------
    RandomForest --> roc_auc: 0.9806

    >>> # Analyze the results
    >>> runner.evaluate()

         accuracy  average_precision   ...  precision  recall  roc_auc
    LR2    0.9006             0.9878   ...     0.9099  0.9352   0.9793
    RF2    0.9474             0.9800   ...     0.9381  0.9815   0.9805
    RF1    0.9532             0.9806   ...     0.9545  0.9722   0.9806

    [3 rows x 9 columns]

    ```

    """

    def __init__(
        self,
        models: str | Predictor | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        skip_runs: INT = 0,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        backend: str = "loky",
        verbose: INT = 0,
        warnings: bool | str = False,
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
        -> score`, a scorer object or a sequence of these. If None, a
        default metric is selected for every task:

        - "f1" for binary classification
        - "f1_weighted" for multiclass(-multioutput) classification
        - "average_precision" for multilabel classification
        - "r2" for regression or multioutput regression

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

        - **cv: int, dict or sequence, default=1**<br>
          Number of folds for the cross-validation. If 1, the training
          set is randomly split in a subtrain and validation set.
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
          optimization after failure without losing previous succesfull
          trials.

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
    atom.api:ATOMRegressor
    atom.training:DirectRegressor
    atom.training:TrainSizingRegressor

    Examples
    --------
    ```pycon
    >>> from atom.training import SuccessiveHalvingRegressor
    >>> from sklearn.datasets import load_digits

    >>> X, y = load_digits(return_X_y=True, as_frame=True)
    >>> train, test = train_test_split(
    ...     X.merge(y.to_frame(), left_index=True, right_index=True),
    ...     test_size=0.3,
    ... )

    >>> runner = SuccessiveHalvingRegressor(["OLS", "RF"], metric="r2", verbose=2)
    >>> runner.run(train, test)

    Training ========================= >>
    Metric: r2


    Run: 0 =========================== >>
    Models: OLS2, RF2
    Size of training set: 398 (50%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.7878
    Test evaluation --> r2: 0.6764
    Time elapsed: 0.007s
    -------------------------------------------------
    Total time: 0.007s


    Results for RandomForest:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.9755
    Test evaluation --> r2: 0.8189
    Time elapsed: 0.132s
    -------------------------------------------------
    Total time: 0.132s


    Final results ==================== >>
    Total time: 0.140s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.6764
    RandomForest         --> r2: 0.8189 !


    Run: 1 =========================== >>
    Models: RF1
    Size of training set: 398 (100%)
    Size of test set: 171


    Results for RandomForest:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.9803
    Test evaluation --> r2: 0.8092
    Time elapsed: 0.217s
    -------------------------------------------------
    Total time: 0.217s


    Final results ==================== >>
    Total time: 0.217s
    -------------------------------------
    RandomForest --> r2: 0.8092

    >>> # Analyze the results
    >>> runner.evaluate()

          neg_mean_absolute_error  ...  neg_root_mean_squared_error
    OLS2                  -0.1982  ...                      -0.2744
    RF2                   -0.0970  ...                      -0.2053
    RF1                   -0.0974  ...                      -0.2107

    [3 rows x 6 columns]

    ```

    """

    def __init__(
        self,
        models: str | Predictor | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        skip_runs: INT = 0,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        backend: str = "loky",
        verbose: INT = 0,
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
        -> score`, a scorer object or a sequence of these. If None, a
        default metric is selected for every task:

        - "f1" for binary classification
        - "f1_weighted" for multiclass(-multioutput) classification
        - "average_precision" for multilabel classification
        - "r2" for regression or multioutput regression

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

        - **cv: int, dict or sequence, default=1**<br>
          Number of folds for the cross-validation. If 1, the training
          set is randomly split in a subtrain and validation set.
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
          optimization after failure without losing previous succesfull
          trials.

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
    atom.api:ATOMRegressor
    atom.training:DirectRegressor
    atom.training:TrainSizingRegressor

    Examples
    --------
    ```pycon
    >>> from atom.training import SuccessiveHalvingRegressor
    >>> from sklearn.datasets import load_digits

    >>> X, y = load_digits(return_X_y=True, as_frame=True)
    >>> train, test = train_test_split(
    ...     X.merge(y.to_frame(), left_index=True, right_index=True),
    ...     test_size=0.3,
    ... )

    >>> runner = SuccessiveHalvingRegressor(["OLS", "RF"], metric="r2", verbose=2)
    >>> runner.run(train, test)

    Training ========================= >>
    Metric: r2


    Run: 0 =========================== >>
    Models: OLS2, RF2
    Size of training set: 398 (50%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.7878
    Test evaluation --> r2: 0.6764
    Time elapsed: 0.007s
    -------------------------------------------------
    Total time: 0.007s


    Results for RandomForest:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.9755
    Test evaluation --> r2: 0.8189
    Time elapsed: 0.132s
    -------------------------------------------------
    Total time: 0.132s


    Final results ==================== >>
    Total time: 0.140s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.6764
    RandomForest         --> r2: 0.8189 !


    Run: 1 =========================== >>
    Models: RF1
    Size of training set: 398 (100%)
    Size of test set: 171


    Results for RandomForest:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.9803
    Test evaluation --> r2: 0.8092
    Time elapsed: 0.217s
    -------------------------------------------------
    Total time: 0.217s


    Final results ==================== >>
    Total time: 0.217s
    -------------------------------------
    RandomForest --> r2: 0.8092

    >>> # Analyze the results
    >>> runner.evaluate()

          neg_mean_absolute_error  ...  neg_root_mean_squared_error
    OLS2                  -0.1982  ...                      -0.2744
    RF2                   -0.0970  ...                      -0.2053
    RF1                   -0.0974  ...                      -0.2107

    [3 rows x 6 columns]

    ```

    """

    def __init__(
        self,
        models: str | Predictor | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        skip_runs: INT = 0,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        backend: str = "loky",
        verbose: INT = 0,
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
        - "r2" for regression or multioutput regression

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

        - **cv: int, dict or sequence, default=1**<br>
          Number of folds for the cross-validation. If 1, the training
          set is randomly split in a subtrain and validation set.
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
          optimization after failure without losing previous succesfull
          trials.

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
    atom.api:ATOMRegressor
    atom.training:DirectRegressor
    atom.training:SuccessiveHalvingRegressor

    Examples
    --------
    ```pycon
    >>> from atom.training import TrainSizingClassifier
    >>> from sklearn.datasets import load_breast_cancer

    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    >>> train, test = train_test_split(
    ...     X.merge(y.to_frame(), left_index=True, right_index=True),
    ...     test_size=0.3,
    ... )

    >>> runner = TrainSizingClassifier(models="LR", metric="auc", verbose=2)
    >>> runner.run(train, test)

    Training ========================= >>
    Metric: roc_auc


    Run: 0 =========================== >>
    Models: LR02
    Size of training set: 79 (20%)
    Size of test set: 171


    Results for LogisticRegression:
    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 0.9846
    Test evaluation --> roc_auc: 0.9737
    Time elapsed: 0.017s
    -------------------------------------------------
    Total time: 0.017s


    Final results ==================== >>
    Total time: 0.018s
    -------------------------------------
    LogisticRegression --> roc_auc: 0.9737


    Run: 1 =========================== >>
    Models: LR04
    Size of training set: 159 (40%)
    Size of test set: 171


    Results for LogisticRegression:
    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 0.9855
    Test evaluation --> roc_auc: 0.9838
    Time elapsed: 0.018s
    -------------------------------------------------
    Total time: 0.018s


    Final results ==================== >>
    Total time: 0.019s
    -------------------------------------
    LogisticRegression --> roc_auc: 0.9838


    Run: 2 =========================== >>
    Models: LR06
    Size of training set: 238 (60%)
    Size of test set: 171


    Results for LogisticRegression:
    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 0.9898
    Test evaluation --> roc_auc: 0.9813
    Time elapsed: 0.018s
    -------------------------------------------------
    Total time: 0.018s


    Final results ==================== >>
    Total time: 0.018s
    -------------------------------------
    LogisticRegression --> roc_auc: 0.9813


    Run: 3 =========================== >>
    Models: LR08
    Size of training set: 318 (80%)
    Size of test set: 171


    Results for LogisticRegression:
    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 0.9936
    Test evaluation --> roc_auc: 0.9816
    Time elapsed: 0.038s
    -------------------------------------------------
    Total time: 0.038s


    Final results ==================== >>
    Total time: 0.038s
    -------------------------------------
    LogisticRegression --> roc_auc: 0.9816


    Run: 4 =========================== >>
    Models: LR10
    Size of training set: 398 (100%)
    Size of test set: 171


    Results for LogisticRegression:
    Fit ---------------------------------------------
    Train evaluation --> roc_auc: 0.9925
    Test evaluation --> roc_auc: 0.9871
    Time elapsed: 0.040s
    -------------------------------------------------
    Total time: 0.040s


    Final results ==================== >>
    Total time: 0.041s
    -------------------------------------
    LogisticRegression --> roc_auc: 0.9871

    >>> # Analyze the results
    >>> runner.evaluate()

          accuracy  average_precision  ...  recall  roc_auc
    LR02    0.8947             0.9835  ...  0.8981   0.9737
    LR04    0.9181             0.9907  ...  0.9352   0.9838
    LR06    0.9415             0.9888  ...  0.9444   0.9813
    LR08    0.9474             0.9878  ...  0.9630   0.9816
    LR10    0.9357             0.9923  ...  0.9444   0.9871

    [5 rows x 9 columns]

    ```

    """

    def __init__(
        self,
        models: str | Predictor | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        train_sizes: INT | SEQUENCE = 5,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        backend: str = "loky",
        verbose: INT = 0,
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
        -> score`, a scorer object or a sequence of these. If None, a
        default metric is selected for every task:

        - "f1" for binary classification
        - "f1_weighted" for multiclass(-multioutput) classification
        - "average_precision" for multilabel classification
        - "r2" for regression or multioutput regression

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

        - **cv: int, dict or sequence, default=1**<br>
          Number of folds for the cross-validation. If 1, the training
          set is randomly split in a subtrain and validation set.
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
          optimization after failure without losing previous succesfull
          trials.

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
    atom.api:ATOMRegressor
    atom.training:DirectRegressor
    atom.training:SuccessiveHalvingRegressor

    Examples
    --------
    ```pycon
    >>> from atom.training import TrainSizingRegressor
    >>> from sklearn.datasets import load_digits

    >>> X, y = load_digits(return_X_y=True, as_frame=True)
    >>> train, test = train_test_split(
    ...     X.merge(y.to_frame(), left_index=True, right_index=True),
    ...     test_size=0.3,
    ... )

    >>> runner = TrainSizingRegressor(models="OLS", metric="r2", verbose=2)
    >>> runner.run(train, test)

    Training ========================= >>
    Metric: r2


    Run: 0 =========================== >>
    Models: OLS02
    Size of training set: 79 (20%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.8554
    Test evaluation --> r2: 0.4273
    Time elapsed: 0.008s
    -------------------------------------------------
    Total time: 0.008s


    Final results ==================== >>
    Total time: 0.107s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.4273 ~


    Run: 1 =========================== >>
    Models: OLS04
    Size of training set: 159 (40%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.7987
    Test evaluation --> r2: 0.653
    Time elapsed: 0.008s
    -------------------------------------------------
    Total time: 0.008s


    Final results ==================== >>
    Total time: 0.129s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.653


    Run: 2 =========================== >>
    Models: OLS06
    Size of training set: 238 (60%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.7828
    Test evaluation --> r2: 0.7161
    Time elapsed: 0.008s
    -------------------------------------------------
    Total time: 0.008s


    Final results ==================== >>
    Total time: 0.156s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.7161


    Run: 3 =========================== >>
    Models: OLS08
    Size of training set: 318 (80%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.7866
    Test evaluation --> r2: 0.7306
    Time elapsed: 0.009s
    -------------------------------------------------
    Total time: 0.009s


    Final results ==================== >>
    Total time: 0.187s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.7306


    Run: 4 =========================== >>
    Models: OLS10
    Size of training set: 398 (100%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.7798
    Test evaluation --> r2: 0.7394
    Time elapsed: 0.009s
    -------------------------------------------------
    Total time: 0.009s


    Final results ==================== >>
    Total time: 0.226s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.7394

    >>> # Analyze the results
    >>> runner.evaluate()

           neg_mean_absolute_error  ...  neg_root_mean_squared_error
    OLS02                  -0.2766  ...                      -0.3650
    OLS04                  -0.2053  ...                      -0.2841
    OLS06                  -0.1957  ...                      -0.2570
    OLS08                  -0.1928  ...                      -0.2504
    OLS10                  -0.1933  ...                      -0.2463

    [5 rows x 6 columns]

    ```

    """

    def __init__(
        self,
        models: str | Predictor | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        train_sizes: INT | SEQUENCE = 5,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        backend: str = "loky",
        verbose: INT = 0,
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
        -> score`, a scorer object or a sequence of these. If None, a
        default metric is selected for every task:

        - "f1" for binary classification
        - "f1_weighted" for multiclass(-multioutput) classification
        - "average_precision" for multilabel classification
        - "r2" for regression or multioutput regression

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

        - **cv: int, dict or sequence, default=1**<br>
          Number of folds for the cross-validation. If 1, the training
          set is randomly split in a subtrain and validation set.
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
          optimization after failure without losing previous succesfull
          trials.

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
    atom.api:ATOMRegressor
    atom.training:DirectRegressor
    atom.training:SuccessiveHalvingRegressor

    Examples
    --------
    ```pycon
    >>> from atom.training import TrainSizingRegressor
    >>> from sklearn.datasets import load_digits

    >>> X, y = load_digits(return_X_y=True, as_frame=True)
    >>> train, test = train_test_split(
    ...     X.merge(y.to_frame(), left_index=True, right_index=True),
    ...     test_size=0.3,
    ... )

    >>> runner = TrainSizingRegressor(models="OLS", metric="r2", verbose=2)
    >>> runner.run(train, test)

    Training ========================= >>
    Metric: r2


    Run: 0 =========================== >>
    Models: OLS02
    Size of training set: 79 (20%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.8554
    Test evaluation --> r2: 0.4273
    Time elapsed: 0.008s
    -------------------------------------------------
    Total time: 0.008s


    Final results ==================== >>
    Total time: 0.107s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.4273 ~


    Run: 1 =========================== >>
    Models: OLS04
    Size of training set: 159 (40%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.7987
    Test evaluation --> r2: 0.653
    Time elapsed: 0.008s
    -------------------------------------------------
    Total time: 0.008s


    Final results ==================== >>
    Total time: 0.129s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.653


    Run: 2 =========================== >>
    Models: OLS06
    Size of training set: 238 (60%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.7828
    Test evaluation --> r2: 0.7161
    Time elapsed: 0.008s
    -------------------------------------------------
    Total time: 0.008s


    Final results ==================== >>
    Total time: 0.156s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.7161


    Run: 3 =========================== >>
    Models: OLS08
    Size of training set: 318 (80%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.7866
    Test evaluation --> r2: 0.7306
    Time elapsed: 0.009s
    -------------------------------------------------
    Total time: 0.009s


    Final results ==================== >>
    Total time: 0.187s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.7306


    Run: 4 =========================== >>
    Models: OLS10
    Size of training set: 398 (100%)
    Size of test set: 171


    Results for OrdinaryLeastSquares:
    Fit ---------------------------------------------
    Train evaluation --> r2: 0.7798
    Test evaluation --> r2: 0.7394
    Time elapsed: 0.009s
    -------------------------------------------------
    Total time: 0.009s


    Final results ==================== >>
    Total time: 0.226s
    -------------------------------------
    OrdinaryLeastSquares --> r2: 0.7394

    >>> # Analyze the results
    >>> runner.evaluate()

           neg_mean_absolute_error  ...  neg_root_mean_squared_error
    OLS02                  -0.2766  ...                      -0.3650
    OLS04                  -0.2053  ...                      -0.2841
    OLS06                  -0.1957  ...                      -0.2570
    OLS08                  -0.1928  ...                      -0.2504
    OLS10                  -0.1933  ...                      -0.2463

    [5 rows x 6 columns]

    ```

    """

    def __init__(
        self,
        models: str | Predictor | SEQUENCE | None = None,
        metric: str | Callable | SEQUENCE | None = None,
        *,
        train_sizes: INT | SEQUENCE = 5,
        est_params: dict | SEQUENCE | None = None,
        n_trials: INT | dict | SEQUENCE = 0,
        ht_params: dict | None = None,
        n_bootstrap: INT | dict | SEQUENCE = 0,
        parallel: bool = False,
        errors: str = "skip",
        n_jobs: INT = 1,
        device: str = "cpu",
        engine: str = "sklearn",
        backend: str = "loky",
        verbose: INT = 0,
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

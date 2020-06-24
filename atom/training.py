# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the training classes.

"""

# Standard packages
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union, Sequence, Tuple

# Own modules
from .basetrainer import BaseTrainer
from .plots import plot_successive_halving, plot_learning_curve
from .utils import TRAIN_TYPES, get_best_score, get_train_test, composed, crash


# Classes =================================================================== >>

class Trainer(BaseTrainer):
    """Train the models in a direct fashion.

    Parameters
    ----------
    models: string, list or tuple
        List of models to fit on the data. Use the predefined acronyms
        in MODEL_LIST to select the models.

    metric: str or callable
        Metric on which the pipeline fits the models. Choose from any of
        the string scorers predefined by sklearn, use a score (or loss)
        function with signature metric(y, y_pred, **kwargs) or use a
        scorer object.

    greater_is_better: bool, optional (default=True)
        whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower is
        better. Will be ignored if the metric is a string or a scorer.

    needs_proba: bool, optional (default=False)
        Whether the metric function requires probability estimates out of a
        classifier. If True, make sure that every model in the pipeline has
        a `predict_proba` method! Will be ignored if the metric is a string
        or a scorer.

    needs_threshold: bool, optional (default=False)
        Whether the metric function takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method. Will
        be ignored if the metric is a string or a scorer.

    n_calls: int or sequence, optional (default=0)
        Maximum number of iterations of the BO (including `random starts`).
        If 0, skip the BO and fit the model on its default Parameters.
        If sequence, the n-th value will apply to the n-th model in the
        pipeline.

    n_random_starts: int or sequence, optional (default=5)
        Initial number of random tests of the BO before fitting the
        surrogate function. If equal to `n_calls`, the optimizer will
        technically be performing a random search. If sequence, the n-th
        value will apply to the n-th model in the pipeline.

    bo_params: dict, optional (default={})
        Dictionary of extra keyword arguments for the BO. These can
        include:
            - max_time: int
                Maximum allowed time for the BO (in seconds).
            - delta_x: int or float
                Maximum distance between two consecutive points.
            - delta_y: int or float
                Maximum score between two consecutive points.
            - cv: int
                Number of folds for the cross-validation. If 1, the
                training set will be randomly split in a subtrain and
                validation set.
            - callback: callable or list of callables
                Callbacks for the BO.
            - dimensions: dict or array
                Custom hyperparameter space for the bayesian optimization.
                Can be an array (only if there is 1 model in the pipeline)
                or a dictionary with the model's name as key.
            - plot_bo: bool
                Whether to plot the BO's progress.
            - Any other parameter from the skopt gp_minimize function.

    bagging: int, sequence or None, optional (default=None)
        Number of data sets (bootstrapped from the training set) to use in
        the bagging algorithm. If None or 0, no bagging is performed.
        If sequence, the n-th value will apply to the n-th model in the
        pipeline.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If -1, use all available cores
            - If <-1, use available_cores - 1 + value

        Beware that using multiple processes on the same machine may
        cause memory issues for large datasets.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print extended information.

    warnings: bool, optional (default=True)
        If False, suppresses all warnings. Note that this will change
        the `PYTHONWARNINGS` environment.

    logger: str, callable or None, optional (default=None)
        - If string: name of the logging file. 'auto' for default name
                     with timestamp. None to not save any log.
        - If callable: python Logger object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Union[str, callable],
                 greater_is_better: bool = True,
                 needs_proba: bool = False,
                 needs_threshold: bool = False,
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_params: dict = {},
                 bagging: Optional[Union[int, Sequence[int]]] = None,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 warnings: Union[bool, str] = True,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts,
                         bo_params, bagging, n_jobs, verbose, warnings,
                         logger, random_state)

    @crash
    def run(self, *arrays):
        """Run the trainer.

        Parameters
        ----------
        arrays: array-like
            Either a train and test set or X_train, X_test, y_train, y_test.

        """
        train, test = get_train_test(*arrays)

        self.log("\nRunning pipeline ==============================>>", 1)
        self.log(f"Models in pipeline: {', '.join(self.models)}", 1)
        self.log(f"Metric: {self.metric.name}", 1)

        self._results = self._run(train, test)


class SuccessiveHalving(BaseTrainer):
    """Train the models in a successive halving fashion.

        If you want to compare similar models, you can choose to use a successive
        halving approach to run the pipeline. This technique is a bandit-based
        algorithm that fits N models to 1/N of the data. The best half are selected
        to go to the next iteration where the process is repeated. This continues
        until only one model remains, which is fitted on the complete dataset.
        Beware that a model's performance can depend greatly on the amount of data
        on which it is trained. For this reason, we recommend only to use this
        technique with similar models, e.g. only using tree-based models.

    Parameters
    ----------
    models: string, list or tuple
        List of models to fit on the data. Use the predefined acronyms
        in MODEL_LIST to select the models.

    metric: str or callable
        Metric on which the pipeline fits the models. Choose from any of
        the string scorers predefined by sklearn, use a score (or loss)
        function with signature metric(y, y_pred, **kwargs) or use a
        scorer object.

    greater_is_better: bool, optional (default=True)
        whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower is
        better. Will be ignored if the metric is a string or a scorer.

    needs_proba: bool, optional (default=False)
        Whether the metric function requires probability estimates out of a
        classifier. If True, make sure that every model in the pipeline has
        a `predict_proba` method! Will be ignored if the metric is a string
        or a scorer.

    needs_threshold: bool, optional (default=False)
        Whether the metric function takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method. Will
        be ignored if the metric is a string or a scorer.

    skip_iter: int, optional (default=0)
        Skip last `skip_iter` iterations of the successive halving. Will be
        ignored if successive_halving=False.

    n_calls: int or sequence, optional (default=0)
        Maximum number of iterations of the BO (including `random starts`).
        If 0, skip the BO and fit the model on its default Parameters.
        If sequence, the n-th value will apply to the n-th model in the
        pipeline.

    n_random_starts: int or sequence, optional (default=5)
        Initial number of random tests of the BO before fitting the
        surrogate function. If equal to `n_calls`, the optimizer will
        technically be performing a random search. If sequence, the n-th
        value will apply to the n-th model in the pipeline.

    bo_params: dict, optional (default={})
        Dictionary of extra keyword arguments for the BO. These can
        include:
            - max_time: int
                Maximum allowed time for the BO (in seconds).
            - delta_x: int or float
                Maximum distance between two consecutive points.
            - delta_y: int or float
                Maximum score between two consecutive points.
            - cv: int
                Number of folds for the cross-validation. If 1, the
                training set will be randomly split in a subtrain and
                validation set.
            - callback: callable or list of callables
                Callbacks for the BO.
            - dimensions: dict or array
                Custom hyperparameter space for the bayesian optimization.
                Can be an array (only if there is 1 model in the pipeline)
                or a dictionary with the model's name as key.
            - plot_bo: bool
                Whether to plot the BO's progress.
            - Any other parameter from the skopt gp_minimize function.

    bagging: int, sequence or None, optional (default=None)
        Number of data sets (bootstrapped from the training set) to use in
        the bagging algorithm. If None or 0, no bagging is performed.
        If sequence, the n-th value will apply to the n-th model in the
        pipeline.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If -1, use all available cores
            - If <-1, use available_cores - 1 + value

        Beware that using multiple processes on the same machine may
        cause memory issues for large datasets.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print extended information.

    warnings: bool, optional (default=True)
        If False, suppresses all warnings. Note that this will change
        the `PYTHONWARNINGS` environment.

    logger: str, callable or None, optional (default=None)
        - If string: name of the logging file. 'auto' for default name
                     with timestamp. None to not save any log.
        - If callable: python Logger object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Union[str, callable],
                 greater_is_better: bool = True,
                 needs_proba: bool = False,
                 needs_threshold: bool = False,
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
        if skip_iter < 0:
            raise ValueError("Invalid value for the skip_iter parameter." +
                             f"Value should be >=0, got {skip_iter}.")
        else:
            self.skip_iter = skip_iter

        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts,
                         bo_params, bagging, n_jobs, verbose, warnings,
                         logger, random_state)

    @crash
    def run(self, *arrays):
        """Run the trainer.

        Parameters
        ----------
        arrays: array-like
            Either a train and test set or X_train, X_test, y_train, y_test.

        """
        train, test = get_train_test(*arrays)

        self.log("\nRunning pipeline ==============================>>", 1)
        self.log(f"Metric: {self.metric.name}", 1)

        run = 0
        results = []  # List of dataframes returned by self._run
        all_models = self.models.copy()
        while len(self.models) > 2 ** self.skip_iter - 1:
            # Select 1/N of training set to use for this iteration
            rs = self.random_state + run if self.random_state is not None else None
            train_subsample = train.sample(frac=1./len(self.models), random_state=rs)

            # Print stats for this subset of the data
            p = round(100. / len(self.models))
            self.log(f"\n\nRun {run} ({p}% of set) {'='*(30-len(str(p)))}>>")
            self.log(f"Models in pipeline: {', '.join(self.models)}")
            self.log(f"Size of training set: {len(train_subsample)}")
            self.log(f"Size of test set: {len(test)}")

            # Run iteration and append to the results list
            df = self._run(train_subsample, test)
            results.append(df)

            # Select best models for halving
            best = df.apply(lambda row: get_best_score(row), axis=1)
            best = best.nlargest(n=int(len(self.models) / 2), keep='all')
            self.models = list(best.index.values)

            run += 1

        # Concatenate all resulting dataframes with multi-index
        self._results = pd.concat([df for df in results], keys=range(len(results)))

        # Renew self.models
        self.models = all_models

    @composed(crash, typechecked)
    def plot_successive_halving(self,
                                models: Union[None, str, Sequence[str]] = None,
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 6),
                                filename: Optional[str] = None,
                                display: bool = True):
        """Plot the models' results per iteration of the successive halving."""
        plot_successive_halving(self, models,
                                title, figsize, filename, display)


class TrainSizing(BaseTrainer):
    """Train the models in a train sizing fashion.

    Parameters
    ----------
    models: string, list or tuple
        List of models to fit on the data. Use the predefined acronyms
        in MODEL_LIST to select the models.

    metric: str or callable
        Metric on which the pipeline fits the models. Choose from any of
        the string scorers predefined by sklearn, use a score (or loss)
        function with signature metric(y, y_pred, **kwargs) or use a
        scorer object.

    greater_is_better: bool, optional (default=True)
        whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower is
        better. Will be ignored if the metric is a string or a scorer.

    needs_proba: bool, optional (default=False)
        Whether the metric function requires probability estimates out of a
        classifier. If True, make sure that every model in the pipeline has
        a `predict_proba` method! Will be ignored if the metric is a string
        or a scorer.

    needs_threshold: bool, optional (default=False)
        Whether the metric function takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method. Will
        be ignored if the metric is a string or a scorer.

    train_sizes: sequence, optional (default=np.linspace(0.2, 1.0, 5))
        Relative or absolute numbers of training examples that will be used
        to generate the learning curve. If the dtype is float, it is
        regarded as a fraction of the maximum size of the training set.
        Otherwise it is interpreted as absolute sizes of the training sets.
        Will be ignored if train_sizing=False.

    n_calls: int or sequence, optional (default=0)
        Maximum number of iterations of the BO (including `random starts`).
        If 0, skip the BO and fit the model on its default Parameters.
        If sequence, the n-th value will apply to the n-th model in the
        pipeline.

    n_random_starts: int or sequence, optional (default=5)
        Initial number of random tests of the BO before fitting the
        surrogate function. If equal to `n_calls`, the optimizer will
        technically be performing a random search. If sequence, the n-th
        value will apply to the n-th model in the pipeline.

    bo_params: dict, optional (default={})
        Dictionary of extra keyword arguments for the BO. These can
        include:
            - max_time: int
                Maximum allowed time for the BO (in seconds).
            - delta_x: int or float
                Maximum distance between two consecutive points.
            - delta_y: int or float
                Maximum score between two consecutive points.
            - cv: int
                Number of folds for the cross-validation. If 1, the
                training set will be randomly split in a subtrain and
                validation set.
            - callback: callable or list of callables
                Callbacks for the BO.
            - dimensions: dict or array
                Custom hyperparameter space for the bayesian optimization.
                Can be an array (only if there is 1 model in the pipeline)
                or a dictionary with the model's name as key.
            - plot_bo: bool
                Whether to plot the BO's progress.
            - Any other parameter from the skopt minimizer.

    bagging: int, sequence or None, optional (default=None)
        Number of data sets (bootstrapped from the training set) to use in
        the bagging algorithm. If None or 0, no bagging is performed.
        If sequence, the n-th value will apply to the n-th model in the
        pipeline.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If -1, use all available cores
            - If <-1, use available_cores - 1 + value

        Beware that using multiple processes on the same machine may
        cause memory issues for large datasets.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print extended information.

    warnings: bool, optional (default=True)
        If False, suppresses all warnings. Note that this will change
        the `PYTHONWARNINGS` environment.

    logger: str, callable or None, optional (default=None)
        - If string: name of the logging file. 'auto' for default name
                     with timestamp. None to not save any log.
        - If callable: python Logger object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Union[str, callable],
                 greater_is_better: bool = True,
                 needs_proba: bool = False,
                 needs_threshold: bool = False,
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
        self.train_sizes = train_sizes
        self._sizes = []  # Number of training samples (attr for plot)

        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts,
                         bo_params, bagging, n_jobs, verbose, warnings,
                         logger, random_state)

    @crash
    def run(self, *arrays):
        """Run the trainer.

        Parameters
        ----------
        arrays: array-like
            Either a train and test set or X_train, X_test, y_train, y_test.

        """
        train, test = get_train_test(*arrays)

        self.log("\nRunning pipeline ==============================>>", 1)
        self.log(f"Models in pipeline: {', '.join(self.models)}", 1)
        self.log(f"Metric: {self.metric.name}", 1)

        results = []  # List of dataframes returned by self._run
        for run, n_rows in enumerate(self.train_sizes):
            # Select fraction of data to use for this iteration
            rs = self.random_state + run if self.random_state is not None else None
            kwargs = {'frac': n_rows} if n_rows <= 1 else {'n': int(n_rows)}
            train_subsample = train.sample(random_state=rs, **kwargs)
            self._sizes.append(len(train_subsample))

            # Print stats for this subset of the data
            p = round(len(train_subsample) * 100. / (len(train)))
            self.log(f"\n\nRun {run} ({p}% of set) {'='*(30-len(str(p)))}>>")
            self.log(f"Size of training set: {len(train_subsample)}")
            self.log(f"Size of test set: {len(test)}")

            # Run iteration and append to the results list
            results.append(self._run(train_subsample, test))

        # Concatenate all resulting dataframes with multi-index
        self._results = pd.concat([df for df in results], keys=range(len(results)))

    @composed(crash, typechecked)
    def plot_learning_curve(
            self,
            models: Union[None, str, Sequence[str]] = None,
            title: Optional[str] = None,
            figsize: Tuple[int, int] = (10, 6),
            filename: Optional[str] = None,
            display: bool = True):
        """Plot the model's learning curve: score vs training samples."""
        plot_learning_curve(self, models, title, figsize, filename, display)


class TrainerClassifier(Trainer):
    """Trainer class for classification tasks."""

    def __init__(self, *args, **kwargs):
        self.goal = 'classification'
        super().__init__(*args, **kwargs)


class TrainerRegressor(Trainer):
    """Trainer class for regression tasks."""

    def __init__(self, *args, **kwargs):
        self.goal = 'regression'
        super().__init__(*args, **kwargs)


class SuccessiveHalvingClassifier(SuccessiveHalving):
    """SuccessiveHalving class for classification tasks."""

    def __init__(self, *args, **kwargs):
        self.goal = 'classification'
        super().__init__(*args, **kwargs)


class SuccessiveHalvingRegressor(SuccessiveHalving):
    """SuccessiveHalving class for regression tasks."""

    def __init__(self, *args, **kwargs):
        self.goal = 'regression'
        super().__init__(*args, **kwargs)


class TrainSizingClassifier(TrainSizing):
    """TrainSizing class for classification tasks."""

    def __init__(self, *args, **kwargs):
        self.goal = 'classification'
        super().__init__(*args, **kwargs)


class TrainSizingRegressor(TrainSizing):
    """TrainSizing class for regression tasks."""

    def __init__(self, *args, **kwargs):
        self.goal = 'regression'
        super().__init__(*args, **kwargs)

# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the training classes.

"""

# Standard packages
import numpy as np
from typeguard import typechecked
from typing import Optional, Union, Sequence, Tuple

# Own modules
from .basetrainer import BaseTrainer
from .plots import plot_successive_halving, plot_learning_curve
from .utils import TRAIN_TYPES, get_best_score, get_train_test, composed, crash


# << ================ Classes ================= >>

class Trainer(BaseTrainer):
    """Train the models in a direct fashion."""

    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Union[str, callable],
                 greater_is_better: bool = True,
                 needs_proba: bool = False,
                 needs_threshold: bool = False,
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_kwargs: dict = {},
                 bagging: Optional[Union[int, Sequence[int]]] = None,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 warnings: Union[bool, str] = True,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        """Initialize class.

        The pipeline method is where the models are fitted to the data and
        their performance is evaluated according to the selected metric. For
        every model, the pipeline applies the following steps:

            1. The optimal hyperParameters are selected using a Bayesian
               Optimization (BO) algorithm with gaussian process as kernel.
               The resulting score of each step of the BO is either computed
               by cross-validation on the complete training set or by randomly
               splitting the training set every iteration into a (sub) training
               set and a validation set. This process can create some data
               leakage but ensures a maximal use of the provided data. The test
               set, however, does not contain any leakage and will be used to
               determine the final score of every model. Note that, if the
               dataset is relatively small, the best score on the BO can
               consistently be lower than the final score on the test set
               (despite the leakage) due to the considerable fewer instances on
               which it is trained.

            2. Once the best hyperParameters are found, the model is trained
               again, now using the complete training set. After this,
               predictions are made on the test set.

            3. You can choose to evaluate the robustness of each model's
            applying a bagging algorithm, i.e. the model will be trained
            multiple times on a bootstrapped training set, returning a
            distribution of its performance on the test set.

        A couple of things to take into account:
            - The metric implementation follows sklearn's API. This means that
              the implementation always tries to maximize the scorer, i.e. loss
              functions will be made negative.
            - If an exception is encountered while fitting a model, the
              pipeline will automatically jump to the side model and save the
              exception in the `errors` attribute.
            - When showing the final results, a `!!` indicates the highest
              score and a `~` indicates that the model is possibly overfitting
              (training set has a score at least 20% higher than the test set).
            - The winning model subclass will be attached to the `winner`
              attribute.

        Parameters
        ----------
        models: string, list or tuple
            List of models to fit on the data. Use the predefined acronyms
            to select the models. Possible values are (case insensitive):
                - 'GNB' for Gaussian Naive Bayes (no hyperparameter tuning)
                - 'MNB' for Multinomial Naive Bayes
                - 'BNB' for Bernoulli Naive Bayes
                - 'GP' for Gaussian Process (no hyperparameter tuning)
                - 'OLS' for Ordinary Least Squares (no hyperparameter tuning)
                - 'Ridge' for Ridge Linear
                - 'Lasso' for Lasso Linear Regression
                - 'EN' for ElasticNet Linear Regression
                - 'BR' for Bayesian Regression (with ridge regularization)
                - 'LR' for Logistic Regression
                - 'LDA' for Linear Discriminant Analysis
                - 'QDA' for Quadratic Discriminant Analysis
                - 'KNN' for K-Nearest Neighbors
                - 'Tree' for a single Decision Tree
                - 'Bag' for Bagging (with decision tree as base estimator)
                - 'ET' for Extra-Trees
                - 'RF' for Random Forest
                - 'AdaB' for AdaBoost (with decision tree as base estimator)
                - 'GBM' for Gradient Boosting Machine
                - 'XGB' for XGBoost (if package is available)
                - 'LGB' for LightGBM (if package is available)
                - 'CatB' for CatBoost (if package is available)
                - 'lSVM' for Linear Support Vector Machine
                - 'kSVM' for Kernel (non-linear) Support Vector Machine
                - 'PA' for Passive Aggressive
                - 'SGD' for Stochastic Gradient Descent
                - 'MLP' for Multilayer Perceptron

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

        bo_kwargs: dict, optional (default={})
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
        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts,
                         bo_kwargs, bagging, n_jobs, verbose, warnings,
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

        self.log('\nRunning pipeline =================>')
        self.log(f"Model{'s' if len(self.models) > 1 else ''} in " +
                 f"pipeline: {', '.join(self.models)}")
        self.log(f"Metric: {self.metric.name}")

        self._results = [self._run(train, test)]


class SuccessiveHalving(BaseTrainer):
    """Train the models in a successive halving fashion."""

    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Union[str, callable],
                 greater_is_better: bool = True,
                 needs_proba: bool = False,
                 needs_threshold: bool = False,
                 skip_iter: int = 0,
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_kwargs: dict = {},
                 bagging: Optional[Union[int, Sequence[int]]] = None,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 warnings: Union[bool, str] = True,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        """Initialize class.

        The pipeline method is where the models are fitted to the data and
        their performance is evaluated according to the selected metric. For
        every model, the pipeline applies the following steps:

            1. The optimal hyperParameters are selected using a Bayesian
               Optimization (BO) algorithm with gaussian process as kernel.
               The resulting score of each step of the BO is either computed
               by cross-validation on the complete training set or by randomly
               splitting the training set every iteration into a (sub) training
               set and a validation set. This process can create some data
               leakage but ensures a maximal use of the provided data. The test
               set, however, does not contain any leakage and will be used to
               determine the final score of every model. Note that, if the
               dataset is relatively small, the best score on the BO can
               consistently be lower than the final score on the test set
               (despite the leakage) due to the considerable fewer instances on
               which it is trained.

            2. Once the best hyperParameters are found, the model is trained
               again, now using the complete training set. After this,
               predictions are made on the test set.

            3. You can choose to evaluate the robustness of each model's
            applying a bagging algorithm, i.e. the model will be trained
            multiple times on a bootstrapped training set, returning a
            distribution of its performance on the test set.

        A couple of things to take into account:
            - The metric implementation follows sklearn's API. This means that
              the implementation always tries to maximize the scorer, i.e. loss
              functions will be made negative.
            - If an exception is encountered while fitting a model, the
              pipeline will automatically jump to the side model and save the
              exception in the `errors` attribute.
            - When showing the final results, a `!!` indicates the highest
              score and a `~` indicates that the model is possibly overfitting
              (training set has a score at least 20% higher than the test set).
            - The winning model subclass will be attached to the `winner`
              attribute.

        Parameters
        ----------
        models: string, list or tuple
            List of models to fit on the data. Use the predefined acronyms
            to select the models. Possible values are (case insensitive):
                - 'GNB' for Gaussian Naive Bayes (no hyperparameter tuning)
                - 'MNB' for Multinomial Naive Bayes
                - 'BNB' for Bernoulli Naive Bayes
                - 'GP' for Gaussian Process (no hyperparameter tuning)
                - 'OLS' for Ordinary Least Squares (no hyperparameter tuning)
                - 'Ridge' for Ridge Linear
                - 'Lasso' for Lasso Linear Regression
                - 'EN' for ElasticNet Linear Regression
                - 'BR' for Bayesian Regression (with ridge regularization)
                - 'LR' for Logistic Regression
                - 'LDA' for Linear Discriminant Analysis
                - 'QDA' for Quadratic Discriminant Analysis
                - 'KNN' for K-Nearest Neighbors
                - 'Tree' for a single Decision Tree
                - 'Bag' for Bagging (with decision tree as base estimator)
                - 'ET' for Extra-Trees
                - 'RF' for Random Forest
                - 'AdaB' for AdaBoost (with decision tree as base estimator)
                - 'GBM' for Gradient Boosting Machine
                - 'XGB' for XGBoost (if package is available)
                - 'LGB' for LightGBM (if package is available)
                - 'CatB' for CatBoost (if package is available)
                - 'lSVM' for Linear Support Vector Machine
                - 'kSVM' for Kernel (non-linear) Support Vector Machine
                - 'PA' for Passive Aggressive
                - 'SGD' for Stochastic Gradient Descent
                - 'MLP' for Multilayer Perceptron

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

        bo_kwargs: dict, optional (default={})
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
        if skip_iter < 0:
            raise ValueError("Invalid value for the skip_iter parameter." +
                             f"Value should be >=0, got {skip_iter}.")
        else:
            self.skip_iter = skip_iter

        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts,
                         bo_kwargs, bagging, n_jobs, verbose, warnings,
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

        self.log('\nRunning pipeline =================>')
        self.log(f"Metric: {self.metric.name}")

        iter_ = 0
        all_models = self.models.copy()
        while len(self.models) > 2 ** self.skip_iter - 1:
            # Select 1/N of training set to use for this iteration
            rs = self.random_state + iter_ if self.random_state else None
            train_subsample = train.sample(frac=1./len(self.models), random_state=rs)

            # Print stats for this subset of the data
            self.log("\n\n<<=============== Iteration {} ==============>>"
                     .format(iter_))
            self.log(f"Model{'s' if len(self.models) > 1 else ''} in " +
                     f"pipeline: {', '.join(self.models)}")
            self.log(f"Percentage of set: {100. / len(self.models):.1f}%")
            self.log(f"Size of training set: {len(train_subsample)}")
            self.log(f"Size of test set: {len(test)}")

            # Run iteration and append to the results list+
            results = self._run(train_subsample, test)
            self._results.append(results)

            # Select best models for halving
            best = results.apply(lambda row: get_best_score(row), axis=1)
            best = best.nlargest(n=int(len(self.models) / 2), keep='all')
            self.models = list(best.index.values)

            iter_ += 1

        # At the end, renew self.models
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
    """Train the models in a train sizing fashion."""

    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Union[str, callable],
                 greater_is_better: bool = True,
                 needs_proba: bool = False,
                 needs_threshold: bool = False,
                 train_sizes: TRAIN_TYPES = np.linspace(0.2, 1.0, 5),
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_kwargs: dict = {},
                 bagging: Optional[Union[int, Sequence[int]]] = None,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 warnings: Union[bool, str] = True,
                 logger: Optional[Union[str, callable]] = None,
                 random_state: Optional[int] = None):
        """Initialize class.

        The pipeline method is where the models are fitted to the data and
        their performance is evaluated according to the selected metric. For
        every model, the pipeline applies the following steps:

            1. The optimal hyperParameters are selected using a Bayesian
               Optimization (BO) algorithm with gaussian process as kernel.
               The resulting score of each step of the BO is either computed
               by cross-validation on the complete training set or by randomly
               splitting the training set every iteration into a (sub) training
               set and a validation set. This process can create some data
               leakage but ensures a maximal use of the provided data. The test
               set, however, does not contain any leakage and will be used to
               determine the final score of every model. Note that, if the
               dataset is relatively small, the best score on the BO can
               consistently be lower than the final score on the test set
               (despite the leakage) due to the considerable fewer instances on
               which it is trained.

            2. Once the best hyperParameters are found, the model is trained
               again, now using the complete training set. After this,
               predictions are made on the test set.

            3. You can choose to evaluate the robustness of each model's
            applying a bagging algorithm, i.e. the model will be trained
            multiple times on a bootstrapped training set, returning a
            distribution of its performance on the test set.

        A couple of things to take into account:
            - The metric implementation follows sklearn's API. This means that
              the implementation always tries to maximize the scorer, i.e. loss
              functions will be made negative.
            - If an exception is encountered while fitting a model, the
              pipeline will automatically jump to the side model and save the
              exception in the `errors` attribute.
            - When showing the final results, a `!!` indicates the highest
              score and a `~` indicates that the model is possibly overfitting
              (training set has a score at least 20% higher than the test set).
            - The winning model subclass will be attached to the `winner`
              attribute.

        Parameters
        ----------
        models: string, list or tuple
            List of models to fit on the data. Use the predefined acronyms
            to select the models. Possible values are (case insensitive):
                - 'GNB' for Gaussian Naive Bayes (no hyperparameter tuning)
                - 'MNB' for Multinomial Naive Bayes
                - 'BNB' for Bernoulli Naive Bayes
                - 'GP' for Gaussian Process (no hyperparameter tuning)
                - 'OLS' for Ordinary Least Squares (no hyperparameter tuning)
                - 'Ridge' for Ridge Linear
                - 'Lasso' for Lasso Linear Regression
                - 'EN' for ElasticNet Linear Regression
                - 'BR' for Bayesian Regression (with ridge regularization)
                - 'LR' for Logistic Regression
                - 'LDA' for Linear Discriminant Analysis
                - 'QDA' for Quadratic Discriminant Analysis
                - 'KNN' for K-Nearest Neighbors
                - 'Tree' for a single Decision Tree
                - 'Bag' for Bagging (with decision tree as base estimator)
                - 'ET' for Extra-Trees
                - 'RF' for Random Forest
                - 'AdaB' for AdaBoost (with decision tree as base estimator)
                - 'GBM' for Gradient Boosting Machine
                - 'XGB' for XGBoost (if package is available)
                - 'LGB' for LightGBM (if package is available)
                - 'CatB' for CatBoost (if package is available)
                - 'lSVM' for Linear Support Vector Machine
                - 'kSVM' for Kernel (non-linear) Support Vector Machine
                - 'PA' for Passive Aggressive
                - 'SGD' for Stochastic Gradient Descent
                - 'MLP' for Multilayer Perceptron

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

        bo_kwargs: dict, optional (default={})
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
        self.train_sizes = train_sizes
        self._sizes = []  # Number of training samples (attr for plot)
        super().__init__(models, metric, greater_is_better, needs_proba,
                         needs_threshold, n_calls, n_random_starts,
                         bo_kwargs, bagging, n_jobs, verbose, warnings,
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

        self.log('\nRunning pipeline =================>')
        self.log(f"Model{'s' if len(self.models) > 1 else ''} in " +
                 f"pipeline: {', '.join(self.models)}")
        self.log(f"Metric: {self.metric.name}")

        for iter_, size in enumerate(self.train_sizes):
            if size > 1:
                raise ValueError("Invalid value for the train_sizes " +
                                 "parameter. All elements should be >1, " +
                                 f"got, {size}.")

            # Select fraction of data to use for this iteration
            rs = self.random_state + iter_ if self.random_state else None
            train_subsample = train.sample(frac=size, random_state=rs)
            self._sizes.append(len(train))

            # Print stats for this subset of the data
            self.log("\n\n<<=============== Iteration {} ==============>>"
                     .format(iter_))
            p = size * 100 if size <= 1 else size * 100./(len(train))
            self.log(f"Percentage of set: {p:.1f}%")
            self.log(f"Size of training set: {len(train_subsample)}")
            self.log(f"Size of test set: {len(test)}")

            # Run iteration and append to the results list
            results = self._run(train_subsample, test)
            self._results.append(results)

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

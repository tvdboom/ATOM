# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the pipeline class.

"""

# Standard packages
import importlib
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from typing import Optional, Union, Sequence
from sklearn.metrics import SCORERS, get_scorer, make_scorer

# Own modules
from .data_cleaning import BaseCleaner
from .models import MODEL_LIST, get_model_name
from .utils import (
    SCALAR, X_TYPES, Y_TYPES, TRAIN_TYPES, OPTIONAL_PACKAGES,
    ONLY_CLASSIFICATION, ONLY_REGRESSION, composed, crash, params_to_log,
    time_to_string, merge, check_property, check_scaling, check_is_fitted,
    variable_return, save
    )


# << ================ Classes ================= >>

class BasePipeline(BaseCleaner):
    """Train and validate the models."""

    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Optional[Union[str, callable]] = None,
                 greater_is_better: bool = True,
                 needs_proba: bool = False,
                 needs_threshold: bool = False,
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_kwargs: dict = {},
                 bagging: Optional[int] = None,
                 **kwargs):
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

        metric: string or callable, optional (default=None)
            Metric on which the pipeline fits the models. Choose from any of
            the string scorers predefined by sklearn, use a score (or loss)
            function with signature metric(y, y_pred, **kwargs) or use a
            scorer object. If None, the default metric per task is selected:
                - 'f1' for binary classification
                - 'f1_weighted' for multiclass classification
                - 'r2' for regression

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

        bagging: int or None, optional (default=None)
            Number of data sets (bootstrapped from the training set) to use in
            the bagging algorithm. If None or 0, no bagging is performed.

        **kwargs
            Additional parameters for the estimator:
                - task: binary, multiclass or regression.
                - verbose: Verbosity level of the output.
                - n_jobs: number of cores to use for parallel processing.
                - random_state: seed used by the random number generator.

        """
        self.errors = {}
        super().__init__(**kwargs)

        # << =================== Check validity models ================== >>

        if isinstance(models, str):
            models = [models]

        # Set models to right name
        self.models = [get_model_name(m) for m in models]

        # Check for duplicates
        if len(self.models) != len(set(self.models)):
            raise ValueError("Duplicate models found in pipeline!")

        # Check if packages for not-sklearn models are available
        for m, package in OPTIONAL_PACKAGES:
            if m in self.models:
                try:
                    importlib.import_module(package)
                except ImportError:
                    raise ValueError(f"Unable to import {package}!")

        # Remove regression/classification-only models from pipeline
        if self.task != 'regression':
            for m in ONLY_REGRESSION:
                if m in self.models:
                    raise ValueError(f"The {m} model can't perform " +
                                     "classification tasks!")
        else:
            for m in ONLY_CLASSIFICATION:
                if m in self.models:
                    raise ValueError(f"The {m} model can't perform " +
                                     "regression tasks!")

        # << ================= Check validity parameters ================ >>

        # Check Parameters
        if isinstance(n_calls, (list, tuple)):
            if len(n_calls) != len(self.models):
                raise ValueError(f"Invalid value for the n_calls parameter. " +
                                 "Length should be equal to the number of mo" +
                                 f"dels, got len(models)={len(self.models)}" +
                                 f"and len(n_calls)={len(n_calls)}.")
        else:
            self.n_calls = [n_calls for _ in self.models]
        if isinstance(n_random_starts, (list, tuple)):
            if len(n_random_starts) != len(self.models):
                raise ValueError("Invalid value for the n_random_starts " +
                                 "parameter. Length should be equal to the " +
                                 "number of models, got len(models)=" +
                                 f"{len(self.models)} and len(n_random_" +
                                 f"starts)={len(n_random_starts)}.")
        else:
            self.n_random_starts = [n_random_starts for _ in self.models]
        self.bagging = bagging if bagging else None  # Is None or 0

        # << ================= Check dimensions kwargs ================== >>

        self.bo_kwargs = bo_kwargs
        if self.bo_kwargs.get('dimensions'):
            # Dimensions can be array for one model or dict if more
            if not isinstance(self.bo_kwargs.get('dimensions'), dict):
                if len(self.models) != 1:
                    raise TypeError("Invalid type for the dimensions " +
                                    "parameter. For multiple models it has " +
                                    "to be a dictionary with the model's " +
                                    "name as key!")
                else:
                    self.bo_kwargs['dimensions'] = \
                        {self.models[0]: self.bo_kwargs['dimensions']}

            # Assign proper model name to key of dimensions dict
            for key in self.bo_kwargs.get('dimensions', []):
                self.bo_kwargs['dimensions'][get_model_name(key)] = \
                    self.bo_kwargs['dimensions'].pop(key)

        # << ================== Check validity metric =================== >>

        if metric is None and not hasattr(self, 'metric'):
            if self.task.startswith('bin'):
                self.metric = get_scorer('f1')
                self.metric.name = 'f1'
            elif self.task.startswith('multi'):
                self.metric = get_scorer('f1_weighted')
                self.metric.name = 'f1_weighted'
            else:
                self.metric = get_scorer('r2')
                self.metric.name = 'r2'
        elif metric is None and hasattr(self, 'metric'):
            pass  # Metric is already defined
        elif isinstance(metric, str):
            if metric not in SCORERS:
                options = ', '.join(SCORERS)
                raise ValueError("Unknown value for the metric parameter, " +
                                 f"got {metric}. Try one of: {options}.")
            self.metric = get_scorer(metric)
            self.metric.name = metric
        elif hasattr(metric, '_score_func'):  # Provided metric is scoring
            self.metric = metric
            self.metric.name = self.metric._score_func.__name__
        else:  # Metric is a function with signature metric(y, y_pred)
            self.metric = make_scorer(metric,
                                      greater_is_better,
                                      needs_proba,
                                      needs_threshold)
            self.metric.name = self.metric._score_func.__name__

        # Add all metrics to the metric attribute as subclasses
        for key, value in SCORERS.items():
            setattr(self.metric, key, value)
            setattr(getattr(self.metric, key), 'name', key)

    def run_iteration(self, train, test):
        """Core iteration.

        Parameters
        ----------
        train: pd.DataFrame
            Training set used for this iteration.

        test: pd.DataFrame
            Test set used for this iteration.

        Returns
        -------
        scores: pd.DataFrame
            Dataframe of the scores for this iteration.

        """
        t_init = time()  # To measure the time the whole pipeline takes

        # If verbose=1, use tqdm to evaluate process
        loop = self.models
        if self.verbose == 1:
            loop = tqdm(self.models, desc='Processing')
            loop.clear()  # Prevent starting a progress bar before the loop

        # Loop over every independent model
        for m, call_, start_ in zip(loop, self.n_calls, self.n_random_starts):

            model_time = time()

            # Define model class
            setattr(self, m, MODEL_LIST[m](self, train, test))
            subclass = getattr(self, m)

            try:  # If errors occurs, just skip the model
                # Run Bayesian Optimization
                # Use copy of kwargs to not delete original in method
                subclass.bayesian_optimization(
                    call_, start_, self.bo_kwargs.copy())

                # Fit the model to the test set
                subclass.fit()

                # Perform bagging
                subclass.bagging(self.bagging)

                # Get the total time spend on this model
                total_time = time_to_string(model_time)
                setattr(subclass, 'total_time', total_time)
                self.log('-' * 49, 1)
                self.log(f'Total time: {total_time}', 1)

            except Exception as ex:
                if call_ > 0:
                    self.log('', 1)  # Add extra line
                self.log("Exception encountered while running the "
                         + f"{m} model. Removing model from "
                         + f"pipeline. \n{type(ex).__name__}: {ex}", 1)

                # Save the exception to model attribute
                exception = type(ex).__name__ + ': ' + str(ex)
                subclass.error = exception

                # Append exception to ATOM errors dictionary
                self.errors[m] = exception

                # Replace model with value X for later removal
                # Can't remove at once to not disturb list order
                self.models[self.models.index(m)] = 'X'

            # Set model attributes for lowercase as well
            setattr(self, m.lower(), subclass)

        # Remove faulty models (replaced with X)
        while 'X' in self.models:
            self.models.remove('X')

        # Check if all models failed (self.models is empty)
        if not self.models:
            raise ValueError('It appears all models failed to run...')

        # << =================== Print final results ================== >>

        # Create dataframe with final results
        scores = pd.DataFrame(
            columns=['name', 'score_train', 'score_test',
                     'time_fit', 'mean_bagging', 'std_bagging',
                     'time_bagging', 'total_time'])

        # Print final results
        self.log("\n\nFinal results ================>>")
        self.log(f"Duration: {time_to_string(t_init)}")
        self.log(f"Metric: {self.metric.name}")
        self.log("--------------------------------")

        # Get max length of the models' names
        maxlen = max([len(getattr(self, m).longname) for m in self.models])

        # List of all scores on the test set (for all models in pipeline!)
        all_scores = []
        for m in self.models:
            if getattr(self, m).score_bagging:
                all_scores.append(getattr(self, m).mean_bagging)
            else:
                all_scores.append(getattr(self, m).score_test)

        for m in self.models:
            name = getattr(self, m).name
            longname = getattr(self, m).longname
            total_time = getattr(self, m).total_time
            score_train = getattr(self, m).score_train
            score_test = getattr(self, m).score_test
            time_fit = getattr(self, m).time_fit
            mean_bagging = getattr(self, m).mean_bagging
            std_bagging = getattr(self, m).std_bagging
            time_bagging = getattr(self, m).time_bagging

            # Append model row to scores
            scores.loc[name] = [longname,
                                score_train,
                                score_test,
                                time_fit,
                                mean_bagging,
                                std_bagging,
                                time_bagging,
                                total_time]

            if self.bagging is None:
                # Create string of the score
                print_ = f"{longname:{maxlen}s} --> {score_test:.3f}"

                # Highlight best score and assign winner attribute
                if score_test == max(all_scores):
                    self.winner = getattr(self, m)
                    print_ += ' !!'

            else:
                # Create string of the score
                print1 = f"{longname:{maxlen}s} --> {mean_bagging:.3f}"
                print2 = f"{std_bagging:.3f}"
                print_ = print1 + u" \u00B1 " + print2

                # Highlight best score and assign winner attribute
                if mean_bagging == max(all_scores):
                    self.winner = getattr(self, m)
                    print_ += ' !!'

            # Annotate if model overfitted when train 20% > test
            if score_train - 0.2 * score_train > score_test:
                print_ += ' ~'

            self.log(print_)  # Print the score

        return scores


class Pipeline(BasePipeline):

    def __init__(self,
                 models: Union[str, Sequence[str]],
                 metric: Optional[Union[str, callable]] = None,
                 greater_is_better: bool = True,
                 needs_proba: bool = False,
                 needs_threshold: bool = False,
                 n_calls: Union[int, Sequence[int]] = 0,
                 n_random_starts: Union[int, Sequence[int]] = 5,
                 bo_kwargs: dict = {},
                 bagging: Optional[int] = None,
                 **kwargs):
        """Initialize class."""
        super().__init__(models,
                         metric,
                         greater_is_better,
                         needs_proba,
                         needs_threshold,
                         n_calls,
                         n_random_starts,
                         bo_kwargs,
                         bagging,
                         **kwargs)

    def run(self, train, test):

        self.log('\nRunning pipeline =================>')
        self.log(f"Model{'s' if len(self.models) > 1 else ''} in " +
                 f"pipeline: {', '.join(self.models)}")
        self.log(f"Metric: {self.metric.name}")

        self.scores = self.run_iteration(train, test)




    if self._has_sh:
        self.clear()
        iter_ = 0
        while len(models_now) > 2 ** skip_iter - 1:
            # Select 1/N of training set to use for this iteration
            rs = self.random_state + iter_ if self.random_state else None
            train_subsample = self.train.sample(frac=1. / len(models_now),
                                                random_state=rs)

            # Print stats for this subset of the data
            self.log("\n\n<<=============== Iteration {} ==============>>"
                     .format(iter_))
            self.log(f"Model{'s' if len(models_now) > 1 else ''} in " +
                     f"pipeline: {', '.join(models_now)}")
            self.log(f"Percentage of set: {100. / len(models_now):.1f}%")
            self.log(f"Size of training set: {len(train_subsample)}")

            # Run iteration and append to the scores list
            scores, models_now = \
                run_iteration(models_now, train_subsample, self.test)
            self._scores.append(scores)

            # Select best models for halving
            col = 'score_test' if not self._has_bag else 'bagging_mean'
            lx = scores.nlargest(n=int(len(models_now) / 2),
                                 columns=col,
                                 keep='all')

            # Keep the models in the same order
            n = []  # List of new models
            [n.append(m) for m in models_now if m in lx.index]
            models_now = n.copy()
            iter_ += 1

    elif self._has_ts:
        self.clear()
        self._sizes = []  # Number of training samples (attr for plot)
        for iter_, size in enumerate(train_sizes):
            if size > 1:
                raise ValueError("Invalid value for the train_sizes " +
                                 "parameter. All elements should be >1, " +
                                 f"got, {size}.")

            # Select fraction of data to use for this iteration
            rs = self.random_state + iter_ if self.random_state else None
            train_subsample = self.train.sample(frac=size, random_state=rs)
            self._sizes.append(len(train_subsample))

            # Print stats for this subset of the data
            self.log("\n\n<<=============== Iteration {} ==============>>"
                     .format(iter_))
            p = size * 100 if size <= 1 else size * 100 / len(self.dataset)
            self.log(f"Percentage of set: {p:.1f}%")
            self.log(f"Size of training set: {len(train_subsample)}")

            # Run iteration and append to the scores list
            scores, models_now = \
                run_iteration(models_now, train_subsample, self.test)
            self._scores.append(scores)

    else:
        scores, _ = run_iteration(models_now, self.train, self.test)
        if not self._has_pl:
            self._scores = [scores]
        else:
            self._has_pl = False
            for index, row in scores.iterrows():
                self._scores[0].loc[index] = row

    # Update self.models again (removed earlier by self.clear())
    self.models = list(self._scores[0].index)

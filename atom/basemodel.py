# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the parent class for all model subclasses.

"""

# Import Packages =========================================================== >>

# Standard packages
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from typeguard import typechecked
from typing import Optional, Union, Sequence, Tuple

# Sklearn
from sklearn.utils import resample
from sklearn.metrics import SCORERS, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# Others
from skopt.utils import use_named_args
from skopt.callbacks import DeadlineStopper, DeltaXStopper, DeltaYStopper
from skopt.optimizer import (
    base_minimize, gp_minimize, forest_minimize, gbrt_minimize
)

# Own package modules
from .utils import (
    CAL, composed, crash, params_to_log, attach_methods,
    save, time_to_string, PlotCallback
    )
from .plots import (
        plot_bagging, plot_successive_halving, plot_learning_curve,
        plot_roc, plot_prc, plot_permutation_importance,
        plot_feature_importance, plot_confusion_matrix, plot_threshold,
        plot_probabilities, plot_calibration, plot_gains, plot_lift, plot_bo
        )


# << ====================== Classes ====================== >>

class BaseModel(object):
    """Parent class of all model subclasses.

   Parameters
    ----------
    data: dict
        Dictionary of the data used for this model (train and test).

    T: class
        Class from which the model is called. To avoid having to pass
        attributes through params.

    """

    def __init__(self, *args, **kwargs):
        # Set attributes from ATOM to the model's parent class
        self.__dict__.update(kwargs)
        self.name, self.longname = None, None

        # BO attributes
        self._iter = 0
        self._init_bo = None
        self._cv = 3  # Default value
        self.bo = pd.DataFrame(
            columns=['call', 'params', 'model', 'score', 'time_iteration', 'time'])

        # BaseModel attributes
        self.best_params = None
        self.best_model = None
        self.best_model_fit = None
        self.time_fit = None
        self.score_bo = None
        self.time_bo = None
        self.predict_train = None
        self.predict_test = None
        self.predict_proba_train = None
        self.predict_proba_test = None
        self.decision_function_train = None
        self.decision_function_test = None
        self.score_train = None
        self.score_test = None
        self.score_bagging = []
        self.mean_bagging = None
        self.std_bagging = None
        self.time_bagging = None

        # Metric attributes
        self.confusion_matrix = None
        self.lift = None
        self.tp, self.fp, self.tn, self.fn = None, None, None, None
        self.fpr, self.tpr, self.sup = None, None, None

        # Attach methods from child class (they need best_model_fit)
        methods = ['_pipe', 'predict', 'decision', 'score']
        attach_methods(self, self.T.__class__, methods)

    @composed(crash, params_to_log, typechecked)
    def bayesian_optimization(self,
                              n_calls: int = 15,
                              n_random_starts: int = 5,
                              bo_params: dict = {},
                              *args):
        """Run the bayesian optimization algorithm.

        Search for the best combination of hyperparameters. The function to
        optimize is evaluated either with a K-fold cross-validation on the
        training set or using a validation set.

        Parameters
        ----------
        n_calls: int or sequence, optional (default=15)
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
            Dictionary of extra keyword arguments for the BO.
            These can include:
                - base_estimator: str
                    Surrogate model to use. Choose from: 'GP', 'ET', 'RF', 'GBRT'.
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
                    or a dictionary with the model names as key.
                - plot_bo: bool
                    Whether to plot the BO's progress.
                - Any other parameter for the skopt estimator.

        *args
            Data to be used, in the form X_train, X_test, y_train, y_test.

        """
        def optimize(**params):
            """Optimization function for the bayesian optimization algorithm.

            Parameters
            ----------
            params: dict
               Model's hyperparameters to be used for this iteration of the BO.

            Returns
            -------
            score: float
                Score achieved by the model.

            """
            t_iter = time()  # Get current time for start of the iteration

            # Print iteration and time
            self._iter += 1
            n = 'Iteration' if self._iter > n_random_starts else 'Random start'

            len_ = '-' * (46 - len(n) - len(str(self._iter)))
            self.T.log(f"{n}: {self._iter} {len_}", 2)
            self.T.log(f"Parameters --> {params}", 2)

            algorithm = self.get_model(params)

            # We want same splits for every model, but different
            # for every iteration of the BO
            random_state = None
            if self.T.random_state is not None:
                random_state = self.T.random_state + self._iter

            if self._cv == 1:
                # Select test_size from ATOM or use default of 0.3
                t_size = self.T._test_size if hasattr(self.T, '_test_size') else 0.3

                # Split each iteration in different train and validation set
                X_subtrain, X_validation, y_subtrain, y_validation = \
                    train_test_split(X_train,
                                     y_train,
                                     test_size=t_size,
                                     shuffle=True,
                                     random_state=random_state)

                # Calculate metric on the validation set
                algorithm.fit(X_subtrain, y_subtrain)
                score = self.T.metric(algorithm, X_validation, y_validation)

            else:  # Use cross validation to get the score

                # Determine number of folds for the cross_val_score
                if self.T.goal.startswith('class'):
                    # Folds are made preserving the % of samples for each class
                    k_fold = StratifiedKFold(n_splits=self._cv,
                                             shuffle=True,
                                             random_state=random_state)
                else:
                    k_fold = KFold(n_splits=self._cv,
                                   shuffle=True,
                                   random_state=random_state)

                # Run cross-validation (get mean of results)
                score = cross_val_score(algorithm,
                                        X_train,
                                        y_train,
                                        cv=k_fold,
                                        scoring=self.T.metric,
                                        n_jobs=self.T.n_jobs).mean()

            # Append row to the BO attribute
            t = time_to_string(t_iter)
            t_tot = time_to_string(self._init_bo)
            self.bo = self.bo.append({'call': n,
                                      'params': params,
                                      'model': algorithm,
                                      'score': score,
                                      'time_iteration': t,
                                      'time': t_tot},
                                     ignore_index=True)

            # Print output of the BO and plot progress
            self.T.log(f"Evaluation --> Score: {score:.4f}   " +
                       f"Best score: {max(self.bo.score):.4f}", 2)
            self.T.log(f"Time iteration: {t}   Total time: {t_tot}", 2)

            return -score  # Negative since skopt tries to minimize

        # Running optimization ============================================== >>

        # Check parameters
        if n_random_starts < 1:
            raise ValueError("Invalid value for the n_random_starts parameter. " +
                             f"Value should be >0, got {n_random_starts}.")
        if n_calls < n_random_starts:
            raise ValueError("Invalid value for the n_calls parameter. Value " +
                             f"should be >n_random_starts, got {n_calls}.")

        self.T.log(f"\n\nRunning BO for {self.longname}...", 1)

        self._init_bo = time()
        X_train, X_test, y_train, y_test = args[0], args[1], args[2], args[3]

        # Prepare callbacks
        callbacks = []
        if bo_params.get('callback'):
            if not isinstance(bo_params['callback'], (list, tuple)):
                callbacks = [bo_params['callback']]
            bo_params.pop('callback')

        if bo_params.get('max_time'):
            if bo_params['max_time'] <= 0:
                raise ValueError("Invalid value for the max_time parameter. " +
                                 f"Value should be >0, got {bo_params['max_time']}.")
            callbacks.append(DeadlineStopper(bo_params['max_time']))
            bo_params.pop('max_time')

        if bo_params.get('delta_x'):
            if bo_params['delta_x'] < 0:
                raise ValueError("Invalid value for the max_time parameter. " +
                                 f"Value should be >=0, got {bo_params['delta_x']}.")
            callbacks.append(DeltaXStopper(bo_params['delta_x']))
            bo_params.pop('delta_x')

        if bo_params.get('delta_y'):
            if bo_params['delta_y'] < 0:
                raise ValueError("Invalid value for the max_time " +
                                 "parameter. Value should be >=0, " +
                                 f"got {bo_params['delta_y']}.")
            callbacks.append(DeltaYStopper(bo_params['delta_y'], n_best=2))
            bo_params.pop('delta_y')

        if bo_params.get('plot_bo'):  # Exists and is True
            callbacks.append(PlotCallback(self))
            bo_params.pop('plot_bo')

        # Prepare additional arguments
        if bo_params.get('cv'):
            if bo_params['cv'] <= 0:
                raise ValueError("Invalid value for the max_time " +
                                 "parameter. Value should be >=0, " +
                                 f"got {bo_params['cv']}.")
            self._cv = bo_params['cv']
            bo_params.pop('cv')

        # Specify model dimensions
        def pre_defined_hyperparameters(x):
            return optimize(**self.get_params(x))

        dimensions = self.get_domain()
        func = pre_defined_hyperparameters  # Default optimization func
        if bo_params.get('dimensions'):
            if bo_params['dimensions'].get(self.name):
                dimensions = bo_params.get('dimensions').get(self.name)

                @use_named_args(dimensions)
                def custom_hyperparameters(**x):
                    return optimize(**x)
                func = custom_hyperparameters  # Use custom hyperparameters
            bo_params.pop('dimensions')

        # If only 1 random start, use the model's default parameters
        if n_random_starts == 1:
            bo_params['x0'] = self.get_init_values()

        # Choose base estimator (GP is chosen as default)
        base = bo_params.pop('base_estimator', 'GP')

        # Prepare keyword arguments for the optimizer
        kwargs = dict(func=func,
                      dimensions=dimensions,
                      n_calls=n_calls,
                      n_random_starts=n_random_starts,
                      callback=callbacks,
                      n_jobs=self.T.n_jobs,
                      random_state=self.T.random_state)
        kwargs.update(**bo_params)

        if isinstance(base, str):
            if base.lower() == 'gp':
                optimizer = gp_minimize(**kwargs)
            elif base.lower() == 'et':
                optimizer = forest_minimize(base_estimator='ET', **kwargs)
            elif base.lower() == 'rf':
                optimizer = forest_minimize(base_estimator='RF', **kwargs)
            elif base.lower() == 'gbrt':
                optimizer = gbrt_minimize(**kwargs)
            else:
                raise ValueError(
                    f"Invalid value for the base_estimator parameter, got {base}. " +
                    "Value should be one of: 'GP', 'ET', 'RF', 'GBRT'.")
        else:
            optimizer = base_minimize(base_estimator=base, **kwargs)

        if plot_bo:
            plt.close()

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
        self.score_bo = -optimizer.fun

        # Save best model (not yet fitted)
        self.best_model = self.get_model(self.best_params)

        # Get the BO duration
        self.time_bo = time_to_string(self._init_bo)

        # Print results
        self.T.log('', 2)  # Print extra line (note verbosity level 2)
        self.T.log(f"Final results for {self.longname}:{' ':9s}", 1)
        self.T.log("Bayesian Optimization ---------------------------", 1)
        self.T.log(f"Best hyperparameters: {self.best_params}", 1)
        self.T.log(f"Best score: {self.score_bo:.4f}", 1)
        self.T.log(f"Time elapsed: {self.time_bo}", 1)

    @composed(crash, params_to_log)
    def fit(self, *args):
        """Fit the best model to the test set.

        Parameters
        ----------
        *args
            Data to be used, in the form X_train, X_test, y_train, y_test.

        """
        t_init = time()
        X_train, X_test, y_train, y_test = args[0], args[1], args[2], args[3]

        # In case the bayesian_optimization method wasn't called
        if self.best_model is None:
            self.best_model = self.get_model()

        # Fit the selected model on the complete training set
        self.best_model_fit = self.best_model.fit(X_train, y_train)

        # Save predictions
        self.predict_train = self.best_model_fit.predict(X_train)
        self.predict_test = self.best_model_fit.predict(X_test)

        # Save probability predictions
        if self.T.goal.startswith('class'):
            if hasattr(self.best_model_fit, 'predict_proba'):
                self.predict_proba_train = \
                    self.best_model_fit.predict_proba(X_train)
                self.predict_proba_test = \
                    self.best_model_fit.predict_proba(X_test)
            if hasattr(self.best_model_fit, 'decision_function'):
                self.decision_function_train = \
                    self.best_model_fit.decision_function(X_train)
                self.decision_function_test = \
                    self.best_model_fit.decision_function(X_test)

        # Save scores on complete training and test set
        self.score_train = self.T.metric(self.best_model_fit, X_train, y_train)
        self.score_test = self.T.metric(self.best_model_fit, X_test, y_test)

        # Calculate custom metrics and attach to attributes
        if self.T.goal.startswith('class'):
            self.confusion_matrix = confusion_matrix(y_test, self.predict_test)
            if self.T.task.startswith('bin'):
                tn, fp, fn, tp = self.confusion_matrix.ravel()

                # Get metrics (lift, true (false) positive rate and support)
                self.lift = (tp/(tp+fp))/((tp+fn)/(tp+tn+fp+fn))
                self.fpr = fp/(fp+tn)
                self.tpr = tp/(tp+fn)
                self.sup = (tp+fp)/(tp+fp+fn+tn)

                self.tn, self.fp, self.fn, self.tp = tn, fp, fn, tp

        # Calculate all pre-defined metrics on the test set
        for key, scorer in SCORERS.items():
            try:
                # Some metrics need probabilities and other need predict
                if type(scorer).__name__ == '_ThresholdScorer':
                    if self.T.task.startswith('reg'):
                        y_pred = self.predict_test
                    elif self.decision_function_test is not None:
                        y_pred = self.decision_function_test
                    else:
                        y_pred = self.predict_proba_test
                        if self.T.task.startswith('bin'):
                            y_pred = y_pred[:, 1]
                elif type(scorer).__name__ == '_ProbaScorer':
                    if self.predict_proba_test is not None:
                        y_pred = self.predict_proba_test
                        if self.T.task.startswith('bin'):
                            y_pred = y_pred[:, 1]
                    else:
                        y_pred = self.decision_function_test
                else:
                    y_pred = self.predict_test

                # Calculate metric on the test set
                scr = scorer._score_func(y_test, y_pred, **scorer._kwargs)
                setattr(self, key, scorer._sign * scr)

            except (ValueError, TypeError):
                msg = f"This metric is unavailable for the {self.name} model!"
                setattr(self, key, msg)

        # Print stats ======================================================= >>

        if self.bo.empty:
            self.T.log('\n', 1)  # Print 2 extra lines
            self.T.log(f"Final results for {self.longname}:{' ':9s}", 1)
        if not hasattr(self, 'get_domain') and not self.bo.empty:
            self.T.log('\n', 1)  # Print 2 extra lines
            self.T.log(f"Final results for {self.longname}:{' ':9s}", 1)
        self.T.log("Fitting -----------------------------------------", 1)
        self.T.log(f"Score on the training set: {self.score_train:.4f}", 1)
        self.T.log(f"Score on the test set: {self.score_test:.4f}", 1)

        # Get duration and print to log
        self.time_fit = time_to_string(t_init)
        self.T.log(f"Time elapsed: {self.time_fit}", 1)

        # Add transformation methods ======================================== >>

        # Attach methods from child class (they need best_model_fit)
        methods = ['_pipe', 'predict', 'decision', 'score']
        attach_methods(self, self.T.__class__, methods)

    @composed(crash, params_to_log, typechecked)
    def bagging(self, bagging: Optional[int] = 5, *args):
        """Perform bagging algorithm.

        Take bootstrap samples from the training set and test them on the test
        set to get a distribution of the model's results.

        Parameters
        ----------
        bagging: int or None, optional (default=5)
            Number of data sets (bootstrapped from the training set) to use in
            the bagging algorithm. If None or 0, no bagging is performed.

        *args
            Data to be used, in the form X_train, X_test, y_train, y_test.

        """
        t_init = time()
        X_train, X_test, y_train, y_test = args[0], args[1], args[2], args[3]

        if bagging < 0:
            raise ValueError("Invalid value for the bagging parameter." +
                             f"Value should be >=0, got {bagging}.")

        for _ in range(bagging):
            # Create samples with replacement
            sample_x, sample_y = resample(X_train, y_train)

            # Fit on bootstrapped set and predict on the independent test set
            algorithm = self.best_model.fit(sample_x, sample_y)
            score = self.T.metric(algorithm, X_test, y_test)

            # Append metric result to list
            self.score_bagging.append(score)

        # Numpy array for mean and std
        self.mean_bagging = np.array(self.score_bagging).mean()
        self.std_bagging = np.array(self.score_bagging).std()
        self.T.log("Bagging -----------------------------------------", 1)
        self.T.log("Mean: {:.4f}   Std: {:.4f}".format(
                    self.mean_bagging, self.std_bagging), 1)

        # Get duration and print to log
        self.time_bagging = time_to_string(t_init)
        self.T.log(f"Time elapsed: {self.time_bagging}", 1)

    # Plot methods ========================================================== >>

    @composed(crash, params_to_log, typechecked)
    def plot_bagging(self,
                     title: Optional[str] = None,
                     figsize: Optional[Tuple[int, int]] = None,
                     filename: Optional[str] = None,
                     display: bool = True):
        """Boxplot of the bagging's results."""
        plot_bagging(self.T, self.name, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_successive_halving(self,
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 6),
                                filename: Optional[str] = None,
                                display: bool = True):
        """Plot the models' scores per iteration of the successive halving."""
        plot_successive_halving(self.T, self.name,
                                title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_learning_curve(
                    self,
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    filename: Optional[str] = None,
                    display: bool = True):
        """Plot the model's learning curve: score vs training samples."""
        plot_learning_curve(self.T, self.name,
                            title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_roc(self,
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the Receiver Operating Characteristics curve."""
        plot_roc(self.T, self.name, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_prc(self,
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the precision-recall curve."""
        plot_prc(self.T, self.name, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_permutation_importance(self,
                                    show: Optional[int] = None,
                                    n_repeats: int = 10,
                                    title: Optional[str] = None,
                                    figsize: Optional[Tuple[int, int]] = None,
                                    filename: Optional[str] = None,
                                    display: bool = True):
        """Plot the feature permutation importance of models."""
        plot_permutation_importance(self.T, self.name, show, n_repeats,
                                    title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_feature_importance(self,
                                show: Optional[int] = None,
                                title: Optional[str] = None,
                                figsize: Optional[Tuple[int, int]] = None,
                                filename: Optional[str] = None,
                                display: bool = True):
        """Plot a tree-based model's normalized feature importances."""
        plot_feature_importance(self.T, self.name,
                                show, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_confusion_matrix(self,
                              normalize: bool = False,
                              title: Optional[str] = None,
                              figsize: Tuple[int, int] = (8, 8),
                              filename: Optional[str] = None,
                              display: bool = True):
        """Plot the confusion matrix.

        For 1 model: plot it's confusion matrix in a heatmap.
        For >1 models: compare TP, FP, FN and TN in a barplot.

        """
        plot_confusion_matrix(self.T, self.name, normalize,
                              title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_threshold(self,
                       metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                       steps: int = 100,
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       filename: Optional[str] = None,
                       display: bool = True):
        """Plot performance metric(s) against threshold values."""
        plot_threshold(self.T, self.name, metric, steps,
                       title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_probabilities(self,
                           target: Union[int, str] = 1,
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           filename: Optional[str] = None,
                           display: bool = True):
        """Plot the distribution of predicted probabilities."""
        plot_probabilities(self.T, self.name, target,
                           title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_calibration(self,
                         n_bins: int = 10,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 10),
                         filename: Optional[str] = None,
                         display: bool = True):
        """Plot the calibration curve for a binary classifier."""
        plot_calibration(self.T, self.name, n_bins,
                         title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_gains(self,
                   title: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 6),
                   filename: Optional[str] = None,
                   display: bool = True):
        """Plot the cumulative gains curve."""
        plot_gains(self.T, self.name, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_lift(self,
                  title: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 6),
                  filename: Optional[str] = None,
                  display: bool = True):
        """Plot the lift curve."""
        plot_lift(self.T, self.name, title, figsize, filename, display)

    @composed(crash, params_to_log, typechecked)
    def plot_bo(self,
                title: Optional[str] = None,
                figsize: Tuple[int, int] = (10, 6),
                filename: Optional[str] = None,
                display: bool = True):
        """Plot the bayesian optimization scoring."""
        plot_bo(self.T, self.name, title, figsize, filename, display)

    # << ============ Utility functions ============ >>

    @composed(crash, params_to_log, typechecked)
    def save(self, filename: Optional[str] = None):
        """Save the class to a pickle file."""
        save(self, 'ATOM_' + self.name if filename is None else filename)
        self.T.log("ATOM's " + self.name + " class saved successfully!", 1)

    @composed(crash, params_to_log, typechecked)
    def save_model(self, filename: Optional[str] = None):
        """Save the best found model (fitted) to a pickle file."""
        filename = 'model_' + self.name if filename is None else filename
        save(self.best_model_fit, filename)
        self.T.log(self.longname + " model saved successfully!", 1)

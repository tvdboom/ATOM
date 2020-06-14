# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the parent class for all training classes.

"""

# Standard packages
import importlib
import pandas as pd
from time import time
from tqdm import tqdm
from copy import deepcopy
from typeguard import typechecked
from typing import Optional, Union, Sequence, Tuple
from sklearn.metrics import SCORERS

# Own modules
from .data_cleaning import BaseCleaner
from .models import MODEL_LIST, get_model_name
from .utils import (
    CAL, X_TYPES, Y_TYPES, OPTIONAL_PACKAGES, ONLY_CLASSIFICATION,
    ONLY_REGRESSION, to_df, time_to_string, check_is_fitted, catch_return,
    get_best_score, get_metric, get_metric_name, infer_task, clear,
    composed, crash
    )
from .plots import (
    plot_roc, plot_prc, plot_bagging, plot_permutation_importance,
    plot_feature_importance, plot_confusion_matrix, plot_threshold,
    plot_probabilities, plot_calibration, plot_gains, plot_lift, plot_bo
    )


# << ================ Classes ================= >>

class BaseTrainer(BaseCleaner):
    """Base estimator for the training classes."""

    def __init__(self, models, metric, greater_is_better, needs_proba,
                 needs_threshold, n_calls, n_random_starts, bo_kwargs,
                 bagging, n_jobs, verbose, warnings, logger, random_state):
        """Initialize class.

        Check parameters and prepare them for the run.

        """
        super().__init__(n_jobs=n_jobs,
                         verbose=verbose,
                         logger=logger,
                         warnings=warnings,
                         random_state=random_state)

        # Model attributes
        self.models = []
        self.errors = {}
        self.winner = None
        self._results = []

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
        if self.goal.startswith('class'):
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
            self.n_calls = n_calls
            if len(self.n_calls) != len(self.models):
                raise ValueError("Invalid value for the n_calls parameter. " +
                                 "Length should be equal to the number of " +
                                 "models, got len(models)=" +
                                 f"{len(self.models)} and len(n_calls)=" +
                                 f"{len(self.n_calls)}.")
        else:
            self.n_calls = [n_calls for _ in self.models]
        if isinstance(n_random_starts, (list, tuple)):
            self.n_random_starts = n_random_starts
            if len(self.n_random_starts) != len(self.models):
                raise ValueError("Invalid value for the n_random_starts " +
                                 "parameter. Length should be equal to the " +
                                 "number of models, got len(models)=" +
                                 f"{len(self.models)} and len(n_random_" +
                                 f"starts)={len(self.n_random_starts)}.")
        else:
            self.n_random_starts = [n_random_starts for _ in self.models]
        if isinstance(bagging, (list, tuple)):
            self.bagging = bagging
            if len(self.bagging) != len(self.models):
                raise ValueError("Invalid value for the bagging parameter. " +
                                 "Length should be equal to the number of " +
                                 "models, got len(models)=" +
                                 f"{len(self.models)} and len(bagging)" +
                                 f"{len(self.bagging)}.")
        else:
            self.bagging = [bagging for _ in self.models]

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

        self.metric = get_metric(
            metric, greater_is_better, needs_proba, needs_threshold)

        # Assign the name corresponding to the scorer
        self.metric.name = get_metric_name(self.metric)

    @property
    def results(self):
        # Return df without bagging cols if all are empty
        if len(self._results) == 1:
            return self._results[0].dropna(axis=1, how='all')
        else:
            return [df.dropna(axis=1, how='all') for df in self._results]

    def _run(self, train, test):
        """Core iteration.

        Parameters
        ----------
        train: pd.DataFrame
            Training set used for this iteration.

        test: pd.DataFrame
            Test set used for this iteration.

        Returns
        -------
        results: pd.DataFrame
            Dataframe of the results for this iteration.

        """
        # Prepare input
        if not isinstance(train, pd.DataFrame):
            train = to_df(deepcopy(train))
        if not isinstance(test, pd.DataFrame):
            train = to_df(deepcopy(test))

        # Assign algorithm's task
        self.task = infer_task(train.iloc[:, -1], goal=self.goal)

        t_init = time()  # To measure the time the whole pipeline takes

        # If verbose=1, use tqdm to evaluate process
        loop = self.models
        if self.verbose == 1:
            loop = tqdm(self.models, desc='Processing')
            loop.clear()  # Prevent starting a progress bar before the loop

        # Loop over every independent model
        to_remove = []
        for m, n_calls, n_random_starts, bagging in zip(
                loop, self.n_calls, self.n_random_starts, self.bagging):

            model_time = time()

            # Define model class
            setattr(self, m, MODEL_LIST[m](self, train, test))
            subclass = getattr(self, m)
            setattr(self, m.lower(), subclass)  # Lowercase as well

            try:  # If errors occurs, just skip the model
                # Run Bayesian Optimization
                # Use copy of kwargs to not delete original in method
                # Shallow copy is enough since we only delete keys
                if hasattr(subclass, 'get_domain') and n_calls > 0:
                    subclass.bayesian_optimization(
                        n_calls, n_random_starts, self.bo_kwargs.copy())
                subclass.fit()
                if bagging:
                    subclass.bagging(bagging)

                # Get the total time spend on this model
                total_time = time_to_string(model_time)
                setattr(subclass, 'time', total_time)
                self.log('-' * 49, 1)
                self.log(f'Total time: {total_time}', 1)

            except Exception as ex:
                if hasattr(subclass, 'get_domain') and n_calls > 0:
                    self.log('', 1)  # Add extra line
                self.log("Exception encountered while running the "
                         + f"{m} model. Removing model from "
                         + f"pipeline. \n{type(ex).__name__}: {ex}", 1)

                # Append exception to ATOM errors dictionary
                exception = type(ex).__name__ + ': ' + str(ex)
                self.errors[m] = exception

                # Add model to "garbage collector"
                # Cannot remove at once to maintain iteration order
                to_remove.append(m)

        # Remove faulty models
        clear(self, to_remove)

        # Check if all models failed (self.models is empty)
        if not self.models:
            raise ValueError('It appears all models failed to run...')

        # << =================== Print final results ================== >>

        # Create dataframe with final results
        results = pd.DataFrame(
            columns=['name', 'score_train', 'score_test', 'time_fit',
                     'mean_bagging', 'std_bagging', 'time_bagging', 'time'])

        # Print final results
        self.log("\n\nFinal results ================>>")
        self.log(f"Duration: {time_to_string(t_init)}")
        self.log(f"Metric: {self.metric.name}")
        self.log("--------------------------------")

        # Create a df to get maxlen of name and best_scores
        all_ = pd.DataFrame(columns=['name', 'score_test', 'mean_bagging'])
        for model in self.models:
            m = getattr(self, model)
            all_.loc[model] = [m.longname, m.score_test, m.mean_bagging]

        # Get max length of the models' names
        maxlen = max(all_['name'].apply(lambda x: len(x)))

        # List of scores the test set
        best_score = all_.apply(lambda row: get_best_score(row), axis=1)

        for m in self.models:
            name = getattr(self, m).name
            longname = getattr(self, m).longname
            score_train = getattr(self, m).score_train
            score_test = getattr(self, m).score_test
            time_fit = getattr(self, m).time_fit
            mean_bagging = getattr(self, m).mean_bagging
            std_bagging = getattr(self, m).std_bagging
            time_bagging = getattr(self, m).time_bagging
            total_time = getattr(self, m).time

            # Append model row to results
            results.loc[name] = [longname,
                                 score_train,
                                 score_test,
                                 time_fit,
                                 mean_bagging,
                                 std_bagging,
                                 time_bagging,
                                 total_time]

            if not mean_bagging:
                # Create string of the score
                print_ = f"{longname:{maxlen}s} --> {score_test:.3f}"

                # Highlight best score and assign winner attribute
                if score_test == max(best_score):
                    self.winner = getattr(self, m)
                    if len(self.models) > 1:
                        print_ += ' !'

            else:
                # Create string of the score
                print1 = f"{longname:{maxlen}s} --> {mean_bagging:.3f}"
                print2 = f"{std_bagging:.3f}"
                print_ = print1 + u" \u00B1 " + print2

                # Highlight best score and assign winner attribute
                if mean_bagging == max(best_score):
                    self.winner = getattr(self, m)
                    if len(self.models) > 1:
                        print_ += ' !'

            # Annotate if model overfitted when train 20% > test
            if score_train - 0.2 * score_train > score_test:
                print_ += ' ~'

            self.log(print_)  # Print the score

        return results

    @composed(crash, typechecked)
    def clear(self, models: Union[str, Sequence[str]] = 'all'):
        """Clear models from the trainer.

        If the winning model is removed. The next best model (through
        score_test or mean_bagging if available) is selected as winner.

        Parameters
        ----------
        models: str, or sequence, optional (default='all')
            Name of the models to clear from the pipeline. If 'all', clear
            all models.

        """
        # Prepare the models parameter
        if models == 'all':
            keyword = 'Pipeline'
            models = self.models.copy()
        elif isinstance(models, str):
            keyword = 'Model'
            models = [get_model_name(models)]
        else:
            keyword = 'Models'
            models = [get_model_name(m) for m in models]

        clear(self, models)

        self.log(keyword + " cleared successfully!", 1)

    @composed(crash, typechecked)
    def outcome(self, metric: Optional[str] = None):
        """Print the trainer's final outcome for a specific metric.

        If a model shows a `XXX`, it means the metric failed for that specific
        model. This can happen if either the metric is unavailable for the task
        or if the model does not have a `predict_proba` method while the metric
        requires it.

        Parameters
        ----------
        metric: string or None, optional (default=None)
            String of one of sklearn's predefined scorers. If None, the metric
            used to fit the trainer is selected and the bagging results will
            be showed (if used).

        """
        check_is_fitted(self, 'winner')

        # Prepare parameters
        if metric is None:
            metric = self.metric.name
        elif metric not in SCORERS:
            raise ValueError("Unknown value for the metric parameter, " +
                             f"got {metric}. Try one of {', '.join(SCORERS)}.")

        # Get max side of the models' names
        max_len = max([len(getattr(self, m).longname) for m in self.models])

        # Get list of scores
        all_scores = []
        for model in self.models:
            m = getattr(self, model)
            if metric == self.metric.name:
                score = m.mean_bagging if m.mean_bagging else m.score_test
                all_scores.append(score)

            # If invalid metric, don't append to scores
            elif not isinstance(getattr(m, metric), str):
                all_scores.append(getattr(m, metric))

        if len(all_scores) == 0:  # All metrics failed
            raise ValueError("Invalid metric selected!")

        self.log("Results ======================>>", -2)
        self.log(f"Metric: {metric}", -2)
        self.log("--------------------------------", -2)

        for model in self.models:
            m = getattr(self, model)

            if metric == self.metric.name and m.mean_bagging:
                score = m.mean_bagging

                # Create string of the score
                print1 = f"{m.longname:{max_len}s} --> {score:.3f}"
                print2 = f"{m.std_bagging:.3f}"
                print_ = print1 + u" \u00B1 " + print2
            else:
                if metric == self.metric.name:
                    score = m.score_test
                else:
                    score = getattr(m, metric)

                # Create string of the score (if wrong metric for model -> XXX)
                if isinstance(score, str):
                    print_ = f"{m.longname:{max_len}s} --> XXX"
                else:
                    print_ = f"{m.longname:{max_len}s} --> {score:.3f}"

            # Highlight best score (if more than one model in pipeline)
            if score == max(all_scores) and len(self.models) > 1:
                print_ += ' !'

            # Annotate if model overfitted when train 20% > test
            if metric == self.metric.name:
                if m.score_train - 0.2 * m.score_train > m.score_test:
                    print_ += ' ~'

            self.log(print_, -2)  # Always print

    # << ================== Transformation methods ================== >>

    def _pipeline_methods(self, X, y=None, method='predict', **kwargs):
        """Apply pipeline methods on new data.

        First transform the new data and apply the attribute on the winning
        model. The model has to have the provided attribute.

        Parameters
        ----------
        self: class
            Class from which we call the method.

        X: dict, sequence, np.array or pd.DataFrame
            Data containing the features, with shape=(n_samples, n_features).

        y: int, str, sequence, np.array, pd.Series, optional (default=None)
            - If None, the target column is not used in the attribute
            - If int: index of the column of X which is selected as target
            - If string: name of the target column in X
            - Else: data target column with shape=(n_samples,)

        method: str, optional (default='predict')
            Method of the model to be applied.

        **kwargs
            Additional parameters for the transform method.

        Returns
        -------
        np.array
            Return of the attribute.

        """
        if hasattr(self, 'winner'):
            check_is_fitted(self, 'winner')
            model = self.winner
        else:
            model = self

        if not hasattr(model.best_model_fit, method):
            raise AttributeError(f"The {model.name} model doesn't have a " +
                                 f"{method} method!")

        # When called from the ATOM class, apply all data transformations first
        if hasattr(self, 'transform'):
            X, y = catch_return(self.transform(X, y, **kwargs))

        if y is None:
            return getattr(model.best_model_fit, method)(X)
        else:
            return getattr(model.best_model_fit, method)(X, y)

    @composed(crash, typechecked)
    def predict(self, X: X_TYPES):
        """Get predictions on new data."""
        return self._pipeline_methods(X, method='predict')

    @composed(crash, typechecked)
    def predict_proba(self, X: X_TYPES):
        """Get probability predictions on new data."""
        return self._pipeline_methods(X, method='predict_proba')

    @composed(crash, typechecked)
    def predict_log_proba(self, X: X_TYPES):
        """Get log probability predictions on new data."""
        return self._pipeline_methods(X, method='predict_log_proba')

    @composed(crash, typechecked)
    def decision_function(self, X: X_TYPES):
        """Get the decision function on new data."""
        return self._pipeline_methods(X, method='decision_function')

    @composed(crash, typechecked)
    def score(self, X: X_TYPES, y: Y_TYPES):
        """Get the score function on new data."""
        return self._pipeline_methods(X, y, method='score')

    # << ======================= Plot methods ======================= >>

    @composed(crash, typechecked)
    def plot_bagging(self,
                     models: Union[None, str, Sequence[str]] = None,
                     title: Optional[str] = None,
                     figsize: Optional[Tuple[int, int]] = None,
                     filename: Optional[str] = None,
                     display: bool = True):
        """Boxplot of the bagging's results."""
        plot_bagging(self, models, title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_roc(self,
                 models: Union[None, str, Sequence[str]] = None,
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the Receiver Operating Characteristics curve."""
        plot_roc(self, models, title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_prc(self,
                 models: Union[None, str, Sequence[str]] = None,
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6),
                 filename: Optional[str] = None,
                 display: bool = True):
        """Plot the precision-recall curve."""
        plot_prc(self, models, title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_permutation_importance(
            self,
            models: Union[None, str, Sequence[str]] = None,
            show: Optional[int] = None,
            n_repeats: int = 10,
            title: Optional[str] = None,
            figsize: Optional[Tuple[int, int]] = None,
            filename: Optional[str] = None,
            display: bool = True):
        """Plot the feature permutation importance of models."""
        plot_permutation_importance(self, models, show, n_repeats,
                                    title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_feature_importance(self,
                                models: Union[None, str, Sequence[str]] = None,
                                show: Optional[int] = None,
                                title: Optional[str] = None,
                                figsize: Optional[Tuple[int, int]] = None,
                                filename: Optional[str] = None,
                                display: bool = True):
        """Plot a tree-based model's normalized feature importances."""
        plot_feature_importance(self, models, show,
                                title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_confusion_matrix(self,
                              models: Union[None, str, Sequence[str]] = None,
                              normalize: bool = False,
                              title: Optional[str] = None,
                              figsize: Tuple[int, int] = (8, 8),
                              filename: Optional[str] = None,
                              display: bool = True):
        """Plot the confusion matrix.

        For 1 model: plot it's confusion matrix in a heatmap.
        For >1 models: compare TP, FP, FN and TN in a barplot.

        """
        plot_confusion_matrix(self, models, normalize,
                              title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_threshold(self,
                       models: Union[None, str, Sequence[str]] = None,
                       metric: Optional[Union[CAL, Sequence[CAL]]] = None,
                       steps: int = 100,
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       filename: Optional[str] = None,
                       display: bool = True):
        """Plot performance metric(s) against threshold values."""
        plot_threshold(self, models, metric, steps,
                       title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_probabilities(self,
                           models: Union[None, str, Sequence[str]] = None,
                           target: Union[int, str] = 1,
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           filename: Optional[str] = None,
                           display: bool = True):
        """Plot the distribution of predicted probabilities."""
        plot_probabilities(self, models, target,
                           title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_calibration(self,
                         n_bins: int = 10,
                         models: Union[None, str, Sequence[str]] = None,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 10),
                         filename: Optional[str] = None,
                         display: bool = True):
        """Plot the calibration curve for a binary classifier."""
        plot_calibration(self, models, n_bins,
                         title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_gains(self,
                   models: Union[None, str, Sequence[str]] = None,
                   title: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 6),
                   filename: Optional[str] = None,
                   display: bool = True):
        """Plot the cumulative gains curve."""
        plot_gains(self, models, title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_lift(self,
                  models: Union[None, str, Sequence[str]] = None,
                  title: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 6),
                  filename: Optional[str] = None,
                  display: bool = True):
        """Plot the lift curve."""
        plot_lift(self, models, title, figsize, filename, display)

    @composed(crash, typechecked)
    def plot_bo(self,
                models: Union[None, str, Sequence[str]] = None,
                title: Optional[str] = None,
                figsize: Tuple[int, int] = (10, 6),
                filename: Optional[str] = None,
                display: bool = True):
        """Plot the bayesian optimization scoring."""
        plot_bo(self, models, title, figsize, filename, display)

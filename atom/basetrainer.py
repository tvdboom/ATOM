# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the parent class for all training classes.

"""

# Standard packages
import importlib
import pandas as pd
from time import time

# Own modules
from .models import MODEL_LIST
from .basepredictor import BasePredictor
from .data_cleaning import BaseTransformer, Scaler
from .utils import (
    OPTIONAL_PACKAGES, ONLY_CLASSIFICATION, ONLY_REGRESSION, merge, to_df,
    to_series, get_best_score, time_to_string, get_model_name, get_metric, clear
    )


class BaseTrainer(BaseTransformer, BasePredictor):
    """Base estimator for the training classes.

    Parameters
    ----------
    models: string or sequence
        List of models to fit on the data. Use the predefined acronyms
        in MODEL_NAMES.

    metric: str, callable or sequence, optional (default=None)
        Metric(s) on which the pipeline fits the models. Choose from any of
        the string scorers predefined by sklearn, use a score (or loss)
        function with signature metric(y, y_pred, **kwargs) or use a
        scorer object. If multiple metrics are selected, only the first will
        be used to optimize the BO. If None, a default metric is selected:
            - 'f1' for binary classification
            - 'f1_weighted' for multiclass classification
            - 'r2' for regression

    greater_is_better: bool or sequence, optional (default=True)
        Whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower is
        better. Will be ignored if the metric is a string or a scorer.
        If sequence, the n-th value will apply to the n-th metric in the
        pipeline.

    needs_proba: bool or sequence, optional (default=False)
        Whether the metric function requires probability estimates out of a
        classifier. If True, make sure that every model in the pipeline has
        a `predict_proba` method. Will be ignored if the metric is a string
        or a scorer. If sequence, the n-th value will apply to the n-th metric
        in the pipeline.

    needs_threshold: bool or sequence, optional (default=False)
        Whether the metric function takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method. Will
        be ignored if the metric is a string or a scorer. If sequence, the
        n-th value will apply to the n-th metric in the pipeline.

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
        Dictionary of extra keyword arguments for the BO. See bayesian_optimization
        in basemodel.py for the available options.

    bagging: int, sequence or None, optional (default=None)
        Number of data sets (bootstrapped from the training set) to use in
        the bagging algorithm. If None or 0, no bagging is performed.
        If sequence, the n-th value will apply to the n-th model in the
        pipeline.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 - n_jobs.

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

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name, False for no logger.
        - If string: name of the logging file. 'auto' for default name.
        - If class: python Logger object.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    def __init__(self, models, metric, greater_is_better, needs_proba,
                 needs_threshold, n_calls, n_random_starts, bo_params,
                 bagging, n_jobs, verbose, warnings, logger, random_state):
        super().__init__(n_jobs=n_jobs,
                         verbose=verbose,
                         warnings=warnings,
                         logger=logger,
                         random_state=random_state)

        # Data attribute
        self._data = None
        self._idx = [None, None]
        self.scaler = None
        self.task = None

        # Model attributes
        self.models = []
        self.errors = {}
        self._results = pd.DataFrame(
            columns=['name', 'score_bo', 'time_bo',
                     'score_train', 'score_test', 'time_fit',
                     'mean_bagging', 'std_bagging', 'time_bagging', 'time'])

        # Check validity models ============================================= >>

        if isinstance(models, str):
            models = [models]

        # Set models to right name
        self.models = [get_model_name(m) for m in models]

        # Check for duplicates
        if len(self.models) != len(set(self.models)):
            raise ValueError("There are duplicate values in the models parameter!")

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
                    raise ValueError(
                        f"The {m} model can't perform classification tasks!")
        else:
            for m in ONLY_CLASSIFICATION:
                if m in self.models:
                    raise ValueError(
                        f"The {m} model can't perform regression tasks!")

        # Check validity parameters ========================================= >>

        # Check Parameters
        if isinstance(n_calls, (list, tuple)):
            self.n_calls = n_calls
            if len(self.n_calls) != len(self.models):
                raise ValueError(
                    "Invalid value for the n_calls parameter. Length should " +
                    "be equal to the number of models, got len(models)=" +
                    f"{len(self.models)} and len(n_calls)={len(self.n_calls)}.")
        else:
            self.n_calls = [n_calls for _ in self.models]
        if isinstance(n_random_starts, (list, tuple)):
            self.n_random_starts = n_random_starts
            if len(self.n_random_starts) != len(self.models):
                raise ValueError(
                    "Invalid value for the n_random_starts parameter. Length " +
                    "should be equal to the number of models, got len(models)=" +
                    f"{len(self.models)} and len(n_random_starts)=" +
                    f"{len(self.n_random_starts)}.")
        else:
            self.n_random_starts = [n_random_starts for _ in self.models]
        if isinstance(bagging, (list, tuple)):
            self.bagging = bagging
            if len(self.bagging) != len(self.models):
                raise ValueError(
                    "Invalid value for the bagging parameter. Length should " +
                    "be equal to the number of models, got len(models)=" +
                    f"{len(self.models)} and len(bagging)={len(self.bagging)}.")
        else:
            self.bagging = [bagging for _ in self.models]

        # Check dimensions params =========================================== >>

        self.bo_params = bo_params
        if self.bo_params.get('dimensions'):
            # Dimensions can be array for one model or dict if more
            if not isinstance(self.bo_params.get('dimensions'), dict):
                if len(self.models) != 1:
                    raise TypeError(
                        "Invalid type for the dimensions parameter. For >1 " +
                        "models, use a dictionary with the model names as keys!")
                else:
                    self.bo_params['dimensions'] = \
                        {self.models[0]: self.bo_params['dimensions']}

            # Assign proper model name to key of dimensions dict
            for key in self.bo_params['dimensions']:
                self.bo_params['dimensions'][get_model_name(key)] = \
                    self.bo_params['dimensions'].pop(key)

        # Check validity metric ============================================= >>

        self.metric = self._prepare_metric(
            metric, greater_is_better, needs_proba, needs_threshold)

    @staticmethod
    def _prepare_metric(metric, gib, needs_proba, needs_threshold):
        """Return a metric scorer given the parameters."""
        if not isinstance(metric, (list, tuple)):
            metric = [metric]
        elif len(metric) > 3:
            raise ValueError("A maximum of 3 metrics are allowed!")

        # Check metric parameters
        if isinstance(gib, (list, tuple)):
            if len(gib) != len(metric):
                raise ValueError("Invalid value for the greater_is_better " +
                                 "parameter. Length should be equal to the number " +
                                 f"of metrics, got len(metric)={len(metric)} " +
                                 f"and len(greater_is_better)={len(gib)}.")
        else:
            gib = [gib for _ in metric]

        if not isinstance(metric, (list, tuple)):
            metric = [metric]

        if isinstance(needs_proba, (list, tuple)):
            if len(needs_proba) != len(metric):
                raise ValueError("Invalid value for the needs_proba " +
                                 "parameter. Length should be equal to the number " +
                                 f"of metrics, got len(metric)={len(metric)} " +
                                 f"and len(needs_proba)={len(needs_proba)}.")
        else:
            needs_proba = [needs_proba for _ in metric]

        if isinstance(needs_threshold, (list, tuple)):
            if len(needs_threshold) != len(metric):
                raise ValueError("Invalid value for the needs_threshold " +
                                 "parameter. Length should be equal to the number " +
                                 f"of metrics, got len(metric)={len(metric)} " +
                                 f"and len(needs_threshold)={len(needs_threshold)}.")
        else:
            needs_threshold = [needs_threshold for _ in metric]

        metric_list = []
        for i, j, m, n in zip(metric, gib, needs_proba, needs_threshold):
            metric_list.append(get_metric(i, j, m, n))

        return metric_list

    def _params_to_attr(self, *args):
        """Attach the provided data as attributes of the class."""
        # Data can be already in attrs
        if len(args) == 0 and self._data is not None:
            return
        if len(args) == 2:
            train, test = to_df(args[0]), to_df(args[1])
        elif len(args) == 4:
            train = merge(to_df(args[0]), to_series(args[2]))
            test = merge(to_df(args[1]), to_series(args[3]))
        else:
            raise ValueError(
                "Invalid parameters. Must be either of the form (train, " +
                "test) or (X_train, X_test, y_train, y_test).")

        # Update the data attributes
        self._data = pd.concat([train, test]).reset_index(drop=True)
        self._idx = [len(train), len(test)]

        # Reset data scaler in case of a rerun with new data
        self.scaler = None

    def _run(self):
        """Core iteration.

        Returns
        -------
        results: pd.DataFrame
            Dataframe of the results for this iteration.

        """
        t_init = time()  # To measure the time the whole pipeline takes

        # Loop over every independent model
        to_remove = []
        for m, n_calls, n_random_starts, bagging in zip(
                self.models, self.n_calls, self.n_random_starts, self.bagging):

            # Check n_calls parameter
            if n_calls < 0:
                raise ValueError("Invalid value for the n_calls parameter. " +
                                 f"Value should be >=0, got {n_calls}.")

            model_time = time()

            # Define model class
            setattr(self, m, MODEL_LIST[m](self))
            subclass = getattr(self, m)
            setattr(self, m.lower(), subclass)  # Lowercase as well

            # Create scaler if model needs scaling and data not already scaled
            if subclass.need_scaling and not self.scaler:
                self.scaler = Scaler().fit(self.X_train)

            try:  # If errors occurs, skip the model
                # Run Bayesian Optimization
                # Use copy of kwargs to not delete original in method
                # Shallow copy is enough since we only delete entries in basemodel
                if hasattr(subclass, 'get_domain') and n_calls > 0:
                    subclass.bayesian_optimization(
                        n_calls, n_random_starts, self.bo_params.copy())

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

                # Append exception to errors dictionary
                self.errors[m] = ex

                # Add model to "garbage collector"
                # Cannot remove at once to maintain iteration order
                to_remove.append(m)

        # Remove faulty models
        clear(self, to_remove)

        # Check if all models failed (self.models is empty)
        if not self.models:
            raise RuntimeError('It appears all models failed to run...')

        # Print final results =============================================== >>

        # Create dataframe with final results
        results = pd.DataFrame(columns=self._results.columns)

        # Print final results
        self.log("\n\nFinal results ========================= >>", 1)
        self.log(f"Duration: {time_to_string(t_init)}", 1)
        self.log('-' * 42, 1)

        # Get max length of the model names
        maxlen = max([len(m.longname) for m in self.models_])

        # Get best score of all the models
        best_score = max([get_best_score(m) for m in self.models_])

        for m in self.models_:
            # Append model row to results
            values = [m.longname, m.score_bo, m.time_bo,
                      m.score_train, m.score_test, m.time_fit,
                      m.mean_bagging, m.std_bagging, m.time_bagging, m.time]
            m._results.loc[m.name] = values
            results.loc[m.name] = values

            # Get the model's final output and highlight best score
            out = f"{m.longname:{maxlen}s} --> {m._final_output()}"
            if get_best_score(m) == best_score and len(self.models) > 1:
                out += ' !'

            self.log(out, 1)  # Print the score

        return results

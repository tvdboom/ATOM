# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the parent class for all training classes.

"""

# Standard packages
import importlib
import pandas as pd
from time import time
import matplotlib.pyplot as plt

# Own modules
from .models import MODEL_LIST
from .basepredictor import BasePredictor
from .data_cleaning import BaseTransformer, Scaler
from .utils import (
    OPTIONAL_PACKAGES, ONLY_CLASS, ONLY_REG, merge, to_df, to_series, get_best_score,
    time_to_string, get_model_name, get_metric, get_default_metric, clear
)


class BaseTrainer(BaseTransformer, BasePredictor):
    """Base estimator for the training classes.

    Parameters
    ----------
    models: string or sequence
        List of models to fit on the data. Use the predefined acronyms
        in MODEL_LIST.

    metric: str, callable or sequence, optional (default=None)
        Metric(s) on which the pipeline fits the models. Choose from any of
        the scorers predefined by sklearn, use a score (or loss) function with
        signature metric(y, y_pred, **kwargs) or use a scorer object.
        If multiple metrics are selected, only the first will be used to
        optimize the BO. If None, a default metric is selected:
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
        classifier. If True, make sure that every estimator in the pipeline has
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

    n_initial_points: int or sequence, optional (default=5)
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

    warnings: bool or str, optional (default=True)
        - If True: Default warning action (equal to 'default' when string).
        - If False: Suppress all warnings (equal to 'ignore' when string).
        - If str: One of the possible actions in python's warnings environment.

        Note that changing this parameter will affect the `PYTHONWARNINGS`
        environment.

        Note that ATOM can't manage warnings that go directly from C/C++ code
        to the stdout/stderr.

    logger: bool, str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If bool: True for logging file with default name. False for no logger.
        - If string: name of the logging file. 'auto' for default name.
        - If class: python `Logger` object'.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` instance used by `numpy.random`.

    """

    def __init__(self, models, metric, greater_is_better, needs_proba,
                 needs_threshold, n_calls, n_initial_points, bo_params,
                 bagging, n_jobs, verbose, warnings, logger, random_state):
        super().__init__(n_jobs=n_jobs,
                         verbose=verbose,
                         warnings=warnings,
                         logger=logger,
                         random_state=random_state)

        # Parameter attributes
        self.models = models
        self.metric_ = metric
        self.greater_is_better = greater_is_better
        self.needs_proba = needs_proba
        self.needs_threshold = needs_threshold
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.bo_params = bo_params
        self.bagging = bagging

        # Data attributes
        self._data = None
        self._idx = [None, None]
        self.mapping = {}
        self.scaler = None
        self.task = None

        # Model attributes
        self.errors = {}
        self._results = pd.DataFrame(
            columns=['name', 'metric_bo', 'time_bo',
                     'metric_train', 'metric_test', 'time_fit',
                     'mean_bagging', 'std_bagging', 'time_bagging', 'time'])

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
                "Invalid parameters. Must be either of the form (train, "
                "test) or (X_train, X_test, y_train, y_test).")

        # Update the data attributes
        self._data = pd.concat([train, test]).reset_index(drop=True)
        self._idx = [len(train), len(test)]

        # Reset data scaler in case of a rerun with new data
        self.scaler = None

    def _check_parameters(self):
        """Check the validity of the input parameters."""
        if isinstance(self.models, str):
            self.models = [self.models]

        # Set models to right name
        models = [get_model_name(m) for m in self.models]

        # Check for duplicates
        if len(models) != len(set(models)):
            raise ValueError("There are duplicate values in the models parameter!")

        # Check if packages for not-sklearn models are available
        for m, package in OPTIONAL_PACKAGES:
            if m in models:
                try:
                    importlib.import_module(package)
                except ImportError:
                    raise ValueError(f"Unable to import {package}!")

        # Remove regression/classification-only models from pipeline
        if self.goal.startswith('class'):
            for m in ONLY_REG:
                if m in models:
                    raise ValueError(
                        f"The {m} model can't perform classification tasks!")
        else:
            for m in ONLY_CLASS:
                if m in models:
                    raise ValueError(
                        f"The {m} model can't perform regression tasks!")

        self.models = models

        # Check validity BO parameters ======================================= >>

        if isinstance(self.n_calls, (list, tuple)):
            if len(self.n_calls) != len(self.models):
                raise ValueError(
                    "Invalid value for the n_calls parameter. Length should "
                    "be equal to the number of models, got len(models)="
                    f"{len(self.models)} and len(n_calls)={len(self.n_calls)}.")
        else:
            self.n_calls = [self.n_calls for _ in self.models]
        if isinstance(self.n_initial_points, (list, tuple)):
            if len(self.n_initial_points) != len(self.models):
                raise ValueError(
                    "Invalid value for the n_initial_points parameter. Length "
                    "should be equal to the number of models, got len(models)="
                    f"{len(self.models)} and len(n_initial_points)="
                    f"{len(self.n_initial_points)}.")
        else:
            self.n_initial_points = [self.n_initial_points for _ in self.models]
        if isinstance(self.bagging, (list, tuple)):
            if len(self.bagging) != len(self.models):
                raise ValueError(
                    "Invalid value for the bagging parameter. Length should "
                    "be equal to the number of models, got len(models)="
                    f"{len(self.models)} and len(bagging)={len(self.bagging)}.")
        else:
            self.bagging = [self.bagging for _ in self.models]

        # Check dimensions params =========================================== >>

        if self.bo_params.get('dimensions'):
            # Dimensions can be array for one model or dict if more
            if not isinstance(self.bo_params.get('dimensions'), dict):
                if len(self.models) != 1:
                    raise TypeError(
                        "Invalid type for the dimensions parameter. For >1 "
                        "models, use a dictionary with the model names as keys!")
                else:
                    self.bo_params['dimensions'] = \
                        {self.models[0]: self.bo_params['dimensions']}

            # Assign proper model name to key of dimensions dict
            dimensions = {}
            for key in self.bo_params['dimensions']:
                dimensions[get_model_name(key)] = self.bo_params['dimensions'][key]
            self.bo_params['dimensions'] = dimensions

        # Check validity metric ============================================= >>

        if not self.metric_:
            self.metric_ = [get_default_metric(self.task)]
        else:
            self.metric_ = self._prepare_metric(
                metric=self.metric_,
                gib=self.greater_is_better,
                needs_proba=self.needs_proba,
                needs_threshold=self.needs_threshold
            )

        # Assign mapping ==================================================== >>

        if not self.task.startswith('reg'):
            self.mapping = {str(i): i for i in self.categories}

    @staticmethod
    def _prepare_metric(metric, gib, needs_proba, needs_threshold):
        """Return a list of metric scorers given the parameters."""
        if not isinstance(metric, (list, tuple)):
            metric = [metric]

        # Check metric parameters
        if isinstance(gib, (list, tuple)):
            if len(gib) != len(metric):
                raise ValueError("Invalid value for the greater_is_better "
                                 "parameter. Length should be equal to the number "
                                 f"of metrics, got len(metric)={len(metric)} "
                                 f"and len(greater_is_better)={len(gib)}.")
        else:
            gib = [gib for _ in metric]

        if isinstance(needs_proba, (list, tuple)):
            if len(needs_proba) != len(metric):
                raise ValueError("Invalid value for the needs_proba " +
                                 "parameter. Length should be equal to the number "
                                 f"of metrics, got len(metric)={len(metric)} "
                                 f"and len(needs_proba)={len(needs_proba)}.")
        else:
            needs_proba = [needs_proba for _ in metric]

        if isinstance(needs_threshold, (list, tuple)):
            if len(needs_threshold) != len(metric):
                raise ValueError("Invalid value for the needs_threshold "
                                 "parameter. Length should be equal to the number "
                                 f"of metrics, got len(metric)={len(metric)} "
                                 f"and len(needs_threshold)={len(needs_threshold)}.")
        else:
            needs_threshold = [needs_threshold for _ in metric]

        metric_list = []
        for i, j, m, n in zip(metric, gib, needs_proba, needs_threshold):
            metric_list.append(get_metric(i, j, m, n))

        return metric_list

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
        for idx, (m, n_calls, n_initial_points, bagging) in enumerate(zip(
                self.models, self.n_calls, self.n_initial_points, self.bagging)):

            # Check n_calls parameter
            if n_calls < 0:
                raise ValueError("Invalid value for the n_calls parameter. "
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
                        n_calls, n_initial_points, self.bo_params.copy())

                subclass.fit()

                if bagging:
                    subclass.bagging(bagging)

                # Get the total time spend on this model
                total_time = time_to_string(model_time)
                setattr(subclass, 'time', total_time)
                self.log('-' * 49, 1)
                self.log(f'Total time: {total_time}', 1)

            except Exception as ex:
                if idx != 0 or (hasattr(subclass, 'get_domain') and n_calls > 0):
                    self.log('', 1)  # Add extra line
                self.log(f"Exception encountered while running the {m} model. Remov"
                         f"ing model from pipeline. \n{type(ex).__name__}: {ex}", 1)

                # Append exception to errors dictionary
                self.errors[m] = ex

                # Add model to "garbage collector"
                # Cannot remove at once to maintain iteration order
                to_remove.append(m)

        # Close the BO plot if there was one
        if self.bo_params.get('plot_bo'):
            plt.close()

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
            values = [m.longname, m.metric_bo, m.time_bo,
                      m.metric_train, m.metric_test, m.time_fit,
                      m.mean_bagging, m.std_bagging, m.time_bagging, m.time]
            m._results.loc[m.name] = values
            results.loc[m.name] = values

            # Get the model's final output and highlight best score
            out = f"{m.longname:{maxlen}s} --> {m._final_output()}"
            if get_best_score(m) == best_score and len(self.models) > 1:
                out += ' !'

            self.log(out, 1)  # Print the score

        return results

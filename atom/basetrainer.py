# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the parent class for the trainers.

"""

# Standard packages
import importlib
import pandas as pd
from time import time
import matplotlib.pyplot as plt

# Own modules
from .branch import Branch
from .models import MODEL_LIST, CustomModel
from .basepredictor import BasePredictor
from .data_cleaning import BaseTransformer
from .utils import (
    SEQUENCE, OPTIONAL_PACKAGES, ONLY_CLASS, ONLY_REG, lst,
    get_best_score, time_to_string, get_metric, get_default_metric,
    fit_init, delete
)


class BaseTrainer(BaseTransformer, BasePredictor):
    """Base class for the trainers.

    Parameters
    ----------
    models: string or sequence
        Models to train on the data. Use the predefined acronyms in
        MODEL_LIST or a custom model.

    metric: str, callable or sequence, optional (default=None)
        Metric on which to fit the models. Choose from any of sklearn's
        SCORERS, a function with signature metric(y, y_pred, **kwargs),
        a scorer object or a sequence of these. If multiple metrics are
        selected, only the first will be used to optimize the BO. If
        None, a default metric is selected:
            - "f1" for binary classification
            - "f1_weighted" for multiclass classification
            - "r2" for regression

    greater_is_better: bool or sequence, optional (default=True)
        Whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower is
        better. Will be ignored if the metric is a string or a scorer.
        If sequence, the n-th value will apply to the n-th metric.

    needs_proba: bool or sequence, optional (default=False)
        Whether the metric function requires probability estimates out
        of a classifier. If True, make sure that every selected model
        has a `predict_proba` method. Will be ignored if the metric is
        a string or a scorer. If sequence, the n-th value will apply to
        the n-th metric.

    needs_threshold: bool or sequence, optional (default=False)
        Whether the metric function takes a continuous decision
        certainty. This only works for estimators that have either a
        `decision_function` or `predict_proba` method. Will be ignored
        if the metric is a string or a scorer. If sequence, the n-th
        value will apply to the n-th metric.

    n_calls: int or sequence, optional (default=15)
        Maximum number of iterations of the BO. It includes the random
        points of `n_initial_points`. If 0, skip the BO and fit the
        model on its default Parameters. If sequence, the n-th value
        will apply to the n-th model.

    n_initial_points: int or sequence, optional (default=5)
        Initial number of random tests of the BO before fitting the
        surrogate function. If equal to `n_calls`, the optimizer will
        technically be performing a random search. If sequence, the
        n-th value will apply to the n-th model.

    est_params: dict, optional (default={})
        Additional parameters for the estimators. See the corresponding
        documentation for the available options. For multiple models,
        use the acronyms as key and a dict of the parameters as value.
        Add _fit to the parameter's name to pass it to the fit method.

    bo_params: dict, optional (default={})
        Additional parameters to for the BO. See bayesian_optimization
        in basemodel.py for the available options.

    bagging: int, sequence or None, optional (default=None)
        Number of data sets (bootstrapped from the training set) to use
        in the bagging algorithm. If None or 0, no bagging is performed.
        If sequence, the n-th value will apply to the n-th model.

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
            - 2 to print detailed information.

    warnings: bool or str, optional (default=True)
        - If True: Default warning action (equal to "default").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of the actions in python's warnings environment.

        Note that changing this parameter will affect the
        `PYTHONWARNINGS` environment.

        Note that ATOM can't manage warnings that go directly
        from C/C++ code to the stdout/stderr.

    logger: str, class or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the logging file. Use "auto" for default name.
        - If class: Python `Logger` object.

        The default name created consists of the class' name
        followed by the timestamp of the logger's creation.

        Note that warnings will not be saved to the logger.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `numpy.random`.

    """

    def __init__(
        self, models, metric, greater_is_better, needs_proba, needs_threshold,
        n_calls, n_initial_points, est_params, bo_params, bagging, n_jobs,
        verbose, warnings, logger, random_state
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            random_state=random_state,
        )

        # Parameter attributes
        self.models = models
        self.metric_ = lst(metric)
        self.greater_is_better = greater_is_better
        self.needs_proba = needs_proba
        self.needs_threshold = needs_threshold
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.est_params = est_params
        self.bo_params = bo_params
        self.bagging = bagging

        # Branching attributes
        self._current = "main"  # Current (and only) branch
        self._branches = {self._current: Branch(self, self._current)}
        self.task = None

        # Model attributes
        self.errors = {}
        self._results = pd.DataFrame(
            columns=[
                "metric_bo",
                "time_bo",
                "metric_train",
                "metric_test",
                "time_fit",
                "mean_bagging",
                "std_bagging",
                "time_bagging",
                "time",
            ]
        )

    def _check_parameters(self):
        """Check the validity of the input parameters."""
        models = []
        for m in lst(self.models):
            if isinstance(m, str):
                # Get the models' right acronym
                acronym = None
                for model in MODEL_LIST:
                    # Use startswith because models can be tagged
                    if m.lower().startswith(model.lower()):
                        acronym = model

                if not acronym:
                    raise ValueError(
                        f"Unknown model: {m}! Choose from: {', '.join(MODEL_LIST)}."
                    )

                # Check if packages for non-sklearn models are available
                if acronym in OPTIONAL_PACKAGES:
                    try:
                        importlib.import_module(OPTIONAL_PACKAGES[acronym])
                    except ImportError:
                        raise ValueError(
                            f"Unable to import the {OPTIONAL_PACKAGES[acronym]} "
                            "package. Make sure it is installed."
                        )

                # Check for regression/classification-only models
                if self.goal.startswith("class") and acronym in ONLY_REG:
                    raise ValueError(
                        f"The {acronym} model can't perform classification tasks!"
                    )
                elif self.goal.startswith("reg") and acronym in ONLY_CLASS:
                    raise ValueError(
                        f"The {acronym} model can't perform regression tasks!"
                    )

                subclass = MODEL_LIST[acronym](self, acronym + m[len(acronym):])

            else:  # Model is custom estimator
                subclass = CustomModel(self, estimator=m)

            # Add the model to the pipeline and check for duplicates
            models.append(subclass.name)
            if len(models) != len(set(models)):
                raise ValueError(
                    "Invalid value for the models parameter. Duplicate model: "
                    f"{subclass.name}. Add a tag to the model's acronym to use "
                    "the same estimator twice."
                )

            # Attach the subclasses to the trainers
            setattr(self, subclass.name, subclass)
            setattr(self, subclass.name.lower(), subclass)  # Lowercase as well

        self.models = models

        # Check validity BO parameters ============================= >>

        if isinstance(self.n_calls, SEQUENCE):
            if len(self.n_calls) != len(self.models):
                raise ValueError(
                    "Invalid value for the n_calls parameter. Length should "
                    "be equal to the number of models, got len(models)="
                    f"{len(self.models)} and len(n_calls)={len(self.n_calls)}."
                )
        else:
            self.n_calls = [self.n_calls for _ in self.models]
        if isinstance(self.n_initial_points, SEQUENCE):
            if len(self.n_initial_points) != len(self.models):
                raise ValueError(
                    "Invalid value for the n_initial_points parameter. Length "
                    "should be equal to the number of models, got len(models)="
                    f"{len(self.models)} and len(n_initial_points)="
                    f"{len(self.n_initial_points)}."
                )
        else:
            self.n_initial_points = [self.n_initial_points for _ in self.models]
        if isinstance(self.bagging, SEQUENCE):
            if len(self.bagging) != len(self.models):
                raise ValueError(
                    "Invalid value for the bagging parameter. Length should "
                    "be equal to the number of models, got len(models)="
                    f"{len(self.models)} and len(bagging)={len(self.bagging)}."
                )
        else:
            self.bagging = [self.bagging for _ in self.models]

        # Check dimensions params ================================== >>

        if self.bo_params.get("dimensions"):
            dimensions = {}
            for model in self.models:
                # If not dict, the dimensions are for all models
                if not isinstance(self.bo_params["dimensions"], dict):
                    dimensions[model] = self.bo_params["dimensions"]
                else:
                    # Dimensions for every specific model
                    for key, value in self.bo_params["dimensions"].items():
                        # Parameters for this model only
                        if key.lower() == model.lower():
                            dimensions[model] = value

            self.bo_params["dimensions"] = dimensions

        # Prepare est_params ======================================= >>

        if self.est_params:
            est_params = {}
            for model in self.models:
                est_params[model] = {}  # Create empty dict for every model

                for key, value in self.est_params.items():
                    # Parameters for this model only
                    if key.lower() == model.lower():
                        est_params[model].update(value)
                    # Parameters for all models
                    elif key.lower() not in map(str.lower, self.models):
                        est_params[model].update({key: value})

            self.est_params = est_params

        # Check validity metric ==================================== >>

        if not self.metric_[0]:
            self.metric_ = [get_default_metric(self.task)]

        # Ignore if it's the same metric as previous call
        elif any([not hasattr(m, "name") for m in self.metric_]):
            self.metric_ = self._prepare_metric(
                metric=self.metric_,
                gib=self.greater_is_better,
                needs_proba=self.needs_proba,
                needs_threshold=self.needs_threshold,
            )

        # Assign mapping =========================================== >>

        # Is already filled if called from atom
        if not self.mapping:
            self.mapping = {str(v): v for v in sorted(self.y.unique())}

    @staticmethod
    def _prepare_metric(metric, gib, needs_proba, needs_threshold):
        """Return a list of scorers given the parameters."""
        if isinstance(gib, SEQUENCE):
            if len(gib) != len(metric):
                raise ValueError(
                    "Invalid value for the greater_is_better parameter. Length "
                    "should be equal to the number of metrics, got len(metric)="
                    f"{len(metric)} and len(greater_is_better)={len(gib)}."
                )
        else:
            gib = [gib for _ in metric]

        if isinstance(needs_proba, SEQUENCE):
            if len(needs_proba) != len(metric):
                raise ValueError(
                    "Invalid value for the needs_proba parameter. Length should "
                    "be equal to the number of metrics, got len(metric)="
                    f"{len(metric)} and len(needs_proba)={len(needs_proba)}."
                )
        else:
            needs_proba = [needs_proba for _ in metric]

        if isinstance(needs_threshold, SEQUENCE):
            if len(needs_threshold) != len(metric):
                raise ValueError(
                    "Invalid value for the needs_threshold parameter. Length should "
                    "be equal to the number of metrics, got len(metric)="
                    f"{len(metric)} and len(needs_threshold)={len(needs_threshold)}."
                )
        else:
            needs_threshold = [needs_threshold for _ in metric]

        metric_list = []
        for i, j, m, n in zip(metric, gib, needs_proba, needs_threshold):
            metric_list.append(get_metric(i, j, m, n))

        return metric_list

    def _core_iteration(self):
        """Core iteration of the trainer.

        Returns
        -------
        results: pd.DataFrame
            Dataframe of the results for this iteration.

        """
        t_init = time()  # To measure the time the whole pipeline takes

        # Loop over every independent model
        to_remove = []
        for idx, (m, n_calls, n_initial_points, bagging) in enumerate(
            zip(self.models, self.n_calls, self.n_initial_points, self.bagging)
        ):
            # Check n_calls parameter
            if n_calls < 0:
                raise ValueError(
                    "Invalid value for the n_calls parameter. "
                    f"Value should be >=0, got {n_calls}."
                )

            model_time = time()
            mdl = getattr(self, m)

            # Add est_params to the model
            mdl._est_params = fit_init(self.est_params.get(mdl.name, {}), False)
            mdl._est_params_fit = fit_init(self.est_params.get(mdl.name, {}), True)

            try:  # If errors occurs, skip the model
                # If it has custom dimensions, run the BO
                bo = False
                if self.bo_params.get("dimensions"):
                    if self.bo_params["dimensions"].get(mdl.name):
                        bo = True

                # Use copy of kwargs to not delete original in method
                # Shallow copy is enough since we only delete entries in basemodel
                if (bo or hasattr(mdl, "get_dimensions")) and n_calls > 0:
                    mdl.bayesian_optimization(
                        n_calls=n_calls,
                        n_initial_points=n_initial_points,
                        bo_params=self.bo_params.copy(),
                    )

                mdl.fit()

                if bagging:
                    mdl.bagging(bagging)

                # Get the total time spend on this model
                total_time = time_to_string(model_time)
                setattr(mdl, "time", total_time)
                self.log("-" * 49, 1)
                self.log(f"Total time: {total_time}", 1)

            except Exception as ex:
                if idx != 0 or ((bo or hasattr(mdl, "get_dimensions")) and n_calls > 0):
                    self.log("", 1)  # Add extra line
                self.log(
                    f"Exception encountered while running the {m} model. Removing "
                    f"model from pipeline. \n{type(ex).__name__}: {ex}", 1
                )

                # Append exception to errors dictionary
                self.errors[m] = ex

                # Add model to "garbage collector"
                # Cannot remove immediately to maintain the iteration order
                to_remove.append(m)

        # Close the BO plot if there was one
        if self.bo_params.get("plot_bo"):
            plt.close()

        # Remove faulty models
        delete(self, to_remove)

        # Raise an exception if all models failed
        if not self.models:
            raise RuntimeError("It appears all models failed to run...")

        # Print final results ====================================== >>

        # Create dataframe with final results
        results = pd.DataFrame(columns=self._results.columns)

        # Print final results
        self.log("\n\nFinal results ========================= >>", 1)
        self.log(f"Duration: {time_to_string(t_init)}", 1)
        self.log("-" * 42, 1)

        # Get max length of the model names
        maxlen = max([len(m.fullname) for m in self.models_])

        # Get best score of all the models
        best_score = max([get_best_score(m) for m in self.models_])

        for m in self.models_:
            # Append model row to results
            values = [
                m.metric_bo,
                m.time_bo,
                m.metric_train,
                m.metric_test,
                m.time_fit,
                m.mean_bagging,
                m.std_bagging,
                m.time_bagging,
                m.time,
            ]
            m._results.loc[m.name] = values
            results.loc[m.name] = values

            # Get the model's final output and highlight best score
            out = f"{m.fullname:{maxlen}s} --> {m._final_output()}"
            if get_best_score(m) == best_score and len(self.models) > 1:
                out += " !"

            self.log(out, 1)  # Print the pipeline's final score

        return results

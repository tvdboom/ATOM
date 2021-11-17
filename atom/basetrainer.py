# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the parent class for the trainers.

"""

# Standard packages
import mlflow
import importlib
import traceback
from datetime import datetime
import matplotlib.pyplot as plt
from skopt.callbacks import DeadlineStopper, DeltaXStopper, DeltaYStopper

# Own modules
from .branch import Branch
from .models import MODELS, CustomModel
from .basepredictor import BasePredictor
from .data_cleaning import BaseTransformer
from .basemodel import BaseModel
from .utils import (
    SEQUENCE, OPTIONAL_PACKAGES, lst, time_to_str, is_multidim,
    get_acronym, get_scorer, get_best_score, check_scaling, delete,
    PlotCallback, CustomDict,
)


class BaseTrainer(BaseTransformer, BasePredictor):
    """Base class for the trainers.

    Parameters
    ----------
    models: str, estimator or sequence, optional (default=None)
        Models to fit to the data. Allowed inputs are: an acronym from
        any of ATOM's predefined models, an ATOMModel or a custom
        estimator as class or instance. If None, all the predefined
        models are used.

    metric: str, func, scorer, sequence or None, optional (default=None)
        Metric on which to fit the models. Choose from any of sklearn's
        SCORERS, a function with signature `metric(y_true, y_pred)`, a
        scorer object or a sequence of these. If multiple metrics are
        selected, only the first is used to optimize the BO. If None, a
        default scorer is selected:
            - "f1" for binary classification
            - "f1_weighted" for multiclass classification
            - "r2" for regression

    greater_is_better: bool or sequence, optional (default=True)
        Whether the metric is a score function or a loss function,
        i.e. if True, a higher score is better and if False, lower
        is better. This parameter is ignored if the metric is a
        string or a scorer. If sequence, the n-th value applies to
        the n-th metric.

    needs_proba: bool or sequence, optional (default=False)
        Whether the metric function requires probability estimates out
        of a classifier. If True, make sure that every selected model
        has a `predict_proba` method. This parameter is ignored if the
        metric is a string or a scorer. If sequence, the n-th value
        applies to the n-th metric.

    needs_threshold: bool or sequence, optional (default=False)
        Whether the metric function takes a continuous decision
        certainty. This only works for estimators that have either a
        `decision_function` or `predict_proba` method. This parameter
        is ignored if the metric is a string or a scorer. If sequence,
        the n-th value applies to the n-th metric.

    n_calls: int or sequence, optional (default=15)
        Maximum number of iterations of the BO. It includes the random
        points of `n_initial_points`. If 0, skip the BO and fit the
        model on its default Parameters. If sequence, the n-th value
        applies to the n-th model.

    n_initial_points: int or sequence, optional (default=5)
        Initial number of random tests of the BO before fitting the
        surrogate function. If equal to `n_calls`, the optimizer will
        technically be performing a random search. If sequence, the
        n-th value applies to the n-th model.

    est_params: dict or None, optional (default=None)
        Additional parameters for the estimators. See the corresponding
        documentation for the available options. For multiple models,
        use the acronyms as key and a dict of the parameters as value.
        Add _fit to the parameter's name to pass it to the fit method
        instead of the initializer.

    bo_params: dict or None, optional (default=None)
        Additional parameters to for the BO. These can include:
            - base_estimator: str, optional (default="GP")
                Surrogate model to use. Choose from:
                    - "GP" for Gaussian Process
                    - "RF" for Random Forest
                    - "ET" for Extra-Trees
                    - "GBRT" for Gradient Boosted Regression Trees
            - max_time: int, optional (default=np.inf)
                Stop the optimization after `max_time` seconds.
            - delta_x: int or float, optional (default=0)
                Stop the optimization when `|x1 - x2| < delta_x`.
            - delta_y: int or float, optional (default=0)
                Stop the optimization if the 5 minima are within
                `delta_y` (skopt always minimizes the function).
            - early_stopping: int, float or None, optional (default=None)
                Training will stop if the model didn't improve in
                last `early_stopping` rounds. If <1, fraction of
                rounds from the total. If None, no early stopping
                is performed. Only available for models that allow
                in-training evaluation.
            - cv: int, optional (default=5)
                Number of folds for the cross-validation. If 1, the
                training set is randomly split in a (sub)train and
                validation set.
            - callback: callable or list of callables, optional (default=None)
                Callbacks for the BO.
            - dimensions: dict, sequence or None, optional (default=None)
                Custom hyperparameter space for the BO. Can be an
                array to share the same dimensions across models
                or a dictionary with the model names as key. If
                None, ATOM's predefined dimensions are used.
            - plot: bool, optional (default=False)
                Whether to plot the BO's progress as it runs.
                Creates a canvas with two plots: the first plot
                shows the score of every trial and the second shows
                the distance between the last consecutive steps.
            - Additional keyword arguments for skopt's optimizer.

    n_bootstrap: int or sequence, optional (default=0)
        Number of data sets (bootstrapped from the training set) to
        use in the bootstrap algorithm. If 0, no bootstrap is performed.
        If sequence, the n-th value will apply to the n-th model.

    n_jobs: int, optional (default=1)
        Number of cores to use for parallel processing.
            - If >0: Number of cores to use.
            - If -1: Use all available cores.
            - If <-1: Use number of cores - 1 + `n_jobs`.

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

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic name.
        - Else: Python `logging.Logger` instance.

        Note that warnings will not be saved to the logger.

    experiment: str or None, optional (default=None)
        Name of the mlflow experiment to use for tracking. If None,
        no mlflow tracking is performed.

    random_state: int or None, optional (default=None)
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `numpy.random`.

    """

    def __init__(
        self, models, metric, greater_is_better, needs_proba, needs_threshold,
        n_calls, n_initial_points, est_params, bo_params, n_bootstrap, n_jobs,
        verbose, warnings, logger, experiment, random_state,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )

        # Parameter attributes
        self._models = CustomDict({m: m for m in lst(models) if m is not None})
        self._metric = CustomDict({m: m for m in lst(metric) if m is not None})
        self.greater_is_better = greater_is_better
        self.needs_proba = needs_proba
        self.needs_threshold = needs_threshold
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.est_params = est_params
        self.bo_params = bo_params
        self.n_bootstrap = n_bootstrap

        # Branching attributes
        self.holdout = None
        self._current = "og"
        self._branches = CustomDict({self._current: Branch(self, self._current)})

        # Training attributes
        self.task = None
        self.scaled = None
        self._bo = {"base_estimator": "GP", "cv": 1, "callback": [], "kwargs": {}}
        self._errors = CustomDict()

    def _check_parameters(self):
        """Check the validity of the input parameters."""
        if self.mapping is None:
            self.mapping = {str(v): v for v in sorted(self.y.unique())}

        if self.scaled is None:
            self.scaled = check_scaling(self.X)

        # Create model subclasses ================================== >>

        # If left to default, select all predefined models per task
        if not self._models:
            if self.goal == "class":
                models = [m(self) for m in MODELS.values() if m.task != "reg"]
            else:
                models = [m(self) for m in MODELS.values() if m.task != "class"]
        else:
            models = []
            for m in self._models.values():
                if isinstance(m, str):
                    acronym = get_acronym(m, must_be_equal=False)

                    if is_multidim(self.X):
                        raise ValueError(
                            "Multidimensional datasets are not supported by ATOM's "
                            "predefined models. Refer to the documentation for the "
                            "use of custom models."
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

                    models.append(MODELS[acronym](self, acronym + m[len(acronym):]))

                    # Check for regression/classification-only models
                    if self.goal == "class" and models[-1].task == "reg":
                        raise ValueError(
                            f"The {acronym} model can't perform classification tasks!"
                        )
                    elif self.goal == "reg" and models[-1].task == "class":
                        raise ValueError(
                            f"The {acronym} model can't perform regression tasks!"
                        )

                elif not isinstance(m, BaseModel):  # Model is custom estimator
                    models.append(CustomModel(self, estimator=m))

                else:  # Model is already a model subclass (can happen with reruns)
                    models.append(m)

        self._models = CustomDict({m.name: m for m in models})

        # Define scorer ============================================ >>

        # Assign default scorer
        if not self._metric:
            if self.task.startswith("bin"):
                self._metric = CustomDict(f1=get_scorer("f1"))
            elif self.task.startswith("multi"):
                self._metric = CustomDict(f1_weighted=get_scorer("f1_weighted"))
            else:
                self._metric = CustomDict(r2=get_scorer("r2"))

        # Ignore if it's the same scorer as previous call
        elif not all([hasattr(m, "name") for m in self._metric.values()]):
            self._metric = self._prepare_metric(
                metric=self._metric,
                greater_is_better=self.greater_is_better,
                needs_proba=self.needs_proba,
                needs_threshold=self.needs_threshold,
            )

        # Check validity sequential parameters ===================== >>

        for param in ("n_calls", "n_initial_points", "n_bootstrap"):
            p = lst(getattr(self, param))
            if len(p) != 1 and len(p) != len(self._models):
                raise ValueError(
                    f"Invalid value for the {param} parameter. Length "
                    "should be equal to the number of models, got len"
                    f"(models)={len(self._models)} and len({param})={len(p)}."
                )

            for i, model in enumerate(self._models.values()):
                value = p[i % len(p)]
                if param == "n_calls" and value == 1 or value < 0:
                    raise ValueError(
                        f"Invalid value for the {param} parameter. "
                        f"Value should be >=2, got {value}."
                    )
                elif param == "n_initial_points" and value <= 0:
                    raise ValueError(
                        f"Invalid value for the {param} parameter. "
                        f"Value should be >=1, got {value}."
                    )
                elif param == "n_bootstrap" and value < 0:
                    raise ValueError(
                        f"Invalid value for the {param} parameter. "
                        f"Value should be >=0, got {value}."
                    )

                setattr(model, "_" + param, value)

        # Prepare est_params ======================================= >>

        if self.est_params:
            for name, model in self._models.items():
                params = {}
                for key, value in self.est_params.items():
                    # Parameters for this model only
                    if key.lower() == name.lower():
                        params.update(value)
                    # Parameters for all models
                    elif key not in self._models:
                        params.update({key: value})

                for key, value in params.items():
                    if key.endswith("_fit"):
                        model._est_params_fit[key[:-4]] = value
                    else:
                        model._est_params[key] = value

        # Prepare bo params ======================================== >>

        base_estimators = ["GP", "RF", "ET", "GBRT"]
        exc = ["max_time", "delta_x", "delta_y", "plot", "early_stopping", "dimensions"]

        if self.bo_params:
            if self.bo_params.get("base_estimator"):
                self._bo["base_estimator"] = self.bo_params["base_estimator"]
                if isinstance(self._bo["base_estimator"], str):
                    if self._bo["base_estimator"].upper() not in base_estimators:
                        raise ValueError(
                            "Invalid value for the base_estimator parameter, "
                            f"got {self.bo_params['base_estimator']}. Choose "
                            f"from: {', '.join(base_estimators)}."
                        )

            if self.bo_params.get("callback"):
                self._bo["callback"].extend(lst(self.bo_params["callback"]))

            if "max_time" in self.bo_params:
                if self.bo_params["max_time"] <= 0:
                    raise ValueError(
                        "Invalid value for the max_time parameter. Value "
                        f"should be >0, got {self.bo_params['max_time']}."
                    )
                max_time_callback = DeadlineStopper(self.bo_params["max_time"])
                self._bo["callback"].append(max_time_callback)

            if "delta_x" in self.bo_params:
                if self.bo_params["delta_x"] < 0:
                    raise ValueError(
                        "Invalid value for the delta_x parameter. "
                        f"Value should be >=0, got {self.bo_params['delta_x']}."
                    )
                delta_x_callback = DeltaXStopper(self.bo_params["delta_x"])
                self._bo["callback"].append(delta_x_callback)

            if "delta_y" in self.bo_params:
                if self.bo_params["delta_y"] < 0:
                    raise ValueError(
                        "Invalid value for the delta_y parameter. Value "
                        f"should be >=0, got {self.bo_params['delta_y']}."
                    )
                delta_y_callback = DeltaYStopper(self.bo_params["delta_y"], n_best=5)
                self._bo["callback"].append(delta_y_callback)

            if self.bo_params.get("plot"):
                self._bo["callback"].append(PlotCallback(self))

            if "cv" in self.bo_params:
                if self.bo_params["cv"] <= 0:
                    raise ValueError(
                        "Invalid value for the cv parameter. Value "
                        f"should be >=0, got {self.bo_params['cv']}."
                    )
                self._bo["cv"] = self.bo_params["cv"]

            if "early_stopping" in self.bo_params:
                if self.bo_params["early_stopping"] <= 0:
                    raise ValueError(
                        "Invalid value for the early_stopping parameter. Value "
                        f"should be >=0, got {self.bo_params['early_stopping']}."
                    )
                for model in self._models.values():
                    if hasattr(m, "custom_fit"):
                        model._early_stopping = self.bo_params["early_stopping"]

            # Add custom dimensions to every model subclass
            if self.bo_params.get("dimensions"):
                for name, model in self._models.items():
                    # If not dict, the dimensions are for all models
                    if not isinstance(self.bo_params["dimensions"], dict):
                        model._dimensions = self.bo_params["dimensions"]
                    else:
                        # Dimensions for every specific model
                        for key, value in self.bo_params["dimensions"].items():
                            # Parameters for this model only
                            if key.lower() == name.lower():
                                model._dimensions = value
                                break

            # The remaining bo_params are added as kwargs to the optimizer
            self._bo["kwargs"] = {
                k: v for k, v in self.bo_params.items() if k not in list(self._bo) + exc
            }

    @staticmethod
    def _prepare_metric(metric, **kwargs):
        """Return a list of scorers given the parameters."""
        metric_params = {}
        for key, value in kwargs.items():
            if isinstance(value, SEQUENCE):
                if len(value) != len(metric):
                    raise ValueError(
                        "Invalid value for the greater_is_better parameter. Length "
                        "should be equal to the number of metrics, got len(metric)="
                        f"{len(metric)} and len({key})={len(value)}."
                    )
            else:
                metric_params[key] = [value for _ in metric]

        metric_dict = CustomDict()
        for args in zip(metric, *metric_params.values()):
            metric = get_scorer(*args)
            metric_dict[metric.name] = metric

        return metric_dict

    def _core_iteration(self):
        """Fit and evaluate the models in the pipeline."""
        t_init = datetime.now()  # Measure the time the whole pipeline takes

        self.log("\nTraining " + "=" * 25 + " >>", 1)
        if not self.__class__.__name__.startswith("SuccessiveHalving"):
            self.log(f"Models: {', '.join(lst(self.models))}", 1)
        self.log(f"Metric: {', '.join(lst(self.metric))}\n", 1)

        to_remove = []
        for i, m in enumerate(self._models.values()):
            model_time = datetime.now()

            try:  # If an error occurs, skip the model
                if self.experiment:  # Start mlflow run
                    m._run = mlflow.start_run(run_name=m.name)

                # If it has predefined or custom dimensions, run the BO
                if (m._dimensions or hasattr(m, "get_dimensions")) and m._n_calls > 0:
                    m.bayesian_optimization()

                m.fit()

                if m._n_bootstrap:
                    m.bootstrap()

                # Get the total time spend on this model
                setattr(m, "time", time_to_str(model_time))
                self.log("-" * 49 + f"\nTotal time: {m.time}\n\n", 1)

            except Exception as ex:
                self.log(
                    "\nException encountered while running the "
                    f"{m.name} model. Removing model from pipeline. ", 1
                )
                self.log("".join(traceback.format_tb(ex.__traceback__))[:-1], 5)
                self.log(f"{type(ex).__name__}: {ex}", 1)

                # Append exception to errors dictionary
                self._errors[m.name] = ex

                # Add model to "garbage collector"
                # Cannot remove immediately to maintain the iteration order
                to_remove.append(m.name)

                if self.bo_params and self.bo_params.get("plot"):
                    PlotCallback.c += 1  # Next model
                    plt.close()  # Close the crashed plot
            finally:
                mlflow.end_run()  # Ends the current run (if any)

        delete(self, to_remove)  # Remove faulty models

        # If there's only one model and it failed, raise that exception
        # If multiple models and all failed, raise RuntimeError
        if not self._models:
            if len(self.errors) == 1:
                raise self.errors[0]
            else:
                raise RuntimeError(
                    "All models failed to run. Use the errors attribute "
                    "or the logging file to investigate the exceptions."
                )

        self.log(f"Final results {'=' * 20} >>", 1)
        self.log(f"Duration: {time_to_str(t_init)}\n{'-' * 37}", 1)

        # Get max length of the model names
        maxlen = max([len(m.fullname) for m in self._models.values()])

        # Get best score of all the models
        best_score = max([get_best_score(m) for m in self._models.values()])

        for m in self._models.values():
            out = f"{m.fullname:{maxlen}s} --> {m._final_output()}"
            if get_best_score(m) == best_score and len(self._models) > 1:
                out += " !"

            self.log(out, 1)

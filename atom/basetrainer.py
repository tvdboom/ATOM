# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseTrainer class.

"""

import traceback
from datetime import datetime
from importlib.util import find_spec
from typing import Optional, Union

import matplotlib.pyplot as plt
import mlflow
from skopt.callbacks import DeadlineStopper, DeltaXStopper, DeltaYStopper

from atom.baserunner import BaseRunner
from atom.branch import Branch
from atom.data_cleaning import BaseTransformer
from atom.models import MODELS, CustomModel
from atom.utils import (
    SEQUENCE, SEQUENCE_TYPES, CustomDict, PlotCallback, check_scaling,
    get_best_score, get_custom_scorer, is_sparse, lst, time_to_str,
)


class BaseTrainer(BaseTransformer, BaseRunner):
    """Base class for trainers.

    Implements methods to check the validity of the parameters,
    create models and metrics, run hyperparameter tuning, model
    training, bootstrap, and display the final output.

    See training.py for a description of the parameters.

    """

    def __init__(
        self, models, metric, greater_is_better, needs_proba, needs_threshold,
        n_calls, n_initial_points, est_params, bo_params, n_bootstrap, n_jobs,
        device, engine, verbose, warnings, logger, experiment, random_state,
    ):
        super().__init__(
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )

        # Parameter attributes
        self._models = models
        self._metric = metric
        self.greater_is_better = greater_is_better
        self.needs_proba = needs_proba
        self.needs_threshold = needs_threshold
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.est_params = est_params
        self.bo_params = bo_params
        self.n_bootstrap = n_bootstrap

        # Branching attributes
        self.index = True
        self.holdout = None
        self._current = "master"
        self._branches = CustomDict({self._current: Branch(self, self._current)})

        # Training attributes
        self.task = None
        self.scaled = None
        self._bo = {"base_estimator": "GP", "cv": 1, "callback": [], "kwargs": {}}
        self._errors = CustomDict()

    @staticmethod
    def _prepare_metric(
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]],
        **kwargs,
    ) -> CustomDict:
        """Check the validity of the metric.

        Makes sure the lengths of the metric parameters are equal
        for multi-metric, and converts all input types to a scorer.

        Parameters
        ----------
        metric: str, func, scorer or sequence
            Metric argument provided by the user.

        **kwargs
            Additional metric descriptors:
                - greater_is_better
                - needs_proba
                - needs_threshold

        Returns
        -------
        CustomDict
            Metric names and corresponding scorers.

        """
        metric_params = {}
        for key, value in kwargs.items():
            if isinstance(value, SEQUENCE):
                if len(value) != len(metric):
                    raise ValueError(
                        f"Invalid value for the {key} parameter. Its length "
                        "should be equal to the number of metrics, got "
                        f"len(metric)={len(metric)} and len({key})={len(value)}."
                    )
            else:
                metric_params[key] = [value for _ in metric]

        metric_dict = CustomDict()
        for args in zip(metric, *metric_params.values()):
            scorer = get_custom_scorer(*args)
            metric_dict[scorer.name] = scorer

        return metric_dict

    def _prepare_parameters(self):
        """Check the validity of the input parameters.

        Creates the models, assigns a metric, prepares est_params
        and the hyperparameter tuning parameters.

        """
        if self.scaled is None and not is_sparse(self.X):
            self.scaled = check_scaling(self.X)

        # Create model subclasses ================================== >>

        # If left to default, select all predefined models per task
        if self._models is None:
            if self.goal == "class":
                models = [m(self) for m in MODELS.values() if "class" in m._estimators]
            else:
                models = [m(self) for m in MODELS.values() if "reg" in m._estimators]
        else:
            models = []
            for m in lst(self._models):
                if isinstance(m, str):
                    # Get the acronym from the called model
                    names = [n for n in MODELS if m.lower().startswith(n.lower())]
                    if not names:
                        raise ValueError(
                            f"Unknown model: {m}. Choose from: {', '.join(MODELS)}."
                        )
                    else:
                        acronym = names[0]

                    # Check if packages for non-sklearn models are available
                    packages = {"XGB": "xgboost", "LGB": "lightgbm", "CatB": "catboost"}
                    if acronym in packages and not find_spec(packages[acronym]):
                        raise ModuleNotFoundError(
                            f"Unable to import the {packages[acronym]} package. "
                            f"Install it using: pip install {packages[acronym]}"
                        )

                    models.append(MODELS[acronym](self, acronym + m[len(acronym):]))

                    # Check for regression/classification-only models
                    if self.goal == "class" and self.goal not in models[-1]._estimators:
                        raise ValueError(
                            f"The {acronym} model can't perform classification tasks!"
                        )
                    elif self.goal == "reg" and self.goal not in models[-1]._estimators:
                        raise ValueError(
                            f"The {acronym} model can't perform regression tasks!"
                        )

                else:  # Model is a custom estimator
                    models.append(CustomModel(self, estimator=m))

        names = [m.name for m in models]
        if len(set(names)) != len(names):
            raise ValueError(
                "Invalid value for the models parameter. It seems there are "
                "duplicate models. Add a tag to a model's acronym to train two "
                "different models with the same estimator, e.g. models=['LR1', 'LR2']."
            )

        self._models = CustomDict({name: model for name, model in zip(names, models)})

        # Define scorer ============================================ >>

        # Assign default scorer
        if self._metric is None:
            if self.task.startswith("bin"):
                self._metric = CustomDict(f1=get_custom_scorer("f1"))
            elif self.task.startswith("multi"):
                self._metric = CustomDict(f1_weighted=get_custom_scorer("f1_weighted"))
            else:
                self._metric = CustomDict(r2=get_custom_scorer("r2"))

        # Ignore if it's the same scorer as previous call
        elif not isinstance(self._metric, CustomDict):
            self._metric = self._prepare_metric(
                metric=lst(self._metric),
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
                if param == "n_calls" and (value == 1 or value < 0):
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
                    if key.lower() == name.lower() or key.lower() == "all":
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
                    if hasattr(model, "custom_fit"):
                        model._early_stopping = self.bo_params["early_stopping"]

            # Add hyperparameter dimensions to every model subclass
            if self.bo_params.get("dimensions"):
                for name, model in self._models.items():
                    # If not dict, the dimensions are for all models
                    if not isinstance(self.bo_params["dimensions"], dict):
                        model._dimensions = lst(self.bo_params["dimensions"])
                    else:
                        # Dimensions for every specific model
                        for key, value in self.bo_params["dimensions"].items():
                            if key.lower() == name.lower() or key.lower() == "all":
                                model._dimensions.extend(list(lst(value)))

            # The remaining bo_params are added as kwargs to the optimizer
            self._bo["kwargs"] = {
                k: v for k, v in self.bo_params.items()
                if k not in list(self._bo) + exc
            }

    def _core_iteration(self):
        """Fit and evaluate the models.

        For every model, runs hyperparameter tuning, fitting and
        bootstrap wrapped in a try-except block. Also displays final
        results.

        """
        t_init = datetime.now()  # Measure the time the whole pipeline takes

        self.log("\nTraining " + "=" * 25 + " >>", 1)
        if not self.__class__.__name__.startswith("SuccessiveHalving"):
            self.log(f"Models: {', '.join(lst(self.models))}", 1)
        self.log(f"Metric: {', '.join(lst(self.metric))}", 1)

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
                self.log("-" * 49 + f"\nTotal time: {m.time}", 1)

            except Exception as ex:
                self.log(
                    "\nException encountered while running the "
                    f"{m.name} model. Removing model from pipeline. ", 1
                )
                self.log("".join(traceback.format_tb(ex.__traceback__))[:-1], 3)
                self.log(f"{ex.__class__.__name__}: {ex}", 1)

                # Append exception to errors dictionary
                self._errors[m.name] = ex

                # Add model to "garbage collector"
                # Cannot remove immediately to maintain the iteration order
                to_remove.append(m.name)

                if self.bo_params and self.bo_params.get("plot"):
                    PlotCallback.c += 1  # Next model
                    plt.close()  # Close the crashed plot
            finally:
                if self.experiment:
                    mlflow.end_run()

        self._delete_models(to_remove)  # Remove faulty models

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

        self.log(f"\n\nFinal results {'=' * 20} >>", 1)
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

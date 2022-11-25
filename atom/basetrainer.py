# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseTrainer class.

"""

import traceback
from datetime import datetime as dt
from importlib.util import find_spec
from typing import Any

import matplotlib.pyplot as plt
import mlflow
from optuna import Study, create_study

from atom.baserunner import BaseRunner
from atom.branch import Branch
from atom.data_cleaning import BaseTransformer
from atom.models import MODELS, CustomModel
from atom.plots import HTPlot, PredictionPlot, ShapPlot
from atom.utils import (
    SEQUENCE, CustomDict, check_scaling, get_best_score, get_custom_scorer,
    is_sparse, lst, sign, time_to_str,
)


class BaseTrainer(BaseTransformer, BaseRunner, HTPlot, PredictionPlot, ShapPlot):
    """Base class for trainers.

    Implements methods to check the validity of the parameters,
    create models and metrics, run hyperparameter tuning, model
    training, bootstrap, and display the final output.

    See training.py for a description of the parameters.

    """

    def __init__(
        self, models, metric, est_params, n_trials, ht_params, n_bootstrap, n_jobs,
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

        super(HTPlot, self).__init__()

        # Parameter attributes
        self._models = models
        self._metric = metric
        self.est_params = est_params
        self.n_trials = n_trials
        self.ht_params = ht_params
        self.n_bootstrap = n_bootstrap

        # Branching attributes
        self.index = True
        self.holdout = None
        self._current = "master"
        self._branches = CustomDict({self._current: Branch(self, self._current)})

        # Training attributes
        self.task = None
        self.scaled = None
        self._n_trials = {}
        self._n_bootstrap = {}
        self._ht_params = {"distributions": {}, "cv": 1, "plot": False, "tags": {}}
        self._errors = CustomDict()

    def _check_param(self, param: str, value: Any) -> dict:
        """Check the validity of one parameter.

        Parameters accept three formats:

        - dict: Each key is the name of a model, and the value applies
          only to that model.
        - sequence: The N-th element applies to the N-th model. Has to
          have the same length as the models.
        - value: Same value applies to all models.

        Parameters
        ----------
        param: str
            Name of the parameter to check.

        value: Any
            Value of the parameter.

        Returns
        -------
        dict
            Parameter with model names as key.

        """
        if isinstance(value, SEQUENCE):
            if len(value) != len(self._models):
                raise ValueError(
                    f"Invalid value for the {param} parameter. The length "
                    "should be equal to the number of models, got len"
                    f"(models)={len(self._models)} and len({param})={len(value)}."
                )
            return {k: v for k, v in zip(self._models, value)}
        elif not isinstance(value, dict):
            return {k: value for k in self._models}

        return value

    def _prepare_parameters(self):
        """Check the validity of the input parameters.

        Creates the models, assigns a metric, prepares the estimator's
        parameters and the parameters for hyperparameter tuning.

        """
        if self.scaled is None and not is_sparse(self.X):
            self.scaled = check_scaling(self.X)

        # Create model subclasses ================================== >>

        # If left to default, select all predefined models per task
        if self._models is None:
            self._models = CustomDict(
                {k: v(self) for k, v in MODELS.items() if self.goal in v._estimators}
            )
        else:
            inc, exc = [], []
            for model in lst(self._models):
                if isinstance(model, str):
                    for m in model.split("+"):
                        if m.startswith("!"):
                            exc.append(m[1:])
                        else:
                            names = [n for n in MODELS if m.lower().startswith(n.lower())]
                            if not names:
                                raise ValueError(
                                    f"Invalid value for the models parameter, got {m}. "
                                    f"Choose from: {', '.join(MODELS)}."
                                )
                            else:
                                acronym = names[0]

                            # Check if libraries for non-sklearn models are available
                            libraries = {
                                "XGB": "xgboost", "LGB": "lightgbm", "CatB": "catboost"
                            }
                            if acronym in libraries and not find_spec(libraries[acronym]):
                                raise ModuleNotFoundError(
                                    f"Unable to import the {libraries[acronym]} package. "
                                    f"Install it using: pip install {libraries[acronym]}"
                                )

                            inc.append(MODELS[acronym](self, acronym + m[len(acronym):]))

                            # Check for regression/classification-only models
                            if self.goal not in inc[-1]._estimators:
                                raise ValueError(
                                    f"The {acronym} model is not "
                                    f"available for {self.task} tasks!"
                                )

                else:  # Model is a custom estimator
                    inc.append(CustomModel(self, estimator=model))

            if inc and exc:
                raise ValueError(
                    "Invalid value for the models parameter. You can either "
                    "include or exclude models, not combinations of these."
                )
            elif inc:
                names = [m.name for m in inc]
                if len(set(names)) != len(names):
                    raise ValueError(
                        "Invalid value for the models parameter. There are duplicate "
                        "models. Add a tag to a model's acronym to train two different "
                        "models with the same estimator, e.g. models=['LR1', 'LR2']."
                    )
                self._models = CustomDict({n: m for n, m in zip(names, inc)})
            elif exc:
                self._models = CustomDict(
                    {
                        k: v(self) for k, v in MODELS.items()
                        if self.goal in v._estimators and v.acronym not in exc
                    }
                )

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
            metrics = []
            for m in lst(self._metric):
                if isinstance(m, str):
                    metrics.extend(m.split("+"))
                else:
                    metrics.append(m)

            self._metric = CustomDict(
                {(s := get_custom_scorer(m)).name: s for m in metrics}
            )

        # Prepare est_params ======================================= >>

        if self.est_params is not None:
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

        # Prepare ht parameters ==================================== >>

        self._n_trials = self._check_param("n_trials", self.n_trials)
        self._n_bootstrap = self._check_param("n_bootstrap", self.n_bootstrap)
        self._ht_params.update(self.ht_params or {})
        for key, value in self._ht_params.items():
            if key in ("cv", "plot"):
                self._ht_params[key] = self._check_param(key, value)
            elif key == "tags":
                self._ht_params[key] = {name: {} for name in self._models}
                for name in self._models:
                    for k, v in self._check_param(key, value).items():
                        if k.lower() == name.lower() or k.lower() == "all":
                            self._ht_params[key][name].update(v)
                        elif k not in self._models:
                            self._ht_params[key][name][k] = v
            elif key == "distributions":
                self._ht_params[key] = {name: {} for name in self._models}
                for name in self._models:
                    if not isinstance(value, dict):
                        # If sequence, it applies to all models
                        self._ht_params[key][name] = {k: None for k in lst(value)}
                    else:
                        # Either one distribution for all or per model
                        for k, v in value.items():
                            if k.lower() == name.lower() or k.lower() == "all":
                                if isinstance(v, dict):
                                    self._ht_params[key][name].update(v)
                                else:
                                    self._ht_params[key][name].update(
                                        {param: None for param in lst(v)}
                                    )
                            elif k not in self._models:
                                self._ht_params[key][name][k] = v
            elif key in {**sign(create_study), **sign(Study.optimize)}:
                self._ht_params[key] = {k: value for k in self._models}
            else:
                raise ValueError(
                    f"Invalid value for the ht_params parameter. Key {key} is invalid."
                )

    def _core_iteration(self):
        """Fit and evaluate the models.

        For every model, runs hyperparameter tuning, fitting and
        bootstrap wrapped in a try-except block. Also displays final
        results.

        """
        t = dt.now()  # Measure the time the whole pipeline takes

        to_remove = []
        for i, m in enumerate(self._models.values()):
            try:  # If an error occurs, skip the model
                if self.experiment:  # Start mlflow run
                    m._run = mlflow.start_run(run_name=m.name)

                self.log("\n", 1)  # Separate output from header

                # If it has predefined or custom dimensions, run the ht
                m._ht = {k: v[m._group] for k, v in self._ht_params.items()}
                if self._n_trials[m._group] > 0:
                    if m._ht["distributions"] or hasattr(m, "_get_distributions"):
                        m.hyperparameter_tuning(self._n_trials[m._group])

                m.fit()

                if self._n_bootstrap[m._group]:
                    m.bootstrapping(self._n_bootstrap[m._group])

                self.log("-" * 49 + f"\nTotal time: {time_to_str(m.time)}", 1)

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

                if self._ht_params["plot"][m._group]:
                    plt.close()  # Close the crashed plot

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
        self.log(f"Total time: {time_to_str((dt.now() - t).total_seconds())}", 1)
        self.log("-" * 37, 1)

        maxlen = 0
        names, scores = [], []
        for model in self._models.values():
            # Add the model name for repeated model classes
            select = filter(lambda x: x.acronym == model.acronym, self._models.values())
            if len(list(select)) > 1:
                names.append(f"{model._fullname} ({model.name})")
            else:
                names.append(model._fullname)
            scores.append(get_best_score(model))
            maxlen = max(maxlen, len(names[-1]))

        for i, m in enumerate(self._models.values()):
            out = f"{names[i]:{maxlen}s} --> {m._final_output()}"
            if scores[i] == max(scores) and len(self._models) > 1:
                out += " !"

            self.log(out, 1)

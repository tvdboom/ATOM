# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BaseTrainer class.

"""

from __future__ import annotations

import re
import traceback
from datetime import datetime as dt
from typing import Any

import mlflow
import ray
from optuna import Study, create_study
from joblib import Parallel, delayed
from atom.baserunner import BaseRunner
from atom.basemodel import BaseModel
from atom.branch import Branch
from atom.data_cleaning import BaseTransformer
from atom.models import MODELS, CatBoost, CustomModel, LightGBM, XGBoost
from atom.plots import HTPlot, PredictionPlot, ShapPlot
from atom.utils import (
    SEQUENCE_TYPES, ClassMap, CustomDict, check_dependency, get_best_score,
    get_custom_scorer, lst, sign, time_to_str,
)


class BaseTrainer(BaseTransformer, BaseRunner, HTPlot, PredictionPlot, ShapPlot):
    """Base class for trainers.

    Implements methods to check the validity of the parameters,
    create models and metrics, run hyperparameter tuning, model
    training, bootstrap, and display the final output.

    See training.py for a description of the parameters.

    """

    def __init__(
            self, models, metric, est_params, n_trials, ht_params, n_bootstrap,
            parallel, n_jobs, device, engine, backend, verbose, warnings, logger,
            experiment, random_state,
    ):
        super().__init__(
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            backend=backend,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )

        self.est_params = est_params
        self.n_trials = n_trials
        self.ht_params = ht_params
        self.n_bootstrap = n_bootstrap
        self.parallel = parallel

        self._models = lst(models) if models is not None else []
        self._metric = lst(metric) if metric is not None else []
        self._errors = CustomDict()

        self._og = None
        self._current = Branch(name="master")
        self._branches = ClassMap(self._current)

        self.index = True
        self.task = None

        self._multioutput = "auto"
        self._n_trials = {}
        self._n_bootstrap = {}
        self._ht_params = {"distributions": {}, "cv": 1, "plot": False, "tags": {}}

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
        if isinstance(value, SEQUENCE_TYPES):
            if len(value) != len(self._models):
                raise ValueError(
                    f"Invalid value for the {param} parameter. The length "
                    "should be equal to the number of models, got len"
                    f"(models)={len(self._models)} and len({param})={len(value)}."
                )
            return {k: v for k, v in zip(lst(self.models), value)}
        elif not isinstance(value, dict):
            return {k: value for k in lst(self.models)}

        return value

    def _prepare_parameters(self):
        """Check the validity of the input parameters.

        Creates the models, assigns a metric, prepares the estimator's
        parameters and the parameters for hyperparameter tuning.

        """
        # Define metric ============================================ >>

        # Assign default scorer
        if not self._metric:
            if self.task.startswith("bin"):
                # Binary classification
                self._metric = ClassMap(get_custom_scorer("f1"))
            elif self.task.startswith("multi") and self.goal.startswith("class"):
                # Multiclass, multilabel, multiclass-multioutput classification
                self._metric = ClassMap(get_custom_scorer("f1_weighted"))
            else:
                # Regression or multioutput regression
                self._metric = ClassMap(get_custom_scorer("r2"))
        elif not isinstance(self._metric, ClassMap):
            metrics = []
            for m in lst(self._metric):
                if isinstance(m, str):
                    metrics.extend(m.split("+"))
                else:
                    metrics.append(m)

            self._metric = ClassMap(get_custom_scorer(m) for m in metrics)

        # Define models ============================================ >>

        kwargs = dict(
            index=self.index,
            goal=self.goal,
            metric=self._metric,
            multioutput=self.multioutput,
            og=self.og,
            branch=self.branch,
            **{attr: getattr(self, attr) for attr in BaseTransformer.attrs},
        )

        inc, exc = [], []
        for model in self._models:
            if isinstance(model, str):
                for m in model.split("+"):
                    if m.startswith("!"):
                        exc.append(m[1:])
                    else:
                        cls = [n for n in MODELS if re.match(n.acronym, m, re.I)]
                        if not cls:
                            raise ValueError(
                                f"Invalid value for the models parameter, got {m}. "
                                f"Choose from: {', '.join(MODELS.keys())}."
                            )
                        else:
                            cls = cls[0]

                        # Check if libraries for non-sklearn models are available
                        if cls in (CatBoost, LightGBM, XGBoost):
                            check_dependency(cls.supports_engines[0])

                        inc.append(cls(name=cls.acronym + m[len(cls.acronym):], **kwargs))

                        # Check for regression/classification-only models
                        if self.goal not in inc[-1]._estimators:
                            raise ValueError(
                                f"The {cls._fullname} model is not "
                                f"available for {self.task} tasks!"
                            )
            elif isinstance(model, BaseModel):  # For reruns
                inc.append(model)
            else:  # Model is a custom estimator
                inc.append(CustomModel(estimator=model, **kwargs))

        if inc and exc:
            raise ValueError(
                "Invalid value for the models parameter. You can either "
                "include or exclude models, not combinations of these."
            )
        elif inc:
            if len(set(names := [m.name for m in inc])) != len(names):
                raise ValueError(
                    "Invalid value for the models parameter. There are duplicate "
                    "models. Add a tag to a model's acronym to train two different "
                    "models with the same estimator, e.g. models=['LR1', 'LR2']."
                )
            self._models = ClassMap(*inc)
        else:
            self._models = ClassMap(
                model(**kwargs) for model in MODELS
                if self.goal in model._estimators and model.acronym not in exc
            )

        # Prepare est_params ======================================= >>

        if self.est_params is not None:
            for model in self._models:
                params = {}
                for key, value in self.est_params.items():
                    # Parameters for this model only
                    if key.lower() == model.name.lower() or key.lower() == "all":
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
                self._ht_params[key] = {name: {} for name in lst(self.models)}
                for name in self._models.keys():
                    for k, v in self._check_param(key, value).items():
                        if k.lower() == name.lower() or k.lower() == "all":
                            self._ht_params[key][name].update(v)
                        elif k not in self._models.keys():
                            self._ht_params[key][name][k] = v
            elif key == "distributions":
                self._ht_params[key] = {name: {} for name in self._models.keys()}
                for name in self._models.keys():
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
                self._ht_params[key] = {k: value for k in self._models.keys()}
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

        def execute_model(m):
            try:  # If an error occurs, skip the model
                if self.experiment:
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
                    f"\nException encountered while running the "
                    f"{m.name} model. Removing model from pipeline. ", 1
                )
                self.log("".join(traceback.format_tb(ex.__traceback__))[:-1], 3)
                self.log(f"{ex.__class__.__name__}: {ex}", 1)

                # Append exception to errors dictionary
                self._errors[m.name] = ex

                # Add model to "garbage collector"
                # Cannot remove immediately to maintain the iteration order
                to_remove.append(m.name)

                if self.experiment:
                    mlflow.end_run()

            finally:
                return m

        t = dt.now()  # Measure the time the whole pipeline takes

        to_remove = []
        if self.parallel:
            # Turn off verbosity
            vb = self.verbose
            self.verbose = 0
            for m in self._models:
                m.verbose = self.verbose

            if self.backend == "ray":
                # This implementation is more efficient than through joblib's ray backend
                # The difference is that in this one you start ray tasks, and in the
                # other, you start ray actors and then has them each run the function
                execute_remote = ray.remote(execute_model)
                self._models = ClassMap(*ray.get(execute_remote.remote(m) for m in self._models))
            else:
                self._models = ClassMap(
                    *Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                        delayed(execute_model)(m) for m in self._models
                    )
                )

            # Reset verbosity
            self.verbose = vb
            for m in self._models:
                m.verbose = self.verbose

        else:
            for m in self._models:
                execute_model(m)

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
        for model in self._models:
            # Add the model name for repeated model classes
            select = filter(lambda x: x.acronym == model.acronym, self._models)
            if len(list(select)) > 1:
                names.append(f"{model._fullname} ({model.name})")
            else:
                names.append(model._fullname)
            scores.append(get_best_score(model))
            maxlen = max(maxlen, len(names[-1]))

        for i, m in enumerate(self._models):
            out = f"{names[i]:{maxlen}s} --> {m._final_output()}"
            if scores[i] == max(scores) and len(self._models) > 1:
                out += " !"

            self.log(out, 1)

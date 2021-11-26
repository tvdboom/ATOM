# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the BasePredictor class.

"""

# Standard packages
import mlflow
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Union, Optional

# Own modules
from .branch import Branch
from .models import MODELS, Stacking, Voting
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, DF_ATTRS, flt, lst,
    check_is_fitted, divide, get_best_score, delete,
    method_to_log, composed, crash,
)


class BasePredictor:
    """Properties and shared methods for the trainers."""

    # Tracking parameters for mlflow
    _tracking_params = dict(
        log_bo=True,
        log_model=True,
        log_plots=True,
        log_data=False,
        log_pipeline=False,
    )

    def __getattr__(self, item):
        """Get attributes from the current branch."""
        if self.__dict__.get("_branches").get(item):
            return self._branches[item]  # Get branch
        elif item in self.branch._get_attrs():
            return getattr(self.branch, item)  # Get attr from branch
        elif self.__dict__.get("_models").get(item.lower()):
            return self._models[item.lower()]  # Get model subclass
        elif item in self.branch.columns:
            return self.branch.dataset[item]  # Get column from dataset
        elif item in DF_ATTRS:
            return getattr(self.branch.dataset, item)  # Get attr from dataset
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {item}."
            )

    def __setattr__(self, item, value):
        """Set some properties to the current branch."""
        if isinstance(getattr(Branch, item, None), property):
            setattr(self.branch, item, value)
        else:
            super().__setattr__(item, value)

    def __delattr__(self, item):
        """Call appropriate methods for model and branch deletion."""
        if item == "branch":
            self.branch.delete()
        elif item in self._branches:
            self._branches[item].delete()
        elif item == "winner" or item in self._models:
            self.delete(item)
        elif item in self.__dict__:
            del self.__dict__[item]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'."
            )

    def __len__(self):
        return len(self.dataset)

    def __contains__(self, item):
        if self.dataset is None:
            return False
        else:
            return item in self.dataset

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self._models:
                return self._models[item]  # Get model
            elif item in self.dataset:
                return self.dataset[item]  # Get column from dataset
            else:
                raise ValueError(
                    f"{self.__class__.__name__} object "
                    f"has no model or column called {item}."
                )
        elif isinstance(item, list):
            return self.dataset[item]  # Get subset of dataset
        else:
            raise TypeError(
                f"{self.__class__.__name__} is only "
                "subscriptable with types str or list."
            )

    # Tracking properties ========================================== >>

    @property
    def log_bo(self):
        return self._tracking_params["log_bo"]

    @log_bo.setter
    @typechecked
    def log_bo(self, value: bool):
        self._tracking_params["log_bo"] = value

    @property
    def log_model(self):
        return self._tracking_params["log_model"]

    @log_model.setter
    @typechecked
    def log_model(self, value: bool):
        self._tracking_params["log_model"] = value

    @property
    def log_plots(self):
        return self._tracking_params["log_plots"]

    @log_plots.setter
    @typechecked
    def log_plots(self, value: bool):
        self._tracking_params["log_plots"] = value

    @property
    def log_data(self):
        return self._tracking_params["log_data"]

    @log_data.setter
    @typechecked
    def log_data(self, value: bool):
        self._tracking_params["log_data"] = value

    @property
    def log_pipeline(self):
        return self._tracking_params["log_pipeline"]

    @log_pipeline.setter
    @typechecked
    def log_pipeline(self, value: bool):
        self._tracking_params["log_pipeline"] = value

    # Utility properties =========================================== >>

    @property
    def branch(self):
        """Return the current branch."""
        return self._branches[self._current]

    @property
    def models(self):
        """Return the names of the models in the pipeline."""
        if self._models:
            return flt([getattr(m, "name", m) for m in self._models.values()])

    @property
    def metric(self):
        """Return the names of the metrics in the pipeline."""
        if self._metric:
            return flt([getattr(m, "name", m) for m in self._metric.values()])

    @property
    def errors(self):
        """Return the errors encountered during model training."""
        return self._errors

    @property
    def winner(self):
        """Return the best performing model."""
        if self._models:  # Returns None if not fitted
            best_index = np.argmax([get_best_score(m) for m in self._models.values()])
            return self._models[best_index]  # CustomDict can select item from index

    @property
    def results(self):
        """Return the results as a pd.DataFrame."""

        def frac(m):
            """Return the fraction of the train set used for the model."""
            n_models = m.branch.idx[0] / m._train_idx
            if n_models == int(n_models):
                return round(1.0 / n_models, 2)
            else:
                return round(m._train_idx / m.branch.idx[0], 2)

        df = pd.DataFrame(
            data=[m.results for m in self._models.values()],
            columns=self._models[0].results.index if self._models else [],
            index=lst(self.models),
        ).dropna(axis=1, how="all")

        # For sh and ts runs, include the fraction of training set
        if any(m._train_idx != len(m.branch.train) for m in self._models.values()):
            df = df.set_index(
                pd.MultiIndex.from_arrays(
                    [[frac(m) for m in self._models.values()], self.models],
                    names=["frac", "model"],
                )
            ).sort_index(level=0, ascending=True)

        return df

    # Prediction methods =========================================== >>

    @composed(crash, method_to_log)
    def reset_predictions(self):
        """Clear the prediction attributes from all models."""
        for m in self._models.values():
            m.reset_predictions()

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Get the winning model's predictions on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.predict(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Get the winning model's probability predictions on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.predict_proba(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Get the winning model's log probability predictions on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.predict_log_proba(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def decision_function(self, X: X_TYPES, **kwargs):
        """Get the winning model's decision function on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.decision_function(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: X_TYPES,
        y: Y_TYPES,
        metric: Optional[Union[str, callable]] = None,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        **kwargs,
    ):
        """Get the winning model's score on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.score(X, y, metric, sample_weight, **kwargs)

    # Utility methods ============================================== >>

    def _get_model_name(self, model):
        """Return a model's name.

        If there are multiple models that start with the same
        acronym, all are returned. If the input is a number,
        select all models that end with that number. The input
        is case-insensitive.

        """
        if model.lower() == "winner":
            return [self.winner.name]

        if model in self._models:
            return [self._models[model].name]

        to_return = []
        for key, value in self._models.items():
            key, model = key.lower(), model.lower()
            condition_1 = key.startswith(model)
            condition_2 = model.replace(".", "").isdigit() and key.endswith(model)
            if condition_1 or condition_2:
                to_return.append(value.name)

        if to_return:
            return to_return
        else:
            raise ValueError(
                f"Model {model} not found in the pipeline! The "
                f"available models are: {', '.join(self._models)}."
            )

    def _get_models(self, models=None):
        """Return models in the pipeline. Duplicate inputs are ignored."""
        if not models:
            if self.models:
                return lst(self.models).copy()
            else:
                return self.models
        elif isinstance(models, str):
            return self._get_model_name(models.lower())
        else:
            to_return = []
            for m1 in models:
                for m2 in self._get_model_name(m1.lower()):
                    to_return.append(m2)

            return list(dict.fromkeys(to_return))  # Avoid duplicates

    @crash
    def available_models(self):
        """Give an overview of the available predefined models.

        Returns
        -------
        overview: pd.DataFrame
            Information about the predefined models available for the
            current task. Columns include:
                - acronym: Model's acronym (used to call the model).
                - fullname: Complete name of the model.
                - estimator: The model's underlying estimator.
                - module: The estimator's module.
                - needs_scaling: Whether the model requires feature scaling.

        """
        overview = pd.DataFrame()
        for model in MODELS.values():
            m = model(self)
            est = m.get_estimator()
            if m.goal[:3] == self.goal[:3] or m.goal == "both":
                overview = overview.append(
                    {
                        "acronym": m.acronym,
                        "fullname": m.fullname,
                        "estimator": est.__class__.__name__,
                        "module": est.__module__,
                        "needs_scaling": str(m.needs_scaling),
                    },
                    ignore_index=True,
                )

        return overview

    @composed(crash, method_to_log, typechecked)
    def delete(self, models: Optional[Union[str, SEQUENCE_TYPES]] = None):
        """Delete models from the trainer's pipeline.

        Removes a model from the trainer. If the winning model is
        removed, the next best model (through `metric_test` or
        `mean_bootstrap`) is selected as winner. If all models are
        removed, the metric and training approach are reset. Use this
        method to drop unwanted models from the pipeline or to free
        some memory before saving. The model is not removed from any
        active mlflow experiment.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Models to delete. If None, delete them all.

        """
        models = self._get_models(models)
        if not models:
            self.log(f"No models to delete.", 1)
        else:
            delete(self, models)
            if len(models) == 1:
                self.log(f"Model {models[0]} successfully deleted.", 1)
            else:
                self.log(f"Deleting {len(models)} models...", 1)
                for m in models:
                    self.log(f" --> Model {m} successfully deleted.", 1)

    @composed(crash, typechecked)
    def evaluate(
        self,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        threshold: Optional[float] = None,
        dataset: str = "test",
    ):
        """Get all models' scores for the provided metrics.

        Parameters
        ----------
        metric: str, func, scorer, sequence or None, optional (default=None)
            Metric to calculate. If None, it returns an overview of
            the most common metrics per task.

        threshold: float or None, optional (default=None)
            Threshold between 0 and 1 to convert predicted probabilities
            to class labels. Only for metrics (and models) that make use
            of the `predict_proba` method. If None or not a binary
            classification task, ignore this parameter.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Choose from:
            "train", "test" or "holdout".

        Returns
        -------
        scores: pd.DataFrame
            Scores of the models.

        """
        check_is_fitted(self, attributes="_models")

        scores = pd.DataFrame()
        for m in self._models.values():
            scores = scores.append(m.evaluate(metric, threshold, dataset))

        return scores

    @composed(crash, typechecked)
    def get_class_weight(self, dataset: str = "train"):
        """Return class weights for a balanced dataset.

        Statistically, the class weights re-balance the data set so
        that the sampled data set represents the target population
        as closely as possible. The returned weights are inversely
        proportional to the class frequencies in the selected data set.

        Parameters
        ----------
        dataset: str, optional (default="train")
            Data set from which to get the weights. Choose from
            "train", "test" or "dataset".

        Returns
        -------
        class_weights: dict
            Classes with the corresponding weights.

        """
        if self.goal != "class":
            raise PermissionError(
                "The balance method is only available for classification tasks!"
            )

        if dataset not in ("train", "test", "dataset"):
            raise ValueError(
                "Invalid value for the dataset parameter. "
                "Choose from: train, test or dataset."
            )

        y = self.classes[dataset]
        return {idx: round(divide(sum(y), value), 3) for idx, value in y.iteritems()}

    @composed(crash, method_to_log, typechecked)
    def stacking(
        self,
        name: str = "Stack",
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Add a Stacking model to the pipeline.

        Parameters
        ----------
        name: str, optional (default="Stack")
            Name of the model. The name is always presided with the
            model's acronym: `stack`.

        models: sequence or None, optional (default=None)
            Models that feed the voting estimator. If None, it selects
            all models trained on the current branch.

        **kwargs
            Additional keyword arguments for sklearn's stacking instance.
            The model's acronyms can be used for the `final_estimator`
            parameter.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_models(models or self.branch._get_depending_models())

        if len(models) < 2:
            raise ValueError(
                "Invalid value for the models parameter. A Stacking model should "
                f"contain at least two underlying estimators, got {models}."
            )

        if not name.lower().startswith("stack"):
            name = f"Stack{name}"

        if isinstance(kwargs.get("final_estimator"), str):
            if kwargs["final_estimator"] not in MODELS:
                raise ValueError(
                    "Invalid value for the final_estimator parameter. Unknown model: "
                    f"{kwargs['final_estimator']}. Choose from: {', '.join(MODELS)}."
                )
            else:
                model = MODELS[kwargs["final_estimator"]](self)
                if model.goal not in (self.goal, "both"):
                    raise ValueError(
                        "Invalid value for the final_estimator parameter. Model "
                        f"{model.fullname} can not perform {self.task} tasks."
                    )

                kwargs["final_estimator"] = model.get_estimator()

        self._models[name] = Stacking(self, name, models=self._models[models], **kwargs)

        if self.experiment:
            self[name]._run = mlflow.start_run(run_name=self[name].name)

        self[name].fit()

    @composed(crash, method_to_log, typechecked)
    def voting(
        self,
        name: str = "Vote",
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        **kwargs,
    ):
        """Add a Voting model to the pipeline.

        Parameters
        ----------
        name: str, optional (default="Vote")
            Name of the model. The name is always presided with the
            model's acronym: `vote`.

        models: sequence or None, optional (default=None)
            Models that feed the voting estimator. If None, it selects
            all models trained on the current branch.

        **kwargs
            Additional keyword arguments for sklearn's voting instance.

        """
        check_is_fitted(self, attributes="_models")
        models = self._get_models(models or self.branch._get_depending_models())

        if len(models) < 2:
            raise ValueError(
                "Invalid value for the models parameter. A Voting model should "
                f"contain at least two underlying estimators, got {models}."
            )

        if not name.lower().startswith("vote"):
            name = f"Vote{name}"

        self._models[name] = Voting(self, name, models=self._models[models], **kwargs)

        if self.experiment:
            self[name]._run = mlflow.start_run(run_name=self[name].name)

        self[name].fit()

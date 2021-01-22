# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the BasePredictor class.

"""

# Standard packages
import numpy as np
import pandas as pd
from typing import Union, Optional
from typeguard import typechecked

# Own modules
from .branch import Branch
from .ensembles import Voting, Stacking
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, METRIC_ACRONYMS, flt, lst,
    divide, check_is_fitted, get_best_score, delete, method_to_log,
    composed, crash,
)


class BasePredictor:
    """Properties and shared methods for the trainers."""

    def __getattr__(self, item):
        """Get some attributes from the current branch."""
        props = [i for i in dir(Branch) if isinstance(getattr(Branch, i), property)]
        if self.__dict__.get("_branches"):  # Add public attrs from branch
            props.extend([k for k in self.branch.__dict__ if k not in Branch.private])
        if item in props:
            return getattr(self.branch, item)  # Get attr from branch
        elif self.__dict__.get("_models").get(item.lower()):
            return self._models[item.lower()]   # Get model subclass
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'."
            )

    def __setattr__(self, item, value):
        """Set some properties to the current branch."""
        props = [i for i in dir(Branch) if isinstance(getattr(Branch, i), property)]
        if item in props:
            setattr(self.branch, item, value)
        else:
            super().__setattr__(item, value)

    def __delattr__(self, item):
        """Call appropriate methods for model and branch deletion."""
        # To delete branches, call the appropriate method
        if item == "branch":
            self.branch.delete()
        else:
            try:
                self.delete(item)
            except ValueError:
                del self.__dict__[item]

    # Utility properties =========================================== >>

    @property
    def branch(self):
        """Return the current branch."""
        return self._branches[self._current]

    @property
    def metric(self):
        """Return the names of the metrics in the pipeline."""
        return flt([getattr(metric, "name", metric) for metric in self._metric])

    @property
    def models(self):
        """Return the names of the models in the pipeline."""
        return flt([getattr(model, "name", model) for model in self._models])

    @property
    def results(self):
        """Return the results as a pd.DataFrame."""
        return pd.DataFrame(
            data=[m.results for m in self._models],
            columns=[m.results.index for m in self._models][0],
            index=lst(self.models),
        ).dropna(axis=1, how="all")

    @property
    def winner(self):
        """Return the best performing model."""
        if self._models:  # Returns None if not fitted
            return self._models[np.argmax([get_best_score(m) for m in self._models])]

    # Prediction methods =========================================== >>

    @composed(crash, method_to_log)
    def reset_predictions(self):
        """Clear the prediction attributes from all models."""
        for m in self._models:
            m._pred_attrs = [None] * 10

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
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        **kwargs,
    ):
        """Get the winning model's score on new data."""
        check_is_fitted(self, attributes="_models")
        return self.winner.score(X, y, sample_weight, **kwargs)

    # Utility methods ============================================== >>

    def _get_model_name(self, model):
        """Return a model's name.

        If there are multiple models that start with the same
        acronym, all will be return. If the input is a number,
        select all models that end with that number. The input
        is case-insensitive.

        """
        model = model.lower()

        if model == "winner":
            return [self.winner.name]

        if model in self._models:
            return [self._models[model].name]

        to_return = []
        for key, value in self._models.items():
            condition_1 = key.startswith(model)
            condition_2 = model.replace(".", "").isdigit() and key.endswith(model)
            if condition_1 or condition_2:
                to_return.append(value.name)

        if to_return:
            return to_return
        else:
            raise ValueError(
                f"Model {model} not found in the pipeline! The "
                f"available models are: {', '.join(self.models)}."
            )

    def _get_models(self, models):
        """Return models in the pipeline. Duplicate inputs are ignored."""
        if not models:
            return lst(self.models).copy()
        elif isinstance(models, str):
            return self._get_model_name(models.lower())
        else:
            to_return = []
            for m1 in models:
                for m2 in self._get_model_name(m1.lower()):
                    to_return.append(m2)

            return list(dict.fromkeys(to_return))  # Avoid duplicates

    @composed(crash, method_to_log, typechecked)
    def voting(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        weights: Optional[SEQUENCE_TYPES] = None,
    ):
        """Add a Voting instance to the models in the pipeline.

        Parameters
        ----------
        models: sequence or None, optional (default=None)
            Models that feed the voting. If None, all models depending
            on the current branch are selected.

        weights: sequence or None, optional (default=None)
            Sequence of weights (int or float) to weight the
            occurrences of predicted class labels (hard voting)
            or class probabilities before averaging (soft voting).
            If None, it uses uniform weights.

        """
        check_is_fitted(self, attributes="_models")

        if not models:
            models = self.branch._get_depending_models()

        self._models["vote"] = Voting(
            self,
            models=self._get_models(models),
            weights=weights
        )
        self.log(f"{self.vote.fullname} added to the models!", 1)

    @composed(crash, method_to_log, typechecked)
    def stacking(
        self,
        models: Optional[Union[str, SEQUENCE_TYPES]] = None,
        estimator: Optional[Union[str, callable]] = None,
        stack_method: str = "auto",
        passthrough: bool = False,
    ):
        """Add a Stacking instance to the models in the pipeline.

        Parameters
        ----------
        models: sequence or None, optional (default=None)
            Models that feed the stacking.

        estimator: str, callable or None, optional (default=None)
            The final estimator, which will be used to combine the base
            estimators. If str, choose from ATOM's predefined models.
            If None, a default estimator is selected:
                - LogisticRegression for classification tasks.
                - Ridge for regression tasks.

        stack_method: str, optional (default="auto")
            Methods called for each base estimator. If "auto", it will
            try to invoke `predict_proba`, `decision_function` or
            `predict` in that order.

        passthrough: bool, optional (default=False)
            When False, only the predictions of estimators will be used
            as training data for the final estimator. When True, the
            estimator is trained on the predictions as well as the
            original training data.

        """
        check_is_fitted(self, attributes="_models")

        if not models:
            models = self.branch._get_depending_models()
        if not estimator:
            estimator = "LR" if self.goal.startswith("class") else "Ridge"

        self._models["stack"] = Stacking(
            self,
            models=self._get_models(models),
            estimator=estimator,
            stack_method=stack_method,
            passthrough=passthrough,
        )
        self.log(f"{self.stack.fullname} added to the models!", 1)

    @composed(crash, typechecked)
    def get_class_weight(self, dataset: str = "train"):
        """Return class weights for a balanced dataset.

        Statistically, the class weights re-balance the data set so
        that the sampled data set represents the target population
        as closely as reasonably possible. The returned weights are
        inversely proportional to class frequencies in the selected
        data set.

        Parameters
        ----------
        dataset: str, optional (default="train")
            Data set from which to get the weights. Choose between
            "train", "test" or "dataset".

        Returns
        -------
        class_weights: dict
            Classes with the corresponding weights.

        """
        if dataset not in ("train", "test", "dataset"):
            raise ValueError(
                "Invalid value for the dataset parameter. "
                "Choose between 'train', 'test' or 'dataset'."
            )

        y = self.classes[dataset]
        return {idx: round(divide(sum(y), value), 3) for idx, value in y.iteritems()}

    @composed(crash, method_to_log)
    def calibrate(self, **kwargs):
        """Calibrate the winning model."""
        check_is_fitted(self, attributes="_models")
        self.winner.calibrate(**kwargs)

    @composed(crash, method_to_log, typechecked)
    def scoring(self, metric: Optional[str] = None, dataset: str = "test", **kwargs):
        """Print all the models' scoring for a specific metric.

        Parameters
        ----------
        metric: str or None, optional (default=None)
            Name of the metric to calculate. Choose from any of
            sklearn's SCORERS or one of the CUSTOM_METRICS. If None,
            returns the pipeline's final results (ignores `dataset`).

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train" or "test".

        **kwargs
            Additional keyword arguments for the metric function.

        """
        check_is_fitted(self, attributes="_models")

        # If a metric acronym is used, assign the correct name
        if metric and metric.lower() in METRIC_ACRONYMS:
            metric = METRIC_ACRONYMS[metric.lower()]

        # Get max length of the model names
        maxlen = max([len(m.fullname) for m in self._models])

        # Get best score of all the models
        best_score = max([get_best_score(m) for m in self._models])

        for m in self._models:
            if not metric:
                out = f"{m.fullname:{maxlen}s} --> {m._final_output()}"
            else:
                score = m.scoring(metric, dataset, **kwargs)

                if isinstance(score, float):
                    out_score = round(score, 3)
                else:  # If confusion matrix...
                    out_score = list(score.ravel())
                out = f"{m.fullname:{maxlen}s} --> {metric}: {out_score}"
                if get_best_score(m) == best_score and len(self._models) > 1:
                    out += " !"

            self.log(out, kwargs.get("_vb", -2))  # Always print if called by user

    @composed(crash, method_to_log, typechecked)
    def delete(self, models: Optional[Union[str, SEQUENCE_TYPES]] = None):
        """Delete models from the trainer's pipeline.

        Removes all traces of a model in the pipeline (except for the
        `errors` attribute). If the winning model is removed. The next
        best model (through metric_test or mean_bagging if available)
        is selected as winner. If all models are removed, the metric and
        approach are reset. Use this method to drop unwanted models from
        the pipeline or to free up some memory.

        Parameters
        ----------
        models: str, sequence or None, optional (default=None)
            Models to remove from the pipeline. If None, delete all.

        """
        models = self._get_models(models)
        delete(self, models)
        self.log(f"Model{'' if len(models) == 1 else 's'} deleted successfully!", 1)

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
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, METRIC_ACRONYMS, flt, divide,
    check_is_fitted, get_best_score, delete, method_to_log, composed,
    crash,
)


class BasePredictor(object):
    """Properties and shared methods for the trainers."""

    def __getattr__(self, item):
        """Get some attributes from the current branch."""
        props = [i for i in dir(Branch) if isinstance(getattr(Branch, i), property)]
        if self.__dict__.get("_branches"):  # Add public attrs from branch
            props.extend([k for k in self.branch.__dict__ if k not in Branch.private])
        if item in props:
            return getattr(self.branch, item)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'"
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
        """Return the pipeline's metric."""
        return flt([getattr(metric, "name", metric) for metric in self.metric_])

    @property
    def models_(self):
        """Return a list of the models in the pipeline."""
        return [getattr(self, model) for model in self.models]

    @property
    def results(self):
        """Return the results dataframe ordered and without empty columns."""
        df = self._results

        # If multi-index, invert the runs and reindex the models
        if isinstance(df.index, pd.MultiIndex):
            df = df.sort_index(level=0).reindex(self.models, level=1)

        return df.dropna(axis=1, how="all")

    @property
    def winner(self):
        """Return the best performing model."""
        if self.models:  # Returns None if not fitted
            return self.models_[np.argmax([get_best_score(m) for m in self.models_])]

    # Prediction methods =========================================== >>

    @composed(crash, method_to_log)
    def reset_predictions(self):
        """Clear the prediction attributes from all models."""
        for m in self.models_:
            m._pred_attrs = [None] * 10

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Get the winning model's predictions on new data."""
        check_is_fitted(self, "results")
        return self.winner.predict(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Get the winning model's probability predictions on new data."""
        check_is_fitted(self, "results")
        return self.winner.predict_proba(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Get the winning model's log probability predictions on new data."""
        check_is_fitted(self, "results")
        return self.winner.predict_log_proba(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def decision_function(self, X: X_TYPES, **kwargs):
        """Get the winning model's decision function on new data."""
        check_is_fitted(self, "results")
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
        check_is_fitted(self, "results")
        return self.winner.score(X, y, sample_weight, **kwargs)

    # Utility methods ============================================== >>

    def _get_model_name(self, model):
        """Return a model's name. Case insensitive."""
        for name in self.models:
            if model.lower() == name.lower():
                return name

        raise ValueError(
            f"Model {model} not found in the pipeline! Available "
            f"models are: {', '.join(self.models)}."
        )

    def _get_models(self, models):
        """Return models in the pipeline."""
        if not models:
            return self.models.copy()
        elif isinstance(models, str):
            return [self._get_model_name(models)]
        else:
            return [self._get_model_name(m) for m in models]

    @composed(crash, method_to_log, typechecked)
    def voting(
        self,
        models: Optional[SEQUENCE_TYPES] = None,
        weights: Optional[SEQUENCE_TYPES] = None,
    ):
        """Add a Voting instance to the models in the pipeline."""
        check_is_fitted(self, "results")

        if not models:
            models = self.branch._get_depending_models()

        self.Vote = self.vote = Voting(
            self,
            models=self._get_models(models),
            weights=weights
        )
        self.models += [self.vote.name]
        self.log(f"{self.vote.fullname} added to the models!", 1)

    @composed(crash, method_to_log, typechecked)
    def stacking(
        self,
        models: Optional[SEQUENCE_TYPES] = None,
        estimator: Optional[Union[str, callable]] = None,
        stack_method: str = "auto",
        passthrough: bool = False,
    ):
        """Add a Stacking instance to the models in the pipeline."""
        check_is_fitted(self, "results")

        if not models:
            models = self.branch._get_depending_models()
        if not estimator:
            estimator = "LR" if self.goal.startswith("class") else "Ridge"

        self.Stack = self.stack = Stacking(
            self,
            models=self._get_models(models),
            estimator=estimator,
            stack_method=stack_method,
            passthrough=passthrough,
        )
        self.models += [self.stack.name]
        self.log(f"{self.stack.fullname} added to the models!", 1)

    @composed(crash, typechecked)
    def get_class_weight(self, dataset: str = "train"):
        """Return class weights for a balanced data set.

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
        check_is_fitted(self, "results")
        return self.winner.calibrate(**kwargs)

    @composed(crash, method_to_log, typechecked)
    def scoring(self, metric: Optional[str] = None, dataset: str = "test", **kwargs):
        """Get the scoring for a specific metric.

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
        check_is_fitted(self, "results")

        # If a metric acronym is used, assign the correct name
        if metric and metric.lower() in METRIC_ACRONYMS:
            metric = METRIC_ACRONYMS[metric.lower()]

        # Get max length of the model names
        maxlen = max([len(m.fullname) for m in self.models_])

        self.log("Results ===================== >>", -2)

        for m in self.models_:
            if not metric:
                out = f"{m.fullname:{maxlen}s} --> {m._final_output()}"
            else:
                score = m.scoring(metric, dataset, **kwargs)

                if isinstance(score, float):
                    out_score = round(score, 3)
                else:  # If confusion matrix...
                    out_score = list(score.ravel())
                out = f"{m.fullname:{maxlen}s} --> {metric}: {out_score}"

            self.log(out, -2)  # Always print

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

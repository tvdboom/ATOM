# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the BasePredictor class.

"""

# Standard packages
import numpy as np
import pandas as pd
from typing import Union, Optional, Sequence
from typeguard import typechecked

# Own modules
from .utils import (
    ARRAY_TYPES, X_TYPES, Y_TYPES, METRIC_ACRONYMS, flt, divide, check_is_fitted,
    get_best_score, clear, method_to_log, composed, crash,
)


class BasePredictor(object):
    """Properties and shared methods for the training classes."""

    # Utility properties =================================================== >>

    @property
    def pipeline(self):
        return self._branches[self._pipe]

    @property
    def metric(self):
        """Return a list of the model subclasses."""
        return flt([getattr(metric, "name", metric) for metric in self.metric_])

    @property
    def models_(self):
        """Return a list of the model subclasses."""
        return [getattr(self, model) for model in self.models]

    @property
    def results(self):
        """Return the _results dataframe ordered and without empty columns."""
        df = self._results

        # If multi-index, invert the runs and reindex the models
        if isinstance(df.index, pd.MultiIndex):
            df = df.sort_index(level=0).reindex(self.models, level=1)

        return df.dropna(axis=1, how="all")

    @property
    def winner(self):
        """Return the model subclass that performed best."""
        if self.models:  # Returns None if not fitted
            return self.models_[np.argmax([get_best_score(m) for m in self.models_])]

    # Data properties ====================================================== >>

    @property
    def dataset(self):
        return self.pipeline.data

    @property
    def train(self):
        return self.pipeline.data[:self.pipeline.idx[0]]

    @property
    def test(self):
        return self.pipeline.data[-self.pipeline.idx[1]:]

    @property
    def X(self):
        return self.pipeline.data.drop(self.target, axis=1)

    @property
    def y(self):
        return self.pipeline.data[self.target]

    @property
    def X_train(self):
        return self.train.drop(self.target, axis=1)

    @property
    def X_test(self):
        return self.test.drop(self.target, axis=1)

    @property
    def y_train(self):
        return self.train[self.target]

    @property
    def y_test(self):
        return self.test[self.target]

    @property
    def shape(self):
        return self.pipeline.data.shape

    @property
    def columns(self):
        return list(self.pipeline.data.columns)

    @property
    def target(self):
        return self.columns[-1]

    @property
    def mapping(self):
        return self.pipeline.mapping

    @property
    def classes(self):
        df = pd.DataFrame({
            "dataset": self.y.value_counts(sort=False, dropna=False),
            "train": self.y_train.value_counts(sort=False, dropna=False),
            "test": self.y_test.value_counts(sort=False, dropna=False),
        }, index=self.mapping.values())

        return df.fillna(0)  # If 0 counts, it doesnt return the row (gets a NaN)

    @property
    def n_classes(self):
        return len(self.y.unique())

    # Prediction methods =================================================== >>

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Get predictions on new data."""
        check_is_fitted(self, "results")
        return self.winner.predict(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Get probability predictions on new data."""
        check_is_fitted(self, "results")
        return self.winner.predict_proba(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Get log probability predictions on new data."""
        check_is_fitted(self, "results")
        return self.winner.predict_log_proba(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def decision_function(self, X: X_TYPES, **kwargs):
        """Get the decision function on new data."""
        check_is_fitted(self, "results")
        return self.winner.decision_function(X, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: X_TYPES,
        y: Y_TYPES,
        sample_weight: Optional[Union[ARRAY_TYPES]] = None,
        **kwargs,
    ):
        """Get the score output on new data."""
        check_is_fitted(self, "results")
        return self.winner.score(X, y, sample_weight, **kwargs)

    # Utility methods ====================================================== >>

    def _get_model(self, model):
        """Return the model's name given."""
        for name in self.models:
            if model.lower() == name.lower():
                return name

        raise ValueError(
            f"Model {model} not found in the pipeline! Available "
            f"models are: {', '.join(self.models)}."
        )

    @composed(crash, typechecked)
    def get_class_weight(self, dataset: str = "train"):
        """Return class weights for a balanced data set.

        Statistically, the class weights re-balance the data set so that the
        sampled data set represents the target population as closely as reasonably
        possible. The returned weights are inversely proportional to class
        frequencies in the selected data set.

        Parameters
        ----------
        dataset: str, optional (default="train")
            Data set from which to get the weights. Choose between "train",
            "test" or "dataset".

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
        """Print the final scoring for a specific metric.

        If a model returns `XXX`, it means the metric failed for that specific
        model. This can happen if either the metric is unavailable for the task
        or if the model does not have a `predict_proba` method while the metric
        requires it.

        Parameters
        ----------
        metric: string or None, optional (default=None)
            String of one of sklearn's predefined scorers. If None, the metric(s)
            used to fit the trainer is selected and the bagging results will be
            showed (if used).

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are "train" or "test".

        **kwargs
            Additional keyword arguments for the metric function.

        """
        check_is_fitted(self, "results")

        # If a metric_ acronym is used, assign the correct name
        if metric and metric.lower() in METRIC_ACRONYMS:
            metric = METRIC_ACRONYMS[metric.lower()]

        # Get max length of the model names
        maxlen = max([len(m.fullname) for m in self.models_])

        # Get list of scores
        all_scores = [m.scoring(metric, dataset, **kwargs) for m in self.models_]

        # Raise an error if the metric was invalid for all models
        if metric and all([isinstance(score, str) for score in all_scores]):
            raise ValueError("Invalid metric selected!")

        self.log("Results ===================== >>", -2)

        for m in self.models_:
            if not metric:
                out = f"{m.fullname:{maxlen}s} --> {m._final_output()}"
            else:
                score = m.scoring(metric, dataset, **kwargs)

                # Create string of the score (if wrong metric for model -> XXX)
                if isinstance(score, str):
                    out = f"{m.fullname:{maxlen}s} --> XXX"
                else:
                    if isinstance(score, float):
                        out_score = round(score, 3)
                    else:  # If confusion matrix...
                        out_score = list(score.ravel())
                    out = f"{m.fullname:{maxlen}s} --> {metric}: {out_score}"

            self.log(out, -2)  # Always print

    @composed(crash, method_to_log, typechecked)
    def clear(self, models: Union[str, Sequence[str]] = "all"):
        """Clear models from the pipeline.

        Removes all traces of a model in the pipeline (except for the `errors`
        attribute). If all models in the pipeline are removed, the metric is reset.
        Use this method to remove unwanted models from the pipeline or to clear
        memory before saving the instance.

        Parameters
        ----------
        models: str or iterable, optional (default="all")
            Model(s) to clear from the pipeline. If "all", clear all models.

        """
        # Prepare the models parameter
        if models == "all":
            keyword = "All models were"
            models = self.models.copy()
        elif isinstance(models, str):
            models = [self._get_model(models)]
            keyword = "Model " + models[0]
        else:
            models = [self._get_model(m) for m in models]
            keyword = "Models " + ", ".join(models) + " were"

        clear(self, models)

        self.log(f"{keyword} cleared successfully!", 1)

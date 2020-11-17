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
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, METRIC_ACRONYMS, flt, divide,
    check_is_fitted, get_best_score, delete, method_to_log, composed, crash
)


class BasePredictor(object):
    """Properties and shared methods for the training classes."""

    # Utility properties =========================================== >>

    @property
    def branch(self):
        """Return the current branch."""
        return self._branches[self._current]

    @property
    def pipeline(self):
        """Return the estimators in the branches pipeline."""
        return self.branch.estimators

    @property
    def feature_importance(self):
        """Return the feature_importance in the branches pipeline."""
        return self.branch.feature_importance

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

    # Data properties ============================================== >>

    @property
    def dataset(self):
        return self.branch.data

    @property
    def train(self):
        return self.branch.data[:self.branch.idx[0]]

    @property
    def test(self):
        return self.branch.data[-self.branch.idx[1]:]

    @property
    def X(self):
        return self.branch.data.drop(self.target, axis=1)

    @property
    def y(self):
        return self.branch.data[self.target]

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
        return self.branch.data.shape

    @property
    def columns(self):
        return list(self.branch.data.columns)

    @property
    def target(self):
        return self.columns[-1]

    @property
    def mapping(self):
        return self.branch.mapping

    @property
    def classes(self):
        """Return the number of samples per class and per data set."""
        df = pd.DataFrame({
            "dataset": self.y.value_counts(sort=False, dropna=False),
            "train": self.y_train.value_counts(sort=False, dropna=False),
            "test": self.y_test.value_counts(sort=False, dropna=False),
        }, index=self.mapping.values())

        return df.fillna(0)  # If 0 counts, it doesnt return the row (gets a NaN)

    @property
    def n_classes(self):
        """Return the number of classes in the target column."""
        return len(self.y.unique())

    # Prediction methods =========================================== >>

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

    def _get_model(self, model):
        """Return a model's name. Case insensitive."""
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
        """Print the final scoring for a specific metric.

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

                if isinstance(score, float):
                    out_score = round(score, 3)
                else:  # If confusion matrix...
                    out_score = list(score.ravel())
                out = f"{m.fullname:{maxlen}s} --> {metric}: {out_score}"

            self.log(out, -2)  # Always print

    @composed(crash, method_to_log, typechecked)
    def delete(self, models: Union[str, SEQUENCE_TYPES] = "all"):
        """Delete models from the trainer's pipeline.

        Removes all traces of a model in the pipeline (except for the
        `errors` attribute). If the winning model is removed. The next
        best model (through metric_test or mean_bagging if available)
        is selected as winner.If all models are removed, the metric and
        approach are reset. Use this method to drop unwanted models from
        the pipeline or to clear memory before saving the instance.

        Parameters
        ----------
        models: str or sequence, optional (default="all")
            Model(s) to remove from the pipeline. If "all", delete all
            models.

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

        delete(self, models)

        self.log(f"{keyword} deleted successfully!", 1)

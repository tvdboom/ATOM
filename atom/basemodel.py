# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: Mavs
Description: Module containing the BaseModel class.

"""

# Standard packages
import pandas as pd
from typeguard import typechecked
from typing import Optional, Union
from mlflow.tracking import MlflowClient

# Own modules
from .data_cleaning import Scaler
from .plots import BaseModelPlotter
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, DF_ATTRS, lst, it, merge, arr,
    get_scorer, custom_transform, composed, crash, method_to_log,
)


class BaseModel(BaseModelPlotter):
    """Base class for all models."""

    def __init__(self, *args):
        self.T = args[0]  # Trainer instance
        self.name = self.acronym if len(args) == 1 else args[1]
        self.scaler = None
        self.estimator = None
        self.explainer = None  # Explainer object for shap plots
        self._run = None  # mlflow run (if experiment is active)
        self._group = self.name  # sh and ts models belong to the same group
        self._pred_attrs = [None] * 10

        # Skip if called from FeatureSelector
        if hasattr(self.T, "_branches"):
            self.branch = self.T._branches[self.T._current]
            self._train_idx = self.branch.idx[0]  # Can change for sh and ts
            if getattr(self, "needs_scaling", None) and not self.T.scaled:
                self.scaler = Scaler().fit(self.X_train)

    def __getattr__(self, item):
        if item in self.__dict__.get("branch")._get_attrs():
            return getattr(self.branch, item)  # Get attr from branch
        elif item in self.__dict__.get("branch").columns:
            return self.branch.dataset[item]  # Get column
        elif item in DF_ATTRS:
            return getattr(self.branch.dataset, item)  # Get attr from dataset
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'."
            )

    def __contains__(self, item):
        return item in self.dataset

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.dataset[item]  # Get a column from the dataset
        else:
            raise TypeError(
                f"'{self.__class__.__name__}' object is "
                "only subscriptable with type str."
            )

    # Utility properties =========================================== >>

    @property
    def results(self):
        """Return the results as a pd.Series."""
        return pd.Series(
            {
                "metric_bo": getattr(self, "metric_bo", None),
                "time_bo": getattr(self, "time_bo", None),
                "metric_train": getattr(self, "metric_train", None),
                "metric_test": getattr(self, "metric_test", None),
                "time_fit": getattr(self, "time_fit", None),
                "mean_bootstrap": getattr(self, "mean_bootstrap", None),
                "std_bootstrap": getattr(self, "std_bootstrap", None),
                "time_bootstrap": getattr(self, "time_bootstrap", None),
                "time": getattr(self, "time", None),
            },
            name=self.name,
        )

    # Prediction methods =========================================== >>

    def _prediction(
        self,
        X,
        y=None,
        metric=None,
        sample_weight=None,
        pipeline=None,
        verbose=None,
        method="predict"
    ):
        """Apply prediction methods on new data.

        First transform the new data and then apply the attribute on
        the best model. The model has to have the provided attribute.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        metric: str, func, scorer or None, optional (default=None)
            Metric to calculate. Choose from any of sklearn's SCORERS,
            a function with signature metric(y_true, y_pred) or a scorer
            object. If None, it returns mean accuracy for classification
            tasks and r2 for regression tasks. Only for method="score".

        sample_weight: sequence or None, optional (default=None)
            Sample weights for the score method.

        pipeline: bool, sequence or None, optional (default=None)
            Transformers to use on the data before predicting.
                - If None: Only transformers that are applied on the
                           whole dataset are used.
                - If False: Don't use any transformers.
                - If True: Use all transformers in the pipeline.
                - If sequence: Transformers to use, selected by their
                               index in the pipeline.

        verbose: int or None, optional (default=None)
            Verbosity level for the transformers. If None, it uses the
            transformer's own verbosity.

        method: str, optional (default="predict")
            Prediction method to be applied to the estimator.

        Returns
        -------
        pred: np.ndarray
            Return of the attribute.

        """
        if not hasattr(self.estimator, method):
            raise AttributeError(
                f"{self.estimator.__class__.__name__} doesn't have a {method} method!"
            )

        if pipeline is None:
            pipeline = [i for i, est in enumerate(self.pipeline) if not est.train_only]
        elif pipeline is False:
            pipeline = []
        elif pipeline is True:
            pipeline = list(range(len(self.pipeline)))

        # When there is a pipeline, apply transformations first
        for idx, est in self.pipeline.iteritems():
            if idx in pipeline:
                X, y = custom_transform(self.T, est, self.branch, (X, y), verbose)

        # Scale the data if needed
        if self.scaler:
            X = self.scaler.transform(X)

        if y is None:
            return getattr(self.estimator, method)(X)
        else:
            if metric is None:
                if self.T.goal.startswith("class"):
                    metric = get_scorer("accuracy")
                else:
                    metric = get_scorer("r2")
            else:
                metric = get_scorer(metric)

            kwargs = {}
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight

            return metric(self.estimator, X, y, **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict(
        self,
        X: X_TYPES,
        pipeline: Optional[Union[bool, SEQUENCE_TYPES]] = None,
        verbose: Optional[int] = None,
    ):
        """Get predictions on new data."""
        return self._prediction(
            X=X,
            pipeline=pipeline,
            verbose=verbose,
            method="predict",
        )

    @composed(crash, method_to_log, typechecked)
    def predict_proba(
        self,
        X: X_TYPES,
        pipeline: Optional[Union[bool, SEQUENCE_TYPES]] = None,
        verbose: Optional[int] = None,
    ):
        """Get probability predictions on new data."""
        return self._prediction(
            X=X,
            pipeline=pipeline,
            verbose=verbose,
            method="predict_proba",
        )

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(
        self,
        X: X_TYPES,
        pipeline: Optional[Union[bool, SEQUENCE_TYPES]] = None,
        verbose: Optional[int] = None,
    ):
        """Get log probability predictions on new data."""
        return self._prediction(
            X=X,
            pipeline=pipeline,
            verbose=verbose,
            method="predict_log_proba",
        )

    @composed(crash, method_to_log, typechecked)
    def decision_function(
        self,
        X: X_TYPES,
        pipeline: Optional[Union[bool, SEQUENCE_TYPES]] = None,
        verbose: Optional[int] = None,
    ):
        """Get the decision function on new data."""
        return self._prediction(
            X=X,
            pipeline=pipeline,
            verbose=verbose,
            method="decision_function",
        )

    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: X_TYPES,
        y: Y_TYPES,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        pipeline: Optional[Union[bool, SEQUENCE_TYPES]] = None,
        verbose: Optional[int] = None,
    ):
        """Get the score function on new data."""
        return self._prediction(
            X=X,
            y=y,
            metric=metric,
            sample_weight=sample_weight,
            pipeline=pipeline,
            verbose=verbose,
            method="score",
        )

    # Prediction properties ======================================== >>

    @composed(crash, method_to_log)
    def reset_predictions(self):
        """Clear all the prediction attributes."""
        self._pred_attrs = [None] * 10

    @property
    def predict_train(self):
        if self._pred_attrs[0] is None:
            self._pred_attrs[0] = self.estimator.predict(arr(self.X_train))
        return self._pred_attrs[0]

    @property
    def predict_test(self):
        if self._pred_attrs[1] is None:
            self._pred_attrs[1] = self.estimator.predict(arr(self.X_test))
        return self._pred_attrs[1]

    @property
    def predict_proba_train(self):
        if self._pred_attrs[2] is None:
            self._pred_attrs[2] = self.estimator.predict_proba(arr(self.X_train))
        return self._pred_attrs[2]

    @property
    def predict_proba_test(self):
        if self._pred_attrs[3] is None:
            self._pred_attrs[3] = self.estimator.predict_proba(arr(self.X_test))
        return self._pred_attrs[3]

    @property
    def predict_log_proba_train(self):
        if self._pred_attrs[4] is None:
            self._pred_attrs[4] = self.estimator.predict_log_proba(arr(self.X_train))
        return self._pred_attrs[4]

    @property
    def predict_log_proba_test(self):
        if self._pred_attrs[5] is None:
            self._pred_attrs[5] = self.estimator.predict_log_proba(arr(self.X_test))
        return self._pred_attrs[5]

    @property
    def decision_function_train(self):
        if self._pred_attrs[6] is None:
            self._pred_attrs[6] = self.estimator.decision_function(arr(self.X_train))
        return self._pred_attrs[6]

    @property
    def decision_function_test(self):
        if self._pred_attrs[7] is None:
            self._pred_attrs[7] = self.estimator.decision_function(arr(self.X_test))
        return self._pred_attrs[7]

    @property
    def score_train(self):
        if self._pred_attrs[8] is None:
            self._pred_attrs[8] = self.estimator.score(arr(self.X_train), self.y_train)
        return self._pred_attrs[8]

    @property
    def score_test(self):
        if self._pred_attrs[9] is None:
            self._pred_attrs[9] = self.estimator.score(arr(self.X_test), self.y_test)
        return self._pred_attrs[9]

    # Data Properties ============================================== >>

    @property
    def dataset(self):
        return merge(self.X, self.y)

    @property
    def train(self):
        return merge(self.X_train, self.y_train)

    @property
    def test(self):
        return merge(self.X_test, self.y_test)

    @property
    def X(self):
        return pd.concat([self.X_train, self.X_test])

    @property
    def y(self):
        return pd.concat([self.y_train, self.y_test])

    @property
    def X_train(self):
        if self.scaler:
            return self.scaler.transform(self.branch.X_train[:self._train_idx])
        else:
            return self.branch.X_train[:self._train_idx]

    @property
    def X_test(self):
        if self.scaler:
            return self.scaler.transform(self.branch.X_test)
        else:
            return self.branch.X_test

    @property
    def y_train(self):
        return self.branch.y_train[:self._train_idx]

    # Utility methods ============================================== >>

    @composed(crash, method_to_log)
    def delete(self):
        """Delete the model from the trainer."""
        self.T.delete(self.name)

    @composed(crash, typechecked)
    def evaluate(
        self,
        metric: Optional[Union[str, callable, SEQUENCE_TYPES]] = None,
        dataset: str = "test",
    ):
        """Get the model's scores for the provided metrics.

        Parameters
        ----------
        metric: str, func, scorer, sequence or None, optional (default=None)
            Metrics to calculate. If None, a selection of the most
            common metrics per task are used.

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train" or "test".

        Returns
        -------
        score: pd.Series
            Scores of the model.

        """
        dataset = dataset.lower()
        if dataset not in ("train", "test"):
            raise ValueError(
                "Unknown value for the dataset parameter. "
                "Choose between 'train' or 'test'."
            )

        # Predefined metrics to show
        if metric is None:
            if self.T.task.startswith("bin"):
                metric = [
                    "accuracy",
                    "ap",
                    "ba",
                    "f1",
                    "jaccard",
                    "mcc",
                    "precision",
                    "recall",
                    "auc",
                ]
            elif self.T.task.startswith("multi"):
                metric = [
                    "ba",
                    "f1_weighted",
                    "jaccard_weighted",
                    "mcc",
                    "precision_weighted",
                    "recall_weighted",
                ]
            else:
                metric = ["mae", "mape", "me", "mse", "msle", "r2", "rmse"]

        scores = pd.Series(name=self.name, dtype=float)
        for met in lst(metric):
            scorer = get_scorer(met)
            if scorer.__class__.__name__ == "_ThresholdScorer":
                if hasattr(self.estimator, "decision_function"):
                    y_pred = getattr(self, f"decision_function_{dataset}")
                else:
                    y_pred = getattr(self, f"predict_proba_{dataset}")
                    if self.T.task.startswith("bin"):
                        y_pred = y_pred[:, 1]
            elif scorer.__class__.__name__ == "_ProbaScorer":
                if hasattr(self.estimator, "predict_proba"):
                    y_pred = getattr(self, f"predict_proba_{dataset}")
                    if self.T.task.startswith("bin"):
                        y_pred = y_pred[:, 1]
                else:
                    y_pred = getattr(self, f"decision_function_{dataset}")
            else:
                y_pred = getattr(self, f"predict_{dataset}")

            scores[scorer.name] = scorer._sign * float(
                scorer._score_func(
                    getattr(self, f"y_{dataset}"), y_pred, **scorer._kwargs
                )
            )

            if self._run:  # Log metric to mlflow run
                MlflowClient().log_metric(
                    run_id=self._run.info.run_id,
                    key=scorer.name,
                    value=it(scores[scorer.name]),
                )

        return scores

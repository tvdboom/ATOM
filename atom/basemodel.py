# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the BaseModel class.

"""

# Standard packages
import pandas as pd
from typing import Optional, Union
from typeguard import typechecked
from sklearn.metrics import SCORERS, confusion_matrix

# Own modules
from .plots import BaseModelPlotter
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, CUSTOM_METRICS, METRIC_ACRONYMS,
    merge, arr, custom_transform, composed, crash, method_to_log,
)


class BaseModel(BaseModelPlotter):
    """Base class for all models.

    Parameters
    ----------
    T: class
        Trainer instance from which the model is called.

    acronym: str
        Model's acronym. Used to call the model from the trainer.
        If None, the predictor's __name__ is used (not recommended).

    fullname: str
        Full model's name. If None, the predictor's __name__ is used.

    needs_scaling: bool
        Whether the model needs scaled features. Can not be True for
        deep learning datasets.

    """

    def __init__(self, *args, **kwargs):
        self.T = args[0]  # Trainer instance
        self.__dict__.update(kwargs)
        self.name = self.acronym if len(args) == 1 else args[1]
        self.scaler = None
        self.estimator = None
        self.explainer = None  # Explainer object for shap plots
        self._group = self.name  # sh and ts models belong to the same group
        self._pred_attrs = [None] * 10

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
                "mean_bagging": getattr(self, "mean_bagging", None),
                "std_bagging": getattr(self, "std_bagging", None),
                "time_bagging": getattr(self, "time_bagging", None),
                "time": getattr(self, "time", None),
            },
            name=self.name,
        )

    # Prediction methods =========================================== >>

    def _prediction(self, X, y=None, sw=None, pl=None, vb=None, method="predict"):
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

        sw: sequence or None, optional (default=None)
            Sample weights for the score method.

        pl: bool, sequence or None, optional (default=None)
            Transformers to use on the data before predicting.
                - If None: Only transformers that are applied on the
                           whole dataset are used.
                - If False: Don't use any transformers.
                - If True: Use all transformers in the pipeline.
                - If sequence: Transformers to use, selected by their
                               index in the pipeline.

        vb: int or None, optional (default=None)
            Verbosity level for the transformers. If None, it uses the
            transformer's own verbosity.

        method: str, optional (default="predict")
            Prediction method to be applied to the estimator.

        Returns
        -------
        np.ndarray
            Return of the attribute.

        """
        if not hasattr(self.estimator, method):
            raise AttributeError(
                f"{self.estimator.__class__.__name__} doesn't have a {method} method!"
            )

        if pl is None:
            pl = [i for i, est in enumerate(self.branch.pipeline) if not est.train_only]
        elif pl is False:
            pl = []
        elif pl is True:
            pl = list(range(len(self.branch.pipeline)))

        # When there is a pipeline, apply transformations first
        for idx, est in self.branch.pipeline.iteritems():
            if idx in pl:
                X, y = custom_transform(self.T, est, self.branch, (X, y), vb)

        # Scale the data if needed
        if self.scaler:
            X = self.scaler.transform(X)

        if y is None:
            return getattr(self.estimator, method)(X)
        else:
            return getattr(self.estimator, method)(X, y, sw)

    @composed(crash, method_to_log, typechecked)
    def predict(
        self,
        X: X_TYPES,
        pipeline: Optional[Union[bool, SEQUENCE_TYPES]] = None,
        verbose: Optional[int] = None,
    ):
        """Get predictions on new data."""
        return self._prediction(X, pl=pipeline, vb=verbose, method="predict")

    @composed(crash, method_to_log, typechecked)
    def predict_proba(
        self,
        X: X_TYPES,
        pipeline: Optional[Union[bool, SEQUENCE_TYPES]] = None,
        verbose: Optional[int] = None,
    ):
        """Get probability predictions on new data."""
        return self._prediction(X, pl=pipeline, vb=verbose, method="predict_proba")

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(
            self,
            X: X_TYPES,
            pipeline: Optional[Union[bool, SEQUENCE_TYPES]] = None,
            verbose: Optional[int] = None,
    ):
        """Get log probability predictions on new data."""
        return self._prediction(X, pl=pipeline, vb=verbose, method="predict_log_proba")

    @composed(crash, method_to_log, typechecked)
    def decision_function(
            self,
            X: X_TYPES,
            pipeline: Optional[Union[bool, SEQUENCE_TYPES]] = None,
            verbose: Optional[int] = None,
    ):
        """Get the decision function on new data."""
        return self._prediction(X, pl=pipeline, vb=verbose, method="decision_function")

    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: X_TYPES,
        y: Y_TYPES,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        pipeline: Optional[Union[bool, SEQUENCE_TYPES]] = None,
        verbose: Optional[int] = None,
    ):
        """Get the score function on new data."""
        return self._prediction(X, y, sample_weight, pipeline, verbose, method="score")

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

    @property
    def y_test(self):
        return self.branch.y_test

    @property
    def shape(self):
        return self.branch.shape

    @property
    def columns(self):
        return self.branch.columns

    @property
    def n_columns(self):
        return self.branch.n_columns

    @property
    def features(self):
        return self.branch.features

    @property
    def n_features(self):
        return self.branch.n_features

    @property
    def target(self):
        return self.branch.target

    # Utility methods ============================================== >>

    @composed(crash, method_to_log)
    def delete(self):
        """Delete the model from the trainer."""
        self.T.delete(self.name)

    @composed(crash, method_to_log, typechecked)
    def scoring(self, metric: Optional[str] = None, dataset: str = "test", **kwargs):
        """Get the scoring for a specific metric.

        Parameters
        ----------
        metric: str, optional (default=None)
            Name of the metric to calculate. Choose from any of
            sklearn's SCORERS or one of the CUSTOM_METRICS. If None,
            returns the model's final results (ignores `dataset`).

        dataset: str, optional (default="test")
            Data set on which to calculate the metric. Options are
            "train" or "test".

        **kwargs
            Additional keyword arguments for the metric function.

        Returns
        -------
        score: float or np.ndarray
            Model's score for the selected metric.

        """
        metric_opts = CUSTOM_METRICS + list(SCORERS)

        # Check metric parameter
        if metric is None:
            return self._final_output()
        elif metric.lower() in METRIC_ACRONYMS:
            metric = METRIC_ACRONYMS[metric.lower()]
        elif metric.lower() not in metric_opts:
            raise ValueError(
                "Unknown value for the metric parameter, "
                f"got {metric}. Try one of {', '.join(metric_opts)}."
            )

        # Check set parameter
        dataset = dataset.lower()
        if dataset not in ("train", "test"):
            raise ValueError(
                "Unknown value for the dataset parameter. "
                "Choose between 'train' or 'test'."
            )

        if metric.lower() == "cm":
            return confusion_matrix(
                getattr(self, f"y_{dataset}"), getattr(self, f"predict_{dataset}")
            )
        elif metric.lower() == "tn":
            return int(self.scoring("cm", dataset).ravel()[0])
        elif metric.lower() == "fp":
            return int(self.scoring("cm", dataset).ravel()[1])
        elif metric.lower() == "fn":
            return int(self.scoring("cm", dataset).ravel()[2])
        elif metric.lower() == "tp":
            return int(self.scoring("cm", dataset).ravel()[3])
        elif metric.lower() == "lift":
            tn, fp, fn, tp = self.scoring("cm", dataset).ravel()
            return float((tp / (tp + fp)) / ((tp + fn) / (tp + tn + fp + fn)))
        elif metric.lower() == "fpr":
            tn, fp, _, _ = self.scoring("cm", dataset).ravel()
            return float(fp / (fp + tn))
        elif metric.lower() == "tpr":
            _, _, fn, tp = self.scoring("cm", dataset).ravel()
            return float(tp / (tp + fn))
        elif metric.lower() == "sup":
            tn, fp, fn, tp = self.scoring("cm", dataset).ravel()
            return float((tp + fp) / (tp + fp + fn + tn))

        # Calculate the scorer via _score_func to use the prediction properties
        scorer = SCORERS[metric]
        if type(scorer).__name__ == "_ThresholdScorer":
            if hasattr(self.estimator, "decision_function"):
                y_pred = getattr(self, f"decision_function_{dataset}")
            else:
                y_pred = getattr(self, f"predict_proba_{dataset}")
                if self.T.task.startswith("bin"):
                    y_pred = y_pred[:, 1]
        elif type(scorer).__name__ == "_ProbaScorer":
            if hasattr(self.estimator, "predict_proba"):
                y_pred = getattr(self, f"predict_proba_{dataset}")
                if self.T.task.startswith("bin"):
                    y_pred = y_pred[:, 1]
            else:
                y_pred = getattr(self, f"decision_function_{dataset}")
        else:
            y_pred = getattr(self, f"predict_{dataset}")

        return scorer._sign * float(
            scorer._score_func(
                getattr(self, f"y_{dataset}"), y_pred, **scorer._kwargs, **kwargs
            )
        )

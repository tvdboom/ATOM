# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the BaseModel class.

"""

# Standard packages
import pickle
import numpy as np
from typeguard import typechecked
from typing import Optional
from sklearn.metrics import SCORERS, confusion_matrix

# Own modules
from .plots import BaseModelPlotter
from .utils import (
    SEQUENCE_TYPES, X_TYPES, Y_TYPES, CUSTOM_METRICS, METRIC_ACRONYMS,
    merge, arr, check_scaling, catch_return, transform, composed,
    crash, method_to_log,
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

    type: str
        Model's type. Used to select shap's explainer for plotting.
        Options are: "linear", "tree" or "kernel".

    """

    def __init__(self, *args, **kwargs):
        self.T = args[0]  # Trainer instance
        self.__dict__.update(kwargs)
        self.name = self.acronym if len(args) == 1 else args[1]
        self.scaler = None
        self.estimator = None
        self._pred_attrs = [None] * 10

    def _get_default(self, x, params):
        """Get the standard parameter from params or the trainer.

        Parameters
        ----------
        x: list
            Standard parameter. Can be n_jobs and/or random_state.

        params: dict
            Parameters for the estimator provided by est_params.

        """
        args = []
        for i in x:
            args.append(params.pop(i) if params.get(i) else getattr(self.T, i))

        if len(args) == 1:
            return args[0]
        else:
            return args[0], args[1]

    # Prediction methods =========================================== >>

    def _prediction(self, X, y=None, sample_weight=None, method="predict", **kwargs):
        """Apply prediction methods on new data.

        First transform the new data and apply the attribute on the
        best model. The model needs to have the provided attribute.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features).

        y: int, str, sequence or None, optional (default=None)
            - If None: y is ignored.
            - If int: Index of the target column in X.
            - If str: Name of the target column in X.
            - Else: Target column with shape=(n_samples,).

        sample_weight: sequence or None, optional (default=None)
            Sample weights for the score method.

        method: str, optional (default="predict")
            Method of the model to be applied.

        **kwargs
            Keyword arguments for the transform method.

        Returns
        -------
        np.ndarray
            Return of the attribute.

        """
        if not hasattr(self.estimator, method):
            raise AttributeError(
                f"{self.estimator.__class__.__name__} doesn't have a {method} method!"
            )

        # When there is a pipeline, apply transformations first
        if not self.branch.pipeline.empty:
            if kwargs.get("verbose") is None:
                kwargs["verbose"] = self.T.verbose
            X, y = catch_return(transform(self.branch.pipeline, X, y, **kwargs))

        # Scale the data if needed
        if self.needs_scaling and not check_scaling(X):
            X = self.scaler.transform(X)

        if y is None:
            return getattr(self.estimator, method)(X)
        else:
            return getattr(self.estimator, method)(X, y, sample_weight)

    @composed(crash, method_to_log, typechecked)
    def predict(self, X: X_TYPES, **kwargs):
        """Get predictions on new data."""
        return self._prediction(X, method="predict", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_proba(self, X: X_TYPES, **kwargs):
        """Get probability predictions on new data."""
        return self._prediction(X, method="predict_proba", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def predict_log_proba(self, X: X_TYPES, **kwargs):
        """Get log probability predictions on new data."""
        return self._prediction(X, method="predict_log_proba", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def decision_function(self, X: X_TYPES, **kwargs):
        """Get the decision function on new data."""
        return self._prediction(X, method="decision_function", **kwargs)

    @composed(crash, method_to_log, typechecked)
    def score(
        self,
        X: X_TYPES,
        y: Y_TYPES,
        sample_weight: Optional[SEQUENCE_TYPES] = None,
        **kwargs,
    ):
        """Get the score function on new data."""
        return self._prediction(X, y, sample_weight, method="score", **kwargs)

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

    # Properties =================================================== >>

    @property
    def dataset(self):
        if self.scaler:
            return merge(self.scaler.transform(self.branch.X), self.y)
        else:
            return self.branch.data

    @property
    def train(self):
        if self.scaler:
            return merge(self.scaler.transform(self.branch.X_train), self.y_train)
        else:
            return self.branch.train

    @property
    def test(self):
        if self.scaler:
            return merge(self.scaler.transform(self.branch.X_test), self.y_test)
        else:
            return self.branch.test

    @property
    def X(self):
        if self.scaler:
            return self.scaler.transform(self.branch.X)
        else:
            return self.branch.X

    @property
    def y(self):
        return self.branch.y

    @property
    def X_train(self):
        if self.scaler:
            return self.scaler.transform(self.branch.X_train)
        else:
            return self.branch.X_train

    @property
    def X_test(self):
        if self.scaler:
            return self.scaler.transform(self.branch.X_test)
        else:
            return self.branch.X_test

    @property
    def y_train(self):
        return self.branch.y_train

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
    def features(self):
        return self.branch.features

    @property
    def target(self):
        return self.branch.target

    @property
    def classes(self):
        return self.branch.classes

    @property
    def n_classes(self):
        return self.branch.n_classes

    # Utility methods ============================================== >>

    @composed(crash, method_to_log)
    def delete(self):
        """Delete model from the trainer."""
        self.T.delete(self.name)

    @composed(crash, method_to_log, typechecked)
    def scoring(self, metric: Optional[str] = None, dataset: str = "test", **kwargs):
        """Get the scoring fora specific metric.

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

        try:
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
                if self.T.task.startswith("reg"):
                    y_pred = getattr(self, f"predict_{dataset}")
                elif hasattr(self.estimator, "decision_function"):
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

            return scorer._sign * float(scorer._score_func(
                getattr(self, f"y_{dataset}"), y_pred, **scorer._kwargs, **kwargs
            ))

        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid value for the metric parameter. Metric {metric} is "
                f"invalid for a {self.fullname} model with a {self.T.task} task!"
            )

    @composed(crash, method_to_log, typechecked)
    def save_estimator(self, filename: Optional[str] = None):
        """Save the estimator to a pickle file.

        Parameters
        ----------
        filename: str, optional (default=None)
            Name of the file. If None or "auto", the estimator's
            __name__ is used.

        """
        if not filename:
            filename = self.estimator.__class__.__name__
        elif filename == "auto" or filename.endswith("/auto"):
            filename = filename.replace("auto", self.estimator.__class__.__name__)

        with open(filename, "wb") as file:
            pickle.dump(self.estimator, file)
        self.T.log(f"{self.fullname} estimator saved successfully!", 1)

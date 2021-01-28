# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: tvdboom
Description: Module containing the Voting and Stacking classes.

"""

# Standard packages
import numpy as np
import pandas as pd
from copy import copy
from typing import Optional
from typeguard import typechecked

# Own modules
from .branch import Branch
from .basemodel import BaseModel
from .data_cleaning import Scaler
from .models import MODEL_LIST
from .utils import (
    flt, lst, arr, merge, check_is_fitted, check_scaling, get_acronym,
    get_best_score, catch_return, transform, composed, method_to_log,
    crash, CustomDict,
)


class Voting(BaseModel):
    """Class for voting with the models in the pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, acronym="Vote", fullname="Voting", **kwargs)
        self._models = CustomDict(
            {k: v for k, v in self.T._models.items() if v.name in self.models}
        )
        if len(self.models) < 2:
            raise ValueError(
                "Invalid value for the models parameter. A Voting "
                "class should contain at least two models."
            )

        if self.weights and len(self.models) != len(self.weights):
            raise ValueError(
                "Invalid value for the weights parameter. Length should "
                "be equal to the number of models, got len(models)="
                f"{len(self.models)} and len(weights)={len(self.weights)}."
            )

        # The Voting instance uses the data from the current branch
        # Note that using models from a different branch can cause
        # unexpected errors
        self.branch = self.T.branch
        self._train_idx = self.branch.idx[0]

    def __repr__(self):
        out = f"{self.fullname}"
        out += f"\n --> Models: {self.models}"
        out += f"\n --> Weights: {self.weights}"

        return out

    def _final_output(self):
        """Returns the average final output of the models."""

        def get_average_score(index):
            """Return the average score of the models on a metric."""
            scores = [get_best_score(m, index) for m in self._models]
            return np.average(scores, weights=self.weights)

        out = "   ".join([
            f"{m.name}: {round(get_average_score(i), 3)}"
            for i, m in enumerate(self.T._metric)
        ])
        return out

    @composed(crash, method_to_log, typechecked)
    def scoring(self, metric: Optional[str] = None, dataset: str = "test", **kwargs):
        """Get the average scoring of a specific metric.

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
        if metric is None:
            return self._final_output()

        pred = [m.scoring(metric, dataset, **kwargs) for m in self._models]
        return np.average(pred, axis=0, weights=self.weights)

    def _prediction(self, X, y=None, sample_weight=None, method="predict", **kwargs):
        """Get the mean of the prediction methods on new data.

        First transform the new data and apply the attribute on all
        the models. All models need to have the provided attribute.

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
        # Attribute check also done in BaseModel but better to
        # do it before all the data transformations
        for m in self._models:
            if not hasattr(m.estimator, method):
                raise AttributeError(
                    f"{m.estimator.__class__.__name__} doesn't have a {method} method!"
                )

        # Verbosity is set to trainer's verbosity if left to default
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = self.T.verbose

        # Apply transformations per branch. Shared transformations are
        # not repeated. A unique copy of the data is stored per branch
        data = {k: (copy(X), copy(y)) for k in self.T._branches}
        step = {}  # Current step in the pipeline per branch
        for b1, v1 in self.T._branches.items():
            if not v1.pipeline.empty:
                self.T.log(f"Transforming data for branch {b1}...", 1)

            for i, est1 in enumerate(v1.pipeline):
                # Skip if the transformation was already applied
                if step.get(b1, -1) < i:
                    kwargs["_one_trans"] = i
                    data[b1] = catch_return(transform(v1.pipeline, *data[b1], **kwargs))

                    for b2, v2 in self.T._branches.items():
                        try:  # Can fail if pipeline is shorter than i
                            if b1 != b2 and est1 is v2.pipeline.iloc[i]:
                                # Update the data and step for the other branch
                                data[b2] = copy(data[b1])
                                step[b2] = i
                        except IndexError:
                            continue

        # Use pipeline=[] to skip the transformations
        if method == "predict":
            pred = np.array([
                m.predict(data[m.branch.name][0], pipeline=[])
                for m in self._models
            ])
            majority = np.apply_along_axis(
                func1d=lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=0,
                arr=pred.astype("int")
            )
            return majority

        elif method in ("predict_proba", "predict_log_proba"):
            pred = np.array([
                getattr(m, method)(data[m.branch.name][0], pipeline=[])
                for m in self._models
            ])
            return np.average(pred, axis=0, weights=self.weights)

        else:
            pred = np.array([
                m.score(*data[m.branch.name], sample_weight, pipeline=[])
                for m in self._models
            ])
            return np.average(pred, axis=0, weights=self.weights)

    # Prediction properties ======================================== >>

    @property
    def metric_train(self):
        if self._pred_attrs[0] is None:
            pred = np.array([m.metric_train for m in self._models])
            self._pred_attrs[0] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[0]

    @property
    def metric_test(self):
        if self._pred_attrs[1] is None:
            pred = np.array([m.metric_test for m in self._models])
            self._pred_attrs[1] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[1]

    @property
    def predict_train(self):
        if self._pred_attrs[2] is None:
            pred = np.array([m.predict_train for m in self._models])
            self._pred_attrs[2] = np.apply_along_axis(
                func1d=lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=0,
                arr=pred.astype("int"),
            )
        return self._pred_attrs[2]

    @property
    def predict_test(self):
        if self._pred_attrs[3] is None:
            pred = np.array([m.predict_test for m in self._models])
            self._pred_attrs[3] = np.apply_along_axis(
                func1d=lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=0,
                arr=pred.astype("int"),
            )
        return self._pred_attrs[3]

    @property
    def predict_proba_train(self):
        if self._pred_attrs[4] is None:
            pred = np.array([m.predict_proba_train for m in self._models])
            self._pred_attrs[4] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[4]

    @property
    def predict_proba_test(self):
        if self._pred_attrs[5] is None:
            pred = np.array([m.predict_proba_test for m in self._models])
            self._pred_attrs[5] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[5]

    @property
    def predict_log_proba_train(self):
        if self._pred_attrs[6] is None:
            pred = np.array([m.predict_log_proba_train for m in self._models])
            self._pred_attrs[6] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[6]

    @property
    def predict_log_proba_test(self):
        if self._pred_attrs[7] is None:
            pred = np.array([m.predict_log_proba_test for m in self._models])
            self._pred_attrs[7] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[7]

    @property
    def score_train(self):
        if self._pred_attrs[8] is None:
            pred = np.array([m.score_train for m in self._models])
            self._pred_attrs[8] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[8]

    @property
    def score_test(self):
        if self._pred_attrs[9] is None:
            pred = np.array([m.score_test for m in self._models])
            self._pred_attrs[9] = np.average(pred, axis=0, weights=self.weights)
        return self._pred_attrs[9]


class Stacking(BaseModel):
    """Class for stacking the models in the pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, acronym="Stack", fullname="Stacking", **kwargs)
        self.estimator = kwargs["estimator"]
        self._models = CustomDict(
            {k: v for k, v in self.T._models.items() if v.name in self.models}
        )
        self._fxs_scaler = None

        if len(self.models) < 2:
            raise ValueError(
                "Invalid value for the models parameter. A Stacking "
                "class should contain at least two models."
            )

        # Create the dataset for the Stacking instance
        dataset = pd.DataFrame()
        for m in self._models:
            attr = self._get_stack_attr(m)
            pred = np.concatenate((
                getattr(m, f"{attr}_train"), getattr(m, f"{attr}_test")
            ))
            if attr == "predict_proba" and self.T.task.startswith("bin"):
                pred = pred[:, 1]

            if pred.ndim == 1:
                dataset[f"{attr}_{m.name}"] = pred
            else:
                for i in range(pred.shape[1]):
                    dataset[f"{attr}_{m.name}_{i}"] = pred[:, i]

        # Add scaled features from the branch to the set
        if self.passthrough:
            fxs = self.T.X
            if not check_scaling(fxs):
                self._fxs_scaler = Scaler().fit(self.T.X_train)
                fxs = self._fxs_scaler.transform(fxs)

            dataset = pd.concat([dataset, fxs], axis=1, join="inner")

        # The Stacking instance has his own internal branch
        self.branch = Branch(self, "StackBranch")
        self.branch.data = merge(dataset, self.T.y)
        self.branch.idx = self.T.branch.idx
        self._train_idx = self.branch.idx[0]

        if isinstance(self.estimator, str):
            # Add goal, n_jobs and random_state from trainer for the estimator
            self.goal = self.T.goal
            self.n_jobs = self.T.n_jobs
            self.random_state = self.T.random_state

            # Get one of ATOM's predefined models
            model = MODEL_LIST[get_acronym(self.estimator)](self)
            self.estimator = model.get_estimator()

        self.estimator.fit(self.X_train, self.y_train)

        # Save metric scores on complete training and test set
        self.metric_train = flt([
            metric(self.estimator, arr(self.X_train), self.y_train)
            for metric in self.T._metric
        ])
        self.metric_test = flt([
            metric(self.estimator, arr(self.X_test), self.y_test)
            for metric in self.T._metric
        ])

    def __repr__(self):
        out = f"{self.fullname}"
        out += f"\n --> Models: {self.models}"
        out += f"\n --> Estimator: {self.estimator.__class__.__name__}"
        out += f"\n --> Stack method: {self.stack_method}"
        out += f"\n --> Passthrough: {self.passthrough}"

        return out

    def _get_stack_attr(self, model):
        """Get the stack attribute for a specific model."""
        if self.stack_method == 'auto':
            if hasattr(model.estimator, "predict_proba"):
                return "predict_proba"
            elif hasattr(model.estimator, "decision_function"):
                return "decision_function"
            else:
                return "predict"

        return self.stack_method

    def _final_output(self):
        """Returns the output of the final estimator."""
        out = "   ".join([
            f"{m.name}: {round(lst(self.metric_test)[i], 3)}"
            for i, m in enumerate(self.T._metric)
        ])

        # Annotate if model overfitted when train 20% > test
        metric_train = lst(self.metric_train)
        metric_test = lst(self.metric_test)
        if metric_train[0] - 0.2 * metric_train[0] > metric_test[0]:
            out += " ~"

        return out

    # Prediction methods =========================================== >>

    def _prediction(self, X, y=None, sample_weight=None, method="predict", **kwargs):
        """Get the prediction methods on new data.

        First transform the new data and apply the attribute on all
        the models. All models need to have the provided attribute.

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
        # Attribute check also done in BaseModel but better to
        # do it before all the data transformations
        for m in self._models:
            if not hasattr(m.estimator, method):
                raise AttributeError(
                    f"{m.estimator.__class__.__name__} doesn't have a {method} method!"
                )

        # Verbosity is set to trainer's verbosity if left to default
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = self.T.verbose

        # Apply transformations per branch. Shared transformations are
        # not repeated. A unique copy of the data is stored per branch
        data = {k: (copy(X), copy(y)) for k in self.T._branches}
        step = {}  # Current step in the pipeline per branch
        for b1, v1 in self.T._branches.items():
            if not v1.pipeline.empty:
                self.T.log(f"Transforming data for branch {b1}...", 1)

            for i, est1 in enumerate(v1.pipeline):
                # Skip if the transformation was already applied
                if step.get(b1, -1) < i:
                    kwargs["_one_trans"] = i
                    data[b1] = catch_return(transform(v1.pipeline, *data[b1], **kwargs))

                    for b2, v2 in self.T._branches.items():
                        try:  # Can fail if pipeline is shorter than i
                            if b1 != b2 and est1 is v2.pipeline.iloc[i]:
                                # Update the data and step for the other branch
                                data[b2] = copy(data[b1])
                                step[b2] = i
                        except IndexError:
                            continue

        # Create the new dataset
        dataset = pd.DataFrame()
        for m in self._models:
            attr = self._get_stack_attr(m)
            pred = getattr(m, attr)(data[m.branch.name][0], pipeline=[])
            if attr == "predict_proba" and self.T.task.startswith("bin"):
                pred = pred[:, 1]

            if pred.ndim == 1:
                dataset[f"{attr}_{m.name}"] = pred
            else:
                for i in range(pred.shape[1]):
                    dataset[f"{attr}_{m.name}_{i}"] = pred[:, i]

        # Add scaled features from the branch to the set
        if self.passthrough:
            fxs = self.T.X
            if self._fxs_scaler:
                fxs = self._fxs_scaler.transform(fxs)

            dataset = pd.concat([dataset, fxs], axis=1, join="inner")

        if y is None:
            return getattr(self.estimator, method)(dataset)
        else:
            y = data[self._models[0].branch.name][1]
            return getattr(self.estimator, method)(dataset, y, sample_weight)

# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing utility classes.

"""

from __future__ import annotations

import os
import pprint
import sys
import tempfile
import warnings
from collections import deque
from collections.abc import MutableMapping
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime as dt
from functools import wraps
from importlib import import_module
from importlib.util import find_spec
from inspect import Parameter, signature
from itertools import cycle
from types import GeneratorType, MappingProxyType
from typing import Any, Callable
from unittest.mock import patch

import mlflow
import modin.pandas as md
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sps
from IPython.display import display
from joblib import Memory
from matplotlib.colors import to_rgba
from mlflow.models.signature import infer_signature
from optuna.study import Study
from optuna.trial import FrozenTrial
from pandas.api.types import is_numeric_dtype
from shap import Explainer, Explanation
from sklearn.metrics import (
    confusion_matrix, get_scorer, get_scorer_names, make_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.utils import _print_elapsed_time

from atom.utils.constants import __version__
from atom.utils.types import (
    BOOL, BRANCH, DATAFRAME, DATAFRAME_TYPES, ESTIMATOR, FEATURES, FLOAT,
    INDEX_SELECTOR, INT, INT_TYPES, MODEL, PANDAS, PANDAS_TYPES, PREDICTOR,
    SCALAR, SCORER, SEQUENCE, SEQUENCE_TYPES, SERIES, SERIES_TYPES, TARGET,
    TRANSFORMER, INDEX
)


# Classes ========================================================== >>

class NotFittedError(ValueError, AttributeError):
    """Exception called when the instance is not yet fitted.

    This class inherits from both ValueError and AttributeError to
    help with exception handling and backward compatibility.

    """


@dataclass
class DataConfig:
    """Stores the data configuration.

    This is a utility class to store the data configuration in one
    attribute and pass it down to the models. The default values are
    the one adopted by trainers.

    """
    index: INDEX_SELECTOR = True
    shuffle: bool = True
    stratify: INDEX_SELECTOR = True
    test_size: SCALAR = 0.2


@dataclass
class IndexConfig:
    """Stores the index configuration.

    This is a utility class to store the configuration of the indices
    in a branch.

    """
    train_idx: INDEX  # Indices in the train set
    test_idx: INDEX  # Indices in the test
    n_cols: INT  # Number of target columns


class PandasModin:
    """Utility class to select the right data engine.

    Returns pandas or modin depending on the env variable
    ATOM_DATA_ENGINE, which is set in BaseTransformer.py.

    """
    def __getattr__(self, item: str) -> Any:
        if os.environ.get("ATOM_DATA_ENGINE") == "modin":
            return getattr(md, item)
        else:
            return getattr(pd, item)


# This instance is used by ATOM to access the data engine
bk = PandasModin()


class CatBMetric:
    """Custom evaluation metric for the CatBoost model.

    Parameters
    ----------
    scorer: Scorer
        Scorer to evaluate. It's always the runner's main metric.

    task: str
        Model's task.

    """
    def __init__(self, scorer: SCORER, task: str):
        self.scorer = scorer
        self.task = task

    def get_final_error(self, error: FLOAT, weight: FLOAT) -> FLOAT:
        """Returns final value of metric based on error and weight.

        Can't be a `staticmethod` because of CatBoost's implementation.

        Parameters
        ----------
        error: float
            Sum of errors in all instances.

        weight: float
            Sum of weights of all instances.

        Returns
        -------
        float
            Metric value.

        """
        return error / (weight + 1e-38)

    @staticmethod
    def is_max_optimal() -> bool:
        """Returns whether great values of metric are better."""
        return True

    def evaluate(self, approxes: list, targets: list, weight: list) -> FLOAT:
        """Evaluates metric value.

        Parameters
        ----------
        approxes: list
            Vectors of approx labels.

        targets: list
            Vectors of true labels.

        weight: list
            Vectors of weights.

        Returns
        -------
        float
            Weighted errors.

        float
            Total weight.

        """
        if is_binary(self.task):
            # Convert CatBoost predictions to probabilities
            e = np.exp(approxes[0])
            y_pred = e / (1 + e)
            if self.scorer.__class__.__name__ == "_PredictScorer":
                y_pred = (y_pred > 0.5).astype(int)

        elif self.task.startswith("multiclass"):
            y_pred = np.array(approxes).T
            if self.scorer.__class__.__name__ == "_PredictScorer":
                y_pred = np.argmax(y_pred, axis=1)

        else:
            y_pred = approxes[0]

        kwargs = {}
        if "sample_weight" in sign(self.scorer._score_func):
            kwargs["sample_weight"] = weight

        score = self.scorer._score_func(targets, y_pred, **self.scorer._kwargs)

        return self.scorer._sign * score, 1.0


class LGBMetric:
    """Custom evaluation metric for the LightGBM model.

    Parameters
    ----------
    scorer: Scorer
        Scorer to evaluate. It's always the runner's main metric.

    task: str
        Model's task.

    """
    def __init__(self, scorer: SCORER, task: str):
        self.scorer = scorer
        self.task = task

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weight: np.ndarray,
    ) -> (str, FLOAT, bool):
        """Evaluates metric value.

        Parameters
        ----------
        y_true: np.array
            Vectors of approx labels.

        y_pred: np.array
            Vectors of true labels.

        weight: np.array
            Vectors of weights.

        Returns
        -------
        str
            Metric name.

        float
            Metric score.

        bool
            Whether higher is better.

        """
        if self.scorer.__class__.__name__ == "_PredictScorer":
            if is_binary(self.task):
                y_pred = (y_pred > 0.5).astype(int)
            elif self.task.startswith("multiclass"):
                y_pred = y_pred.reshape(len(np.unique(y_true)), len(y_true)).T
                y_pred = np.argmax(y_pred, axis=1)

        kwargs = {}
        if "sample_weight" in sign(self.scorer._score_func):
            kwargs["sample_weight"] = weight

        score = self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs, **kwargs)

        return self.scorer.name, self.scorer._sign * score, True


class XGBMetric:
    """Custom evaluation metric for the XGBoost model.

    Parameters
    ----------
    scorer: Scorer
        Scorer to evaluate. It's always the runner's main metric.

    task: str
        Model's task.

    """
    def __init__(self, scorer: SCORER, task: str):
        self.scorer = scorer
        self.task = task

    @property
    def __name__(self):
        return self.scorer.name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> FLOAT:
        if self.scorer.__class__.__name__ == "_PredictScorer":
            if is_binary(self.task):
                y_pred = (y_pred > 0.5).astype(int)
            elif self.task.startswith("multiclass"):
                y_pred = np.argmax(y_pred, axis=1)

        score = self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs)
        return -self.scorer._sign * score  # Negative because XGBoost minimizes


class Table:
    """Class to print nice tables per row.

    Parameters
    ----------
    headers: sequence
        Name of each column in the table. If an element is a tuple,
        the second element, should be the position of the text in the
        cell (left or right).

    spaces: sequence
        Width of each column. Should have the same length as `headers`.

    default_pos: str, default="right"
        Default position of the text in the cell.

    """

    def __init__(self, headers: SEQUENCE, spaces: SEQUENCE, default_pos: str = "right"):
        assert len(headers) == len(spaces)

        self.headers = []
        self.positions = []
        for header in headers:
            if isinstance(header, tuple):
                self.headers.append(header[0])
                self.positions.append(header[1])
            else:
                self.headers.append(header)
                self.positions.append(default_pos)

        self.spaces = spaces

    @staticmethod
    def to_cell(text: SCALAR | str, position: str, space: INT) -> str:
        """Get the string format for one cell.

        Parameters
        ----------
        text: int, float or str
            Value to add to the cell.

        position: str
            Position of text in cell. Choose from: right, left.

        space: int
            Maximum char length in the cell.

        Returns
        -------
        str
            Value to add to cell.

        """
        text = str(text)
        if len(text) > space:
            text = text[:space - 2] + ".."

        if position == "right":
            return text.rjust(space)
        else:
            return text.ljust(space)

    def print_header(self) -> str:
        """Print the header.

        Returns
        -------
        str
            New row with column names.

        """
        return self.print({k: k for k in self.headers})

    def print_line(self) -> str:
        """Print a line with dashes.

        Use this method after printing the header for a nice table
        structure.

        Returns
        -------
        str
            New row with dashes.

        """
        return self.print({k: "-" * s for k, s in zip(self.headers, self.spaces)})

    def print(self, sequence: dict) -> str:
        """Convert a sequence to a nice formatted table row.

        Parameters
        ----------
        sequence: dict
            Column names and value to add to row.

        Returns
        -------
        str
            New row with values.

        """
        out = []
        for header, pos, space in zip(self.headers, self.positions, self.spaces):
            out.append(self.to_cell(sequence.get(header, "---"), pos, space))

        return "| " + " | ".join(out) + " |"


class TrialsCallback:
    """Display the trials' overview as the study runs.

    Callback for the hyperparameter tuning study where a table with the
    trial's information is displayed. Additionally, the model's `trials`
    attribute is filled.

    Parameters
    ----------
    model: Model
        Model from which the study is created.

    n_jobs: int
        Number of parallel jobs for the study. If >1, no output is
        showed.

    """

    def __init__(self, model: MODEL, n_jobs: INT):
        self.T = model
        self.n_jobs = n_jobs

        if self.n_jobs == 1:
            self._table = self.create_table()
            self.T.log(self._table.print_header(), 2)
            self.T.log(self._table.print_line(), 2)

    def __call__(self, study: Study, trial: FrozenTrial):
        # The trial values are None when it fails
        if len(self.T._metric) == 1:
            score = [trial.value if trial.value is not None else np.NaN]
        else:
            score = trial.values or [np.NaN] * len(self.T._metric)

        if trial.state.name == "PRUNED" and self.T.acronym == "XGB":
            # XGBoost's eval_metric minimizes the function
            score = np.negative(score)

        params = self.T._trial_to_est(trial.params)
        estimator = trial.user_attrs.get("estimator", None)

        # Add row to the trials attribute
        time_trial = (dt.now() - trial.datetime_start).total_seconds()
        time_ht = self.T._trials["time_trial"].sum() + time_trial
        self.T._trials.loc[trial.number] = pd.Series(
            {
                "params": dict(params),  # To dict because of how pandas prints it
                "estimator": estimator,
                "score": flt(score),
                "time_trial": time_trial,
                "time_ht": time_ht,
                "state": trial.state.name,
            }
        )

        # Save trials to mlflow experiment as nested runs
        if self.T.experiment and self.T.log_ht:
            with mlflow.start_run(run_id=self.T._run.info.run_id):
                run_name = f"{self.T.name} - {trial.number}"
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.set_tags(
                        {
                            "name": self.T.name,
                            "model": self.T._fullname,
                            "branch": self.T.branch.name,
                            "trial_state": trial.state.name,
                            **self.T._ht["tags"],
                        }
                    )

                    # Mlflow only accepts params with char length <=250
                    pars = estimator.get_params() if estimator else params
                    mlflow.log_params(
                        {k: v for k, v in pars.items() if len(str(v)) <= 250}
                    )

                    mlflow.log_metric("time_trial", time_trial)
                    for i, name in enumerate(self.T._metric.keys()):
                        mlflow.log_metric(f"{name}_validation", score[i])

                    if estimator and self.T.log_model:
                        mlflow.sklearn.log_model(
                            sk_model=estimator,
                            artifact_path=estimator.__class__.__name__,
                            signature=infer_signature(
                                model_input=pd.DataFrame(self.T.X),
                                model_output=estimator.predict(self.T.X.iloc[[0], :]),
                            ),
                            input_example=pd.DataFrame(self.T.X.iloc[[0], :]),
                        )

        if self.n_jobs == 1:
            sequence = {"trial": trial.number, **trial.params}
            for i, m in enumerate(self.T._metric):
                best_score = rnd(np.nanmax([lst(s)[i] for s in self.T.trials["score"]]))
                sequence.update({m.name: rnd(score[i]), f"best_{m.name}": best_score})
            sequence["time_trial"] = time_to_str(time_trial)
            sequence["time_ht"] = time_to_str(time_ht)
            sequence["state"] = trial.state.name

            self.T.log(self._table.print(sequence), 2)

    def create_table(self) -> Table:
        """Create the trial table.

        Returns
        -------
        Table
            Object to display the trial overview.

        """
        headers = [("trial", "left")] + list(self.T._ht["distributions"])
        for m in self.T._metric:
            headers.extend([m.name, "best_" + m.name])
        headers.extend(["time_trial", "time_ht", "state"])

        # Define the width op every column in the table
        spaces = [len(str(headers[0][0]))]
        for name, dist in self.T._ht["distributions"].items():
            # If the distribution is categorical, take the mean of the widths
            # Else take the max of 7 (minimum width) and the width of the name
            if hasattr(dist, "choices"):
                options = np.mean([len(str(x)) for x in dist.choices], axis=0, dtype=int)
            else:
                options = 0

            spaces.append(max(7, len(name), options))

        spaces.extend(
            [
                max(7, len(column))
                for column in headers[1 + len(self.T._ht["distributions"]):-1]
            ]
        )

        return Table(headers, spaces + [8])


class PlotCallback:
    """Plot the hyperparameter tuning's progress as it runs.

    Creates a figure with two plots: the first plot shows the score
    of every trial and the second shows the distance between the scores
    of the last consecutive steps.

    Parameters
    ----------
    name: str
        Model's name.

    metric: list of str
        Name(s) of the metrics to plot.

    aesthetics: Aesthetics
        Properties that define the plot's aesthetics.

    """

    max_len = 15  # Maximum trials to show at once in the plot

    def __init__(self, name: str, metric: list[str], aesthetics: Any):
        self.y1 = {i: deque(maxlen=self.max_len) for i in range(len(metric))}
        self.y2 = {i: deque(maxlen=self.max_len) for i in range(len(metric))}

        traces = []
        colors = cycle(aesthetics.palette)
        for met in metric:
            color = next(colors)
            traces.extend(
                [
                    go.Scatter(
                        mode="lines+markers",
                        line=dict(width=aesthetics.line_width, color=color),
                        marker=dict(
                            symbol="circle",
                            size=aesthetics.marker_size,
                            line=dict(width=1, color="white"),
                            opacity=1,
                        ),
                        name=met,
                        legendgroup=met,
                        xaxis="x2",
                        yaxis="y1",
                    ),
                    go.Scatter(
                        mode="lines+markers",
                        line=dict(width=aesthetics.line_width, color=color),
                        marker=dict(
                            line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                            symbol="circle",
                            size=aesthetics.marker_size,
                            opacity=1,
                        ),
                        name=met,
                        legendgroup=met,
                        showlegend=False,
                        xaxis="x2",
                        yaxis="y2",
                    )
                ]
            )

        self.figure = go.FigureWidget(
            data=traces,
            layout=dict(
                xaxis1=dict(domain=(0, 1), anchor="y1", showticklabels=False),
                yaxis1=dict(
                    domain=(0.31, 1.0),
                    title=dict(text="Score", font_size=aesthetics.label_fontsize),
                    anchor="x1",
                ),
                xaxis2=dict(
                    domain=(0, 1),
                    title=dict(text="Trial", font_size=aesthetics.label_fontsize),
                    anchor="y2",
                ),
                yaxis2=dict(
                    domain=(0, 0.29),
                    title=dict(text="d", font_size=aesthetics.label_fontsize),
                    anchor="x2",
                ),
                title=dict(
                    text=f"Hyperparameter tuning for {name}",
                    x=0.5,
                    y=1,
                    pad=dict(t=15, b=15),
                    xanchor="center",
                    yanchor="top",
                    xref="paper",
                    font_size=aesthetics.title_fontsize,
                ),
                legend=dict(
                    x=0.99,
                    y=0.99,
                    xanchor="right",
                    yanchor="top",
                    font_size=aesthetics.label_fontsize,
                    bgcolor="rgba(255, 255, 255, 0.5)",
                ),
                hovermode="x unified",
                hoverlabel=dict(font_size=aesthetics.label_fontsize),
                font_size=aesthetics.tick_fontsize,
                margin=dict(l=0, b=0, r=0, t=25 + aesthetics.title_fontsize, pad=0),
                width=900,
                height=800,
            )
        )

        display(self.figure)

    def __call__(self, study: Study, trial: FrozenTrial):
        """Calculates new values for lines and plots them.

        Parameters
        ----------
        study: Study
            Current study.

        trial: FrozenTrial
            Finished trial.

        """
        x = range(x_min := max(0, trial.number - self.max_len), x_min + self.max_len)

        for i, score in enumerate(lst(trial.value or trial.values)):
            self.y1[i].append(score)
            if self.y2[i]:
                self.y2[i].append(abs(self.y1[i][-1] - self.y1[i][-2]))
            else:
                self.y2[i].append(None)

            # Update trace data
            self.figure.data[i * 2].x = list(x[:len(self.y1[i])])
            self.figure.data[i * 2].y = list(self.y1[i])
            self.figure.data[i * 2 + 1].x = list(x[:len(self.y1[i])])
            self.figure.data[i * 2 + 1].y = list(self.y2[i])


class ShapExplanation:
    """SHAP Explanation wrapper to avoid recalculating shap values.

    Calculating shap or interaction values is computationally expensive.
    This class 'remembers' all calculated values and reuses them when
    needed.

    Parameters
    ----------
    estimator: Predictor
        Estimator to get the shap values from.

    task: str
        Model's task.

    branch: Branch
        Data to get the shap values from.

    random_state: int or None, default=None
        Random seed for reproducibility.

    """

    def __init__(
        self,
        estimator: PREDICTOR,
        task: str,
        branch: BRANCH,
        random_state: INT | None = None,
    ):
        self.estimator = estimator
        self.task = task
        self.branch = branch
        self.random_state = random_state

        self._explainer = None
        self._explanation = None
        self._expected_value = None
        self._shap_values = pd.Series(dtype="object")
        self._interaction_values = pd.Series(dtype="object")

    @property
    def attr(self) -> str:
        """Get the model's main prediction method.

        Returns
        -------
        str
            Name of the prediction method.

        """
        for attr in ("predict_proba", "decision_function", "predict"):
            if hasattr(self.estimator, attr):
                return attr

    @property
    def explainer(self) -> Explainer:
        """Get shap's explainer.

        Returns
        -------
        Explainer
            Get the initialized explainer object.

        """
        if self._explainer is None:
            # Pass masker as np.array and feature names separately for modin frames
            kwargs = dict(
                masker=self.branch.X_train.to_numpy(),
                feature_names=list(self.branch.features),
                seed=self.random_state,
            )
            try:  # Fails when model does not fit standard explainers (e.g. ensembles)
                self._explainer = Explainer(self.estimator, **kwargs)
            except TypeError:
                # If method is provided as first arg, selects always Permutation
                self._explainer = Explainer(getattr(self.estimator, self.attr), **kwargs)

        return self._explainer

    def get_explanation(
        self,
        df: DATAFRAME,
        target: tuple,
    ) -> Explanation:
        """Get an Explanation object.

        Parameters
        ----------
        df: dataframe
            Data set to look at (subset of the complete dataset).

        target: tuple
            Indices of the target column and the class in the target
            column to look at.

        Returns
        -------
        shap.Explanation
            Object containing all information (values, base_values, data).

        """
        # Get rows that still need to be calculated
        calculate = df.loc[~df.index.isin(self._shap_values.index)]
        if not calculate.empty:
            kwargs = {}

            # Minimum of 2 * n_features + 1 evals required (default=500)
            if "max_evals" in sign(self.explainer.__call__):
                kwargs["max_evals"] = "auto"

            # Additivity check fails sometimes for no apparent reason
            if "check_additivity" in sign(self.explainer.__call__):
                kwargs["check_additivity"] = False

            with warnings.catch_warnings():
                # Avoid warning feature names mismatch in sklearn due to passing np.array
                warnings.filterwarnings("ignore", message="X does not have valid.*")

                # Calculate the new shap values
                try:
                    self._explanation = self.explainer(calculate.to_numpy(), **kwargs)
                except (ValueError, AssertionError):
                    raise ValueError(
                        "Failed to get shap's explainer for estimator "
                        f"{self.estimator} with task {self.task}."
                    )

            # Remember shap values in the _shap_values attribute
            values = pd.Series(
                data=list(self._explanation.values),
                index=calculate.index,
                dtype="object",
            )
            self._shap_values = pd.concat([self._shap_values, values])

        # Don't use attribute to not save plot-specific changes
        # Shallow copy to not copy the data in the branch
        explanation = copy(self._explanation)

        # Update the explanation object
        explanation.values = np.stack(self._shap_values.loc[df.index].values)
        explanation.base_values = self._explanation.base_values[0]
        explanation.data = self.branch.X.loc[df.index].to_numpy()

        if is_multioutput(self.task):
            if explanation.values.shape[-1] == self.branch.y.shape[1]:
                # One explanation per column
                explanation.values = explanation.values[:, :, target[0]]
                explanation.base_values = explanation.base_values[target[0]]
            else:
                # For native multilabel or multiclass-multioutput, the values
                # have shape[-1]=n_targets x max(n_cls)
                n_classes = explanation.values.shape[-1] // self.branch.y.shape[1]

                select = target[0] * n_classes + target[1]
                explanation.values = explanation.values[:, :, select]
                explanation.base_values = explanation.base_values[select]
        elif explanation.values.ndim > 2 and len(target) == 2:
            explanation.values = explanation.values[:, :, target[1]]
            explanation.base_values = explanation.base_values[target[1]]

        return explanation


class ClassMap:
    """List for classes with mapping from attribute.

    This class works similar to a list, where all elements should
    have a certain attribute. You can access the class by index or
    by attribute. The access is case-insensitive.

    """

    @staticmethod
    def _conv(key):
        return key.lower() if isinstance(key, str) else key

    def _get_data(self, key):
        if isinstance(key, INT_TYPES) and key not in self.keys():
            try:
                return self.__data[key]
            except IndexError:
                raise KeyError(key)
        else:
            for data in self.__data:
                if self._conv(getattr(data, self.__key)) == self._conv(key):
                    return data

        raise KeyError(key)

    def _check(self, elem):
        if not hasattr(elem, self.__key):
            raise ValueError(f"Element {elem} has no attribute {self.__key}.")
        else:
            return elem

    def __init__(self, *args, key: str = "name"):
        """Assign key and data.

        Mimics a list's initialization and accepts an extra argument
        to specify the attribute to use as key.

        """

        self.__key = key
        self.__data = []
        for elem in args:
            if isinstance(elem, GeneratorType):
                self.__data.extend(self._check(e) for e in elem)
            else:
                self.__data.append(self._check(elem))

    def __getitem__(self, key):
        if isinstance(key, SEQUENCE_TYPES):
            return self.__class__(*[self._get_data(k) for k in key], key=self.__key)
        elif isinstance(key, slice):
            return self.__class__(*self.__data[key], key=self.__key)
        else:
            return self._get_data(key)

    def __setitem__(self, key, value):
        if isinstance(key, INT_TYPES):
            self.__data[key] = self._check(value)
        else:
            try:
                self.__data = [e if self[key] == e else value for e in self.__data]
            except KeyError:
                assert key == getattr(value, self.__key)
                self.append(value)

    def __delitem__(self, key):
        del self.__data[self._get_data(key)]

    def __iter__(self):
        yield from self.__data

    def __len__(self):
        return len(self.__data)

    def __contains__(self, key):
        return key in self.__data or self._conv(key) in self.keys_lower()

    def __repr__(self):
        return self.__data.__repr__()

    def __reversed__(self):
        yield from reversed(list(self.__data))

    def __eq__(self, other):
        return self.__data == other

    def __add__(self, other):
        self.__data += other
        return self

    def __bool__(self):
        return bool(self.__data)

    def keys(self) -> list:
        return [getattr(x, self.__key) for x in self.__data]

    def keys_lower(self) -> list:
        return list(map(self._conv, self.keys()))

    def append(self, value):
        self.__data.append(self._check(value))

    def extend(self, value):
        self.__data.extend(list(map(self._check, value)))

    def remove(self, value):
        if value in self.__data:
            self.__data.remove(value)
        else:
            self.__data.remove(self._get_data(value))

    def clear(self):
        self.__data = []

    def index(self, key):
        if key in self.__data:
            return self.__data.index(key)
        else:
            return self.__data.index(self._get_data(key))


class CustomDict(MutableMapping):
    """Custom ordered dictionary.

    The main differences with the Python dictionary are:

    - It has ordered entries.
    - Key requests are case-insensitive.
    - Returns a subset of itself using getitem with a list of keys or slice.
    - It allows getting an item from an index position.
    - It can insert key value pairs at a specific position.
    - Replace method to change a key or value if key exists.

    """

    @staticmethod
    def _conv(key):
        return key.lower() if isinstance(key, str) else key

    def _get_key(self, key):
        # Get key from index
        if isinstance(key, INT_TYPES) and key not in self.__keys:
            return self.__keys[key]
        else:
            # Get key from name
            for k in self.__keys:
                if self._conv(k) == self._conv(key):
                    return k

        raise KeyError(key)

    def __init__(self, iterable_or_mapping=None, **kwargs):
        """Creates keys and data.

        Mimics a dictionary's initialization and accepts the same
        arguments. You have to pass an ordered iterable or mapping
        unless you want the order to be arbitrary.

        """
        self.__keys = []  # States the order
        self.__data = {}  # Contains the values

        if iterable_or_mapping is not None:
            try:
                iterable = iterable_or_mapping.items()
            except AttributeError:
                iterable = iterable_or_mapping

            for key, value in iterable:
                self.__keys.append(key)
                self.__data[self._conv(key)] = value

        for key, value in kwargs.items():
            self.__keys.append(key)
            self.__data[self._conv(key)] = value

    def __getitem__(self, key):
        if isinstance(key, SEQUENCE_TYPES):
            return self.__class__({self._get_key(k): self[k] for k in key})
        elif isinstance(key, slice):
            return self.__class__({k: self[k] for k in self.__keys[key]})
        else:
            return self.__data[self._conv(self._get_key(key))]

    def __setitem__(self, key, value):
        if key not in self:
            self.__keys.append(key)
        self.__data[self._conv(key)] = value

    def __delitem__(self, key):
        key = self._get_key(key)
        self.__keys.remove(key)
        del self.__data[self._conv(key)]

    def __iter__(self):
        yield from self.__keys

    def __len__(self):
        return len(self.__keys)

    def __contains__(self, key):
        return self._conv(key) in self.__data

    def __repr__(self):
        return pprint.pformat(dict(self), sort_dicts=False)

    def __reversed__(self):
        yield from reversed(list(self.keys()))

    def __bool__(self):
        return bool(self.__keys)

    def keys(self):
        yield from self.__keys

    def items(self):
        for key in self.__keys:
            yield key, self.__data[self._conv(key)]

    def values(self):
        for key in self.__keys:
            yield self.__data[self._conv(key)]

    def insert(self, pos, new_key, value):
        # If key already exists, remove old first
        if new_key in self:
            self.__delitem__(new_key)
        self.__keys.insert(pos, new_key)
        self.__data[self._conv(new_key)] = value

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def pop(self, key, default=None):
        if key in self:
            value = self[key]
            self.__delitem__(key)
            return value
        else:
            return default

    def popitem(self):
        try:
            return self.__data.pop(self._conv(self.__keys.pop()))
        except IndexError:
            raise KeyError(f"{self.__class__.__name__} is empty.")

    def clear(self):
        self.__keys = []
        self.__data = {}

    def update(self, iterable_or_mapping=None, **kwargs):
        if iterable_or_mapping is not None:
            try:
                iterable = iterable_or_mapping.items()
            except AttributeError:
                iterable = iterable_or_mapping

            for key, value in iterable:
                self[key] = value

        for key, value in kwargs.items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def copy(self):
        return copy(self)

    def index(self, key):
        return self.__keys.index(self._get_key(key))

    def reorder(self, keys):
        self.__keys = [k for k in keys if k in self.__keys]
        self.__data = {k: self[k] for k in self.__keys}

    def replace_key(self, key, new_key):
        if key in self:
            self.insert(self.__keys.index(self._get_key(key)), new_key, self[key])
            self.__delitem__(key)

    def replace_value(self, key, value=None):
        if key in self:
            self[key] = value


# Functions ======================================================== >>

def flt(x: Any) -> Any:
    """Return item from sequence with just that item.

    Parameters
    ----------
    x: Any
        Item or sequence.

    Returns
    -------
    Any
        Object.

    """
    return x[0] if isinstance(x, SEQUENCE_TYPES) and len(x) == 1 else x


def lst(x: Any) -> SEQUENCE:
    """Make a sequence from an item if not a sequence already.

    Parameters
    ----------
    x: Any
        Item or sequence.

    Returns
    -------
    sequence
        Item as sequence with length 1 or provided sequence.

    """
    return x if isinstance(x, (dict, CustomDict, ClassMap, *SEQUENCE_TYPES)) else [x]


def it(x: Any) -> Any:
    """Convert rounded floats to int.

    If the provided item is not numerical, return as is.

    Parameters
    ----------
    x: Any
        Item to check for rounding.

    Returns
    -------
    Any
        Rounded float or non-numerical item.

    """
    try:
        is_equal = int(x) == float(x)
    except ValueError:  # Item may not be numerical
        return x

    return int(x) if is_equal else float(x)


def rnd(x: Any, decimals: INT = 4) -> Any:
    """Round a float to the `decimals` position.

    If the value is not a float, return as is.

    Parameters
    ----------
    x: Any
        Numerical item to round.

    decimals: int, default=4
        Decimal position to round to.

    Returns
    -------
    Any
        Rounded float or non-numerical item.

    """
    return round(x, decimals) if np.issubdtype(type(x), np.floating) else x


def divide(a: SCALAR, b: SCALAR) -> SCALAR:
    """Divide two numbers and return 0 if division by zero.

    If the value is not a float, return as is.

    Parameters
    ----------
    a: int or float
        Numerator.

    b: int or float
        Denominator.

    Returns
    -------
    int or float
        Result of division or 0.

    """
    return np.divide(a, b) if b != 0 else 0


def to_rgb(c: str) -> str:
    """Convert a color name or hex to rgb.

    Parameters
    ----------
    c: str
        Color name or code.

    Returns
    -------
    str
        Color's RGB representation.

    """
    if not c.startswith("rgb"):
        colors = to_rgba(c)[:3]
        return f"rgb({colors[0]}, {colors[1]}, {colors[2]})"

    return c


def sign(obj: Callable) -> MappingProxyType:
    """Get the parameters of an object.

    Parameters
    ----------
    obj: Callable
        Object from which to get the parameters.

    Returns
    -------
    mappingproxy
        Object's parameters.

    """
    return signature(obj).parameters


def merge(*args) -> DATAFRAME:
    """Concatenate pandas objects column-wise.

    None and empty objects are ignored.

    Parameters
    ----------
    *args
        Objects to concatenate.

    Returns
    -------
    dataframe
        Concatenated dataframe.

    """
    if len(args := [x for x in args if x is not None and not x.empty]) == 1:
        return bk.DataFrame(args[0])
    else:
        return bk.DataFrame(bk.concat([*args], axis=1))


def get_cols(elem: PANDAS) -> list[SERIES]:
    """Get a list of columns in dataframe / series.

    Parameters
    ----------
    elem: series or dataframe
        Element to get the columns from.

    Returns
    -------
    list of series
        Columns in elem.

    """
    if isinstance(elem, SERIES_TYPES):
        return [elem]
    else:
        return [elem[col] for col in elem]


def variable_return(
    X: DATAFRAME | None,
    y: SERIES | None,
) -> DATAFRAME | SERIES | tuple[DATAFRAME, PANDAS]:
    """Return one or two arguments depending on which is None.

    This utility is used to make methods return only the provided
    data set.

    Parameters
    ----------
    X: dataframe or None
        Feature set.

    y: series, dataframe or None
        Target column.

    Returns
    -------
    dataframe, series or tuple
        Data sets that are not None.

    """
    if y is None:
        return X
    elif X is None:
        return y
    else:
        return X, y


def is_sparse(obj: PANDAS) -> bool:
    """Check if the dataframe is sparse.

    A data set is considered sparse if any of its columns is sparse.

    Parameters
    ----------
    obj: series or dataframe
        Data set to check.

    Returns
    -------
    bool
        Whether the data set is sparse.

    """
    return any(pd.api.types.is_sparse(col) for col in get_cols(obj))


def check_empty(obj: PANDAS) -> PANDAS | None:
    """Check if a pandas object is empty.

    Parameters
    ----------
    obj: series or dataframe
        Pandas object to check.

    Returns
    -------
    series, dataframe or None
        Same object or None if empty.

    """
    return obj if not obj.empty else None


def check_dependency(name: str):
    """Raise an error if a package is not installed.

    Parameters
    ----------
    name: str
        Name of the package to check.

    """
    if not find_spec(name.replace("-", "_")):
        raise ModuleNotFoundError(
            f"Unable to import the {name} package. Install it using "
            f"`pip install {name}` or install all of atom's optional "
            "dependencies with `pip install atom-ml[full]`."
        )


def check_canvas(is_canvas: bool, method: str):
    """Raise an error if a model doesn't have a `predict_proba` method.

    Parameters
    ----------
    is_canvas: bool
        Whether the plot is in a canvas. If True, an error is raised.

    method: str
        Name of the method from which the check is called.

    """
    if is_canvas:
        raise PermissionError(
            f"The {method} method can not be called from a "
            "canvas because it uses the matplotlib backend."
        )


def check_hyperparams(models: MODEL | SEQUENCE, method: str) -> list[MODEL]:
    """Check if the models ran hyperparameter tuning.

    If no models did, raise an exception.

    Parameters
    ----------
    models: model or sequence
        Models to check.

    method: str
        Name of the method from which the check is called.

    Returns
    -------
    list of models
        Models that ran hyperparameter tuning.

    """
    if not (models := list(filter(lambda x: x.trials is not None, lst(models)))):
        raise ValueError(
            f"The {method} method is only available "
            "for models that ran hyperparameter tuning."
        )

    return models


def check_predict_proba(models: MODEL | SEQUENCE, method: str):
    """Raise an error if a model doesn't have a `predict_proba` method.

    Parameters
    ----------
    models: model or sequence of models
        Models to check for the attribute.

    method: str
        Name of the method from which the check is called.

    """
    for m in lst(models):
        if not hasattr(m.estimator, "predict_proba"):
            raise AttributeError(
                f"The {method} method is only available for "
                f"models with a predict_proba method, got {m.name}."
            )


def check_scaling(X: PANDAS, pipeline: Any | None = None) -> bool:
    """Check if the data is scaled.

    A data set is considered scaled when the mean of the mean of
    all columns lies between -0.05 and 0.05 and the mean of the
    standard deviation of all columns lies between 0.85 and 1.15.
    Binary columns are excluded from the calculation.

    Additionally, if a pipeline is provided and there's a scaler in
    the pipeline, it also returns False.

    Parameters
    ----------
    X: series or dataframe
        Data set to check.

    pipeline: Pipeline or None, default=None
        Pipeline in which to check for a scaler (any estimator whose
        name contains the word scaler).

    Returns
    -------
    bool
        Whether the data set is scaled.

    """
    has_scaler = False
    if pipeline is not None:
        est_names = [est.__class__.__name__.lower() for est in pipeline]
        has_scaler = any("scaler" in name for name in est_names)

    df = to_df(X)  # Convert to dataframe

    # Remove binary columns (thus also sparse columns)
    df = df[[c for c in df if ~np.isin(df[c].unique(), [0, 1]).all()]]

    if df.empty:  # All columns are binary -> no scaling needed
        return True
    else:
        mean = df.mean(numeric_only=True).mean()
        std = df.std(numeric_only=True).mean()
        return has_scaler or (-0.05 < mean < 0.05 and 0.85 < std < 1.15)


@contextmanager
def keep_attrs(estimator: ESTIMATOR):
    """Contextmanager to save an estimator's custom attributes.

    ATOM's pipeline uses two custom attributes for its transformers:
    _train_only, and _cols. Since some transformers reset their
    attributes during fit (like those from sktime), we wrap the fit
    method in a contextmanager that saves and restores the attrs.

    """
    try:
        train_only = getattr(estimator, "_train_only", None)
        cols = getattr(estimator, "_cols", None)
        yield
    finally:
        if train_only is not None:
            estimator._train_only = train_only
        if cols is not None:
            estimator._cols = cols


def get_versions(models: ClassMap) -> dict:
    """Get the versions of ATOM and the models' packages.

    Parameters
    ----------
    models: ClassMap
        Models for which to check the version.

    Returns
    -------
    dict
        Current versions of ATOM and models.

    """
    versions = {"atom": __version__}
    for model in models:
        module = model._est_class.__module__.split(".")[0]
        versions[module] = sys.modules.get(module, import_module(module)).__version__

    return versions


def get_corpus(df: DATAFRAME) -> SERIES:
    """Get text column from a dataframe.

    The text column should be called `corpus` (case-insensitive).

    Parameters
    ----------
    df: dataframe
        Data set from which to get the corpus.

    Returns
    -------
    series
        Column with text values.

    """
    try:
        return next(col for col in df if col.lower() == "corpus")
    except StopIteration:
        raise ValueError("The provided dataset does not contain a text corpus!")


def get_best_score(item: MODEL | SERIES, metric: int = 0) -> FLOAT:
    """Returns the best score for a model.

    The best score is the `score_bootstrap` or `score_test`, checked
    in that order.

    Parameters
    ----------
    item: model or series
        Model or row from the results dataframe to get the score from.

    metric: int, default=0
        Index of the metric to use (for multi-metric runs).

    Returns
    -------
    float
        Best score.

    """
    if item.score_bootstrap:
        return lst(item.score_bootstrap)[metric]
    else:
        return lst(item.score_test)[metric]


def time_to_str(t: SCALAR) -> str:
    """Convert time to a nice string representation.

    The resulting string is of format 00h:00m:00s or 1.000s if
    under 1 min.

    Parameters
    ----------
    t: int or float
        Time to convert (in seconds).

    Returns
    -------
    str
        Time representation.

    """
    h = int(t) // 3600
    m = int(t) % 3600 // 60
    s = t % 3600 % 60
    if not h and not m:  # Only seconds
        return f"{s:.3f}s"
    elif not h:  # Also minutes
        return f"{m:02.0f}m:{s:02.0f}s"
    else:  # Also hours
        return f"{h:02.0f}h:{m:02.0f}m:{s:02.0f}s"


def n_cols(data: FEATURES | TARGET | None) -> int:
    """Get the number of columns in a dataset.

    Parameters
    ----------
    data: sequence or dataframe-like
        Dataset to check.

    Returns
    -------
    int or None
        Number of columns.

    """
    if data is not None:
        if (array := np.array(data, dtype="object")).ndim > 1:
            return array.shape[1]
        else:
            return array.ndim  # Can be 0 when input is a dict


def to_pyarrow(column: SERIES, inverse: bool = False) -> str:
    """Get the pyarrow dtype corresponding to a series.

    Parameters
    ----------
    column: series
        Column to get the dtype from. If it already has a pyarrow
        dtype, return original dtype.

    inverse: bool, default=False
        Whether to convert to pyarrow or back from pyarrow.

    Returns
    -------
    str
        Name of the converted dtype.

    """
    if not inverse and not column.dtype.name.endswith("[pyarrow]"):
        if column.dtype.name == "object":
            return "string[pyarrow]"  # pyarrow doesn't support object
        else:
            return f"{column.dtype.name}[pyarrow]"
    elif inverse and column.dtype.name.endswith("[pyarrow]"):
        return column.dtype.name[:-9]

    return column.dtype.name


def to_df(
    data: FEATURES | None,
    index: SEQUENCE | None = None,
    columns: SEQUENCE | None = None,
    dtype: str | dict | np.dtype | None = None,
) -> DATAFRAME | None:
    """Convert a dataset to a dataframe.

    Parameters
    ----------
    data: dataframe-like or None
        Dataset to convert to a dataframe.  If None or already a
        dataframe, return unchanged.

    index: sequence, index or None, default=None
        Values for the index.

    columns: sequence or None, default=None
        Name of the columns. Use None for automatic naming.

    dtype: str, dict, np.dtype or None, default=None
        Data types for the output columns. If None, the types are
        inferred from the data.

    Returns
    -------
    dataframe or None
        Dataset as dataframe of type given by the backend.

    """
    if data is not None:
        if not isinstance(data, bk.DataFrame):
            # Assign default column names (dict already has column names)
            if not isinstance(data, (dict, PANDAS_TYPES)) and columns is None:
                columns = [f"x{str(i)}" for i in range(n_cols(data))]

            if hasattr(data, "to_pandas") and bk.__name__ == "pandas":
                data = data.to_pandas()  # Convert cuML to pandas
            elif sps.issparse(data):
                # Create dataframe from sparse matrix
                data = pd.DataFrame.sparse.from_spmatrix(data, index, columns)
            else:
                data = pd.DataFrame(data, index, columns)

            if dtype is not None:
                data = data.astype(dtype)

        if os.environ.get("ATOM_DATA_ENGINE") == "pyarrow":
            data = data.astype({name: to_pyarrow(col) for name, col in data.items()})

    return data


def to_series(
    data: SEQUENCE | None,
    index: SEQUENCE | None = None,
    name: str = "target",
    dtype: str | np.dtype | None = None,
) -> SERIES | None:
    """Convert a sequence to a series.

    Parameters
    ----------
    data: sequence or None
        Data to convert. If None, return unchanged.

    index: sequence, index or None, default=None
        Values for the index.

    name: str, default="target"
        Name of the series.

    dtype: str, np.dtype or None, default=None
        Data type for the output series. If None, the type is
        inferred from the data.

    Returns
    -------
    series or None
        Sequence as series of type given by the backend.

    """
    if data is not None:
        if not isinstance(data, bk.Series):
            if hasattr(data, "to_pandas") and bk.__name__ == "pandas":
                data = data.to_pandas()  # Convert cuML to pandas
            else:
                # Flatten for arrays with shape (n_samples, 1), sometimes returned by cuML
                data = pd.Series(
                    data=np.array(data, dtype="object").ravel().tolist(),
                    index=index,
                    name=getattr(data, "name", name),
                    dtype=dtype,
                )

        if os.environ.get("ATOM_DATA_ENGINE") == "pyarrow":
            data = data.astype(to_pyarrow(data))

    return data


def to_pandas(
    data: SEQUENCE | None,
    index: SEQUENCE | None = None,
    columns: SEQUENCE | None = None,
    name: str = "target",
    dtype: str | dict | np.dtype | None = None,
) -> PANDAS | None:
    """Convert a sequence or dataset to a dataframe or series object.

    If the data is 1-dimensional, convert to series, else to a dataframe.

    Parameters
    ----------
    data: sequence or None
        Data to convert. If None, return unchanged.

    index: sequence, index or None, default=None
        Values for the index.

    columns: sequence or None, default=None
        Name of the columns. Use None for automatic naming.

    name: str, default="target"
        Name of the series.

    dtype: str, dict, np.dtype or None, default=None
        Data type for the output series. If None, the type is
        inferred from the data.

    Returns
    -------
    series, dataframe or None
        Data as pandas object.

    """
    if n_cols(data) == 1:
        return to_series(data, index=index, name=name, dtype=dtype)
    else:
        return to_df(data, index=index, columns=columns, dtype=dtype)


def check_is_fitted(
    estimator: ESTIMATOR,
    exception: bool = True,
    attributes: str | SEQUENCE | None = None,
) -> bool:
    """Check whether an estimator is fitted.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (not None or empty). Otherwise, it raises a
    NotFittedError. Extension on sklearn's function that accounts
    for empty dataframes and series and returns a boolean.

    Parameters
    ----------
    estimator: Estimator
        Instance to check.

    exception: bool, default=True
        Whether to raise an exception if the estimator is not fitted.
        If False, it returns False instead.

    attributes: str, sequence or None, default=None
        Attribute(s) to check. If None, the estimator is considered
        fitted if there exist an attribute that ends with a underscore
        and does not start with double underscore.

    Returns
    -------
    bool
        Whether the estimator is fitted.

    """

    def check_attr(attr: str) -> bool:
        """Return whether an attribute is False or empty.

        Parameters
        ----------
        attr: str
            Name of the attribute to check.

        Returns
        -------
        bool
            Whether the attribute's value is False.

        """
        if attr:
            if isinstance(value := getattr(estimator, attr), PANDAS_TYPES):
                return value.empty
            else:
                return not value

    is_fitted = False
    if hasattr(estimator, "_is_fitted"):
        is_fitted = estimator._is_fitted
    elif attributes is None:
        # Check for attributes from a fitted object
        for k, v in vars(estimator).items():
            if k.endswith("_") and not k.startswith("__") and v is not None:
                is_fitted = True
                break
    elif not all(check_attr(attr) for attr in lst(attributes)):
        is_fitted = True

    if not is_fitted:
        if exception:
            raise NotFittedError(
                f"This {type(estimator).__name__} instance is not fitted yet. "
                f"Call {'run' if hasattr(estimator, 'run') else 'fit'} with "
                "appropriate arguments before using this estimator."
            )
        else:
            return False

    return True


def get_custom_scorer(metric: str | Callable | SCORER) -> SCORER:
    """Get a scorer from a str, func or scorer.

    Scorers used by ATOM have a name attribute.

    Parameters
    ----------
    metric: str, func or scorer
        Name, function or scorer to get the scorer from. If it's a
        function, the scorer is created using the default parameters
        of sklearn's `make_scorer`.

    Returns
    -------
    scorer
        Custom sklearn scorer with name attribute.

    """
    if isinstance(metric, str):
        custom_acronyms = dict(
            ap="average_precision",
            ba="balanced_accuracy",
            auc="roc_auc",
            logloss="neg_log_loss",
            ev="explained_variance",
            me="max_error",
            mae="neg_mean_absolute_error",
            mse="neg_mean_squared_error",
            rmse="neg_root_mean_squared_error",
            msle="neg_mean_squared_log_error",
            mape="neg_mean_absolute_percentage_error",
            medae="neg_median_absolute_error",
            poisson="neg_mean_poisson_deviance",
            gamma="neg_mean_gamma_deviance",
        )

        custom_scorers = dict(
            tn=true_negatives,
            fp=false_positives,
            fn=false_negatives,
            tp=true_positives,
            fpr=false_positive_rate,
            tpr=true_positive_rate,
            tnr=true_negative_rate,
            fnr=false_negative_rate,
            mcc=matthews_corrcoef,
        )

        metric = metric.lower()
        if metric in get_scorer_names():
            scorer = get_scorer(metric)
            scorer.name = metric
        elif metric in custom_acronyms:
            scorer = get_scorer(custom_acronyms[metric])
            scorer.name = custom_acronyms[metric]
        elif metric in custom_scorers:
            scorer = make_scorer(custom_scorers[metric])
        else:
            raise ValueError(
                f"Unknown value for the metric parameter, got {metric}. "
                f"Choose from: {', '.join(get_scorer_names())}."
            )

    elif hasattr(metric, "_score_func"):  # Scoring is a scorer
        scorer = copy(metric)

        # Some scorers use default kwargs
        default_kwargs = ("precision", "recall", "f1", "jaccard")
        if any(scorer._score_func.__name__.startswith(name) for name in default_kwargs):
            if not scorer._kwargs:
                scorer._kwargs = {"average": "binary"}

        for key in get_scorer_names():
            if scorer.__dict__ == get_scorer(key).__dict__:
                scorer.name = key
                break

    else:  # Scoring is a function with signature metric(y_true, y_pred)
        scorer = make_scorer(score_func=metric)

    # If no name was assigned, use the name of the function
    if not hasattr(scorer, "name"):
        scorer.name = scorer._score_func.__name__

    return scorer


def infer_task(y: PANDAS, goal: str = "class") -> str:
    """Infer the task corresponding to a target column.

    Parameters
    ----------
    y: series or dataframe
        Target column(s).

    goal: str, default="class"
        Classification or regression goal.

    Returns
    -------
    str
        Inferred task.

    """
    if goal == "reg":
        if y.ndim == 1:
            return "regression"
        else:
            return "multioutput regression"
    elif goal == "fc":
        if y.ndim == 1:
            return "univariate forecast"
        else:
            return "multivariate forecast"

    if y.ndim > 1:
        if all(y[col].nunique() == 2 for col in y):
            return "multilabel classification"
        else:
            return "multiclass-multioutput classification"
    elif isinstance(y.iloc[0], SEQUENCE_TYPES):
        return "multilabel classification"
    elif y.nunique() == 1:
        raise ValueError(f"Only found 1 target value: {y.unique()[0]}")
    elif y.nunique() == 2:
        return "binary classification"
    else:
        return "multiclass classification"


def is_binary(task) -> bool:
    """Return whether the task is binary or multilabel."""
    return task in ("binary classification", "multilabel classification")


def is_multioutput(task) -> bool:
    """Return whether the task is binary or multilabel."""
    return any(t in task for t in ("multilabel", "multioutput", "multivariate"))


def get_feature_importance(
    est: PREDICTOR,
    attributes: SEQUENCE | None = None,
) -> np.ndarray | None:
    """Return the feature importance from an estimator.

    Gets the feature importance from the provided attribute. For
    meta-estimators, it gets the mean of the values of the underlying
    estimators.

    Parameters
    ----------
    est: Predictor
        Instance from which to get the feature importance.

    attributes: sequence or None, default=None
        Attributes to get, in order of importance. If None, it
        uses scores_ > coef_ > feature_importances_.

    Returns
    -------
    np.array or None
        Estimator's feature importances.

    """
    norm = lambda x: np.linalg.norm(x, axis=np.argmin(x.shape), ord=1)

    data = None
    if not attributes:
        attributes = ("scores_", "coef_", "feature_importances_")

    try:
        data = getattr(est, next(x for x in attributes if hasattr(est, x)))
    except StopIteration:
        # Get the mean value for meta-estimators
        if hasattr(est, "estimators_"):
            if all(hasattr(x, "feature_importances_") for x in est.estimators_):
                data = [fi.feature_importances_ for fi in est.estimators_]
            elif all(hasattr(x, "coef_") for x in est.estimators_):
                data = [norm(fi.coef_) for fi in est.estimators_]
            else:
                # For ensembles that mix attributes
                raise ValueError(
                    "Failed to calculate the feature importance for meta-estimator "
                    f"{est.__class__.__name__}. The underlying estimators have a mix "
                    f"of feature_importances_ and coef_ attributes."
                )

            # Trim each coef to the number of features in the 1st estimator
            # ClassifierChain adds features to subsequent estimators
            min_length = min(map(len, data))
            data = np.mean([c[:min_length] for c in data], axis=0)

    if data is None:
        return data
    else:
        return np.abs(data.flatten())


def export_pipeline(
    pipeline: pd.Series,
    model: MODEL | None = None,
    memory: BOOL | str | Memory | None = None,
    verbose: INT | None = None,
) -> Any:
    """Export a pipeline to a sklearn-like object.

    Optionally, you can add a model as final estimator.

    Parameters
    ----------
    pipeline: pd.Series
        Transformers to add to the pipeline.

    model: str, Model or None, default=None
        Model for which to export the pipeline. If the model used
        [automated feature scaling][], the [Scaler][] is added to
        the pipeline. If None, the pipeline in the current branch
        is exported.

    memory: bool, str, Memory or None, default=None
        Used to cache the fitted transformers of the pipeline.
            - If None or False: No caching is performed.
            - If True: A default temp directory is used.
            - If str: Path to the caching directory.
            - If Memory: Object with the joblib.Memory interface.

    verbose: int or None, default=None
        Verbosity level of the transformers in the pipeline. If
        None, it leaves them to their original verbosity. Note
        that this is not the pipeline's own verbose parameter.
        To change that, use the `set_params` method.

    Returns
    -------
    Pipeline
        Current branch as a sklearn-like Pipeline object.

    """
    from atom.pipeline import Pipeline

    steps = []
    for transformer in pipeline:
        est = deepcopy(transformer)  # Not clone to keep fitted

        # Set the new verbosity (if possible)
        if verbose is not None and hasattr(est, "verbose"):
            est.verbose = verbose

        # Check if there already exists an estimator with that
        # name. If so, add a counter at the end of the name
        counter = 1
        name = est.__class__.__name__.lower()
        while name in (elem[0] for elem in steps):
            counter += 1
            name = est.__class__.__name__.lower() + str(counter)

        steps.append((name, est))

    if model:
        steps.append((model.name, deepcopy(model.estimator)))

    if not memory:  # None or False
        memory = None
    elif memory is True:
        memory = tempfile.gettempdir()

    return Pipeline(steps, memory=memory)  # ATOM's pipeline, not sklearn's


# Pipeline functions =============================================== >>

def name_cols(
    array: np.ndarray,
    original_df: DATAFRAME,
    col_names: list[str],
) -> list[str]:
    """Get the column names after a transformation.

    If the number of columns is unchanged, the original
    column names are returned. Else, give the column a
    default name if the column values changed.

    Parameters
    ----------
    array: np.array
        Transformed dataset.

    original_df: dataframe
        Original dataset.

    col_names: list of str
        Names of the columns used in the transformer.

    Returns
    -------
    list of str
        Column names.

    """
    # If columns were only transformed, return og names
    if array.shape[1] == len(col_names):
        return col_names

    # If columns were added or removed
    temp_cols = []
    for i, col in enumerate(array.T):
        # equal_nan=True fails for non-numeric dtypes
        mask = original_df.apply(
            lambda c: np.array_equal(
                a1=c,
                a2=col,
                equal_nan=is_numeric_dtype(c) and np.issubdtype(col.dtype, np.number)),
        )

        if any(mask) and mask[mask].index.values[0] not in temp_cols:
            # If the column is equal, use the existing name
            temp_cols.append(mask[mask].index.values[0])
        else:
            # If the column is new, use a default name
            counter = 0
            while True:
                n = f"x{i + counter + original_df.shape[1] - len(col_names)}"
                if (n not in original_df or n in col_names) and n not in temp_cols:
                    temp_cols.append(n)
                    break
                else:
                    counter += 1

    return temp_cols


def reorder_cols(
    transformer: TRANSFORMER,
    df: DATAFRAME,
    original_df: DATAFRAME,
    col_names: list[str],
) -> DATAFRAME:
    """Reorder the columns to their original order.

    This function is necessary in case only a subset of the
    columns in the dataset was used. In that case, we need
    to reorder them to their original order.

    Parameters
    ----------
    transformer: TRANSFORMER
        Instance that transformed `df`.

    df: dataframe
        Dataset to reorder.

    original_df: dataframe
        Original dataset (states the order).

    col_names: list of str
        Names of the columns used in the transformer.

    Returns
    -------
    dataframe
        Dataset with reordered columns.

    """
    # Check if columns returned by the transformer are already in the dataset
    for col in df:
        if col in original_df and col not in col_names:
            raise ValueError(
                f"Column '{col}' returned by transformer {transformer} "
                "already exists in the original dataset."
            )

    # Force new indices on old dataset for merge
    try:
        original_df.index = df.index
    except ValueError:  # Length mismatch
        raise IndexError(
            f"Length of values ({len(df)}) does not match length of "
            f"index ({len(original_df)}). This usually happens when "
            "transformations that drop rows aren't applied on all "
            "the columns."
        )

    # Define new column order
    columns = []
    for col in original_df:
        if col in df or col not in col_names:
            columns.append(col)

        # Add all derivative columns: columns that originate from another
        # and start with its progenitor name, e.g. one-hot encoded columns
        columns.extend(
            [c for c in df.columns if c.startswith(f"{col}_") and c not in original_df]
        )

    # Add remaining new columns (non-derivatives)
    columns.extend([col for col in df if col not in columns])

    # Merge the new and old datasets keeping the newest columns
    new_df = df.merge(
        right=original_df[[col for col in original_df if col in columns]],
        how="outer",
        left_index=True,
        right_index=True,
        suffixes=("", "__drop__"),
    )
    new_df = new_df.drop(new_df.filter(regex="__drop__$").columns, axis=1)

    return new_df[columns]


def fit_one(
    transformer: TRANSFORMER,
    X: FEATURES | None = None,
    y: TARGET | None = None,
    message: str | None = None,
    **fit_params,
):
    """Fit the data using one estimator.

    Parameters
    ----------
    transformer: TRANSFORMER
        Instance to fit.

    X: dataframe-like or None, default=None
        Feature set with shape=(n_samples, n_features). If None,
        X is ignored.

    y: int, str, dict, sequence, dataframe or None, default=None
        Target column corresponding to X.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - If sequence: Target array with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    message: str or None
        Short message. If None, nothing will be printed.

    **fit_params
        Additional keyword arguments for the fit method.

    """
    X = to_df(X, index=getattr(y, "index", None))
    y = to_pandas(y, index=getattr(X, "index", None))

    with _print_elapsed_time("Pipeline", message):
        if hasattr(transformer, "fit"):
            kwargs = {}
            inc = getattr(transformer, "_cols", getattr(X, "columns", []))
            if "X" in (params := sign(transformer.fit)):
                if X is not None and (cols := [c for c in inc if c in X]):
                    kwargs["X"] = X[cols]

                # X is required but has not been provided
                if len(kwargs) == 0:
                    if y is not None and hasattr(transformer, "_cols"):
                        kwargs["X"] = to_df(y)[inc]
                    elif params["X"].default != Parameter.empty:
                        kwargs["X"] = params["X"].default  # Fill X with default
                    else:
                        raise ValueError(
                            "Exception while trying to fit transformer "
                            f"{transformer.__class__.__name__}. Parameter "
                            "X is required but has not been provided."
                        )

            if "y" in params and y is not None:
                kwargs["y"] = y

            # Keep custom attrs since some transformers reset during fit
            with keep_attrs(transformer):
                transformer.fit(**kwargs, **fit_params)


def transform_one(
    transformer: TRANSFORMER,
    X: FEATURES | None = None,
    y: TARGET | None = None,
    method: str = "transform",
) -> (DATAFRAME | None, SERIES | None):
    """Transform the data using one estimator.

    Parameters
    ----------
    transformer: TRANSFORMER
        Instance to fit.

    X: dataframe-like or None, default=None
        Feature set with shape=(n_samples, n_features). If None,
        X is ignored.

    y: int, str, dict, sequence, dataframe or None, default=None
        Target column corresponding to X.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - If sequence: Target array with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    method: str, default="transform"
        Method to apply: transform or inverse_transform.

    Returns
    -------
    dataframe or None
        Feature set. Returns None if not provided.

    series or None
        Target column. Returns None if not provided.

    """

    def prepare_df(out: FEATURES, og: DATAFRAME) -> DATAFRAME:
        """Convert to df and set correct column names and order.

        Parameters
        ----------
        out: dataframe-like
            Data returned by the transformation.

        og: dataframe
            Original dataframe, prior to transformations.

        Returns
        -------
        dataframe
            Transformed dataset.

        """
        use_cols = [c for c in inc if c in og.columns]

        # Convert to pandas and assign proper column names
        if not isinstance(out, DATAFRAME_TYPES):
            if hasattr(transformer, "get_feature_names_out"):
                columns = transformer.get_feature_names_out()
            elif hasattr(transformer, "get_feature_names"):
                # Some estimators have legacy method, e.g. cuml, category-encoders...
                columns = transformer.get_feature_names()
            else:
                columns = name_cols(out, og, use_cols)

            out = to_df(out, index=og.index, columns=columns)

        # Reorder columns if only a subset was used
        if len(use_cols) != og.shape[1]:
            return reorder_cols(transformer, out, og, use_cols)
        else:
            return out

    X = to_df(
        data=X,
        index=getattr(y, "index", None),
        columns=getattr(transformer, "feature_names_in_", None),
    )
    y = to_pandas(y, index=getattr(X, "index", None))

    kwargs = {}
    inc = getattr(transformer, "_cols", getattr(X, "columns", []))
    if "X" in (params := sign(getattr(transformer, method))):
        if X is not None and (cols := [c for c in inc if c in X]):
            kwargs["X"] = X[cols]

        # X is required but has not been provided
        if len(kwargs) == 0:
            if y is not None and hasattr(transformer, "_cols"):
                kwargs["X"] = to_df(y)[inc]
            elif params["X"].default != Parameter.empty:
                kwargs["X"] = params["X"].default  # Fill X with default
            else:
                return X, y  # If X is needed, skip the transformer

    if "y" in params:
        if y is not None:
            kwargs["y"] = y
        elif "X" not in params:
            return X, y  # If y is None and no X in transformer, skip the transformer

    try:
        out = getattr(transformer, method)(**kwargs)
    except TypeError as ex:
        try:
            # Duck type using args instead of kwargs
            out = getattr(transformer, method)(*kwargs.values())
        except TypeError:
            raise ex

    # Transform can return X, y or both
    if isinstance(out, tuple):
        X_new = prepare_df(out[0], X)
        y_new = to_pandas(
            data=out[1],
            index=X.index,
            name=getattr(y, "name", None),
            columns=getattr(y, "columns", None),
        )
        if isinstance(y, DATAFRAME_TYPES):
            y_new = prepare_df(y_new, y)
    elif "X" in params and X is not None and any(c in X for c in inc):
        # X in -> X out
        X_new = prepare_df(out, X)
        y_new = y if y is None else y.set_axis(X_new.index)
    else:
        # Output must be y
        y_new = to_pandas(
            data=out,
            index=y.index,
            name=getattr(y, "name", None),
            columns=getattr(y, "columns", None),
        )
        X_new = X if X is None else X.set_index(y_new.index)
        if isinstance(y, DATAFRAME_TYPES):
            y_new = prepare_df(y_new, y)

    return X_new, y_new


def fit_transform_one(
    transformer: TRANSFORMER,
    X: FEATURES | None = None,
    y: TARGET | None = None,
    message: str | None = None,
    **fit_params,
) -> (DATAFRAME | None, SERIES | None, TRANSFORMER):
    """Fit and transform the data using one estimator.

    Parameters
    ----------
    transformer: TRANSFORMER
        Instance to fit.

    X: dataframe-like or None, default=None
        Feature set with shape=(n_samples, n_features). If None,
        X is ignored.

    y: int, str, dict, sequence, dataframe or None, default=None
        Target column corresponding to X.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - If sequence: Target array with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    message: str or None
        Short message. If None, nothing will be printed.

    **fit_params
        Additional keyword arguments for the fit method.

    Returns
    -------
    dataframe or None
        Feature set. Returns None if not provided.

    series or None
        Target column. Returns None if not provided.

    Transformer
        Fitted transformer.

    """
    fit_one(transformer, X, y, message, **fit_params)
    X, y = transform_one(transformer, X, y)

    return X, y, transformer


def custom_transform(
    transformer: TRANSFORMER,
    branch: BRANCH,
    data: tuple[DATAFRAME, PANDAS] | None = None,
    verbose: int | None = None,
    method: str = "transform",
) -> (DATAFRAME, PANDAS):
    """Applies a transformer on a branch.

    This function is generic and should work for all
    methods with parameters X and/or y.

    Parameters
    ----------
    transformer: TRANSFORMER
        Estimator to apply to the data.

    branch: Branch
        Transformer's branch.

    data: tuple or None
        New data to transform on. If tuple, should have form
        (X, y). If None, the transformation is applied directly
        on the branch.

    verbose: int or None, default=None
        Verbosity level for the transformation. If None, the
        transformer's verbosity is used.

    method: str, default="transform"
        Method to apply to the transformer. Choose from: transform
        or inverse_transform.

    Returns
    -------
    dataframe
        Feature set.

    series or dataframe
        Target column(s).

    """
    # Select provided data or from the branch
    if data:
        X_og, y_og = to_df(data[0]), to_pandas(data[1])
    else:
        if transformer._train_only:
            X_og, y_og = branch.X_train, branch.y_train
        else:
            X_og, y_og = branch.X, branch.y

    # Adapt the transformer's verbosity
    if verbose is not None and hasattr(transformer, "verbose"):
        vb = transformer.verbose  # Save original verbosity
        transformer.verbose = verbose

    X, y = transform_one(transformer, X_og, y_og, method)

    # Apply changes to the branch
    if not data:
        if transformer._train_only:
            branch.train = merge(
                branch.X_train if X is None else X,
                branch.y_train if y is None else y,
            )
        else:
            branch._data = merge(
                branch.X if X is None else X,
                branch.y if y is None else y,
            )

            # y can change the number of columns or remove rows -> reassign index
            branch._idx = IndexConfig(
                train_idx=branch._idx.train_idx.intersection(branch._data.index),
                test_idx=branch._idx.test_idx.intersection(branch._data.index),
                n_cols=len(get_cols(y)),
            )

    # Back to the original verbosity
    if verbose is not None and hasattr(transformer, "verbose"):
        transformer.verbose = vb

    return X, y


# Patches ========================================================== >>

def fit_and_score(*args, **kwargs) -> dict:
    """Wrapper for sklearn's _fit_and_score function.

    Wrap the function sklearn.model_selection._validation._fit_and_score
    to, in turn, path sklearn's _score function to accept pipelines that
    drop samples during transforming, within a joblib parallel context.

    """

    def wrapper(*args, **kwargs) -> dict:
        with patch("sklearn.model_selection._validation._score", score(_score)):
            return _fit_and_score(*args, **kwargs)

    return wrapper(*args, **kwargs)


def score(f: Callable) -> Callable:
    """Patch decorator for sklearn's _score function.

    Monkey patch for sklearn.model_selection._validation._score
    function to score pipelines that drop samples during transforming.

    """

    def wrapper(*args, **kwargs) -> FLOAT | dict[str, FLOAT]:
        args = list(args)  # Convert to list for item assignment
        if len(args[0]) > 1:  # Has transformers
            args[1], args[2] = args[0][:-1].transform(args[1], args[2])

        # Return f(final_estimator, X_transformed, y_transformed, ...)
        return f(args[0][-1], *tuple(args[1:]), **kwargs)

    return wrapper


# Decorators ======================================================= >>

def has_task(task: str | list[str], inverse: bool = False) -> Callable:
    """Check that the instance is from a specific task.

    Parameters
    ----------
    task: str or list of str
        Task(s) to check.

    inverse: bool, default=False
        If True, checks that the parameter is not part of the task.

    """

    def check(self: Any) -> bool:
        if inverse:
            return not any(t in self.task for t in lst(task))
        else:
            return any(t in self.task for t in lst(task))

    return check


def has_attr(attr: str) -> Callable:
    """Check that the instance has attribute `attr`.

    Parameters
    ----------
    attr: str
        Name of the attribute to check.

    """

    def check(self: Any) -> bool:
        # Raise original `AttributeError` if `attr` does not exist
        getattr(self, attr)
        return True

    return check


def estimator_has_attr(attr: str) -> Callable:
    """Check that the estimator has attribute `attr`.

    Parameters
    ----------
    attr: str
        Name of the attribute to check.

    """

    def check(self: Any) -> bool:
        # Raise original `AttributeError` if `attr` does not exist
        getattr(self.estimator, attr)
        return True

    return check


def composed(*decs) -> Callable:
    """Add multiple decorators in one line.

    Parameters
    ----------
    *decs
        Decorators to run.

    """

    def decorator(f: Callable) -> Callable:
        for dec in reversed(decs):
            f = dec(f)
        return f

    return decorator


def crash(f: Callable, cache: dict = {"last_exception": None}) -> Callable:
    """Save program crashes to log file.

    We use a mutable argument to cache the last exception raised. If
    the current exception is the same (happens when there is an error
    catch or multiple calls to crash), it's not re-written in the logger.

    """

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        try:  # Run the function
            return f(*args, **kwargs)

        except Exception as ex:
            # If exception is not same as last, write to log
            if ex is not cache["last_exception"] and args[0].logger:
                cache["last_exception"] = ex
                args[0].logger.exception("Exception encountered:")

            raise ex

    return wrapper


def method_to_log(f: Callable) -> Callable:
    """Save called functions to log file."""

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        if args[0].logger:
            if f.__name__ != "__init__":
                args[0].logger.info("")
            args[0].logger.info(f"{args[0].__class__.__name__}.{f.__name__}()")

        return f(*args, **kwargs)

    return wrapper


def plot_from_model(
    func: Callable | None = None,
    max_one: bool = False,
    ensembles: bool = True,
    check_fitted: bool = True,
) -> Callable:
    """If a plot is called from a model, adapt the `models` parameter.

    Parameters
    ----------
    func: callable or None
        Function to decorate. When the decorator is called with no
        optional arguments, the function is passed as the first argument
        and the decorator just returns the decorated function.

    max_one: bool, default=False
        Whether one or multiple models are allowed. If True, return
        the model instead of a list of models.

    ensembles: bool, default=True
        If False, drop ensemble models from selection.

    check_fitted: bool, default=True
        Raise an exception if the runner isn't fitted (has no models).

    """

    @wraps(func)
    def wrapper(f: Callable) -> Callable:

        @wraps(f)
        def wrapped_f(*args, **kwargs) -> Any:
            if hasattr(args[0], "_get_models"):
                # Called from runner, get models for `models` parameter
                if check_fitted:
                    check_is_fitted(args[0], attributes="_models")

                if len(args) > 1:
                    models = args[1]
                    args = [args[0]] + list(args[2:])  # Remove models from args
                elif "models" in kwargs:
                    models = kwargs.pop("models")
                else:
                    models = None

                models = args[0]._get_models(models, ensembles=ensembles)
                if max_one:
                    if len(models) > 1:
                        raise ValueError(f"The {f.__name__} plot only accepts one model.")
                    else:
                        models = models[0]

                return f(args[0], models, *args[1:], **kwargs)
            else:
                # Called from model, send model directly to `models` parameter
                return f(args[0], args[0] if max_one else [args[0]], *args[1:], **kwargs)

        return wrapped_f

    if func:
        return wrapper(func)

    return wrapper


# Custom scorers =================================================== >>

def true_negatives(y_true: SEQUENCE, y_pred: SEQUENCE) -> INT:
    return confusion_matrix(y_true, y_pred).ravel()[0]


def false_positives(y_true: SEQUENCE, y_pred: SEQUENCE) -> INT:
    return confusion_matrix(y_true, y_pred).ravel()[1]


def false_negatives(y_true: SEQUENCE, y_pred: SEQUENCE) -> INT:
    return confusion_matrix(y_true, y_pred).ravel()[2]


def true_positives(y_true: SEQUENCE, y_pred: SEQUENCE) -> INT:
    return confusion_matrix(y_true, y_pred).ravel()[3]


def false_positive_rate(y_true: SEQUENCE, y_pred: SEQUENCE) -> FLOAT:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def true_positive_rate(y_true: SEQUENCE, y_pred: SEQUENCE) -> FLOAT:
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)


def true_negative_rate(y_true: SEQUENCE, y_pred: SEQUENCE) -> FLOAT:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def false_negative_rate(y_true: SEQUENCE, y_pred: SEQUENCE) -> FLOAT:
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp)

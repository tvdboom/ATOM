# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing utility classes.

"""

from __future__ import annotations

import os
import pprint
import sys
import warnings
from collections import deque
from collections.abc import MutableMapping
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from importlib import import_module
from importlib.util import find_spec
from inspect import Parameter, signature
from itertools import cycle
from types import GeneratorType, MappingProxyType
from unittest.mock import patch

import mlflow
import modin.pandas as md
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sps
from beartype.typing import Any, Callable, Hashable, TypeVar
from IPython.display import display
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
    Bool, DataFrame, DataFrameTypes, Estimator, Features, Float, Index,
    IndexSelector, Int, IntTypes, MetricConstructor, Model, Pandas,
    PandasTypes, Predictor, Scalar, Scorer, Segment, SegmentTypes, Sequence,
    SequenceTypes, Series, SeriesTypes, Target, Transformer, Verbose,
)


T = TypeVar("T")
T_Pandas = TypeVar("T_Pandas", Series, DataFrame)


# Classes ========================================================== >>

class NotFittedError(ValueError, AttributeError):
    """Exception called when the instance is not yet fitted.

    This class inherits from both ValueError and AttributeError to
    help with exception handling and backward compatibility.

    """


class Goal(Enum):
    """Supported goals by ATOM."""
    classification = 1
    regression = 2
    forecast = 3

    def infer_task(self, y: Pandas) -> Task:
        """Infer the task corresponding to a target column.

        Parameters
        ----------
        y: series or dataframe
            Target column(s).

        Returns
        -------
        Task
            Inferred task.

        """
        if self.value == 2:
            if y.ndim == 1:
                return Task.regression
            else:
                return Task.multioutput_regression
        elif self.value == 3:
            if y.ndim == 1:
                return Task.univariate_forecast
            else:
                return Task.multivariate_forecast

        if y.ndim > 1:
            if all(y[col].nunique() == 2 for col in y):
                return Task.multilabel_classification
            else:
                return Task.multiclass_multioutput_classification
        elif isinstance(y.iloc[0], SequenceTypes):
            return Task.multilabel_classification
        elif y.nunique() == 1:
            raise ValueError(f"Only found 1 target value: {y.unique()[0]}")
        elif y.nunique() == 2:
            return Task.binary_classification
        else:
            return Task.multiclass_classification


class Task(Enum):
    """Supported tasks by ATOM."""
    binary_classification = 1
    multiclass_classification = 2
    multilabel_classification = 3
    multiclass_multioutput_classification = 4
    regression = 5
    multioutput_regression = 6
    univariate_forecast = 7
    multivariate_forecast = 8

    def __str__(self) -> str:
        return self.name.replace("_", " ").capitalize()

    @property
    def is_classification(self) -> Bool:
        """Return whether the task is a classification task."""
        return self.value in (1, 2, 3, 4)

    @property
    def is_regression(self) -> Bool:
        """Return whether the task is a regression task."""
        return self.value in (5, 6)

    @property
    def is_forecast(self) -> Bool:
        """Return whether the task is a forecast task."""
        return self.value in (7, 8)

    @property
    def is_binary(self) -> Bool:
        """Return whether the task is binary or multilabel."""
        return self.value in (1, 3)

    @property
    def is_multiclass(self) -> Bool:
        """Return whether the task is multiclass or multiclass-multioutput."""
        return self.value in (2, 4)

    @property
    def is_multioutput(self) -> Bool:
        """Return whether the task has more than one target column."""
        return self.value in (3, 4, 6, 8)


@dataclass
class DataConfig:
    """Stores the data configuration.

    This is a utility class to store the data configuration in one
    attribute and pass it down to the models. The default values are
    the one adopted by trainers.

    """
    index: IndexSelector = True
    shuffle: Bool = True
    stratify: IndexSelector = True
    test_size: Scalar = 0.2


@dataclass
class DataContainer:
    """Stores a branch's data."""
    data: DataFrame  # Complete dataset
    train_idx: Index  # Indices in the train set
    test_idx: Index  # Indices in the test
    n_cols: Int  # Number of target columns


@dataclass
class Aesthetics:
    """Keeps track of plot aesthetics."""
    palette: str | Sequence[str] | dict[str, str]  # Sequence of colors
    title_fontsize: Int  # Fontsize for titles
    label_fontsize: Int  # Fontsize for labels, legend and hoverinfo
    tick_fontsize: Int  # Fontsize for ticks
    line_width: Int  # Width of the line plots
    marker_size: Int  # Size of the markers


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


# ATOM uses this instance to access the data engine
bk = PandasModin()


class CatBMetric:
    """Custom evaluation metric for the CatBoost model.

    Parameters
    ----------
    scorer: Scorer
        Scorer to evaluate. It's always the runner's main metric.

    task: Task
        Model's task.

    """
    def __init__(self, scorer: Scorer, task: Task):
        self.scorer = scorer
        self.task = task

    def get_final_error(self, error: Float, weight: Float) -> Float:
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

    def evaluate(self, approxes: list, targets: list, weight: list) -> Float:
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
        if self.task.is_binary:
            # Convert CatBoost predictions to probabilities
            e = np.exp(approxes[0])
            y_pred = e / (1 + e)
            if self.scorer.__class__.__name__ == "_PredictScorer":
                y_pred = (y_pred > 0.5).astype(int)

        elif self.task.is_multiclass:
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

    task: Task
        Model's task.

    """
    def __init__(self, scorer: Scorer, task: Task):
        self.scorer = scorer
        self.task = task

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weight: np.ndarray,
    ) -> tuple[str, Float, bool]:
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
            if self.task.is_binary:
                y_pred = (y_pred > 0.5).astype(int)
            elif self.task.is_multiclass:
                y_pred = y_pred.reshape(len(np.unique(y_true)), len(y_true)).T
                y_pred = np.argmax(y_pred, axis=1)

        kwargs = {}
        if "sample_weight" in sign(self.scorer._score_func):
            kwargs["sample_weight"] = weight

        score = self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs, **kwargs)

        return self.scorer.name, self.scorer._sign * score, True


class Table:
    """Class to print nice tables per row.

    Parameters
    ----------
    headers: sequence
        Name of each column in the table. If an element is a tuple,
        the second element should be the position of the text in the
        cell (left or right).

    spaces: sequence
        Width of each column. Should have the same length as `headers`.

    default_pos: str, default="right"
        Default position of the text in the cell.

    """

    def __init__(self, headers: Sequence, spaces: Sequence, default_pos: str = "right"):
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
    def to_cell(text: Scalar | str, position: str, space: Int) -> str:
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

    def print(self, sequence: dict[str, Any]) -> str:
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
            out.append(self.to_cell(rnd(sequence.get(header, "---")), pos, space))

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
        Number of parallel jobs. If >1, no output is shown.

    """

    def __init__(self, model: Model, n_jobs: Int):
        self.T = model
        self.n_jobs = n_jobs

        if self.n_jobs == 1:
            self._table = self.create_table()
            self.T._log(self._table.print_header(), 2)
            self.T._log(self._table.print_line(), 2)

    def __call__(self, study: Study, trial: FrozenTrial):
        trial_info = self.T.trials.reset_index(names="trial").loc[trial.number]

        # Save trials to mlflow experiment as nested runs
        if self.T.experiment and self.T.log_ht:
            with mlflow.start_run(run_id=self.T.run.info.run_id):
                run_name = f"{self.T.name} - {trial.number}"
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.set_tags(
                        {
                            "name": self.T.name,
                            "model": self.T._fullname,
                            "branch": self.T.branch.name,
                            "trial_state": trial_info.state,
                            **self.T._ht["tags"],
                        }
                    )

                    mlflow.log_metric("time_trial", trial_info["time_trial"])
                    for met in self.T._metric.keys():
                        mlflow.log_metric(f"{met}_validation", trial_info[met])

                    if estimator := trial_info["estimator"]:
                        # Mlflow only accepts params with char length <=250
                        mlflow.log_params(
                            {
                                k: v for k, v in estimator.get_params().items()
                                if len(str(v)) <= 250
                            }
                        )

                        mlflow.sklearn.log_model(
                            sk_model=estimator,
                            artifact_path=estimator.__class__.__name__,
                            signature=infer_signature(
                                model_input=pd.DataFrame(self.T.X),
                                model_output=estimator.predict(self.T.X.iloc[[0], :]),
                            ),
                            input_example=pd.DataFrame(self.T.X.iloc[[0], :]),
                        )
                    else:
                        mlflow.log_params(
                            {
                                k: v for k, v in trial.params.items()
                                if len(str(v)) <= 250
                            }
                        )

        if self.n_jobs == 1:
            # Print overview of trials
            trial_info["time_trial"] = time_to_str(trial_info["time_trial"])
            trial_info["time_ht"] = time_to_str(trial_info["time_ht"])
            self.T._log(self._table.print(trial_info), 2)

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
            # Else take the max of seven (minimum width) and the width of the name
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

    def __init__(self, name: str, metric: list[str], aesthetics: Aesthetics):
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

    task: Task
        Model's task.

    branch: Branch
        Data to get the shap values from.

    random_state: int or None, default=None
        Random seed for reproducibility.

    """

    def __init__(
        self,
        estimator: Predictor,
        task: Task,
        branch: Any,
        random_state: Int | None = None,
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
            try:  # Fails when model does not fit standard explainers (e.g., ensembles)
                self._explainer = Explainer(self.estimator, **kwargs)
            except TypeError:
                # If method is provided as first arg, selects always Permutation
                self._explainer = Explainer(getattr(self.estimator, self.attr), **kwargs)

        return self._explainer

    def get_explanation(
        self,
        df: DataFrame,
        target: tuple[Int, Int],
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

            # Additivity check sometimes fails for no apparent reason
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
        explanation.data = self.branch._all.loc[df.index].to_numpy()

        if self.task.is_multioutput:
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
        if isinstance(key, IntTypes) and key not in self.keys():
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
        if isinstance(key, SequenceTypes):
            return self.__class__(*[self._get_data(k) for k in key], key=self.__key)
        elif isinstance(key, SegmentTypes):
            return self.__class__(*get_segment(self.__data, key), key=self.__key)
        elif isinstance(key, slice):
            return self.__class__(*self.__data[key], key=self.__key)
        else:
            return self._get_data(key)

    def __setitem__(self, key, value):
        if isinstance(key, IntTypes):
            self.__data[key] = self._check(value)
        else:
            try:
                self.__data = [e if self[key] == e else value for e in self.__data]
            except KeyError:
                assert key == getattr(value, self.__key)
                self.append(value)

    def __delitem__(self, key):
        del self.__data[self.index(self._get_data(key))]

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

    def values(self) -> list:
        return self.__data

    def keys_lower(self) -> list:
        return list(map(self._conv, self.keys()))

    def append(self, value):
        self.__data.append(self._check(value))
        return value

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
        if isinstance(key, SequenceTypes):
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

    @staticmethod
    def _conv(key):
        return key.lower() if isinstance(key, str) else key

    def _get_key(self, key):
        # Get key from index
        if isinstance(key, IntTypes) and key not in self.__keys:
            return self.__keys[key]
        else:
            # Get key from name
            for k in self.__keys:
                if self._conv(k) == self._conv(key):
                    return k

        raise KeyError(key)

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

    def deepcopy(self):
        return deepcopy(self)

    def index(self, key):
        return self.__keys.index(self._get_key(key))

    def reorder(self, keys):
        self.__keys = [k for k in keys if k in self]
        self.__data = {self._conv(k): self[k] for k in keys if k in self}

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
    return x[0] if isinstance(x, SequenceTypes) and len(x) == 1 else x


def lst(x: Any) -> list[Any]:
    """Make a list from an item if not a sequence already.

    Parameters
    ----------
    x: Any
        Item or sequence.

    Returns
    -------
    list
        Item as list with length 1 or provided sequence as list.

    """
    return list(x) if isinstance(x, (dict, CustomDict, ClassMap, *SequenceTypes)) else [x]


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


def rnd(x: Any, decimals: Int = 4) -> Any:
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


def divide(a: Scalar, b: Scalar) -> Scalar:
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


def merge(*args) -> DataFrame:
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


def replace_missing(X: T_Pandas, missing_values: list[Any] | None = None) -> T_Pandas:
    """Replace all values considered 'missing' in a dataset.

    This method replaces the missing values in columns with pandas'
    nullable dtypes with `pd.NA`, else with `np.NaN`.

    Parameters
    ----------
    X: series or dataframe
        Data set to replace.

    missing_values: list or None, default=None
        Values considered 'missing'. If None, use only the default
        values.

    Returns
    -------
    series or dataframe
        Data set without missing values.

    """
    # Always convert these values
    default_values = [None, pd.NA, pd.NaT, np.NaN, np.inf, -np.inf]

    get_nan = lambda dtype: np.NaN if isinstance(dtype, np.dtype) else pd.NA

    if isinstance(X, SeriesTypes):
        return X.replace(
            to_replace=(missing_values or []) + default_values,
            value=get_nan(X.dtype),
        )
    else:
        return X.replace(
            to_replace={k: (missing_values or []) + default_values for k in X},
            value={k: get_nan(X[k].dtype) for k in X},
        )


def get_cols(elem: Pandas) -> list[Series]:
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
    if isinstance(elem, SeriesTypes):
        return [elem]
    else:
        return [elem[col] for col in elem]


def variable_return(
    X: DataFrame | None,
    y: Series | None,
) -> DataFrame | Series | tuple[DataFrame, Pandas]:
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


def get_segment(obj: Sequence[T], segment: Segment) -> Sequence[T]:
    """Get a subset of a sequence by range or slice.

    Parameters
    ----------
    obj: sequence
        Object to slice.

    segment: range or slice
        Segment to extract from the sequence.

    Returns
    -------
    sequence
        Subset of the original sequence.

    """
    if isinstance(segment, range):
        return obj[segment]
    else:
        return obj[slice(segment.start, segment.stop, segment.step)]


def is_sparse(obj: Pandas) -> bool:
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


def check_empty(obj: Pandas) -> Pandas | None:
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


def check_canvas(is_canvas: Bool, method: str):
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


def check_predict_proba(models: Model | Sequence, method: str):
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
            raise PermissionError(
                f"The {method} method is only available for "
                f"models with a predict_proba method, got {m.name}."
            )


def check_scaling(X: Pandas, pipeline: Any | None = None) -> bool:
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
        has_scaler = any("scaler" in name.lower() for name in pipeline.named_steps)

    df = to_df(X)  # Convert to dataframe
    df = df.loc[:, (~df.isin([0, 1])).any(axis=0)]  # Remove binary columns

    if df.empty:  # All columns are binary -> no scaling needed
        return True
    else:
        mean = df.mean(numeric_only=True).mean()
        std = df.std(numeric_only=True).mean()
        return has_scaler or (-0.05 < mean < 0.05 and 0.85 < std < 1.15)


@contextmanager
def keep_attrs(estimator: Estimator):
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


@contextmanager
def adjust_verbosity(estimator: Estimator, verbose: Verbose | None):
    """Contextmanager to save an estimator's custom attributes.

    ATOM's pipeline uses two custom attributes for its transformers:
    _train_only, and _cols. Since some transformers reset their
    attributes during fit (like those from sktime), we wrap the fit
    method in a contextmanager that saves and restores the attrs.

    Parameters
    ----------
    estimator: Estimator
        Temporarily change the verbosity of this estimator.

    verbose: int or None, default=None
        Verbosity level of the transformers in the pipeline. If
        None, it leaves them to their original verbosity.

    """
    try:
        if verbose is not None and hasattr(estimator, "verbose"):
            verbosity = estimator.verbose
            estimator.verbose = verbose
        yield estimator
    finally:
        if verbose is not None and hasattr(estimator, "verbose"):
            estimator.verbose = verbosity


def get_versions(models: ClassMap) -> dict[str, str]:
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


def get_corpus(df: DataFrame) -> Series:
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


def get_best_score(item: Model | Series, metric: int = 0) -> Float:
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


def time_to_str(t: Scalar) -> str:
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


def n_cols(data: Features | Target | None) -> int:
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


def to_pyarrow(column: Series, inverse: bool = False) -> str:
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
    data: Features | None,
    index: Sequence[Hashable] | None = None,
    columns: Sequence[str] | None = None,
    dtype: str | np.dtype | dict[str, str | np.dtype] | None = None,
) -> DataFrame | None:
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
            if not isinstance(data, (dict, PandasTypes)) and columns is None:
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
    data: Sequence[Any] | None,
    index: Sequence[Hashable] | None = None,
    name: str = "target",
    dtype: str | np.dtype | None = None,
) -> Series | None:
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
    data: dict[str, Sequence[Any]] | Sequence[Sequence[Any]] | DataFrame | None,
    index: Sequence[Hashable] | None = None,
    columns: Sequence[str] | None = None,
    name: str = "target",
    dtype: str | dict | np.dtype | None = None,
) -> Pandas | None:
    """Convert a sequence or dataset to a dataframe or series object.

    If the data is 1-dimensional, convert to series, else to a dataframe.

    Parameters
    ----------
    data: sequence, dataframe or None
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
    obj: Any,
    exception: Bool = True,
    attributes: str | Sequence[str] | None = None,
) -> Bool:
    """Check whether an estimator is fitted.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (not None or empty). Otherwise, it raises a
    NotFittedError. Extension on sklearn's function that accounts
    for empty dataframes and series and returns a boolean.

    Parameters
    ----------
    obj: object
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

    def check_attr(attr: str) -> Bool:
        """Return whether an attribute is False or empty.

        Parameters
        ----------
        attr: str
            Name of the attribute to check.

        Returns
        -------
        bool
            Whether the attribute's value is False or empty.

        """
        if attr:
            if isinstance(value := getattr(obj, attr), PandasTypes):
                return value.empty
            else:
                return not value

    is_fitted = False
    if hasattr(obj, "__sklearn_is_fitted__"):
        return obj.__sklearn_is_fitted__()
    elif attributes is None:
        # Check for attributes from a fitted object
        for k, v in vars(obj).items():
            if k.endswith("_") and not k.startswith("__") and v is not None:
                is_fitted = True
                break
    elif not all(check_attr(attr) for attr in lst(attributes)):
        is_fitted = True

    if not is_fitted:
        if exception:
            raise NotFittedError(
                f"This {type(obj).__name__} instance is not fitted yet. "
                f"Call {'run' if hasattr(obj, 'run') else 'fit'} with "
                "appropriate arguments before using this object."
            )
        else:
            return False

    return True


def get_custom_scorer(metric: MetricConstructor) -> Scorer:
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


def get_feature_importance(
    est: Predictor,
    attributes: Sequence[str] | None = None,
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


# Pipeline functions =============================================== >>

def name_cols(
    array: np.ndarray,
    original_df: DataFrame,
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
    transformer: Transformer,
    df: DataFrame,
    original_df: DataFrame,
    col_names: list[str],
) -> DataFrame:
    """Reorder the columns to their original order.

    This function is necessary in case only a subset of the
    columns in the dataset was used. In that case, we need
    to reorder them to their original order.

    Parameters
    ----------
    transformer: Transformer
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
        # and start with its progenitor name, e.g., one-hot encoded columns
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
    estimator: Estimator,
    X: Features | None = None,
    y: Target | None = None,
    message: str | None = None,
    **fit_params,
) -> Estimator:
    """Fit the data using one estimator.

    Parameters
    ----------
    estimator: Estimator
        Instance to fit.

    X: dataframe-like or None, default=None
        Feature set with shape=(n_samples, n_features). If None,
        X is ignored.

    y: int, str, dict, sequence, dataframe or None, default=None
        Target column corresponding to X.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    message: str or None
        Short message. If None, nothing will be printed.

    **fit_params
        Additional keyword arguments for the fit method.

    Returns
    -------
    Estimator
        Fitted estimator.

    """
    X = to_df(X, index=getattr(y, "index", None))
    y = to_pandas(y, index=getattr(X, "index", None))

    with _print_elapsed_time("Pipeline", message):
        if hasattr(estimator, "fit"):
            kwargs = {}
            inc = getattr(estimator, "_cols", getattr(X, "columns", []))
            if "X" in (params := sign(estimator.fit)):
                if X is not None and (cols := [c for c in inc if c in X]):
                    kwargs["X"] = X[cols]

                # X is required but has not been provided
                if len(kwargs) == 0:
                    if y is not None and hasattr(estimator, "_cols"):
                        kwargs["X"] = to_df(y)[inc]
                    elif params["X"].default != Parameter.empty:
                        kwargs["X"] = params["X"].default  # Fill X with default
                    else:
                        raise ValueError(
                            "Exception while trying to fit transformer "
                            f"{estimator.__class__.__name__}. Parameter "
                            "X is required but has not been provided."
                        )

            if "y" in params and y is not None:
                kwargs["y"] = y

            # Keep custom attrs since some transformers reset during fit
            with keep_attrs(estimator):
                estimator.fit(**kwargs, **fit_params)

        return estimator


def transform_one(
    transformer: Transformer,
    X: Features | None = None,
    y: Target | None = None,
    method: str = "transform",
) -> tuple[DataFrame | None, Pandas | None]:
    """Transform the data using one estimator.

    Parameters
    ----------
    transformer: Transformer
        Instance to fit.

    X: dataframe-like or None, default=None
        Feature set with shape=(n_samples, n_features). If None,
        X is ignored.

    y: int, str, dict, sequence, dataframe or None, default=None
        Target column corresponding to X.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    method: str, default="transform"
        Method to apply: transform or inverse_transform.

    Returns
    -------
    dataframe or None
        Feature set. Returns None if not provided.

    series, dataframe or None
        Target column. Returns None if not provided.

    """

    def prepare_df(out: Features, og: DataFrame) -> DataFrame:
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
        if not isinstance(out, DataFrameTypes):
            if hasattr(transformer, "get_feature_names_out"):
                columns = transformer.get_feature_names_out()
            elif hasattr(transformer, "get_feature_names"):
                # Some estimators have legacy method, e.g., cuml, category-encoders...
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

    # Transform can return X, y or both
    if isinstance(out := getattr(transformer, method)(**kwargs), tuple):
        X_new = prepare_df(out[0], X)
        y_new = to_pandas(
            data=out[1],
            index=X.index,
            name=getattr(y, "name", None),
            columns=getattr(y, "columns", None),
        )
        if isinstance(y, DataFrameTypes):
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
        if isinstance(y, DataFrameTypes):
            y_new = prepare_df(y_new, y)

    return X_new, y_new


def fit_transform_one(
    transformer: Transformer,
    X: Features | None = None,
    y: Target | None = None,
    message: str | None = None,
    **fit_params,
) -> tuple[DataFrame | None, Series | None, Transformer]:
    """Fit and transform the data using one estimator.

    Parameters
    ----------
    transformer: Transformer
        Instance to fit.

    X: dataframe-like or None, default=None
        Feature set with shape=(n_samples, n_features). If None,
        X is ignored.

    y: int, str, dict, sequence, dataframe or None, default=None
        Target column corresponding to X.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - If sequence: Target column with shape=(n_samples,) or
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

    def wrapper(*args, **kwargs) -> Float | dict[str, Float]:
        args = list(args)  # Convert to list for item assignment
        if len(args[0]) > 1:  # Has transformers
            args[1], args[2] = args[0][:-1].transform(args[1], args[2])

        # Return f(final_estimator, X_transformed, y_transformed, ...)
        return f(args[0][-1], *tuple(args[1:]), **kwargs)

    return wrapper


# Decorators ======================================================= >>

def has_task(task: str | Sequence[str]) -> Callable:
    """Checks that the instance has a specific task.

    If the check returns False, the decorated function becomes
    unavailable for the instance.

    Parameters
    ----------
    task: str or sequence
        Tasks to check. Choose from: classification, regression,
        forecast, binary, multioutput. Add the character `!` before
        a task to not allow that task instead.

    Returns
    -------
    func
        Function to check.

    """

    def check(runner: Any) -> Bool:
        checks = []
        for t in lst(task):
            if t.startswith("!"):
                checks.append(not getattr(runner.task, f"is_{t[1:]}"))
            else:
                checks.append(getattr(runner.task, f"is_{t}"))

        return all(checks)

    return check


def has_attr(attr: str) -> Callable:
    """Check that the instance has attribute `attr`.

    Parameters
    ----------
    attr: str
        Name of the attribute to check.

    """

    def check(runner: Any) -> bool:
        # Raise original `AttributeError` if `attr` does not exist
        getattr(runner, attr)
        return True

    return check


def estimator_has_attr(attr: str) -> Callable:
    """Check that the estimator has attribute `attr`.

    Parameters
    ----------
    attr: str
        Name of the attribute to check.

    """

    def check(runner: Any) -> bool:
        # Raise original `AttributeError` if `attr` does not exist
        getattr(runner.estimator, attr)
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


def crash(
    f: Callable,
    cache: dict[str, Exception | None] = {"last_exception": None},
) -> Callable:
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
            # If exception is not the same as last, write to log
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


# Custom scorers =================================================== >>

def true_negatives(y_true: Sequence, y_pred: Sequence) -> Int:
    return confusion_matrix(y_true, y_pred).ravel()[0]


def false_positives(y_true: Sequence, y_pred: Sequence) -> Int:
    return confusion_matrix(y_true, y_pred).ravel()[1]


def false_negatives(y_true: Sequence, y_pred: Sequence) -> Int:
    return confusion_matrix(y_true, y_pred).ravel()[2]


def true_positives(y_true: Sequence, y_pred: Sequence) -> Int:
    return confusion_matrix(y_true, y_pred).ravel()[3]


def false_positive_rate(y_true: Sequence, y_pred: Sequence) -> Float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def true_positive_rate(y_true: Sequence, y_pred: Sequence) -> Float:
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)


def true_negative_rate(y_true: Sequence, y_pred: Sequence) -> Float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def false_negative_rate(y_true: Sequence, y_pred: Sequence) -> Float:
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp)

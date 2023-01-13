# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing utility constants, classes and functions.

"""

from __future__ import annotations

import pprint
import sys
from collections import OrderedDict, deque
from collections.abc import MutableMapping
from copy import copy
from datetime import datetime as dt
from functools import wraps
from importlib import import_module
from importlib.util import find_spec
from inspect import Parameter, signature
from itertools import cycle
from typing import Any, Callable, Protocol, Union

import mlflow
import modin.pandas as md
import numpy as np
import pandas as pd
import pandas as bk
import plotly.graph_objects as go
from IPython.display import display
from matplotlib.colors import to_rgba
from mlflow.models.signature import infer_signature
from optuna.study import Study
from optuna.trial import FrozenTrial
from ray import serve
from scipy import sparse
from shap import Explainer, Explanation
from sklearn.metrics import (
    confusion_matrix, get_scorer, get_scorer_names, make_scorer,
    matthews_corrcoef,
)
from sklearn.utils import _print_elapsed_time
from starlette.requests import Request

from atom.pipeline import Pipeline


# Constants ======================================================== >>

# Current library version
__version__ = "5.1.0"

# Group of variable types for isinstance
# TODO: From Python 3.10, isinstance accepts union operator (change by then)
INT = (int, np.integer)
FLOAT = (float, np.floating)
SCALAR = (*INT, *FLOAT)
INDEX = (pd.Index, md.Index, pd.MultiIndex, md.MultiIndex)
SERIES = (pd.Series, md.Series)
DATAFRAME = (pd.DataFrame, md.DataFrame)
PANDAS = (*SERIES, *DATAFRAME)
SEQUENCE = (list, tuple, np.ndarray, pd.Series, md.Series)

# Groups of variable types for type hinting
INT_TYPES = Union[INT]
FLOAT_TYPES = Union[FLOAT]
SCALAR_TYPES = Union[SCALAR]
INDEX_TYPES = Union[INDEX]
SERIES_TYPES = Union[SERIES]
DATAFRAME_TYPES = Union[DATAFRAME]
PANDAS_TYPES = Union[PANDAS]
SEQUENCE_TYPES = Union[SEQUENCE]
X_TYPES = Union[iter, dict, list, tuple, np.ndarray, sparse.spmatrix, DATAFRAME_TYPES]
Y_TYPES = Union[INT_TYPES, str, dict, SEQUENCE_TYPES, DATAFRAME_TYPES]

# Attributes shared between atom and pd.DataFrame
DF_ATTRS = (
    "size",
    "head",
    "tail",
    "loc",
    "iloc",
    "describe",
    "iterrows",
    "dtypes",
    "at",
    "iat",
    "memory_usage",
    "empty",
    "ndim",
)

# Default color palette (discrete color, continuous scale)
PALETTE = {
    "rgb(0, 98, 98)": "Teal",
    "rgb(56, 166, 165)": "Teal",
    "rgb(115, 175, 72)": "Greens",
    "rgb(237, 173, 8)": "Oranges",
    "rgb(225, 124, 5)": "Oranges",
    "rgb(204, 80, 62)": "OrRd",
    "rgb(148, 52, 110)": "PuRd",
    "rgb(111, 64, 112)": "Purples",
    "rgb(102, 102, 102)": "Greys",
}


# Classes ========================================================== >>

class NotFittedError(ValueError, AttributeError):
    """Exception called when the instance is not yet fitted.

    This class inherits from both ValueError and AttributeError to
    help with exception handling and backward compatibility.

    """


class Runner(Protocol):
    """Protocol for all runners."""
    def run(self, **params): ...


class Model(Protocol):
    """Protocol for all models."""
    def est_class(self): ...
    def get_estimator(self, **params): ...


class Scorer(Protocol):
    """Protocol for all scorers."""
    def _score(self, method_caller, clf, X, y, sample_weight=None): ...


class Estimator(Protocol):
    """Protocol for all estimators."""
    def fit(self, **params): ...


class Predictor(Protocol):
    """Protocol for all predictors."""
    def fit(self, **params): ...
    def predict(self, **params): ...


class Transformer(Protocol):
    """Protocol for all predictors."""
    def fit(self, **params): ...
    def transform(self, **params): ...


@serve.deployment
class ServeModel:
    """Model deployment class.

    Parameters
    ----------
    model: Pipeline
        Transformers + estimator to make inference on.

    method: str, default="predict"
        Estimator's method to do inference on.

    """

    def __init__(self, model: Pipeline, method: str = "predict"):
        self.model = model
        self.method = method

    async def __call__(self, request: Request) -> str:
        """Inference call.

        Parameters
        ----------
        request: Request.
            HTTP request. Should contain the rows to predict
            in a json body.

        Returns
        -------
        str
            Model predictions as string.

        """
        payload = await request.json()
        return getattr(self.model, self.method)(pd.read_json(payload))


class CatBMetric:
    """Custom evaluation metric for the CatBoost model.

    Parameters
    ----------
    scorer: Scorer
        Scorer to evaluate. It's always the runner's main metric.

    task: str
        Model's task.

    """
    def __init__(self, scorer: Scorer, task: str):
        self.scorer = scorer
        self.task = task

    @staticmethod
    def get_final_error(error: FLOAT_TYPES, weight: FLOAT_TYPES) -> FLOAT_TYPES:
        """Returns final value of metric based on error and weight.

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

    def evaluate(self, approxes: list, targets: list, weight: list) -> FLOAT_TYPES:
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
        if self.task.startswith("bin"):
            # Convert CatBoost predictions to probabilities
            e = np.exp(approxes[0])
            y_pred = e / (1 + e)
            if self.scorer.__class__.__name__ == "_PredictScorer":
                y_pred = (y_pred > 0.5).astype(int)

        elif self.task.startswith("multi"):
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
    def __init__(self, scorer: Scorer, task: str):
        self.scorer = scorer
        self.task = task

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weight: np.ndarray,
    ) -> tuple[str, FLOAT_TYPES, bool]:
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
            if self.task.startswith("bin"):
                y_pred = (y_pred > 0.5).astype(int)
            elif self.task.startswith("multi"):
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
    def __init__(self, scorer: Scorer, task: str):
        self.scorer = scorer
        self.task = task

    @property
    def __name__(self):
        return self.scorer.name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> FLOAT_TYPES:
        if self.scorer.__class__.__name__ == "_PredictScorer":
            if self.task.startswith("bin"):
                y_pred = (y_pred > 0.5).astype(int)
            elif self.task.startswith("multi"):
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

    def __init__(
        self,
        headers: SEQUENCE_TYPES,
        spaces: SEQUENCE_TYPES,
        default_pos: str = "right",
    ):
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
    def to_cell(text: SCALAR_TYPES | str, position: str, space: INT_TYPES) -> str:
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

    def __init__(self, model: Model, n_jobs: INT_TYPES):
        self.model = model
        self.n_jobs = n_jobs
        self.T = self.model.T  # Parent runner

        if self.n_jobs == 1:
            self._table = self.create_table()
            self.T.log(self._table.print_header(), 2)
            self.T.log(self._table.print_line(), 2)

    def __call__(self, study: Study, trial: FrozenTrial):
        # The trial values are None when it fails
        if len(self.T._metric) == 1:
            score = [trial.value or np.NaN]
        else:
            score = trial.values or [np.NaN] * len(self.T._metric)

        if trial.state.name == "PRUNED" and self.model.acronym == "XGB":
            # XGBoost's eval_metric minimizes the function
            score = np.negative(score)

        params = self.model._trial_to_est(trial.user_attrs["params"])
        estimator = trial.user_attrs.get("estimator", None)

        # Add row to the trials attribute
        time_trial = (dt.now() - trial.datetime_start).total_seconds()
        time_ht = self.model._trials["time_trial"].sum() + time_trial
        self.model._trials.loc[trial.number] = pd.Series(
            {
                "params": dict(params),  # To dict because of how pandas prints it
                "estimator": estimator,
                "score": flt(score),
                "time_trial": time_trial,
                "time_ht": time_ht,
                "state": trial.state.name,
            }
        )

        # Save trials to experiment as nested runs
        if self.model._run and self.T.log_ht:
            run_name = f"{self.model.name} - {trial.number}"
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.set_tag("name", self.model.name)
                mlflow.set_tag("model", self.model._fullname)
                mlflow.set_tag("branch", self.model.branch.name)
                mlflow.set_tag("trial_state", trial.state.name)
                mlflow.set_tags(self.model._ht.get("tags", {}))

                # Mlflow only accepts params with char length <250
                pars = estimator.get_params() if estimator else params
                mlflow.log_params({k: v for k, v in pars.items() if len(str(v)) <= 250})

                mlflow.log_metric("time_trial", time_trial)
                for i, name in enumerate(self.T._metric):
                    mlflow.log_metric(f"{name}_validation", score[i])

                if estimator and self.T.log_model:
                    mlflow.sklearn.log_model(
                        sk_model=estimator,
                        artifact_path=estimator.__class__.__name__,
                        signature=infer_signature(
                            model_input=self.model.X,
                            model_output=estimator.predict(self.model.X.iloc[[0], :]),
                        ),
                        input_example=self.model.X.iloc[[0], :],
                    )

        if self.n_jobs == 1:
            sequence = {"trial": trial.number, **trial.user_attrs["params"]}
            for i, m in enumerate(self.T._metric.values()):
                best_score = rnd(max([lst(s)[i] for s in self.model.trials["score"]]))
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
        headers = [("trial", "left")] + list(self.model._ht["distributions"])
        for m in self.T._metric.values():
            headers.extend([m.name, "best_" + m.name])
        headers.extend(["time_trial", "time_ht", "state"])

        # Define the width op every column in the table
        spaces = [len(str(headers[0][0]))]
        for name, dist in self.model._ht["distributions"].items():
            # If the distribution is categorical, take the mean of the widths
            # Else take the max of 7 (minimum width) and the width of the name
            if hasattr(dist, "choices"):
                options = np.mean([len(str(x)) for x in dist.choices], dtype=int)
            else:
                options = 0

            spaces.append(max(7, len(name), options))

        spaces.extend(
            [
                max(7, len(column))
                for column in headers[1 + len(self.model._ht["distributions"]):-1]
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
    *args
        Model from which the callback is called.

    """

    max_len = 15  # Maximum trials to show at once in the plot

    def __init__(self, *args):
        self.M = args[0]
        self.T = self.M.T

        self.y1 = {i: deque(maxlen=self.max_len) for i in range(len(self.T._metric))}
        self.y2 = {i: deque(maxlen=self.max_len) for i in range(len(self.T._metric))}

        traces = []
        colors = cycle(self.T.palette)
        for met in self.T._metric:
            color = next(colors)
            traces.extend(
                [
                    go.Scatter(
                        mode="lines+markers",
                        line=dict(width=self.T.line_width, color=color),
                        marker=dict(
                            symbol="circle",
                            size=self.T.marker_size,
                            line=dict(width=1, color="white"),
                            opacity=1,
                        ),
                        # hovertemplate=f"(%{{x}}, %{{y}})<extra>{met}</extra>",
                        name=met,
                        legendgroup=met,
                        xaxis="x2",
                        yaxis="y1",
                    ),
                    go.Scatter(
                        mode="lines+markers",
                        line=dict(width=self.T.line_width, color=color),
                        marker=dict(
                            line=dict(width=1, color="rgba(255, 255, 255, 0.9)"),
                            symbol="circle",
                            size=self.T.marker_size,
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
                    title=dict(text="Score", font_size=self.T.label_fontsize),
                    anchor="x1",
                ),
                xaxis2=dict(
                    domain=(0, 1),
                    title=dict(text="Trial", font_size=self.T.label_fontsize),
                    anchor="y2",
                ),
                yaxis2=dict(
                    domain=(0, 0.29),
                    title=dict(text="d", font_size=self.T.label_fontsize),
                    anchor="x2",
                ),
                title=dict(
                    text=f"Hyperparameter tuning for {self.M._fullname}",
                    x=0.5,
                    y=1,
                    pad=dict(t=15, b=15),
                    xanchor="center",
                    yanchor="top",
                    xref="paper",
                    font_size=self.T.title_fontsize,
                ),
                legend=dict(
                    x=0.99,
                    y=0.99,
                    xanchor="right",
                    yanchor="top",
                    font_size=self.T.label_fontsize,
                    bgcolor="rgba(255, 255, 255, 0.5)",
                ),
                hovermode="x unified",
                hoverlabel=dict(font_size=self.T.label_fontsize),
                font_size=self.T.tick_fontsize,
                margin=dict(l=0, b=0, r=0, t=25 + self.T.title_fontsize, pad=0),
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

        for i, score in enumerate(lst(self.M.trials["score"][trial.number])):
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

    Calculating shap values can take much time and computational
    resources. This class 'remembers' all calculated shap values
    and reuses them when appropriate.

    Parameters
    ----------
    T: model subclass
        Model from which the instance is created.

    """

    def __init__(self, *args):
        self.T = args[0]

        self._explainer = None
        self._explanation = None
        self._shap_values = pd.Series(dtype="object")
        self._expected_value = None

    @property
    def attr(self) -> str:
        """Get the model's main prediction method.

        Returns
        -------
        str
            Name of the prediction method.

        """
        return get_attr(self.T.estimator)

    @property
    def explainer(self) -> Explainer:
        """Get shap's explainer.

        Returns
        -------
        Explainer
            Get the initialized explainer object.

        """
        if self._explainer is None:
            try:  # Fails when model does not fit standard explainers (e.g. ensembles)
                self._explainer = Explainer(self.T.estimator, self.T.X_train)
            except TypeError:
                # If method is provided as first arg, selects always Permutation
                self._explainer = Explainer(
                    model=getattr(self.T.estimator, self.attr),
                    masker=self.T.X_train,
                )

        return self._explainer

    def get_explanation(
        self,
        df: DATAFRAME_TYPES,
        target: INT_TYPES = 1,
        column: str | None = None,
        only_one: bool = False,
    ) -> Explanation:
        """Get an Explanation object.

        Parameters
        ----------
        df: dataframe
            Data set to look at (subset of the complete dataset).

        target: int, default=1
            Index of the class in the target column to look at.
            Only for multi-class classification tasks.

        column: str or None, default=None
            Column to look at. If None, look at all features.

        only_one: bool, default=False
            Whether only one row is accepted.

        Returns
        -------
        shap.Explanation
            Object containing all information (values, base_values, data).

        """
        # Get rows that still need to be calculated
        calculate = df.loc[[i for i in df.index if i not in self._shap_values.index]]
        if not calculate.empty:
            kwargs = {}

            # Minimum of 2 * n_features + 1 evals required (default=500)
            if "max_evals" in sign(self.explainer.__call__):
                kwargs["max_evals"] = 2 * self.T.n_features + 1

            # Additivity check fails sometimes for no apparent reason
            if "check_additivity" in sign(self.explainer.__call__):
                kwargs["check_additivity"] = False

            # Calculate the new shap values
            self._explanation = self.explainer(calculate, **kwargs)

            # Remember shap values in the _shap_values attribute
            for i, idx in enumerate(calculate.index):
                self._shap_values.at[idx] = self._explanation.values[i]

        # Don't use attribute to not save plot-specific changes
        explanation = copy(self._explanation)

        # Update the explanation object
        explanation.values = np.stack(self._shap_values.loc[df.index].values)
        explanation.base_values = explanation.base_values[0]
        explanation.data = self.T.X.loc[df.index, :].to_numpy()

        # Select the target values from the array
        if explanation.values.ndim > 2:
            explanation.values = explanation.values[:, :, target]
        if only_one:  # Attributes should be 1-dimensional
            explanation.values = explanation.values[0]
            explanation.data = explanation.data[0]

            # For some models like LGB it's a scalar already
            if isinstance(explanation.base_values, np.ndarray):
                explanation.base_values = explanation.base_values[target]

        if column is None:
            return explanation
        else:
            return explanation[:, df.columns.get_loc(column)]

    def get_shap_values(
        self,
        df: DATAFRAME_TYPES,
        target: INT_TYPES = 1,
        return_all_classes: bool = False,
    ) -> FLOAT_TYPES | SEQUENCE_TYPES:
        """Get shap values from the Explanation object.

        Parameters
        ----------
        df: dataframe
            Data set to look at.

        target: int, default=1
            Index of the class in the target column to look at.
            Only for multi-class classification tasks.

        return_all_classes: bool, default=False
            Whether to return one or all classes.

        Returns
        -------
        float or sequence
            Shap values.

        """
        values = self.get_explanation(df, target).values
        if return_all_classes:
            if self.T.T.task.startswith("bin") and len(values) != self.T.y.nunique():
                values = [np.array(1 - values), values]

        return values

    def get_interaction_values(self, df: DATAFRAME_TYPES) -> np.ndarray:
        """Get shap interaction values from the Explanation object.

        Parameters
        ----------
        df: dataframe
            Data set to get the interaction values from.

        Returns
        -------
        np.ndarray
            Interaction values.

        """
        return self.explainer.shap_interaction_values(df)

    def get_expected_value(
        self,
        target: INT_TYPES = 1,
        return_all_classes: bool = False,
    ) -> FLOAT_TYPES | SEQUENCE_TYPES:
        """Get the expected value of the training set.

        The expected value is either retrieved from the explainer's
        `expected_value` attribute or calculated as the mean of all
        predictions.

        Parameters
        ----------
        target: int, default=1
            Index of the class in the target column to look at.
            Only for multi-class classification tasks.

        return_all_classes: bool, default=False
            Whether to return one or all classes.

        Returns
        -------
        float or sequence
            Expected value.

        """
        if self._expected_value is None:
            # Some explainers like Permutation don't have expected_value attr
            if hasattr(self.explainer, "expected_value"):
                self._expected_value = self.explainer.expected_value
            else:
                # The expected value is the average of the model output
                self._expected_value = np.mean(getattr(self.T, f"{self.attr}_train"))

        if not return_all_classes and isinstance(self._expected_value, SEQUENCE):
            if len(self._expected_value) == self.T.y.nunique():
                return self._expected_value[target]  # Return target expected value
        elif return_all_classes and isinstance(self._expected_value, float):
            # For binary tasks, shap returns the expected value of positive class
            return [1 - self._expected_value, self._expected_value]

        return self._expected_value


class CustomDict(MutableMapping):
    """Custom ordered dictionary.

    The main differences with the Python dictionary are:

    - It has ordered entries.
    - Key requests are case-insensitive.
    - Returns a subset of itself using getitem with a list of keys or slice.
    - It allows getting an item from an index position.
    - It can insert key value pairs at a specific position.
    - Replace method to change a key or value if key exists.
    - Min method to return all elements except one.

    """

    @staticmethod
    def _conv(key):
        return key.lower() if isinstance(key, str) else key

    def _get_key(self, key):
        if isinstance(key, INT) and key not in self.__keys:
            return self.__keys[key]
        else:
            for k in self.__keys:
                if self._conv(k) == self._conv(key):
                    return k

        raise KeyError(key)

    def __init__(self, iterable_or_mapping=None, **kwargs):
        """Class initializer.

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
        if isinstance(key, list):
            return self.__class__({self._get_key(k): self[k] for k in key})
        elif isinstance(key, slice):
            return self.__class__({k: self[k] for k in self.__keys[key]})
        elif self._conv(key) in self.__data:
            return self.__data[self._conv(key)]  # From key
        else:
            try:
                return self.__data[self._conv(self.__keys[key])]  # From index
            except (TypeError, IndexError):
                raise KeyError(key)

    def __setitem__(self, key, value):
        if key not in self:
            self.__keys.append(key)
        self.__data[self._conv(key)] = value

    def __delitem__(self, key):
        key = self._get_key(key)
        self.__keys.remove(key)
        del self.__data[self._conv(key)]

    def __iter__(self):
        yield from self.keys()

    def __len__(self):
        return len(self.__keys)

    def __contains__(self, key):
        return self._conv(key) in self.__data

    def __repr__(self):
        return pprint.pformat(dict(self), sort_dicts=False)

    def __reversed__(self):
        yield from reversed(list(self.keys()))

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

    def replace_key(self, key, new_key):
        if key in self:
            self.insert(self.__keys.index(self._get_key(key)), new_key, self[key])
            self.__delitem__(key)

    def replace_value(self, key, value=None):
        if key in self:
            self[key] = value

    def min(self, key):
        return self.__class__(
            {k: v for k, v in self.items() if self._conv(k) != self._conv(key)}
        )


# Functions ======================================================== >>

def flt(item: Any) -> Any:
    """Return item from sequence with just that item.

    Parameters
    ----------
    item: Any
        Item or sequence.

    Returns
    -------
    Any
        Object.

    """
    return item[0] if isinstance(item, SEQUENCE) and len(item) == 1 else item


def lst(item: Any) -> SEQUENCE_TYPES:
    """Make a sequence from an item if not a sequence already.

    Parameters
    ----------
    item: Any
        Item or sequence.

    Returns
    -------
    sequence
        Item as sequence with length 1 or provided sequence.

    """
    return item if isinstance(item, (dict, CustomDict, *SEQUENCE)) else [item]


def it(item: Any) -> Any:
    """Convert rounded floats to int.

    If the provided item is not numerical, return as is.

    Parameters
    ----------
    item: Any
        Item to check for rounding.

    Returns
    -------
    Any
        Rounded float or non-numerical item.

    """
    try:
        is_equal = int(item) == float(item)
    except ValueError:  # Item may not be numerical
        return item

    return int(item) if is_equal else float(item)


def rnd(item: Any, decimals: INT_TYPES = 4) -> Any:
    """Round a float to the `decimals` position.

    If the value is not a float, return as is.

    Parameters
    ----------
    item: Any
        Numerical item to round.

    decimals: int, default=4
        Decimal position to round to.

    Returns
    -------
    Any
        Rounded float or non-numerical item.

    """
    return round(item, decimals) if np.issubdtype(type(item), np.floating) else item


def divide(a: SCALAR_TYPES, b: SCALAR_TYPES) -> SCALAR_TYPES:
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


def sign(obj: callable) -> OrderedDict:
    """Get the parameters of an object.

    Parameters
    ----------
    obj: Callable
        Object from which to get the parameters.

    Returns
    -------
    OrderedDict
        Object's parameters.

    """
    return signature(obj).parameters


def merge(*args) -> DATAFRAME_TYPES:
    """Concatenate pandas objects column-wise.

    Empty objects are ignored.

    Parameters
    ----------
    *args
        Objects to concatenate.

    Returns
    -------
    dataframe
        Concatenated dataframe.

    """
    if len(args := [elem for elem in args if not elem.empty]) == 1:
        return args[0]
    else:
        return bk.concat([*args], axis=1)


def get_cols(elem: PANDAS_TYPES) -> list[SERIES]:
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
    if isinstance(elem, SERIES):
        return [elem]
    else:
        return [elem[col] for col in elem]


def variable_return(
    X: DATAFRAME_TYPES | None,
    y: SERIES_TYPES | None,
) -> DATAFRAME_TYPES | SERIES_TYPES | tuple[DATAFRAME_TYPES, SERIES_TYPES]:
    """Return one or two arguments depending on which is None.

    This utility is used to make methods return only the provided
    data set.

    Parameters
    ----------
    X: dataframe or None
        Feature set.

    y: series or None
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


def is_sparse(df: DATAFRAME_TYPES) -> bool:
    """Check if the dataframe is sparse.

    A data set is considered sparse if any of its columns is sparse.

    Parameters
    ----------
    df: dataframe
        Data set to check.

    Returns
    -------
    bool
        Whether the data set is sparse.

    """
    return any(bk.api.types.is_sparse(df[col]) for col in df)


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


def check_predict_proba(models: SEQUENCE_TYPES, method: str):
    """Raise an error if a model doesn't have a `predict_proba` method.

    Parameters
    ----------
    models: sequence of [Models][]
        Models to check for the attribute.

    method: str
        Name of the method from which the check is called.

    """
    for m in [m for m in models if m.name != "Vote"]:
        if not hasattr(m.estimator, "predict_proba"):
            raise AttributeError(
                f"The {method} method is only available for "
                f"models with a predict_proba method, got {m.name}."
            )


def check_scaling(df: DATAFRAME_TYPES) -> bool:
    """Check if the data is scaled.

    A data set is considered scaled when the mean of the mean of
    all columns lies between -0.05 and 0.05 and the mean of the
    standard deviation of all columns lies between 0.85 and 1.15.
    Binary columns are excluded from the calculation.

    Parameters
    ----------
    df: dataframe
        Data set to check.

    Returns
    -------
    bool
        Whether the data set is scaled.

    """
    # Remove binary columns
    df = df[[c for c in df if ~np.isin(df[c].unique(), [0, 1]).all()]]

    mean = df.mean(numeric_only=True).mean()
    std = df.std(numeric_only=True).mean()
    return -0.05 < mean < 0.05 and 0.85 < std < 1.15


def get_attr(estimator: Estimator) -> str:
    """Get the main estimator's prediction method.

    Get predict_proba, decision_function or predict in that order
    when available.

    Returns
    -------
    str
        Name of the prediction method.

    """
    for attr in ("predict_proba", "decision_function", "predict"):
        if hasattr(estimator, attr):
            return attr


def get_versions(models: CustomDict) -> dict:
    """Get the versions of ATOM and the models' packages.

    Parameters
    ----------
    models: CustomDict
        Models for which to check the version.

    Returns
    -------
    dict
        Current versions of ATOM and models.

    """
    versions = {"atom": __version__}
    for name, model in models.items():
        module = model.estimator.__module__.split(".")[0]
        versions[module] = sys.modules.get(module, import_module(module)).__version__

    return versions


def get_corpus(df: DATAFRAME_TYPES) -> SERIES_TYPES:
    """Get text column from a dataframe.

    The text column should be called `corpus` (case insensitive).

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


def get_pl_name(name: str, steps: tuple[str, Estimator], counter: int = 1) -> str:
    """Get the estimator name for a pipeline.

    This utility checks if there already exists an estimator with
    that name. If so, add a counter at the end of the name.

    Parameters
    ----------
    name: str
        Name of the estimator.

    steps: tuple
        Current steps in the pipeline.

    counter: int, default=1
        Numerical counter to add to the step's name.

    Returns
    -------
    str
        New name for the pipeline's step.

    """
    og_name = name
    while name.lower() in [elem[0] for elem in steps]:
        counter += 1
        name = og_name + str(counter)

    return name.lower()


def get_best_score(item: Model | SERIES_TYPES, metric: int = 0) -> FLOAT_TYPES:
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


def time_to_str(t: int):
    """Convert time to a nice string representation.

    The resulting string is of format 00h:00m:00s or 1.000s if
    under 1 min.

    Parameters
    ----------
    t: int
        Time to convert (in seconds).

    Returns
    -------
    str
        Time representation.

    """
    h = t // 3600
    m = t % 3600 // 60
    s = t % 3600 % 60
    if not h and not m:  # Only seconds
        return f"{s:.3f}s"
    elif not h:  # Also minutes
        return f"{m:02.0f}m:{s:02.0f}s"
    else:  # Also hours
        return f"{h:02.0f}h:{m:02.0f}m:{s:02.0f}s"


def n_cols(data: X_TYPES | Y_TYPES | None) -> int:
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


def to_df(
    data: X_TYPES | None,
    index: SEQUENCE_TYPES | INDEX_TYPES = None,
    columns: SEQUENCE_TYPES | None = None,
    dtype: str | dict | np.dtype | None = None,
) -> DATAFRAME_TYPES | None:
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
    if data is not None and not isinstance(data, bk.DataFrame):
        # Assign default column names (dict already has column names)
        if not isinstance(data, dict) and columns is None:
            columns = [f"x{str(i)}" for i in range(n_cols(data))]

        if hasattr(data, "to_pandas") and bk.__name__ == "pandas":
            data = data.to_pandas()  # Convert cuML to pandas
        elif sparse.issparse(data):
            # Create dataframe from sparse matrix
            data = bk.DataFrame.sparse.from_spmatrix(
                data=data,
                index=getattr(data, "index", index),
                columns=getattr(data, "columns", columns),
            )
        else:
            # Get attributes from pandas df for modin or vice-versa
            data = bk.DataFrame(
                data=data,
                index=getattr(data, "index", index),
                columns=getattr(data, "columns", columns),
            )

        if dtype is not None:
            data = data.astype(dtype)

    return data


def to_series(
    data: SEQUENCE_TYPES | None,
    index: SEQUENCE_TYPES | INDEX_TYPES | None = None,
    name: str = "target",
    dtype: str | np.dtype | None = None,
) -> SERIES_TYPES | None:
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
    if data is not None and not isinstance(data, bk.Series):
        if hasattr(data, "to_pandas") and bk.__name__ == "pandas":
            data = data.to_pandas()  # Convert cuML to pandas
        else:
            # Flatten for arrays with shape (n_samples, 1), sometimes returned by cuML
            # Get attributes from pandas series for modin or vice-versa
            data = bk.Series(
                data=np.array(data, dtype="object").ravel().tolist(),
                index=getattr(data, "index", index),
                name=getattr(data, "name", name),
                dtype=getattr(data, "dtype", dtype),
            )

    return data


def to_pandas(
    data: SEQUENCE_TYPES | None,
    index: SEQUENCE_TYPES | INDEX_TYPES | None = None,
    columns: SEQUENCE_TYPES | None = None,
    name: str = "target",
    dtype: str | dict | np.dtype | None = None,
) -> PANDAS_TYPES | None:
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
    estimator: Estimator,
    exception: bool = True,
    attributes: str | SEQUENCE_TYPES | None = None,
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
            if isinstance(value := getattr(estimator, attr), PANDAS):
                return value.empty
            else:
                return not value

    is_fitted = False
    if hasattr(estimator, "_is_fitted"):
        is_fitted = estimator._is_fitted
    elif attributes is None:
        # Check for attributes from a fitted object
        for v in vars(estimator):
            if v.endswith("_") and not v.startswith("__"):
                is_fitted = True
                break
    elif not all(check_attr(attr) for attr in lst(attributes)):
        is_fitted = True

    if not is_fitted:
        if exception:
            raise NotFittedError(
                f"This {type(estimator).__name__} instance is not"
                " fitted yet. Call 'fit' or 'run' with appropriate"
                " arguments before using this estimator."
            )
        else:
            return False

    return True


def create_acronym(fullname: str) -> str:
    """Create an acronym for an estimator.

    The acronym consists of the capital letters in the name if
    there are at least two. If not, the entire name is used.

    Parameters
    ----------
    fullname: str
        Estimator's __name__.

    Returns
    -------
    str
        Created acronym.

    """
    from atom.models import MODELS

    acronym = "".join([c for c in fullname if c.isupper()])
    if len(acronym) < 2 or acronym.lower() in MODELS:
        return fullname
    else:
        return acronym


def get_custom_scorer(metric: str | Callable | Scorer) -> Scorer:
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
            scorer.name = scorer._score_func.__name__
        else:
            raise ValueError(
                f"Unknown value for the metric parameter, got {metric}. "
                f"Choose from: {', '.join(get_scorer_names())}."
            )

    elif hasattr(metric, "_score_func"):  # Scoring is a scorer
        scorer = copy(metric)

        # Some scorers use default kwargs
        default_kwargs = ("precision", "recall", "f1", "jaccard")
        if any(name in scorer._score_func.__name__ for name in default_kwargs):
            if not scorer._kwargs:
                scorer._kwargs = {"average": "binary"}

        for key in get_scorer_names():
            if scorer.__dict__ == get_scorer(key).__dict__:
                scorer.name = key
                break

    else:  # Scoring is a function with signature metric(y, y_pred)
        scorer = make_scorer(score_func=metric)
        scorer.name = scorer._score_func.__name__

    return scorer


def infer_task(y: PANDAS_TYPES, goal: str = "class") -> str:
    """Infer the task corresponding to a target column.

    If goal is provided, only look at number of unique values to
    determine the classification task.

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

    if y.ndim > 1:
        if y.isin([0, 1]).all().all():
            return "multilabel classification"
        else:
            return "multiclass-multioutput classification"
    elif isinstance(y.iloc[0], SEQUENCE):
        return "multilabel classification"
    elif y.nunique() == 1:
        raise ValueError(f"Only found 1 target value: {y.unique()[0]}")
    elif y.nunique() == 2:
        return "binary classification"
    else:
        return "multiclass classification"


def get_feature_importance(
    est: Predictor,
    attributes: SEQUENCE_TYPES | None = None,
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
    data = None
    if not attributes:
        attributes = ("scores_", "coef_", "feature_importances_")

    try:
        data = getattr(est, next(attr for attr in attributes if hasattr(est, attr)))
    except StopIteration:
        # Get the mean value for meta-estimators
        if hasattr(est, "estimators_"):
            if all(hasattr(x, "feature_importances_") for x in est.estimators_):
                data = np.mean(
                    [fi.feature_importances_ for fi in est.estimators_],
                    axis=0,
                )
            elif all(hasattr(x, "coef_") for x in est.estimators_):
                data = np.mean([fi.coef_ for fi in est.estimators_], axis=0)

    if data is not None:
        if data.ndim == 1:
            data = np.abs(data)
        else:
            data = np.linalg.norm(data, axis=np.argmin(data.shape), ord=1)

        return data


# Pipeline functions =============================================== >>

def name_cols(
    array: np.ndarray,
    original_df: DATAFRAME_TYPES,
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
        mask = original_df.apply(lambda c: np.array_equal(c, col, equal_nan=True))
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
    df: DATAFRAME_TYPES,
    original_df: DATAFRAME_TYPES,
    col_names: list[str],
) -> DATAFRAME_TYPES:
    """Reorder th   e columns to their original order.

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
    transformer: Transformer,
    X: X_TYPES | None = None,
    y: Y_TYPES | None = None,
    message: str | None = None,
    **fit_params,
):
    """Fit the data using one estimator.

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
            args = []
            if "X" in (params := sign(transformer.fit)):
                if X is not None:
                    inc, exc = getattr(transformer, "_cols", (list(X.columns), None))
                    if inc or exc:  # Skip if inc=[] (happens when columns=-1)
                        args.append(X[inc or [c for c in X.columns if c not in exc]])

                # X is required but has not been provided
                if len(args) == 0:
                    if params["X"].default != Parameter.empty:
                        args.append(params["X"].default)  # Fill X with default
                    else:
                        raise ValueError(
                            "Exception while trying to fit transformer "
                            f"{transformer.__class__.__name__}. Parameter "
                            "X is required but not provided."
                        )

            if "y" in params and y is not None:
                args.append(y)

            transformer.fit(*args, **fit_params)


def transform_one(
    transformer: Transformer,
    X: X_TYPES | None = None,
    y: Y_TYPES | None = None,
    method: str = "transform",
) -> tuple[DATAFRAME_TYPES | None, SERIES_TYPES | None]:
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

    def prepare_df(out: X_TYPES) -> DATAFRAME_TYPES:
        """Convert to df and set correct column names and order.

        Parameters
        ----------
        out: dataframe-like
            Data returned by the transformation.

        Returns
        -------
        dataframe
            Final dataset.

        """
        use_cols = inc or [c for c in X.columns if c not in exc]

        # Convert to pandas and assign proper column names
        if not isinstance(out, DATAFRAME):
            if hasattr(transformer, "get_feature_names"):
                columns = transformer.get_feature_names()
            elif hasattr(transformer, "get_feature_names_out"):
                columns = transformer.get_feature_names_out()
            else:
                columns = name_cols(out, X, use_cols)

            out = to_df(out, index=X.index, columns=columns)

        # Reorder columns if only a subset was used
        if len(use_cols) != X.shape[1]:
            return reorder_cols(transformer, out, X, use_cols)
        else:
            return out

    X = to_df(X, index=getattr(y, "index", None))
    y = to_pandas(y, index=getattr(X, "index", None))

    args = []
    if "X" in (params := sign(getattr(transformer, method))):
        if X is not None:
            inc, exc = getattr(transformer, "_cols", (list(X.columns), []))
            if inc or exc:  # Skip if inc=[] (happens when columns=-1)
                args.append(X[inc or [c for c in X.columns if c not in exc]])

        # X is required but has not been provided
        if len(args) == 0:
            if params["X"].default != Parameter.empty:
                args.append(params["X"].default)  # Fill X with default
            else:
                return X, y  # If X is needed, skip the transformer

    if "y" in params:
        if y is not None:
            args.append(y)
        elif "X" not in params:
            return X, y  # If y is None and no X in transformer, skip the transformer

    # Transform can return X, y or both
    if isinstance(out := getattr(transformer, method)(*args), tuple):
        new_X = prepare_df(out[0])
        new_y = to_pandas(
            data=out[1],
            index=new_X.index,
            name=getattr(y, "name", None),
            columns=getattr(y, "columns", None),
        )
    elif "X" in params and X is not None and (inc != [] or exc != []):
        # X in -> X out
        new_X = prepare_df(out)
        new_y = y if y is None else y.set_axis(new_X.index)
    else:
        # X not in -> output must be y
        new_y = to_pandas(
            data=out,
            index=y.index,
            name=getattr(y, "name", None),
            columns=getattr(y, "columns", None),
        )
        new_X = X if X is None else X.set_index(new_y.index)

    return new_X, new_y


def fit_transform_one(
    transformer: Transformer,
    X: X_TYPES | None = None,
    y: Y_TYPES | None = None,
    message: str | None = None,
    **fit_params,
) -> tuple[DATAFRAME_TYPES | None, SERIES_TYPES | None]:
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
    transformer: Transformer,
    branch: Any,
    data: tuple[DATAFRAME_TYPES, SERIES_TYPES] | None = None,
    verbose: int | None = None,
    method: str = "transform",
) -> tuple[DATAFRAME_TYPES, SERIES_TYPES]:
    """Applies a transformer on a branch.

    This function is generic and should work for all
    methods with parameters X and/or y.

    Parameters
    ----------
    transformer: Transformer
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

    series
        Target column.

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
    if verbose is not None:
        if verbose < 0 or verbose > 2:
            raise ValueError(
                "Invalid value for the verbose parameter."
                f"Value should be between 0 and 2, got {verbose}."
            )
        elif hasattr(transformer, "verbose"):
            vb = transformer.verbose  # Save original verbosity
            transformer.verbose = verbose

    X, y = transform_one(transformer, X_og, y_og, method)

    # Apply changes to the branch
    if not data:
        if transformer._train_only:
            branch.train = merge(X, branch.y_train if y is None else y)
        else:
            branch._data = merge(X, branch.y if y is None else y)

            # Since y can change number of columns, reassign index
            branch._idx[0] = len(get_cols(y))

            # Since rows can be removed from train and test, reassign indices
            branch._idx[1] = [idx for idx in branch._idx[1] if idx in X.index]
            branch._idx[2] = [idx for idx in branch._idx[2] if idx in X.index]

        if branch.T.index is False:
            branch._data = branch.dataset.reset_index(drop=True)
            branch._idx = [
                branch._idx[0],
                branch._data.index[:len(branch._idx[1])],
                branch._data.index[-len(branch._idx[2]):],
            ]

    # Back to the original verbosity
    if verbose is not None and hasattr(transformer, "verbose"):
        transformer.verbose = vb

    return X, y


# Patches ========================================================== >>

def score(f: callable) -> callable:
    """Patch decorator for sklearn's _score function.

    Monkey patch for sklearn.model_selection._validation._score
    function to score pipelines that drop samples during transforming.

    """

    def wrapper(*args, **kwargs):
        args = list(args)  # Convert to list for item assignment
        if len(args[0]) > 1:  # Has transformers
            args[1], args[2] = args[0][:-1].transform(args[1], args[2])

        # Return f(final_estimator, X_transformed, y_transformed, ...)
        return f(args[0][-1], *tuple(args[1:]), **kwargs)

    return wrapper


# Decorators ======================================================= >>

def has_task(task: str) -> callable:
    """Check that the instance has a specific task.

    Parameters
    ----------
    task: str
        Task to check.

    """

    def check(self: Any) -> bool:
        if hasattr(self, "task"):
            return task in self.task
        else:
            return task in self.T.task

    return check


def has_attr(attr: str) -> callable:
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


def estimator_has_attr(attr: str) -> callable:
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


def composed(*decs) -> callable:
    """Add multiple decorators in one line.

    Parameters
    ----------
    decs: tuple
        Decorators to run.

    """

    def decorator(f: callable) -> callable:
        for dec in reversed(decs):
            f = dec(f)
        return f

    return decorator


def crash(f: callable, cache: dict = {"last_exception": None}) -> callable:
    """Save program crashes to log file.

    We use a mutable argument to cache the last exception raised. If
    the current exception is the same (happens when there is an error
    catch or multiple calls to crash), it's not re-written in the logger.

    """

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        logger = args[0].logger if hasattr(args[0], "logger") else args[0].T.logger

        if logger is not None:
            try:  # Run the function
                return f(*args, **kwargs)

            except Exception as ex:
                # If exception is not same as last, write to log
                if ex is not cache["last_exception"]:
                    cache["last_exception"] = ex
                    logger.exception("Exception encountered:")

                raise ex
        else:
            return f(*args, **kwargs)

    return wrapper


def method_to_log(f: callable) -> callable:
    """Save called functions to log file."""

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        # Get logger for calls from models
        logger = args[0].logger if hasattr(args[0], "logger") else args[0].T.logger

        if logger is not None:
            if f.__name__ != "__init__":
                logger.info("")
            logger.info(f"{args[0].__class__.__name__}.{f.__name__}()")

        return f(*args, **kwargs)

    return wrapper


def plot_from_model(f: callable) -> callable:
    """If a plot is called from a model, adapt the `models` parameter."""

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        if hasattr(args[0], "T"):
            return f(args[0].T, args[0].name, *args[1:], **kwargs)
        else:
            return f(*args, **kwargs)

    return wrapper


# Custom scorers =================================================== >>

def true_negatives(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> INT_TYPES:
    return confusion_matrix(y_true, y_pred).ravel()[0]


def false_positives(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> INT_TYPES:
    return confusion_matrix(y_true, y_pred).ravel()[1]


def false_negatives(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> INT_TYPES:
    return confusion_matrix(y_true, y_pred).ravel()[2]


def true_positives(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> INT_TYPES:
    return confusion_matrix(y_true, y_pred).ravel()[3]


def false_positive_rate(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> FLOAT_TYPES:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def true_positive_rate(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> FLOAT_TYPES:
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)


def true_negative_rate(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> FLOAT_TYPES:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def false_negative_rate(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> FLOAT_TYPES:
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp)

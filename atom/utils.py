# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing utility constants, classes and functions.

"""

import pprint
import sys
from collections import OrderedDict, deque
from collections.abc import MutableMapping
from copy import copy
from datetime import datetime as dt
from functools import wraps
from importlib import import_module
from inspect import Parameter, signature
from itertools import cycle
from logging import DEBUG, FileHandler, Formatter, Logger, getLogger
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from matplotlib.colors import to_rgba
from mlflow.models.signature import infer_signature
from optuna.study import Study
from optuna.trial import FrozenTrial
from scipy import sparse
from shap import Explainer, Explanation
from sklearn.inspection._partial_dependence import (
    _grid_from_X, _partial_dependence_brute,
)
from sklearn.metrics import (
    confusion_matrix, get_scorer, get_scorer_names, make_scorer,
    matthews_corrcoef,
)
from sklearn.utils import _print_elapsed_time, _safe_indexing


# Constants ======================================================== >>

# Current library version
__version__ = "5.0.1"

# Group of variable types for isinstance
SEQUENCE = (list, tuple, np.ndarray, pd.Series)

# Groups of variable types for type hinting
INT = Union[int, np.integer]
FLOAT = Union[float, np.floating]
SCALAR = Union[INT, FLOAT]
SEQUENCE_TYPES = Union[SEQUENCE]
PANDAS_TYPES = Union[pd.Series, pd.DataFrame]
X_TYPES = Union[iter, dict, list, tuple, np.ndarray, sparse.spmatrix, pd.DataFrame]
Y_TYPES = Union[INT, str, dict, SEQUENCE_TYPES]

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
    def get_final_error(error: FLOAT, weight: FLOAT) -> FLOAT:
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
    ) -> Tuple[str, FLOAT, bool]:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> FLOAT:
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
    def to_cell(text: Union[SCALAR, str], position: str, space: int) -> str:
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

    def __init__(self, model: Model, n_jobs: int):
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
        df: pd.DataFrame,
        target: int = 1,
        column: Optional[str] = None,
        only_one: bool = False,
    ) -> Explanation:
        """Get an Explanation object.

        Parameters
        ----------
        df: pd.DataFrame
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
        df: pd.DataFrame,
        target: int = 1,
        return_all_classes: bool = False,
    ) -> Union[FLOAT, SEQUENCE_TYPES]:
        """Get shap values from the Explanation object.

        Parameters
        ----------
        df: pd.DataFrame
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

    def get_interaction_values(self, df: pd.DataFrame) -> np.ndarray:
        """Get shap interaction values from the Explanation object.

        Parameters
        ----------
        df: pd.DataFrame
            Data set to get the interaction values from.

        Returns
        -------
        np.ndarray
            Interaction values.

        """
        return self.explainer.shap_interaction_values(df)

    def get_expected_value(
        self,
        target: int = 1,
        return_all_classes: bool = False,
    ) -> Union[FLOAT, SEQUENCE_TYPES]:
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
        if isinstance(key, (int, np.integer)) and key not in self.__keys:
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


def rnd(item: Any, decimals: int = 4) -> Any:
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


def sign(obj: Callable) -> OrderedDict:
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


def merge(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Add a target column to the feature set.

    Parameters
    ----------
    X: pd.DataFrame
        Feature set.

    y: pd.Series
        Target column.

    Returns
    -------
    pd.DataFrame
        Complete dataset.

    """
    return X.merge(y.to_frame(), left_index=True, right_index=True)


def variable_return(
    X: Optional[pd.DataFrame], y: Optional[pd.Series]
) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.Series]]:
    """Return one or two arguments depending on which is None.

    This utility is used to mnake methods return only the provided
    data set.

    Parameters
    ----------
    X: pd.DataFrame or None
        Feature set.

    y: pd.Series or None
        Target column.

    Returns
    -------
    pd.DataFrame, pd.Series or tuple
        Data sets that are not None.

    """
    if y is None:
        return X
    elif X is None:
        return y
    else:
        return X, y


def is_sparse(df: pd.DataFrame) -> bool:
    """Check if the dataframe is sparse.

    A data set is considered sparse if any of its columns is sparse.

    Parameters
    ----------
    df: pd.DataFrame
        Data set to check.

    Returns
    -------
    bool
        Whether the data set is sparse.

    """
    return any(pd.api.types.is_sparse(df[col]) for col in df)


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


def check_scaling(df: pd.DataFrame) -> bool:
    """Check if the data is scaled.

    A data set is considered scaled when the mean of the mean of
    all columns lies between -0.05 and 0.05 and the mean of the
    standard deviation of all columns lies between 0.85 and 1.15.
    Binary columns are excluded from the calculation.

    Parameters
    ----------
    df: pd.DataFrame
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


def get_corpus(df: pd.DataFrame) -> pd.Series:
    """Get text column from a dataframe.

    The text column should be called `corpus` (case insensitive).

    Parameters
    ----------
    df: pd.DataFrame
        Data set from which to get the corpus.

    Returns
    -------
    pd.Series
        Column with text values.

    """
    try:
        return next(col for col in df if col.lower() == "corpus")
    except StopIteration:
        raise ValueError("The provided dataset does not contain a text corpus!")


def get_pl_name(name: str, steps: Tuple[str, Estimator], counter: int = 1) -> str:
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


def get_best_score(item: Union[Model, pd.Series], metric: int = 0) -> FLOAT:
    """Returns the best score for a model.

    The best score is the `score_bootstrap` or `score_test`, checked
    in that order.

    Parameters
    ----------
    item: model or pd.Series
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


def to_df(
    data: Optional[X_TYPES],
    index: Optional[Union[SEQUENCE_TYPES, pd.Index, pd.MultiIndex]] = None,
    columns: Optional[SEQUENCE_TYPES] = None,
    dtypes: Optional[Union[str, dict, np.dtype]] = None,
) -> Optional[pd.DataFrame]:
    """Convert a dataset to pd.Dataframe.

    If the data is None, a pd.Dataframe or cudf.Dataframe, returns
    as is.

    Parameters
    ----------
    data: dataframe-like or None
        Dataset to convert to a dataframe.  If None or already a
        dataframe, return unchanged.

    index: sequence, pd.Index, pd.Multiindex or None, default=None
        Values for the index.

    columns: sequence or None, default=None
        Name of the columns. Use None for automatic naming.

    dtypes: str, dict, np.dtype or None, default=None
        Data types for the output columns. If None, the types are
        inferred from the data.

    Returns
    -------
    pd.DataFrame or None
        Transformed dataframe.

    """
    # Get number of columns (list/tuple have no shape and sp.matrix has no index)
    n_cols = lambda data: data.shape[1] if hasattr(data, "shape") else len(data[0])

    if data is not None and not data.__class__.__name__ == "DataFrame":
        # Assign default column names (dict already has column names)
        if not isinstance(data, dict) and columns is None:
            columns = [f"x{str(i)}" for i in range(n_cols(data))]

        # Create dataframe from sparse matrix or directly from data
        if sparse.issparse(data):
            data = pd.DataFrame.sparse.from_spmatrix(data, index, columns)
        else:
            data = pd.DataFrame(data, index, columns)

        if dtypes is not None:
            data = data.astype(dtypes)

    return data


def to_series(
    data: Optional[SEQUENCE_TYPES],
    index: Optional[Union[SEQUENCE_TYPES, pd.Index, pd.MultiIndex]] = None,
    name: str = "target",
    dtype: Optional[Union[str, np.dtype]] = None,
) -> Optional[pd.Series]:
    """Convert a sequence to pd.Series.

    If the data is None, a pd.Series or cudf.Series, returns as is.

    Parameters
    ----------
    data: sequence or None
        Data to convert. If None, return unchanged.

    index: sequence, pd.Index, pd.Multiindex or None, default=None
        Values for the index.

    name: str, default="target"
        Name of the series.

    dtype: str, np.dtype or None, default=None
        Data type for the output series. If None, the type is
        inferred from the data.

    Returns
    -------
    pd.Series or None
        Transformed series.

    """
    if data is not None and not data.__class__.__name__ == "Series":
        data = pd.Series(data, index=index, name=name, dtype=dtype)

    return data


def prepare_logger(
    logger: Optional[Union[str, Logger]], class_name: str,
) -> Optional[Logger]:
    """Create a new logger and corresponding `.log` file.

    Parameters
    ----------
    logger: str, Logger or None
        - If None: Doesn't create a logging file.
        - If str: Name of the log file. Use "auto" for automatic name.
        - Else: Python `logging.Logger` instance.

    class_name: str
        Name of the class from which the function is called.
        Used for default name creation when log="auto".

    Returns
    -------
    Logger or None
        Logger object.

    """
    if not logger:  # Empty string or None
        return None
    elif isinstance(logger, str):
        # Prepare the FileHandler's name
        if not logger.endswith(".log"):
            logger += ".log"
        if logger == "auto.log" or logger.endswith("/auto.log"):
            current = dt.now().strftime("%d%b%y_%Hh%Mm%Ss")
            logger = logger.replace("auto", class_name + "_" + current)

        # Define file handler and set formatter
        file_handler = FileHandler(logger)
        formatter = Formatter("%(asctime)s - %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)

        # Define logger
        logger = getLogger(class_name + "_logger")
        logger.setLevel(DEBUG)
        logger.propagate = False
        if logger.hasHandlers():  # Remove existing handlers
            logger.handlers.clear()
        logger.addHandler(file_handler)  # Add file handler to logger

    return logger


def check_is_fitted(
    estimator: Estimator,
    exception: bool = True,
    attributes: Optional[Union[str, SEQUENCE_TYPES]] = None,
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
            if isinstance(value := getattr(estimator, attr), (pd.DataFrame, pd.Series)):
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


def get_custom_scorer(metric: Union[str, Callable, Scorer]) -> Scorer:
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


def infer_task(y: pd.Series, goal: str = "class") -> str:
    """Infer the task corresponding to a target column.

    If goal is provided, only look at number of unique values to
    determine the classification task.

    Parameters
    ----------
    y: pd.Series
        Target column from which to infer the task.

    goal: str, default="class"
        Classification or regression goal.

    Returns
    -------
    str
        Inferred task.

    """
    if goal == "reg":
        return "regression"

    if y.nunique() == 1:
        raise ValueError(f"Only found 1 target value: {y.unique()[0]}")
    elif y.nunique() == 2:
        return "binary classification"
    else:
        return "multiclass classification"


def partial_dependence(
    estimator: Predictor, X: pd.DataFrame, features: Union[int, SEQUENCE_TYPES]
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Calculate the partial dependence of features.

    Partial dependence of a feature (or a set of features) corresponds
    to the average response of an estimator for each possible value of
    the feature. Code from sklearn's _partial_dependence.py. Note that
    this implementation always uses method="brute", grid_resolution=100
    and percentiles=(0.05, 0.95).

    Parameters
    ----------
    estimator: Predictor
        Model estimator to use.

    X: pd.DataFrame
        Feature set used to generate a grid of values for the target
        features (where the partial dependence is evaluated), and
        also to generate values for the complement features.

    features: int or sequence
        The feature or pair of interacting features for which the
        partial dependency should be computed.

    Returns
    -------
    np.array
        Average of the predictions.

    np.array
        All predictions.

    list
        Values used for the predictions.

    """
    grid, values = _grid_from_X(_safe_indexing(X, features, axis=1), (0.05, 0.95), 100)

    avg_pred, pred = _partial_dependence_brute(estimator, grid, features, X, "auto")

    # Reshape to (n_targets, n_values_feature,)
    avg_pred = avg_pred.reshape(-1, *[val.shape[0] for val in values])

    # Reshape to (n_targets, n_rows, n_values_feature)
    pred = pred.reshape(-1, X.shape[0], *[val.shape[0] for val in values])

    return avg_pred, pred, values


def get_feature_importance(
    est: Predictor, attributes: Optional[SEQUENCE_TYPES] = None
) -> list:
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
    list
        Estimator's feature importance.

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
            data = np.linalg.norm(data, axis=0, ord=1)

        return list(data)


# Pipeline functions =============================================== >>

def name_cols(
    array: np.ndarray, original_df: pd.DataFrame, col_names: List[str]
) -> List[str]:
    """Get the column names after a transformation.

    If the number of columns is unchanged, the original
    column names are returned. Else, give the column a
    default name if the column values changed.

    Parameters
    ----------
    array: np.array
        Transformed dataset.

    original_df: pd.DataFrame
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
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    col_names: List[str],
) -> pd.DataFrame:
    """Reorder th   e columns to their original order.

    This function is necessary in case only a subset of the
    columns in the dataset was used. In that case, we need
    to reorder them to their original order.

    Parameters
    ----------
    transformer: Transformer
        Instance that transformed `df`.

    df: pd.DataFrame
        Dataset to reorder.

    original_df: pd.DataFrame
        Original dataset (states the order).

    col_names: list of str
        Names of the columns used in the transformer.

    Returns
    -------
    pd.DataFrame
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
    X: Optional[X_TYPES] = None,
    y: Optional[Y_TYPES] = None,
    message: Optional[str] = None,
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

    y: int, str, dict, sequence or None, default=None
        Target column corresponding to X.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - Else: Array with shape=(n_samples,) to use as target.

    message: str or None
        Short message. If None, nothing will be printed.

    **fit_params
        Additional keyword arguments for the fit method.

    """
    X = to_df(X, index=getattr(y, "index", None))
    y = to_series(y, index=getattr(X, "index", None))

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
    X: Optional[X_TYPES] = None,
    y: Optional[Y_TYPES] = None,
    method: str = "transform",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Transform the data using one estimator.

    Parameters
    ----------
    transformer: Transformer
        Instance to fit.

    X: dataframe-like or None, default=None
        Feature set with shape=(n_samples, n_features). If None,
        X is ignored.

    y: int, str, dict, sequence or None, default=None
        Target column corresponding to X.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - Else: Array with shape=(n_samples,) to use as target.

    method: str, default="transform"
        Method to apply: transform or inverse_transform.

    Returns
    -------
    pd.DataFrame or None
        Feature set. Returns None if not provided.

    pd.Series or None
        Target column. Returns None if not provided.

    """

    def prepare_df(out: X_TYPES) -> pd.DataFrame:
        """Convert to df and set correct column names and order.

        Parameters
        ----------
        out: dataframe-like
            Data returned by the transformation.

        Returns
        -------
        pd.DataFrame
            Final dataset.

        """
        use_cols = inc or [c for c in X.columns if c not in exc]

        # Convert to pandas and assign proper column names
        if not isinstance(out, pd.DataFrame):
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
    y = to_series(y, index=getattr(X, "index", None))

    args = []
    if "X" in (params := sign(getattr(transformer, method))):
        if X is not None:
            inc, exc = getattr(transformer, "_cols", (list(X.columns), None))
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

    output = getattr(transformer, method)(*args)

    # Transform can return X, y or both
    if isinstance(output, tuple):
        new_X = prepare_df(output[0])
        new_y = to_series(output[1], index=new_X.index, name=y.name)
    else:
        if output.ndim > 1:
            new_X = prepare_df(output)
            new_y = y if y is None else y.set_axis(new_X.index)
        else:
            new_y = to_series(output, index=y.index, name=y.name)
            new_X = X if X is None else X.set_index(new_y.index)

    return new_X, new_y


def fit_transform_one(
    transformer: Transformer,
    X: Optional[X_TYPES] = None,
    y: Optional[Y_TYPES] = None,
    message: Optional[str] = None,
    **fit_params,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Fit and transform the data using one estimator.

    Parameters
    ----------
    transformer: Transformer
        Instance to fit.

    X: dataframe-like or None, default=None
        Feature set with shape=(n_samples, n_features). If None,
        X is ignored.

    y: int, str, dict, sequence or None, default=None
        Target column corresponding to X.

        - If None: y is ignored.
        - If int: Position of the target column in X.
        - If str: Name of the target column in X.
        - Else: Array with shape=(n_samples,) to use as target.

    message: str or None
        Short message. If None, nothing will be printed.

    **fit_params
        Additional keyword arguments for the fit method.

    Returns
    -------
    pd.DataFrame or None
        Feature set. Returns None if not provided.

    pd.Series or None
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
    data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    verbose: Optional[int] = None,
    method: str = "transform",
) -> Tuple[pd.DataFrame, pd.Series]:
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
    pd.DataFrame
        Feature set.

    pd.Series
        Target column.

    """
    # Select provided data or from the branch
    if data:
        X_og, y_og = to_df(data[0]), to_series(data[1])
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

            # Since rows can be removed from train and test, reassign indices
            branch._idx[0] = [idx for idx in branch._idx[0] if idx in X.index]
            branch._idx[1] = [idx for idx in branch._idx[1] if idx in X.index]

        if branch.T.index is False:
            branch._data = branch.dataset.reset_index(drop=True)
            branch._idx = [
                branch._data.index[:len(branch._idx[0])],
                branch._data.index[-len(branch._idx[1]):],
            ]

    # Back to the original verbosity
    if verbose is not None and hasattr(transformer, "verbose"):
        transformer.verbose = vb

    return X, y


# Patches ========================================================== >>

def score(f):
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

def has_task(task: str) -> Callable:
    """Check that the instance has a specific task.

    Parameters
    ----------
    task: str
        Task to check.

    """

    def check(self) -> bool:
        if hasattr(self, "task"):
            return task in self.task
        else:
            return task in self.T.task

    return check


def has_attr(attr: str) -> Callable:
    """Check that the instance has attribute `attr`.

    Parameters
    ----------
    attr: str
        Name of the attribute to check.

    """

    def check(self) -> bool:
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

    def check(self):
        # Raise original `AttributeError` if `attr` does not exist
        getattr(self.estimator, attr)
        return True

    return check


def composed(*decs) -> Callable:
    """Add multiple decorators in one line.

    Parameters
    ----------
    decs: tuple
        Decorators to run.

    """

    def decorator(f):
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
    def wrapper(*args, **kwargs):
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


def method_to_log(f: Callable) -> Callable:
    """Save called functions to log file."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        # Get logger for calls from models
        logger = args[0].logger if hasattr(args[0], "logger") else args[0].T.logger

        if logger is not None:
            if f.__name__ != "__init__":
                logger.info("")
            logger.info(f"{args[0].__class__.__name__}.{f.__name__}()")

        return f(*args, **kwargs)

    return wrapper


def plot_from_model(f: Callable) -> Callable:
    """If a plot is called from a model, adapt the `models` parameter."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if hasattr(args[0], "T"):
            return f(args[0].T, args[0].name, *args[1:], **kwargs)
        else:
            return f(*args, **kwargs)

    return wrapper


# Custom scorers =================================================== >>

def true_negatives(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> int:
    return int(confusion_matrix(y_true, y_pred).ravel()[0])


def false_positives(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> int:
    return int(confusion_matrix(y_true, y_pred).ravel()[1])


def false_negatives(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> int:
    return int(confusion_matrix(y_true, y_pred).ravel()[2])


def true_positives(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> int:
    return int(confusion_matrix(y_true, y_pred).ravel()[3])


def false_positive_rate(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return float(fp / (fp + tn))


def true_positive_rate(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> float:
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tp / (tp + fn))


def true_negative_rate(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return float(tn / (tn + fp))


def false_negative_rate(y_true: SEQUENCE_TYPES, y_pred: SEQUENCE_TYPES) -> float:
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fn / (fn + tp))

"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing utility classes.

"""

from __future__ import annotations

import functools
import sys
import warnings
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import cached_property, wraps
from importlib import import_module
from importlib.util import find_spec
from inspect import Parameter, signature
from itertools import cycle
from types import GeneratorType, MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload, cast

import numpy as np
import pandas as pd
import scipy.sparse as sps
from beartype.door import is_bearable
from IPython.display import display
from pandas._libs.missing import NAType
from pandas._typing import Axes, Dtype
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator
from sklearn.base import OneToOneFeatureMixin as FMixin
from sklearn.metrics import (
    confusion_matrix, get_scorer, get_scorer_names, make_scorer,
    matthews_corrcoef,
)
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import _is_fitted

from atom.utils.constants import CAT_TYPES, __version__
from atom.utils.types import (
    Bool, EngineDataOptions, EngineTuple, Estimator, FeatureNamesOut, Float,
    IndexSelector, Int, IntLargerEqualZero, MetricFunction, Model, Pandas,
    Predictor, Scalar, Scorer, Segment, Sequence, SPTuple, Transformer,
    Verbose, XConstructor, YConstructor, int_t, segment_t, sequence_t, XReturn, YReturn
)
from pandas.core.generic import NDFrame


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial
    from shap import Explainer, Explanation

    from atom.basemodel import BaseModel
    from atom.baserunner import BaseRunner
    from atom.data import Branch


T = TypeVar("T")
T_Pandas = TypeVar("T_Pandas", bound=NDFrame)
T_Transformer = TypeVar("T_Transformer", bound=Transformer)
T_Estimator = TypeVar("T_Estimator", bound=Estimator)


# Classes ========================================================== >>

class NotFittedError(ValueError, AttributeError):
    """Exception called when the instance is not yet fitted.

    This class inherits from both ValueError and AttributeError to
    help with exception handling and backward compatibility.

    """


class Goal(Enum):
    """Supported goals by ATOM."""

    classification = 0
    regression = 1
    forecast = 2

    def infer_task(self, y: Pandas) -> Task:
        """Infer the task corresponding to a target column.

        Parameters
        ----------
        y: pd.Series or pd.DataFrame
            Target column(s).

        Returns
        -------
        Task
            Inferred task.

        """
        if self.value == 1:
            if isinstance(y, pd.Series):
                return Task.regression
            else:
                return Task.multioutput_regression
        elif self.value == 2:
            if isinstance(y, pd.Series):
                return Task.univariate_forecast
            else:
                return Task.multivariate_forecast

        if isinstance(y, pd.DataFrame):
            if all(y[col].nunique() == 2 for col in y.columns):
                return Task.multilabel_classification
            else:
                return Task.multiclass_multioutput_classification
        elif isinstance(y.iloc[0], sequence_t):
            return Task.multilabel_classification
        elif y.nunique() == 1:  # noqa: PD101
            raise ValueError(f"Only found 1 target value: {y.unique()[0]}")
        elif y.nunique() == 2:
            return Task.binary_classification
        else:
            return Task.multiclass_classification


class Task(Enum):
    """Supported tasks by ATOM."""

    binary_classification = 0
    multiclass_classification = 1
    multilabel_classification = 2
    multiclass_multioutput_classification = 3
    regression = 4
    multioutput_regression = 5
    univariate_forecast = 6
    multivariate_forecast = 7

    def __str__(self) -> str:
        """Print the task capitalized."""
        return self.name.replace("_", " ").capitalize()

    @property
    def is_classification(self) -> bool:
        """Return whether the task is a classification task."""
        return self.value in (0, 1, 2, 3)

    @property
    def is_regression(self) -> bool:
        """Return whether the task is a regression task."""
        return self.value in (4, 5)

    @property
    def is_forecast(self) -> bool:
        """Return whether the task is a forecast task."""
        return self.value in (6, 7)

    @property
    def is_binary(self) -> bool:
        """Return whether the task is binary or multilabel."""
        return self.value in (0, 2)

    @property
    def is_multiclass(self) -> bool:
        """Return whether the task is multiclass or multiclass-multioutput."""
        return self.value in (1, 3)

    @property
    def is_multioutput(self) -> bool:
        """Return whether the task has more than one target column."""
        return self.value in (2, 3, 5, 7)


class SeasonalPeriod(IntEnum):
    """Seasonal periodicity.

    Covers pandas' aliases for periods.
    See: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases

    """

    B = 5  # business day
    D = 7  # calendar day
    W = 52  # week
    M = 12  # month
    Q = 4  # quarter
    A = 1  # year
    Y = 1  # year
    H = 24  # hours
    T = 60  # minutes
    S = 60  # seconds
    L = 1e3  # milliseconds
    U = 1e6  # microseconds
    N = 1e9  # nanoseconds


@dataclass
class DataContainer:
    """Stores a branch's data."""

    data: pd.DataFrame  # Complete dataset
    train_idx: pd.Index  # Indices in the train set
    test_idx: pd.Index  # Indices in the test
    n_targets: int  # Number of target columns


@dataclass
class TrackingParams:
    """Tracking parameters for a mlflow experiment."""

    log_ht: bool  # Track every trial of the hyperparameter tuning
    log_plots: bool  # Save plot artifacts
    log_data: bool  # Save the train and test sets
    log_pipeline: bool  # Save the model's pipeline


@dataclass
class Aesthetics:
    """Keeps track of plot aesthetics."""

    palette: str | Sequence[str]  # Sequence of colors
    title_fontsize: Scalar  # Fontsize for titles
    label_fontsize: Scalar  # Fontsize for labels, legend and hoverinfo
    tick_fontsize: Scalar  # Fontsize for ticks
    line_width: Scalar  # Width of the line plots
    marker_size: Scalar  # Size of the markers


@dataclass
class DataConfig:
    """Stores the data configuration.

    This is a utility class to store the data configuration in one
    attribute and pass it down to the models. The default values are
    the ones adopted by trainers.

    """

    index: bool = False
    ignore: tuple[str, ...] = ()
    sp: SPTuple = SPTuple()  # noqa: RUF009
    shuffle: Bool = False
    stratify: IndexSelector = True
    n_rows: Scalar = 1
    test_size: Scalar = 0.2
    holdout_size: Scalar | None = None

    def get_stratify_columns(self, df: pd.DataFrame, y: Pandas) -> pd.DataFrame | None:
        """Get columns to stratify by.

        Parameters
        ----------
        df: pd.DataFrame
            Dataset from which to get the columns.

        y: pd.Series or pd.DataFrame
            Target column(s).

        Returns
        -------
        pd.DataFrame or None
            Dataset with subselection of columns. Returns None if
            there's no stratification.

        """
        # Stratification is not possible when the data cannot change order
        if self.stratify is False:
            return None
        elif self.shuffle is False:
            return None
        elif self.stratify is True:
            return df[[c.name for c in get_cols(y)]]
        else:
            inc = []
            for col in lst(self.stratify):
                if isinstance(col, int_t):
                    if -df.shape[1] <= col <= df.shape[1]:
                        inc.append(df.columns[int(col)])
                    else:
                        raise ValueError(
                            f"Invalid value for the stratify parameter. Value {col} "
                            f"is out of range for a dataset with {df.shape[1]} columns."
                        )
                elif isinstance(col, str):
                    if col in df:
                        inc.append(col)
                    else:
                        raise ValueError(
                            "Invalid value for the stratify parameter. "
                            f"Column {col} not found in the dataset."
                        )

            return df[inc]


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
        """Return final value of metric based on error and weight.

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
        """Return whether great values of metric are better."""
        return True

    def evaluate(
        self,
        approxes: list[Float],
        targets: list[Float],
        weight: list[Float],
    ) -> tuple[Float, Float]:
        """Evaluate metric value.

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
            if self.scorer._response_method == "predict":
                y_pred = (y_pred > 0.5).astype(int)

        elif self.task.is_multiclass:
            y_pred = np.array(approxes).T
            if self.scorer._response_method == "predict":
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
        """Evaluate metric value.

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
        if self.scorer._response_method == "predict":
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


class XGBMetric:
    """Custom evaluation metric for the XGBoost model.

    Parameters
    ----------
    scorer: Scorer
        Scorer to evaluate. It's always the runner's main metric.

    task: str
        Model's task.

    """

    def __init__(self, scorer: Scorer, task: Task):
        self.scorer = scorer
        self.task = task

    @property
    def __name__(self) -> str:  # noqa: A003
        """Return the scorer's name."""
        return self.scorer.name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Float:
        """Calculate the score.

        Parameters
        ----------
        y_true: np.array
            Vectors of approx labels.

        y_pred: np.array
            Vectors of true labels.

        Returns
        -------
        float
            Metric score.

        """
        if self.scorer._response_method == "predict":
            if self.task.is_binary:
                y_pred = (y_pred > 0.5).astype(int)
            elif self.task.is_multiclass:
                y_pred = np.argmax(y_pred, axis=1)

        score = self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs)
        return -self.scorer._sign * score  # Negative because XGBoost minimizes


class Table:
    """Class to print nice tables per row.

    Parameters
    ----------
    headers: sequence
        Name of each column in the table.

    spaces: sequence
        Width of each column. Should have the same length as `headers`.

    """

    def __init__(self, headers: Sequence[str], spaces: Sequence[Int]):
        self.headers = headers
        self.spaces = spaces
        self.positions = ["left"] + (len(headers) - 1) * ["right"]

    @staticmethod
    def to_cell(text: Scalar | str, position: str, space: Int) -> str:
        """Get the string format for one cell.

        Parameters
        ----------
        text: int, float or str
            Value to add to the cell.

        position: str
            Position of the text in cell. Choose from: right, left.

        space: int
            Maximum char length in the cell.

        Returns
        -------
        str
            Value to add to cell.

        """
        text = str(text)
        if len(text) > space:
            text = text[: space - 2] + ".."

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
        return self.pprint({k: k for k in self.headers})

    def print_line(self) -> str:
        """Print a line with dashes.

        Use this method after printing the header for a nice table
        structure.

        Returns
        -------
        str
            New row with dashes.

        """
        return self.pprint({k: "-" * s for k, s in zip(self.headers, self.spaces, strict=True)})

    def pprint(self, sequence: dict[str, Any] | pd.Series) -> str:
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
        for h, p, s in zip(self.headers, self.positions, self.spaces, strict=True):
            out.append(self.to_cell(rnd(sequence.get(h, "---")), p, s))

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

    def __init__(self, model: BaseModel, n_jobs: Int):
        self.T = model
        self.n_jobs = n_jobs

        if self.n_jobs == 1:
            self._table = self.create_table()
            self.T._log(self._table.print_header(), 2)
            self.T._log(self._table.print_line(), 2)

    def __call__(self, study: Study, trial: FrozenTrial):
        """Print trial info and store in mlflow experiment."""
        try:  # Fails when there are no successful trials
            trials = self.T.trials.reset_index(names="trial")
            trial_info = cast(pd.Series, trials.loc[trial.number])  # Loc returns df or series
        except KeyError:
            return

        # Save trials to mlflow experiment as nested runs
        if self.T.experiment and self.T.log_ht:
            import mlflow

            with mlflow.start_run(run_id=self.T.run.info.run_id):
                run_name = f"{self.T.name} - {trial.number}"
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.set_tags(
                        {
                            "name": self.T.name,
                            "model": self.T.fullname,
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
                            {k: v for k, v in estimator.get_params().items() if len(str(v)) <= 250}
                        )

                        mlflow.sklearn.log_model(
                            sk_model=estimator,
                            artifact_path=estimator.__class__.__name__,
                            signature=mlflow.models.signature.infer_signature(
                                model_input=pd.DataFrame(self.T.branch.X),
                                model_output=estimator.predict(self.T.branch.X.iloc[[0]]),
                            ),
                            input_example=pd.DataFrame(self.T.branch.X.iloc[[0], :]),
                        )
                    else:
                        mlflow.log_params(
                            {k: v for k, v in trial.params.items() if len(str(v)) <= 250}
                        )

        if self.n_jobs == 1:
            # Print overview of trials
            trial_info["time_trial"] = time_to_str(trial_info["time_trial"])
            trial_info["time_ht"] = time_to_str(trial_info["time_ht"])
            self.T._log(self._table.pprint(trial_info), 2)

    def create_table(self) -> Table:
        """Create the trial table.

        Returns
        -------
        Table
            Object to display the trial overview.

        """
        headers = ["trial", *self.T._ht["distributions"]]
        for m in self.T._metric:
            headers.extend([m.name, "best_" + m.name])
        headers.extend(["time_trial", "time_ht", "state"])

        # Define the width op every column in the table
        spaces = [len(headers[0])]
        for name, dist in self.T._ht["distributions"].items():
            # If the distribution is categorical, take the mean of the widths
            # Else take the max of seven (minimum width) and the width of the name
            if hasattr(dist, "choices"):
                options = np.mean([len(str(x)) for x in dist.choices], axis=0, dtype=int)
            else:
                options = 0

            spaces.append(max(7, len(name), options))

        spaces.extend(
            [max(7, len(column)) for column in headers[1 + len(self.T._ht["distributions"]): -1]]
        )

        return Table(headers, [*spaces, 8])


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
        import plotly.graph_objects as go

        self.y1: dict[int, deque] = {i: deque(maxlen=self.max_len) for i in range(len(metric))}
        self.y2: dict[int, deque] = {i: deque(maxlen=self.max_len) for i in range(len(metric))}

        traces = []
        colors = cycle(aesthetics.palette)
        for met in metric:
            color = next(colors)
            traces.extend(
                [
                    go.Scatter(
                        mode="lines+markers",
                        line={"width": aesthetics.line_width, "color": color},
                        marker={
                            "symbol": "circle",
                            "size": aesthetics.marker_size,
                            "line": {"width": 1, "color": "white"},
                            "opacity": 1,
                        },
                        name=met,
                        legendgroup=met,
                        xaxis="x2",
                        yaxis="y1",
                    ),
                    go.Scatter(
                        mode="lines+markers",
                        line={"width": aesthetics.line_width, "color": color},
                        marker={
                            "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                            "symbol": "circle",
                            "size": aesthetics.marker_size,
                            "opacity": 1,
                        },
                        name=met,
                        legendgroup=met,
                        showlegend=False,
                        xaxis="x2",
                        yaxis="y2",
                    ),
                ]
            )

        self.figure = go.FigureWidget(
            data=traces,
            layout={
                "xaxis1": {"domain": (0, 1), "anchor": "y1", "showticklabels": False},
                "yaxis1": {
                    "domain": (0.31, 1.0),
                    "title": {"text": "Score", "font_size": aesthetics.label_fontsize},
                    "anchor": "x1",
                },
                "xaxis2": {
                    "domain": (0, 1),
                    "title": {"text": "Trial", "font_size": aesthetics.label_fontsize},
                    "anchor": "y2",
                },
                "yaxis2": {
                    "domain": (0, 0.29),
                    "title": {"text": "d", "font_size": aesthetics.label_fontsize},
                    "anchor": "x2",
                },
                "title": {
                    "text": f"Hyperparameter tuning for {name}",
                    "x": 0.5,
                    "y": 1,
                    "pad": {"t": 15, "b": 15},
                    "xanchor": "center",
                    "yanchor": "top",
                    "xref": "paper",
                    "font_size": aesthetics.title_fontsize,
                },
                "legend": {
                    "x": 0.99,
                    "y": 0.99,
                    "xanchor": "right",
                    "yanchor": "top",
                    "font_size": aesthetics.label_fontsize,
                    "bgcolor": "rgba(255, 255, 255, 0.5)",
                },
                "hovermode": "x unified",
                "hoverlabel": {"font_size": aesthetics.label_fontsize},
                "font_size": aesthetics.tick_fontsize,
                "margin": {
                    "l": 0,
                    "b": 0,
                    "r": 0,
                    "t": 25 + aesthetics.title_fontsize,
                    "pad": 0,
                },
                "width": 900,
                "height": 800,
            },
        )

        display(self.figure)

    def __call__(self, study: Study, trial: FrozenTrial):
        """Calculate new values for lines and plots them.

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
            self.figure.data[i * 2].x = list(x[: len(self.y1[i])])
            self.figure.data[i * 2].y = list(self.y1[i])
            self.figure.data[i * 2 + 1].x = list(x[: len(self.y1[i])])
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
        branch: Branch,
        random_state: IntLargerEqualZero | None = None,
    ):
        self.estimator = estimator
        self.task = task
        self.branch = branch
        self.random_state = random_state

        self._explanation: Explainer
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
        if hasattr(self.estimator, "predict_proba"):
            return "predict_proba"
        elif hasattr(self.estimator, "decision_function"):
            return "decision_function"
        else:
            return "predict"

    @cached_property
    def explainer(self) -> Explainer:
        """Get shap's explainer.

        Returns
        -------
        shap.Explainer
            Get the initialized explainer object.

        """
        from shap import Explainer

        kwargs = {
            "masker": self.branch.X_train,
            "feature_names": list(self.branch.features),
            "seed": self.random_state,
        }

        try:  # Fails when model does not fit standard explainers (e.g., ensembles)
            return Explainer(self.estimator, **kwargs)
        except TypeError:
            # If a method is provided as first arg, selects always Permutation
            return Explainer(getattr(self.estimator, self.attr), **kwargs)

    def get_explanation(
        self,
        df: pd.DataFrame,
        target: tuple[Int, ...],
    ) -> Explanation:
        """Get an Explanation object.

        Shap values are memoized to not repeat calculations on same rows.

        Parameters
        ----------
        df: pd.DataFrame
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
        if not (calculate := df.loc[~df.index.isin(self._shap_values.index)]).empty:
            kwargs: dict[str, Any] = {}

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
                except (ValueError, AssertionError) as ex:
                    raise ValueError(
                        "Failed to get shap's explainer for estimator "
                        f"{self.estimator} with task {self.task}. Exception: {ex}"
                    ) from None

            # Remember shap values in the _shap_values attribute
            self._shap_values = pd.concat(
                [
                    self._shap_values,
                    pd.Series(list(self._explanation.values), index=calculate.index),
                ]
            )

        # Don't use attribute to not save plot-specific changes
        # Shallow copy to not copy the data in the branch
        explanation = copy(self._explanation)

        # Update the explanation object
        explanation.values = np.stack(self._shap_values.loc[df.index].tolist())
        explanation.base_values = self._explanation.base_values[0]

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
    def _conv(key: Any) -> Any:
        return key.lower() if isinstance(key, str) else key

    def _get_data(self, key: Any) -> Any:
        if isinstance(key, int_t) and key not in self.keys():
            try:
                return self.__data[key]
            except IndexError:
                raise KeyError(key) from None
        else:
            for data in self.__data:
                if self._conv(getattr(data, self.__key)) == self._conv(key):
                    return data

        raise KeyError(key)

    def _check(self, elem: T) -> T:
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
        self.__data: list[Any] = []
        for elem in args:
            if isinstance(elem, GeneratorType):
                self.__data.extend(self._check(e) for e in elem)
            else:
                self.__data.append(self._check(elem))

    def __getitem__(self, key: Any) -> Any:
        """Get a value or subset of the mapping."""
        if isinstance(key, sequence_t):
            return self.__class__(*[self._get_data(k) for k in key], key=self.__key)
        elif isinstance(key, segment_t):
            return self.__class__(*get_segment(self.__data, key), key=self.__key)
        else:
            return self._get_data(key)

    def __setitem__(self, key: Any, value: Any):
        """Add a new item to the mapping."""
        if isinstance(key, int_t):
            self.__data[key] = self._check(value)
        else:
            try:
                self.__data = [e if self[key] == e else value for e in self.__data]
            except KeyError:
                self.append(value)

    def __delitem__(self, key: Any):
        """Delete an item."""
        del self.__data[self.index(self._get_data(key))]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the values."""
        yield from self.__data

    def __len__(self) -> int:
        """Length of the mapping."""
        return len(self.__data)

    def __contains__(self, key: Any) -> bool:
        """Whether the key or value exists."""
        return key in self.__data or self._conv(key) in self.keys_lower()

    def __repr__(self) -> str:
        """Print the mapping representation."""
        return self.__data.__repr__()

    def __reversed__(self) -> Iterator[Any]:
        """Reverse order of the mapping."""
        yield from reversed(list(self.__data))

    def __eq__(self, other: object) -> bool:
        """Compare equality of the instances."""
        return self.__data == other

    def __add__(self, other: ClassMap) -> ClassMap:
        """Merge two mappings."""
        self.__data += other
        return self

    def __bool__(self) -> bool:
        """Whether the mapping has values."""
        return bool(self.__data)

    def keys(self) -> list[Any]:
        """Return the mapping keys."""
        return [getattr(x, self.__key) for x in self.__data]

    def values(self) -> list[Any]:
        """Return the mapping values."""
        return self.__data

    def keys_lower(self) -> list[Any]:
        """Return the map keys in lower-case."""
        return list(map(self._conv, self.keys()))

    def append(self, value: T) -> T:
        """Add an item to the mapping."""
        self.__data.append(self._check(value))
        return value

    def extend(self, value: Any):
        """Extend the mapping with another sequence."""
        self.__data.extend(list(map(self._check, value)))

    def remove(self, value: Any):
        """Remove an item."""
        if value in self.__data:
            self.__data.remove(value)
        else:
            self.__data.remove(self._get_data(value))

    def clear(self):
        """Clear the content."""
        self.__data = []

    def index(self, key: Any) -> Any:
        """Return the key's index."""
        if key in self.__data:
            return self.__data.index(key)
        else:
            return self.__data.index(self._get_data(key))


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
    return x[0] if isinstance(x, sequence_t) and len(x) == 1 else x


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
    return list(x) if isinstance(x, (dict, *sequence_t, ClassMap)) else [x]


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


def divide(a: Scalar, b: Scalar, decimals: Int = 4) -> int | float:
    """Divide two numbers and return 0 if division by zero.

    Parameters
    ----------
    a: int or float
        Numerator.

    b: int or float
        Denominator.

    decimals: int, default=4
        Decimal position to round to.

    Returns
    -------
    int or float
        Result of division or 0.

    """
    return float(np.round(np.divide(a, b), decimals)) if b != 0 else 0


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
    from matplotlib.colors import to_rgba

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


def merge(*args) -> pd.DataFrame:
    """Concatenate pandas objects column-wise.

    None and empty objects are ignored.

    Parameters
    ----------
    *args
        Objects to concatenate.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe.

    """
    if len(args_c := [x for x in args if x is not None and not x.empty]) == 1:
        return pd.DataFrame(args_c[0])
    else:
        return pd.DataFrame(pd.concat(args_c, axis=1))


def replace_missing(X: T_Pandas, missing_values: list[Any] | None = None) -> T_Pandas:
    """Replace all values considered 'missing' in a dataset.

    This method replaces the missing values in columns with pandas'
    nullable dtypes with `pd.NA`, else with `np.NaN`.

    Parameters
    ----------
    X: pd.Series or pd.DataFrame
        Data set to replace.

    missing_values: list or None, default=None
        Values considered 'missing'. If None, use only the default
        values.

    Returns
    -------
    pd.Series or pd.DataFrame
        Data set without missing values.

    """

    def get_nan(dtype: Dtype) -> float | NAType:
        """Get NaN type depending on a column's type.

        Parameters
        ----------
        dtype: Dtype
            Type of the column.

        Returns
        -------
        np.NaN or pd.NA
            Missing value indicator.

        """
        return np.NaN if isinstance(dtype, np.dtype) else pd.NA

    # Always convert these values
    default_values = [None, pd.NA, pd.NaT, np.NaN, np.inf, -np.inf]

    if isinstance(X, pd.Series):
        return X.replace(
            to_replace=(missing_values or []) + default_values,
            value=get_nan(X.dtype),
        )
    else:
        return X.replace(
            to_replace={c: (missing_values or []) + default_values for c in X.columns},
            value={c: get_nan(d) for c, d in X.dtypes.items()},
        )


def n_cols(obj: YConstructor | None) -> int:
    """Get the number of columns in a dataset.

    Parameters
    ----------
    obj: dict, sequence, dataframe-like or None
        Dataset to check.

    Returns
    -------
    int
        Number of columns.

    """
    if hasattr(obj, "shape"):
        return obj.shape[1] if len(obj.shape) > 1 else 1  # type: ignore[union-attr]
    elif isinstance(obj, dict):
        return 2  # Dict always goes to dataframe

    try:
        if (array := np.asarray(obj)).ndim > 1:
            return array.shape[1]
        else:
            return array.ndim
    except ValueError:
        # Fails for inhomogeneous data, return series
        return 1


def get_cols(obj: Pandas) -> list[pd.Series]:
    """Get a list of columns in dataframe / series.

    Parameters
    ----------
    obj: pd.Series or pd.DataFrame
        Element to get the columns from.

    Returns
    -------
    list of pd.Series
        Columns.

    """
    if isinstance(obj, pd.Series):
        return [obj]
    else:
        return [obj[col] for col in obj.columns]


def get_col_names(obj: Any) -> list[str] | None:
    """Get a list of column names in tabular objects.

    Parameters
    ----------
    obj: object
        Element to get the column names from.

    Returns
    -------
    list of str
        Names of the columns. Returns None when the object passed is
        no pandas object.

    """
    if isinstance(obj, pd.DataFrame):
        return list(obj.columns)
    elif isinstance(obj, pd.Series):
        return [str(obj.name)]
    else:
        return None


def variable_return(
    X: XReturn | None,
    y: YReturn | None,
) -> XReturn | tuple[XReturn, YReturn]:
    """Return one or two arguments depending on which is None.

    This utility is used to make methods return only the provided
    data set.

    Parameters
    ----------
    X: dataframe or None
        Feature set.

    y: series, dataframe or None
        Target column(s).

    Returns
    -------
    series, dataframe or tuple
        Data sets that are not None.

    """
    if y is None and X is not None:
        return X
    elif X is None and y is not None:
        return y
    elif X is not None and y is not None:
        return X, y
    else:
        raise ValueError("Both X and y can't be None.")


def get_segment(obj: list[T], segment: Segment) -> list[T]:
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
    if isinstance(segment, slice):
        return obj[segment]
    else:
        return obj[slice(segment.start, segment.stop, segment.step)]


def is_sparse(obj: Pandas) -> bool:
    """Check if the dataframe is sparse.

    A data set is considered sparse if any of its columns is sparse.

    Parameters
    ----------
    obj: pd.Series or pd.DataFrame
        Data set to check.

    Returns
    -------
    bool
        Whether the data set is sparse.

    """
    return any(isinstance(col.dtype, pd.SparseDtype) for col in get_cols(obj))


def check_empty(obj: Pandas | None) -> Pandas | None:
    """Check if a pandas object is empty.

    Parameters
    ----------
    obj: pd.Series, pd.DataFrame or None
        Pandas object to check.

    Returns
    -------
    pd.Series, pd.DataFrame or None
        Same object or None if empty or obj is None.

    """
    return obj if isinstance(obj, pd.DataFrame) and not obj.empty else None


def check_dependency(name: str):
    """Check an optional dependency.

    Raise an error if the package is not installed.

    Parameters
    ----------
    name: str
        Name of the package to check.

    """
    if not find_spec(name):
        raise ModuleNotFoundError(
            f"Unable to import the {name} package. Install it using "
            f"`pip install {name}` or install all of atom's optional "
            "dependencies with `pip install atom-ml[full]`."
        )


def check_nltk_module(module: str, *, quiet: bool):
    """Check if a module for the NLTK package is avaialble.

    If the module isn't available, it's downloaded.

    Parameters
    ----------
    module: str
        Name of the module to check.

    quiet: bool
        Whether to show logs when downloading.

    """
    import nltk

    try:
        nltk.data.find(module)
    except LookupError:
        nltk.download(module.split("/")[-1], quiet=quiet)


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


def check_predict_proba(models: Model | Sequence[Model], method: str):
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


def check_scaling(obj: Pandas) -> bool:
    """Check if the data is scaled.

    A data set is considered scaled when the mean of the mean of
    all columns lies between -0.05 and 0.05 and the mean of the
    standard deviation of all columns lies between 0.85 and 1.15.
    Categorical and binary columns are excluded from the calculation.

    Parameters
    ----------
    obj: pd.Series or pd.DataFrame
        Data set to check.

    Returns
    -------
    bool
        Whether the data set is scaled.

    """
    if isinstance(obj, pd.DataFrame):
        mean = obj.mean(numeric_only=True).mean()
        std = obj.std(numeric_only=True).mean()
    else:
        mean = obj.mean()
        std = obj.std()

    return bool(-0.05 < mean < 0.05 and 0.85 < std < 1.15)


@contextmanager
def keep_attrs(estimator: Estimator):
    """Temporarily save an estimator's custom attributes.

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
def adjust(
    estimator: Estimator,
    *,
    transform: EngineDataOptions | None = None,
    verbose: Verbose | None = None,
):
    """Temporarily adjust output parameters of an estimator.

    The estimator's data engine and verbosity are temporarily changed
    to the provided values.

    Parameters
    ----------
    estimator: Estimator
        Temporarily change the verbosity of this estimator.

    transform: str or None, default=None
        Data engine for the estimator. If None, it leaves it to
        its original engine.

    verbose: int or None, default=None
        Verbosity level for the estimator. If None, it leaves it to
        its original verbosity.

    """
    try:
        if transform is not None and hasattr(estimator, "set_output"):
            output = getattr(estimator, "_engine", EngineTuple())
            estimator.set_output(transform=transform)
        if verbose is not None and hasattr(estimator, "verbose"):
            verbosity = estimator.verbose
            estimator.verbose = verbose
        yield estimator
    finally:
        if transform is not None and hasattr(estimator, "set_output"):
            estimator._engine = output
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


def get_corpus(df: pd.DataFrame) -> str:
    """Get text column from a dataframe.

    The text column should be called `corpus` (case-insensitive). Also
    checks if the column consists of a string or sequence of strings.

    Parameters
    ----------
    df: pd.DataFrame
        Data set from which to get the corpus.

    Returns
    -------
    str
        Name of the corpus column.

    """
    try:
        corpus = next(col for col in df.columns if col.lower() == "corpus")

        if not is_bearable(df[corpus].iloc[0], (str, Sequence[str])):
            raise TypeError("The corpus should consist of a string or sequence of strings.")
        else:
            return corpus
    except StopIteration as ex:
        raise ValueError("The provided dataset does not contain a column named corpus.") from ex


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


@overload
def to_df(
    data: Literal[None],
    index: Axes | None = ...,
    columns: Axes | None = ...,
) -> None: ...


@overload
def to_df(
    data: XConstructor,
    index: Axes | None = ...,
    columns: Axes | None = ...,
) -> pd.DataFrame: ...


def to_df(
    data: XConstructor | None,
    index: Axes | None = None,
    columns: Axes | None = None,
) -> pd.DataFrame | None:
    """Convert a dataset to a pandas dataframe.

    Parameters
    ----------
    data: dataframe-like or None
        Dataset to convert to a dataframe. If None or already a
        pandas dataframe, return unchanged.

    index: sequence or None, default=None
        Values for the index.

    columns: sequence or None, default=None
        Names of the columns. Use None for automatic naming.

    Returns
    -------
    pd.DataFrame or None
        Data as dataframe. Returns None if data is None.

    """
    if data is not None:
        if isinstance(data, pd.DataFrame):
            data_c = data.copy()
        elif hasattr(data, "to_pandas"):
            data_c = data.to_pandas()
        elif hasattr(data, "__dataframe__"):
            # Transform from dataframe interchange protocol
            data_c = pd.api.interchange.from_dataframe(data.__dataframe__())
        else:
            # Assign default column names (dict and series already have names)
            if columns is None and not isinstance(data, dict | pd.Series):
                columns = [f"x{i}" for i in range(n_cols(data))]

            if sps.issparse(data):
                data_c = pd.DataFrame.sparse.from_spmatrix(data, index, columns)
            else:
                data_c = pd.DataFrame(
                    data=data,  # type: ignore[misc, arg-type]
                    index=index,
                    columns=columns,
                    copy=True,
                )

        # If text dataset, change the name of the column to corpus
        if list(data_c.columns) == ["x0"] and data_c.dtypes[0].name in CAT_TYPES:
            data_c = data_c.rename(columns={data_c.columns[0]: "corpus"})
        else:
            # Convert all column names to str
            data_c.columns = data_c.columns.astype(str)

            # No duplicate rows nor column names are allowed
            if data_c.columns.duplicated().any():
                raise ValueError("Duplicate column names found in X.")

            if columns is not None:
                # Reorder columns to the provided order
                try:
                    data_c = data_c[list(columns)]  # Force order determined by columns
                except KeyError:
                    raise ValueError(
                        f"The columns are different than seen at fit time. Features "
                        f"{set(data_c.columns) - set(columns)} "  # type: ignore[arg-type]
                        "are missing in X."
                    ) from None

        return data_c
    else:
        return None


@overload
def to_series(
    data: Literal[None],
    index: Axes | None = ...,
    name: str | None = ...,
) -> None: ...


@overload
def to_series(
    data: dict[str, Any] | Sequence[Any] | pd.DataFrame,
    index: Axes | None = ...,
    name: str | None = ...,
) -> pd.Series: ...


def to_series(
    data: dict[str, Any] | Sequence[Any] | pd.DataFrame | None,
    index: Axes | None = None,
    name: str | None = None,
) -> pd.Series | None:
    """Convert a sequence to a pandas series.

    Parameters
    ----------
    data: dict, sequence, pd.DataFrame or None
        Data to convert. If None or already a pandas series, return
        unchanged.

    index: sequence, index or None, default=None
        Values for the index.

    name: str or None, default=None
        Name of the series.

    Returns
    -------
    pd.Series or None
        Data as series. Returns None if data is None.

    """
    if data is not None:
        if isinstance(data, pd.Series):
            data_c = data.copy()
        elif isinstance(data, pd.DataFrame):
            data_c = data.iloc[:, 0].copy()
        elif hasattr(data, "to_pandas"):
            data_c = data.to_pandas()
        else:
            try:
                # Flatten for arrays with shape=(n_samples, 1)
                array = np.asarray(data).ravel().tolist()
            except ValueError:
                # Fails for inhomogeneous data
                array = data

            data_c = pd.Series(array, index=index, name=name or "target", copy=True)

        return data_c
    else:
        return None


@overload
def to_tabular(
    data: Literal[None],
    index: Axes | None = ...,
    columns: str | Axes | None = ...,
) -> None: ...


@overload
def to_tabular(
    data: YConstructor,
    index: Axes | None = ...,
    columns: str | Axes | None = ...,
) -> Pandas: ...


def to_tabular(
    data: YConstructor | None,
    index: Axes | None = None,
    columns: str | Axes | None = None,
) -> Pandas | None:
    """Convert to a tabular pandas type.

    If the data is one-dimensional, convert to series, else to a
    dataframe.

    Parameters
    ----------
    data: dict, sequence, pd.DataFrame or None
        Data to convert. If None, return unchanged.

    index: sequence, index or None, default=None
        Values for the index.

    columns: str, sequence or None, default=None
        Name of the columns. Use None for automatic naming.

    Returns
    -------
    pd.Series, pd.DataFrame or None
        Data as a pandas object.

    """
    if (n_targets := n_cols(data)) == 1:
        return to_series(data, index=index, name=flt(columns))  # type: ignore[misc, arg-type]
    else:
        if columns is None and not hasattr(data, "__dataframe__"):
            columns = [f"y{i}" for i in range(n_targets)]

        return to_df(data, index=index, columns=columns)  # type: ignore[misc, arg-type]


def check_is_fitted(
    obj: Any,
    *,
    exception: Bool = True,
    attributes: str | Sequence[str] | None = None,
) -> bool:
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
    if hasattr(obj, "_is_fitted"):
        is_fitted = obj._is_fitted
    else:
        is_fitted = _is_fitted(obj, attributes)

    if not is_fitted:
        if exception:
            raise NotFittedError(
                f"This {type(obj).__name__} instance is not yet fitted. "
                f"Call {'run' if hasattr(obj, 'run') else 'fit'} with "
                "appropriate arguments before using this object."
            )
        else:
            return False

    return True


def get_custom_scorer(metric: str | MetricFunction | Scorer) -> Scorer:
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
        custom_acronyms = {
            "ap": "average_precision",
            "ba": "balanced_accuracy",
            "auc": "roc_auc",
            "logloss": "neg_log_loss",
            "ev": "explained_variance",
            "me": "max_error",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "msle": "neg_mean_squared_log_error",
            "mape": "neg_mean_absolute_percentage_error",
            "medae": "neg_median_absolute_error",
            "poisson": "neg_mean_poisson_deviance",
            "gamma": "neg_mean_gamma_deviance",
        }

        custom_scorers = {
            "tn": true_negatives,
            "fp": false_positives,
            "fn": false_negatives,
            "tp": true_positives,
            "fpr": false_positive_rate,
            "tpr": true_positive_rate,
            "tnr": true_negative_rate,
            "fnr": false_negative_rate,
            "mcc": matthews_corrcoef,
        }

        metric = metric.lower()
        if metric in get_scorer_names():
            scorer = get_scorer(metric)
        elif metric in custom_acronyms:
            scorer = get_scorer(custom_acronyms[metric])
        elif metric in custom_scorers:
            scorer = make_scorer(custom_scorers[metric])
        else:
            raise ValueError(
                f"Unknown value for the metric parameter, got {metric}. "
                f"Choose from: {', '.join(get_scorer_names())}."
            )

        scorer.name = metric

    elif hasattr(metric, "_score_func"):  # Scoring is a scorer
        scorer = copy(metric)

        # Some scorers use default kwargs
        if scorer._score_func.__name__.startswith(("precision", "recall", "f1", "jaccard")):
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
    if not hasattr(scorer, "fullname"):
        scorer.fullname = scorer._score_func.__name__

    return scorer


# Pipeline functions =============================================== >>

def name_cols(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    col_names: list[str],
) -> pd.Index:
    """Get the column names after a transformation.

    If the number of columns is unchanged, the original
    column names are returned. Else, give the column a
    default name if the column values changed.

    Parameters
    ----------
    df: pd.DataFrame
        Transformed dataset.

    original_df: pd.DataFrame
        Original dataset.

    col_names: list of str
        Columns used in the transformer.

    Returns
    -------
    pd.Index
        Column names.

    """
    # If columns were only transformed, return og names
    if df.shape[1] == len(col_names):
        return pd.Index(col_names)

    # If columns were added or removed
    temp_cols = []
    for i, (name, column) in enumerate(df.items()):
        # equal_nan=True fails for non-numeric dtypes
        mask = original_df.apply(  # type: ignore[type-var]
            lambda c: np.array_equal(
                a1=c,
                a2=column,
                equal_nan=is_numeric_dtype(c) and np.issubdtype(column.dtype.name, np.number),
            )
        )

        if any(mask) and mask[mask].index[0] not in temp_cols:
            # If the column is equal, use the existing name
            temp_cols.append(mask[mask].index[0])
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

    return pd.Index(temp_cols)


def get_col_order(
    new_columns: list[str],
    og_columns: list[str],
    col_names: list[str],
) -> np.ndarray:
    """Determine column order for a dataframe.

    The column order is determined by the order in the original
    dataset. Derivative columns are placed in the position of their
    progenitor.

    Parameters
    ----------
    new_columns: list of str
        Columns in the new dataframe.

    og_columns: list of str
        Columns in the original dataframe.

    col_names: list of str
        Names of the columns used in the transformer.

    Returns
    -------
    np.ndarray
        New column order.

    """
    columns: list[str] = []
    for col in og_columns:
        if col in new_columns or col not in col_names:
            columns.append(col)

        # Add all derivative columns: columns that originate from another
        # and start with its progenitor name, e.g., one-hot encoded columns
        columns.extend([c for c in new_columns if c.startswith(f"{col}_") and c not in og_columns])

    # Add remaining new columns (non-derivatives)
    columns.extend([col for col in new_columns if col not in columns])

    return np.array(columns)


def reorder_cols(
    transformer: Transformer,
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    col_names: list[str],
) -> pd.DataFrame:
    """Reorder the columns to their original order.

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
    except ValueError as ex:  # Length mismatch
        raise IndexError(
            f"Length of values ({len(df)}) does not match length of "
            f"index ({len(original_df)}). This usually happens when "
            "transformations that drop rows aren't applied on all "
            "the columns."
        ) from ex

    columns = get_col_order(df.columns.tolist(), original_df.columns.tolist(), col_names)

    # Merge the new and old datasets keeping the newest columns
    new_df = df.merge(
        right=original_df[[col for col in original_df if col in columns]],
        how="outer",
        left_index=True,
        right_index=True,
        suffixes=("", "__drop__"),
    )
    new_df = new_df.drop(columns=new_df.filter(regex="__drop__$").columns)

    return new_df[columns]


def fit_one(
    estimator: Estimator,
    X: pd.DataFrame | None = None,
    y: Pandas | None = None,
    message: str | None = None,
    **fit_params,
) -> Estimator:
    """Fit the data on an estimator.

    Parameters
    ----------
    estimator: Estimator
        Instance to fit.

    X: pd.DataFrame or None, default=None
        Feature set with shape=(n_samples, n_features). If None,
        `X` is ignored.

    y: pd.Series, pd.DataFrame or None, default=None
        Target column(s) corresponding to `X`.

    message: str or None
        Short message. If None, nothing will be printed.

    **fit_params
        Additional keyword arguments passed to the `fit` method.

    Returns
    -------
    Estimator
        Fitted estimator.

    """
    with _print_elapsed_time("Pipeline", message):
        if hasattr(estimator, "fit"):
            kwargs: dict[str, Pandas] = {}
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
                    elif X is None:
                        raise ValueError(
                            "Exception while trying to fit transformer "
                            f"{estimator.__class__.__name__}. Parameter "
                            "X is required but has not been provided."
                        )
                    elif X.empty:
                        raise ValueError(
                            "Exception while trying to fit transformer "
                            f"{estimator.__class__.__name__}. Parameter X is "
                            "required but the provided feature set is empty. "
                            "Use the columns parameter to only transform the "
                            "target column, e.g., atom.decompose(columns=-1)."
                        )

            if "y" in params and y is not None:
                kwargs["y"] = y

            # Keep custom attrs since some transformers reset during fit
            with keep_attrs(estimator):
                estimator.fit(**kwargs, **fit_params)

        return estimator


def transform_one(
    transformer: Transformer,
    X: pd.DataFrame | None = None,
    y: Pandas | None = None,
    method: Literal["transform", "inverse_transform"] = "transform",
    **transform_params,
) -> tuple[pd.DataFrame | None, Pandas | None]:
    """Transform the data using one estimator.

    Parameters
    ----------
    transformer: Transformer
        Instance to fit.

    X: pd.DataFrame or None, default=None
        Feature set with shape=(n_samples, n_features). If None,
        `X` is ignored.

    y: pd.Series, pd.DataFrame or None, default=None
        Target column(s) corresponding to `X`.

    method: str, default="transform"
        Method to apply: transform or inverse_transform.

    **transform_params
        Additional keyword arguments passed to the method.

    Returns
    -------
    pd.DataFrame or None
        Feature set. Returns None if not provided.

    pd.Series, pd.DataFrame or None
        Target column(s). Returns None if not provided.

    """

    def prepare_df(out: XConstructor, og: pd.DataFrame) -> pd.DataFrame:
        """Convert to df and set the correct column names.

        Parameters
        ----------
        out: dataframe-like
            Data returned by the transformation.

        og: pd.DataFrame
            Original dataframe, prior to transformations.

        Returns
        -------
        pd.DataFrame
            Transformed dataset.

        """
        out_c = to_df(out, index=og.index)

        # Assign proper column names
        use_cols = [c for c in inc if c in og.columns]
        if not isinstance(out, pd.DataFrame):
            if hasattr(transformer, "get_feature_names_out"):
                out_c.columns = transformer.get_feature_names_out()
            else:
                out_c.columns = name_cols(out_c, og, use_cols)

        # Reorder columns if only a subset was used
        if len(use_cols) != og.shape[1]:
            return reorder_cols(transformer, out_c, og, use_cols)
        else:
            return out_c

    use_y = True

    kwargs: dict[str, Any] = {}
    inc = list(getattr(transformer, "_cols", getattr(X, "columns", [])))
    if "X" in (params := sign(getattr(transformer, method))):
        if X is not None and (cols := [c for c in inc if c in X]):
            kwargs["X"] = X[cols]

        # X is required but has not been provided
        if len(kwargs) == 0:
            if y is not None and hasattr(transformer, "_cols"):
                kwargs["X"] = to_df(y)[inc]
                use_y = False
            elif params["X"].default != Parameter.empty:
                kwargs["X"] = params["X"].default  # Fill X with default
            else:
                return X, y  # If X is needed, skip the transformer

    if "y" in params:
        # We skip `y` when already added to `X`
        if y is not None and use_y:
            kwargs["y"] = y
        elif "X" not in params:
            return X, y  # If y is None and no X in transformer, skip the transformer

    out: YConstructor | tuple[XConstructor, YConstructor] = getattr(transformer, method)(**kwargs, **transform_params)

    # Transform can return X, y or both
    X_new: pd.DataFrame | None
    y_new: Pandas | None
    if isinstance(out, tuple) and X is not None:
        X_new = prepare_df(out[0], X)
        y_new = to_tabular(out[1], index=X_new.index)
        if isinstance(y, pd.DataFrame) and isinstance(y_new, pd.DataFrame):
            y_new = prepare_df(y_new, y)
    elif "X" in params and X is not None and any(c in X for c in inc):
        # X in -> X out
        X_new = prepare_df(out, X)  # type: ignore[arg-type]
        y_new = y if y is None else y.set_axis(X_new.index, axis=0)
    elif y is not None:
        y_new = to_tabular(out)
        X_new = X if X is None else X.set_index(y_new.index)
        if isinstance(y, pd.DataFrame) and isinstance(y_new, pd.DataFrame):
            y_new = prepare_df(y_new, y)

    return X_new, y_new


def fit_transform_one(
    transformer: Transformer,
    X: pd.DataFrame | None,
    y: Pandas | None,
    message: str | None = None,
    **fit_params,
) -> tuple[pd.DataFrame | None, Pandas | None, Transformer]:
    """Fit and transform the data using one estimator.

    Estimators without a `transform` method aren't transformed.

    Parameters
    ----------
    transformer: Transformer
        Instance to fit.

    X: pd.DataFrame or None
        Feature set with shape=(n_samples, n_features). If None,
        `X` is ignored.

    y: pd.Series, pd.DataFrame or None
        Target column(s) corresponding to `X`.

    message: str or None, default=None
        Short message. If None, nothing will be printed.

    **fit_params
        Additional keyword arguments passed to the `fit` method.

    Returns
    -------
    pd.DataFrame or None
        Feature set. Returns None if not provided.

    pd.Series, pd.DataFrame or None
        Target column(s). Returns None if not provided.

    Transformer
        Fitted transformer.

    """
    fit_one(transformer, X, y, message, **fit_params)
    Xt, yt = transform_one(transformer, X, y)

    return Xt, yt, transformer


# Decorators ======================================================= >>

def cache(f: Callable) -> Callable:
    """Cache method utility.

    This decorator checks if `functools.cache` works (fails when args
    are not hashable), and else returns the result without caching.

    """

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return memoize(*args, **kwargs)
        except TypeError:
            return f(*args, **kwargs)

    memoize = functools.cache(f)

    # Add methods from memoizer to the decorator
    wrapper.cache_info = memoize.cache_info
    wrapper.clear_cache = memoize.cache_clear

    return wrapper


def has_task(task: str | Sequence[str]) -> Callable:
    """Check that the instance has a specific task.

    If the check returns False, the decorated function becomes
    unavailable for the instance.

    Parameters
    ----------
    task: str or sequence
        Tasks to check. Choose from: classification, regression,
        forecast, binary, multioutput. Add the character `!` before
        a task to not allow that task instead.

    """

    def check(runner: BaseRunner) -> bool:
        checks = []
        for t in lst(task):
            if t.startswith("!"):
                checks.append(not getattr(runner.task, f"is_{t[1:]}"))
            else:
                checks.append(getattr(runner.task, f"is_{t}"))

        return all(checks)

    return check


def estimator_has_attr(attr: str) -> Callable:
    """Check that the estimator has attribute `attr`.

    Parameters
    ----------
    attr: str
        Name of the attribute to check.

    """

    def check(runner: BaseRunner) -> bool:
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
    cache: dict[str, Exception | None] = {"last_exception": None},  # noqa: B006
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
            if ex is not cache["last_exception"] and getattr(args[0], "logger", None):
                cache["last_exception"] = ex
                args[0].logger.exception("Exception encountered:")

            raise ex

    return wrapper


def method_to_log(f: Callable) -> Callable:
    """Save called functions to log file."""

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        if getattr(args[0], "logger", None):
            if f.__name__ != "__init__":
                args[0].logger.info("")
            args[0].logger.info(f"{args[0].__class__.__name__}.{f.__name__}()")

        return f(*args, **kwargs)

    return wrapper


def make_sklearn(
    obj: T_Estimator,
    feature_names_out: FeatureNamesOut = "one-to-one",
) -> T_Estimator:
    """Add functionality to a class to adhere to sklearn's API.

    The `fit` method of non-sklearn objects is wrapped to always add
    the `n_features_in_` and `feature_names_in_` attributes, and the
    `get-feature_names_out` method is added to transformers that
    don't have it already.

    Parameters
    ----------
    obj: Estimator
        Object to wrap.

    feature_names_out: "one-to-one", callable or None, default="one-to-one"
        Determines the list of feature names that will be returned
        by the `get_feature_names_out` method.

        - If None: The `get_feature_names_out` method is not defined.
        - If "one-to-one": The output feature names will be equal to
          the input feature names.
        - If callable: Function that takes positional arguments self
          and a sequence of input feature names. It must return a
          sequence of output feature names.

    Returns
    -------
    Estimator
        Object with wrapped fit method.

    """

    def wrap_fit(f: Callable) -> Callable:

        @wraps(f)
        def wrapper(self, *args, **kwargs):
            out = f(self, *args, **kwargs)

            # For sktime estimators, we are interested in y, not X
            X = args[0] if len(args) > 0 else kwargs.get("X")

            # We add the attributes and methods after running fit
            # to avoid deleting them with .reset() calls
            if X is not None:
                if not hasattr(self, "feature_names_in_"):
                    BaseEstimator._check_feature_names(self, X, reset=True)
                if not hasattr(self, "n_features_in_"):
                    BaseEstimator._check_n_features(self, X, reset=True)

                if hasattr(self, "transform") and not hasattr(self, "get_feature_names_out"):
                    if feature_names_out == "one-to-one":
                        self.get_feature_names_out = FMixin.get_feature_names_out.__get__(self)
                    elif callable(feature_names_out):
                        self.get_feature_names_out = feature_names_out.__get__(self)

            return out

        # Avoid double wrapping
        if getattr(f, "_fit_wrapped", False):
            return f
        else:
            wrapper._fit_wrapped = True

        return wrapper

    if not obj.__module__.startswith(("atom.", "sklearn.", "imblearn.")):
        if isinstance(obj, type) and hasattr(obj, "fit"):
            obj.fit = wrap_fit(obj.fit)
        elif hasattr(obj.__class__, "fit"):
            obj.fit = wrap_fit(obj.__class__.fit).__get__(obj)  # type: ignore[method-assign]

    return obj


# Custom scorers =================================================== >>

def true_negatives(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Int:
    """Outcome where the model correctly predicts the negative class."""
    return confusion_matrix(y_true, y_pred).ravel()[0]


def false_positives(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Int:
    """Outcome where the model wrongly predicts the negative class."""
    return confusion_matrix(y_true, y_pred).ravel()[1]


def false_negatives(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Int:
    """Outcome where the model wrongly predicts the negative class."""
    return confusion_matrix(y_true, y_pred).ravel()[2]


def true_positives(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Int:
    """Outcome where the model correctly predicts the positive class."""
    return confusion_matrix(y_true, y_pred).ravel()[3]


def false_positive_rate(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Float:
    """Probability that an actual negative tests positive."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def true_positive_rate(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Float:
    """Probability that an actual positive tests positive (sensitivity)."""
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)


def true_negative_rate(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Float:
    """Probability that an actual negative tests negative (specificity)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def false_negative_rate(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Float:
    """Probability that an actual positive tests negative."""
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp)
